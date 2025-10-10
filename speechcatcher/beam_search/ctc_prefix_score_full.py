"""Full CTC Prefix Scoring Implementation.

Based on ESPnet's CTCPrefixScoreTH implementation, following Algorithm 2 in:
WATANABE et al. "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION"

This is the complete, mathematically correct implementation with:
- Full probability matrix storage (2, T, B, O)
- Forward algorithm over all time steps
- extend_prob() for streaming accumulation
- extend_state() for hypothesis state extension
- Proper numerical stability with logzero and logsumexp
"""

import torch
from typing import Optional, Tuple, List, Any


class CTCPrefixScoreTH:
    """Batch processing of CTC prefix scoring with streaming support.

    This implements the full forward algorithm for CTC prefix scoring,
    maintaining forward variables r^n and r^b across all time steps.

    Extended for streaming by accumulating probability matrices across blocks
    via extend_prob() and extending forward variables via extend_state().

    Args:
        x: CTC log probabilities (B, T, O) where O includes blank
        xlens: Sequence lengths (B,)
        blank: Blank token ID (usually 0)
        eos: End-of-sequence token ID
        margin: Windowing margin for attention-based pruning (0=disabled)
    """

    def __init__(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        blank: int,
        eos: int,
        margin: int = 0
    ):
        """Initialize CTC prefix scorer with probability matrix."""
        # Use very negative number instead of -inf to avoid NaN issues
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos

        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.dtype = x.dtype
        self.device = x.device

        # Pad probabilities beyond valid lengths
        # This ensures we don't use garbage values for varying-length sequences
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0  # log(P(blank)) = log(1) = 0

        # Store full probability matrix: (2, T, B, O)
        # Dimension 0: [0] = non-blank emissions, [1] = blank emissions
        # This duplication optimizes the forward algorithm computation
        xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x = torch.stack([xn, xb])  # (2, T, B, O)

        self.end_frames = torch.as_tensor(xlens, device=self.device) - 1

        # Windowing for attention-based pruning (optional)
        self.margin = margin
        if margin > 0:
            self.frame_ids = torch.arange(
                self.input_length, dtype=self.dtype, device=self.device
            )

        # Pre-computed indices for efficient batch operations
        self.idx_bh = None  # Batch-hypothesis indices (allocated on first use)
        self.idx_b = torch.arange(self.batch, device=self.device)
        self.idx_bo = (self.idx_b * self.odim).unsqueeze(1)

    def __call__(
        self,
        y: List[torch.Tensor],
        state: Optional[Tuple],
        scoring_ids: Optional[torch.Tensor] = None,
        att_w: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """Compute CTC prefix scores for next tokens.

        This is the core forward algorithm that computes:
        P(y·c | X) for each candidate token c

        Args:
            y: List of prefix sequences, each (prefix_len,), batch*hyps sequences
            state: Previous state (r_prev, s_prev, f_min, f_max) or None
            scoring_ids: (batch*hyps, n_candidates) for partial scoring, or None for full vocab
            att_w: Attention weights (batch*hyps, T) for windowing, or None

        Returns:
            Tuple of:
            - log_psi - s_prev: (batch*hyps, vocab) incremental prefix scores
            - new_state: (r, log_psi, f_min, f_max, scoring_idmap)
        """
        # Extract prefix information
        output_length = len(y[0]) - 1  # Ignore SOS token
        last_ids = [yi[-1].item() if isinstance(yi, torch.Tensor) else yi[-1] for yi in y]
        n_bh = len(last_ids)  # Total number of hypotheses (batch * beam_width)
        n_hyps = n_bh // self.batch  # Hypotheses per batch item

        # Determine scoring vocabulary size
        self.scoring_num = scoring_ids.size(-1) if scoring_ids is not None else 0

        # Initialize or restore previous state
        if state is None:
            # Initial state: r^b cumulates blank probabilities
            r_prev = torch.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype,
                device=self.device
            )
            # r^b(empty) = cumulative blank probs: sum_{t=0}^T log P(blank|t)
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, n_bh)
            s_prev = 0.0  # No previous score
            f_min_prev = 0
            f_max_prev = 1
        else:
            # State can be 4 or 5 elements depending on source
            if len(state) == 4:
                # From batched individual states (after select_state): (r, s, f_min, f_max)
                r_prev, s_prev, f_min_prev, f_max_prev = state
            elif len(state) == 5:
                # From previous __call__: (r, log_psi, f_min, f_max, scoring_idmap)
                r_prev, s_prev, f_min_prev, f_max_prev, _ = state
            else:
                raise ValueError(f"Expected state with 4 or 5 elements, got {len(state)}")

        # Select scoring dimensions (partial scoring for efficiency)
        if self.scoring_num > 0:
            # Partial scoring: only score top-K candidates
            scoring_idmap = torch.full(
                (n_bh, self.odim), -1, dtype=torch.long, device=self.device
            )
            snum = self.scoring_num

            # Allocate batch-hypothesis indices if needed
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = torch.arange(n_bh, device=self.device).view(-1, 1)

            # Map scoring_ids to consecutive indices 0..snum-1
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = torch.arange(
                snum, device=self.device
            )

            # Extract only the probabilities we need to score
            scoring_idx = (
                scoring_ids + self.idx_bo.repeat(1, n_hyps).view(-1, 1)
            ).view(-1)
            x_ = torch.index_select(
                self.x.view(2, -1, self.batch * self.odim), 2, scoring_idx
            ).view(2, -1, n_bh, snum)
        else:
            # Full scoring: score entire vocabulary
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            # Expand x for all hypotheses: (2, T, B, O) -> (2, T, B*H, O)
            x_ = self.x.unsqueeze(3).repeat(1, 1, 1, n_hyps, 1).view(2, -1, n_bh, snum)

        # Allocate forward variables: r[t, {n,b}, hypothesis, token]
        r = torch.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero,
            dtype=self.dtype,
            device=self.device
        )

        # Initialize at t=0 for empty prefix
        if output_length == 0:
            r[0, 0] = x_[0, 0]  # r^n(empty) = P(token | x_0)

        # Compute r_sum = log(r^n + r^b) for all previous time steps
        # This is the total probability of the prefix
        r_sum = torch.logsumexp(r_prev, 1)  # (T, n_bh)

        # Setup log_phi: transition probabilities for extending prefix
        # log_phi[t, h, c] = log P(prefix_h at time t before emitting c)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)  # (T, n_bh, snum)

        # Special case: extending with same token as last token
        # Must go through blank: only r^b contributes
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # Determine time window based on attention (optional)
        if att_w is not None and self.margin > 0:
            # Use attention to focus CTC forward pass on relevant frames
            f_arg = torch.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min().cpu()), f_min_prev)
            f_max = max(int(f_arg.max().cpu()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            # No windowing: process all frames
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # ============================================
        # FORWARD ALGORITHM (THE CORE CTC COMPUTATION)
        # ============================================
        # Compute r^n[t] and r^b[t] for all t in [start, end)
        for t in range(start, end):
            rp = r[t - 1]  # Previous forward variables: (2, n_bh, snum)

            # Transition matrix: what states can lead to current states?
            # r^n[t] can come from: r^n[t-1] (same token) or log_phi[t-1] (new token)
            # r^b[t] can come from: r^n[t-1] (emit blank) or r^b[t-1] (stay blank)
            rr = torch.stack([
                rp[0],           # r^n[t-1] -> r^n[t] (continuing same token)
                log_phi[t - 1],  # any state -> r^n[t] (new token)
                rp[0],           # r^n[t-1] -> r^b[t] (emit blank after token)
                rp[1]            # r^b[t-1] -> r^b[t] (stay in blank)
            ]).view(2, 2, n_bh, snum)

            # Forward update:
            # r[t] = logsumexp(transitions) + emission_prob
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]
            # r[t, 0] = log(exp(rp[0]) + exp(log_phi[t-1])) + x[0, t]  # r^n
            # r[t, 1] = log(exp(rp[0]) + exp(rp[1])) + x[1, t]          # r^b

        # ============================================
        # COMPUTE PREFIX PROBABILITIES
        # ============================================
        # log_psi[h, c] = log P(prefix_h · c | X)

        # log_phi_x = log_phi + emission = prob of emitting c at each time
        log_phi_x = torch.cat((
            log_phi[0].unsqueeze(0),
            log_phi[:-1]
        ), dim=0) + x_[0]  # (T, n_bh, snum)

        # Sum over all possible ending times
        if scoring_ids is not None:
            log_psi = torch.full(
                (n_bh, self.odim), self.logzero, dtype=self.dtype, device=self.device
            )
            log_psi_ = torch.logsumexp(
                torch.cat((
                    log_phi_x[start:end],
                    r[start - 1, 0].unsqueeze(0)
                ), dim=0),
                dim=0
            )
            # Map back to full vocabulary
            for si in range(n_bh):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = torch.logsumexp(
                torch.cat((
                    log_phi_x[start:end],
                    r[start - 1, 0].unsqueeze(0)
                ), dim=0),
                dim=0
            )  # (n_bh, snum)

        # EOS has special probability: r_sum at final frame
        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # Blank token should not be predicted (already handled internally)
        log_psi[:, self.blank] = self.logzero

        # Return incremental scores (relative to previous prefix score)
        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def extend_prob(self, x: torch.Tensor):
        """Extend CTC probability matrix with new encoder output.

        This is called when a new encoder block arrives in streaming mode.
        The probability matrix grows from T to T+T_new.

        Args:
            x: New CTC posteriors (B, T_new, O)
        """
        if self.x.shape[1] < x.shape[1]:
            # New time length is longer than current
            # Pad beyond sequence lengths
            xlens = [x.size(1)]
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0

            # Store old matrix
            tmp_x = self.x

            # Reshape new probabilities
            xn = x.transpose(0, 1)  # (T_new, B, O)
            xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
            self.x = torch.stack([xn, xb])  # (2, T_new, B, O)

            # Copy old values, new frames already set
            self.x[:, :tmp_x.shape[1], :, :] = tmp_x

            # Update metadata
            self.input_length = x.size(1)
            self.end_frames = torch.as_tensor(xlens, device=self.device) - 1

    def extend_state(self, state: Optional[Tuple]) -> Optional[Tuple]:
        """Extend forward variables when probability matrix grows.

        When extend_prob() adds new time steps, we need to extend
        the forward variables r to cover those new time steps.

        Args:
            state: (r_prev, s_prev, f_min, f_max) or None

        Returns:
            Extended state with r covering new time length
        """
        if state is None:
            return None

        # Handle both 4 and 5 element states
        if len(state) == 4:
            r_prev, s_prev, f_min_prev, f_max_prev = state
        elif len(state) == 5:
            r_prev, s_prev, f_min_prev, f_max_prev, _ = state
        else:
            raise ValueError(f"Expected state with 4 or 5 elements, got {len(state)}")

        # Allocate new r with extended time dimension
        r_prev_new = torch.full(
            (self.input_length, 2),
            self.logzero,
            dtype=self.dtype,
            device=self.device
        )

        # Copy existing forward variables
        start = max(r_prev.shape[0], 1)
        r_prev_new[0:start] = r_prev

        # Fill new time steps with cumulative blank probabilities
        # r^b[t] = r^b[t-1] + log P(blank | x_t)
        for t in range(start, self.input_length):
            r_prev_new[t, 1] = r_prev_new[t - 1, 1] + self.x[0, t, :, self.blank]

        # Return 4 elements (consistent with select_state output)
        return (r_prev_new, s_prev, f_min_prev, f_max_prev)

    def index_select_state(
        self,
        state: Tuple,
        best_ids: torch.Tensor
    ) -> Tuple:
        """Select CTC states according to beam pruning.

        When beam search prunes hypotheses, we need to select only the
        forward variables for the surviving hypotheses.

        Args:
            state: (r, s, f_min, f_max, scoring_idmap)
            best_ids: (B, W) indices of best hypotheses

        Returns:
            Pruned state with only selected hypotheses
        """
        r, s, f_min, f_max, scoring_idmap = state
        n_bh = len(s)
        n_hyps = n_bh // self.batch

        # Convert best_ids to flat indices in batch-hyp-output space
        vidx = (best_ids + (self.idx_b * (n_hyps * self.odim)).view(-1, 1)).view(-1)

        # Select hypothesis scores
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(n_bh, self.odim)

        # Select forward probabilities
        if scoring_idmap is not None:
            snum = self.scoring_num
            # Get hypothesis and label indices
            hyp_idx = (best_ids // self.odim + (self.idx_b * n_hyps).view(-1, 1)).view(-1)
            label_ids = torch.fmod(best_ids, self.odim).view(-1)
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim

        r_new = torch.index_select(
            r.view(-1, 2, n_bh * snum), 2, vidx
        ).view(-1, 2, n_bh)

        return r_new, s_new, f_min, f_max
