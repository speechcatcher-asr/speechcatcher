"""CTC Prefix Scoring for beam search.

Based on "First-Pass Large Vocabulary Continuous Speech Recognition using
Bi-Directional Recurrent DNNs" (Graves 2012) and ESPnet implementation.

The algorithm maintains prefix probabilities incrementally by tracking:
- r[y]: log probability of prefix y ending with a non-blank token
- log_psi[y]: log probability of prefix y ending with blank

This allows efficient O(T * B * V) beam search instead of O(T * B^2 * V).
"""

import numpy as np
import torch
from typing import List, Optional, Tuple


def log_add(a: float, b: float) -> float:
    """Compute log(exp(a) + exp(b)) in numerically stable way.

    Args:
        a: First log value
        b: Second log value

    Returns:
        log(exp(a) + exp(b))
    """
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a

    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))


class CTCPrefixScore:
    """CTC prefix scorer for beam search.

    This implements incremental CTC prefix probability computation.
    For each prefix, we track:
    - r: log prob of ending with non-blank
    - log_psi: log prob of ending with blank

    Args:
        x: CTC log probabilities (T, vocab_size) where T is time, vocab_size includes blank
        blank_id: ID of blank token (usually 0)
        eos_id: ID of end-of-sentence token
        space_id: ID of space token (optional)
    """

    def __init__(
        self,
        x: np.ndarray,
        blank_id: int = 0,
        eos_id: int = 2,
        space_id: Optional[int] = None,
    ):
        """Initialize CTC prefix scorer.

        Args:
            x: CTC log probabilities (T, vocab_size)
            blank_id: Blank token ID
            eos_id: End-of-sentence token ID
            space_id: Space token ID (optional)
        """
        self.x = x  # (T, vocab_size) log probabilities
        self.blank_id = blank_id
        self.eos_id = eos_id
        self.space_id = space_id
        self.vocab_size = x.shape[1]

        # Current time step being processed
        self.t = 0

    def initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial CTC state.

        Returns:
            Tuple of (r, log_psi) where:
            - r: (vocab_size,) log prob of ending with each non-blank token
            - log_psi: (vocab_size,) log prob of ending with blank after each token
        """
        # Initialize: empty prefix has prob 1.0 (log prob 0.0) of ending with blank
        r = np.full(self.vocab_size, -np.inf, dtype=np.float32)
        log_psi = np.full(self.vocab_size, -np.inf, dtype=np.float32)

        # Empty prefix: prob 1.0 of being in blank state
        r[self.blank_id] = -np.inf  # Can't end with blank as non-blank token
        log_psi[self.blank_id] = 0.0  # Empty prefix is in blank state

        return r, log_psi

    def __call__(
        self,
        y: torch.Tensor,
        cs: torch.Tensor,
        state: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[torch.Tensor, Tuple[np.ndarray, np.ndarray]]:
        """Score next tokens for given prefix.

        Args:
            y: Prefix sequence (prefix_len,) - token IDs
            cs: Candidate next tokens (n_candidates,) to score
            state: Current CTC state (r, log_psi) or None

        Returns:
            Tuple of:
            - scores: (n_candidates,) log probabilities for each candidate
            - new_state: Updated (r, log_psi) after consuming prefix y
        """
        # Convert to numpy
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if isinstance(cs, torch.Tensor):
            cs = cs.cpu().numpy()

        # Get current state
        if state is None:
            r_prev, log_psi_prev = self.initial_state()
        else:
            r_prev, log_psi_prev = state

        # Get prefix length (excluding SOS if present)
        prefix_len = len(y)
        if prefix_len > 0 and y[0] == 1:  # SOS token
            prefix_len -= 1

        # Forward to current time step (one step per prefix token)
        # In streaming, we process incrementally
        t = min(self.t + 1, self.x.shape[0] - 1)
        self.t = t

        # Get log probs at time t
        log_probs = self.x[t]  # (vocab_size,)

        # Compute scores for each candidate
        scores = np.full(len(cs), -np.inf, dtype=np.float32)

        # Last token in prefix (or blank if empty)
        last_token = y[-1] if len(y) > 0 else self.blank_id

        for i, c in enumerate(cs):
            c = int(c)

            if c == self.blank_id:
                # Extending with blank: both r and log_psi contribute
                # P(y + blank) = (P_nb(y) + P_b(y)) * p(blank|t)
                log_psi_c = log_add(r_prev[self.blank_id], log_psi_prev[self.blank_id])
                scores[i] = log_psi_c + log_probs[self.blank_id]

            elif c == last_token:
                # Extending with same token: only r contributes (forced blank between)
                # P(y + c) = P_nb(y) * p(c|t)
                scores[i] = r_prev[c] + log_probs[c]

            else:
                # Extending with different token: both contribute
                # P(y + c) = (P_nb(y) + P_b(y)) * p(c|t)
                r_c = log_add(r_prev[self.blank_id], log_psi_prev[self.blank_id])
                scores[i] = r_c + log_probs[c]

        # Update state for this prefix
        # New state after consuming current prefix
        new_r = np.full(self.vocab_size, -np.inf, dtype=np.float32)
        new_log_psi = np.full(self.vocab_size, -np.inf, dtype=np.float32)

        # Compute new states for all possible next tokens
        for c in range(self.vocab_size):
            if c == self.blank_id:
                new_log_psi[c] = log_add(r_prev[self.blank_id], log_psi_prev[self.blank_id]) + log_probs[self.blank_id]
            elif c == last_token:
                new_r[c] = r_prev[c] + log_probs[c]
                new_log_psi[c] = log_psi_prev[c] + log_probs[self.blank_id]
            else:
                new_r[c] = log_add(r_prev[self.blank_id], log_psi_prev[self.blank_id]) + log_probs[c]
                new_log_psi[c] = log_add(r_prev[c], log_psi_prev[c]) + log_probs[self.blank_id]

        new_state = (new_r, new_log_psi)

        # Convert scores to torch
        scores_tensor = torch.from_numpy(scores).float()

        return scores_tensor, new_state


class CTCPrefixScoreTH:
    """Torch-based batched CTC prefix scorer.

    This is a more efficient torch-based implementation for batched scoring.

    Args:
        x: CTC log probabilities (1, T, vocab_size)
        xlens: Sequence lengths (1,)
        blank_id: Blank token ID
        eos_id: End-of-sentence token ID
    """

    def __init__(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        blank_id: int = 0,
        eos_id: int = 2,
    ):
        """Initialize batched CTC prefix scorer."""
        self.x = x  # (1, T, vocab_size)
        self.xlens = xlens
        self.blank_id = blank_id
        self.eos_id = eos_id
        self.vocab_size = x.shape[2]
        self.device = x.device

        # Time tracking
        self.t = 0
        self.xlen = xlens[0].item()

    def initial_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial state for empty prefix.

        Returns:
            Tuple of (r, log_psi):
            - r: (vocab_size,) non-blank ending probs
            - log_psi: (vocab_size,) blank ending probs
        """
        r = torch.full((self.vocab_size,), -float('inf'), device=self.device)
        log_psi = torch.full((self.vocab_size,), -float('inf'), device=self.device)

        # Empty prefix is in blank state
        log_psi[self.blank_id] = 0.0

        return r, log_psi

    def __call__(
        self,
        y: torch.Tensor,
        state: Optional[Tuple],
        cs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """Batch score next tokens.

        Args:
            y: Batch of prefix sequences (batch, max_prefix_len)
            state: Batch state (r, log_psi, frame_start, frame_end) or None
            cs: Candidate tokens (batch, n_candidates) or None (score all)

        Returns:
            Tuple of (scores, new_state):
            - scores: (batch, vocab_size) or (batch, n_candidates)
            - new_state: Updated state
        """
        batch_size = y.shape[0]

        # Initialize state if needed
        if state is None:
            # Initial state for empty prefixes
            r = torch.full((batch_size, self.vocab_size), -float('inf'), device=self.device)
            log_psi = torch.full((batch_size, self.vocab_size), -float('inf'), device=self.device)
            log_psi[:, self.blank_id] = 0.0

            frame_start = 0
            frame_end = 0
        else:
            r, log_psi, frame_start, frame_end = state

        # Advance time (one step per forward call)
        t = min(frame_end + 1, self.xlen - 1)

        # Get log probs at time t
        log_probs = self.x[0, t, :]  # (vocab_size,)

        # Get last token in each prefix
        last_tokens = torch.full((batch_size,), self.blank_id, dtype=torch.long, device=self.device)
        for b in range(batch_size):
            # Find last non-padding token
            valid_mask = y[b] != 0  # Assuming 0 is padding
            if valid_mask.any():
                last_pos = valid_mask.nonzero(as_tuple=True)[0][-1]
                last_tokens[b] = y[b, last_pos]

        # Compute scores for all tokens
        # Shape: (batch, vocab_size)
        scores = torch.full((batch_size, self.vocab_size), -float('inf'), device=self.device)

        # For blank: both r and log_psi contribute
        combined = torch.logaddexp(r[:, self.blank_id], log_psi[:, self.blank_id])
        scores[:, self.blank_id] = combined + log_probs[self.blank_id]

        # For non-blank tokens
        for c in range(self.vocab_size):
            if c == self.blank_id:
                continue

            # Check if c == last_token for each batch item
            same_as_last = (last_tokens == c)
            diff_from_last = ~same_as_last

            # Same as last: only r contributes
            scores[same_as_last, c] = r[same_as_last, c] + log_probs[c]

            # Different from last: both contribute
            combined_diff = torch.logaddexp(
                r[diff_from_last, self.blank_id],
                log_psi[diff_from_last, self.blank_id]
            )
            scores[diff_from_last, c] = combined_diff + log_probs[c]

        # Update state
        new_r = torch.full((batch_size, self.vocab_size), -float('inf'), device=self.device)
        new_log_psi = torch.full((batch_size, self.vocab_size), -float('inf'), device=self.device)

        # Update for all tokens
        new_log_psi[:, self.blank_id] = torch.logaddexp(
            r[:, self.blank_id], log_psi[:, self.blank_id]
        ) + log_probs[self.blank_id]

        for c in range(self.vocab_size):
            if c == self.blank_id:
                continue

            same_as_last = (last_tokens == c)
            diff_from_last = ~same_as_last

            # Update r
            new_r[same_as_last, c] = r[same_as_last, c] + log_probs[c]
            combined_diff = torch.logaddexp(
                r[diff_from_last, self.blank_id],
                log_psi[diff_from_last, self.blank_id]
            )
            new_r[diff_from_last, c] = combined_diff + log_probs[c]

            # Update log_psi
            new_log_psi[:, c] = torch.logaddexp(r[:, c], log_psi[:, c]) + log_probs[self.blank_id]

        new_state = (new_r, new_log_psi, frame_start, t)

        # If cs is provided, select only those scores
        if cs is not None:
            # cs: (batch, n_candidates)
            batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
            scores = scores[batch_idx, cs]

        return scores, new_state
