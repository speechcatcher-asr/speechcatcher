"""Scorer modules for beam search."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ScorerInterface(ABC):
    """Base interface for beam search scorers."""

    @abstractmethod
    def score(
        self,
        yseq: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Score a single hypothesis.

        Args:
            yseq: Token sequence (seq_len,)
            state: Previous state
            x: Encoder output (enc_len, dim)

        Returns:
            Tuple of (log_probs, new_state)
                - log_probs: (vocab_size,)
                - new_state: Updated state
        """
        pass

    @abstractmethod
    def batch_score(
        self,
        yseqs: torch.Tensor,
        states: List[Optional[List[torch.Tensor]]],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Optional[List[torch.Tensor]]]]:
        """Score a batch of hypotheses.

        Args:
            yseqs: Token sequences (batch, seq_len)
            states: List of states for each hypothesis
            xs: Encoder outputs (batch, enc_len, dim)

        Returns:
            Tuple of (log_probs, new_states)
                - log_probs: (batch, vocab_size)
                - new_states: List of updated states
        """
        pass


class DecoderScorer(ScorerInterface):
    """Attention decoder scorer (wraps TransformerDecoder).

    Args:
        decoder: Decoder module with score/batch_score methods
        sos_id: Start-of-sentence token ID
        eos_id: End-of-sentence token ID
    """

    def __init__(self, decoder: nn.Module, sos_id: int = 1, eos_id: int = 2):
        self.decoder = decoder
        self.sos_id = sos_id
        self.eos_id = eos_id

    def score(
        self,
        yseq: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Score using decoder."""
        return self.decoder.score(yseq, state, x)

    def batch_score(
        self,
        yseqs: torch.Tensor,
        states: List[Optional[List[torch.Tensor]]],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Optional[List[torch.Tensor]]]]:
        """Batch score using decoder."""
        return self.decoder.batch_score(yseqs, states, xs)


class CTCPrefixScorer(ScorerInterface):
    """CTC prefix scorer with full forward algorithm.

    This implements the complete CTC prefix scoring algorithm from:
    WATANABE et al. "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION"

    Following ESPnet's CTCPrefixScoreTH implementation with:
    - Full probability matrix storage across all time steps
    - Forward algorithm with r^n and r^b variables
    - extend_prob() for streaming accumulation
    - extend_state() for hypothesis state extension

    Args:
        ctc: CTC module with ctc_lo() method
        blank_id: Blank token ID (default: 0)
        eos_id: End-of-sentence token ID
        margin: Windowing margin for attention-based pruning (0=disabled)
    """

    def __init__(self, ctc: nn.Module, blank_id: int = 0, eos_id: int = 2, margin: int = 0):
        self.ctc = ctc
        self.blank_id = blank_id
        self.eos_id = eos_id
        self.margin = margin

        # Full CTC prefix scorer (created per batch with batch_init_state)
        self.impl = None

    def batch_init_state(self, x: torch.Tensor) -> None:
        """Initialize CTC scorer with encoder output.

        This must be called before scoring to set up the probability matrix.
        In streaming mode, call extend_prob() to add new encoder blocks.

        Args:
            x: Encoder output (batch, enc_len, enc_dim)
        """
        from speechcatcher.beam_search.ctc_prefix_score_full import CTCPrefixScoreTH

        batch_size = x.size(0)
        enc_len = x.size(1)

        # Get CTC log probabilities for entire sequence
        with torch.no_grad():
            logits = self.ctc.ctc_lo(x)  # (batch, enc_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

        # Create sequence lengths (assume all frames are valid initially)
        xlens = torch.full((batch_size,), enc_len, dtype=torch.long, device=x.device)

        # Initialize CTCPrefixScoreTH with full probability matrix
        self.impl = CTCPrefixScoreTH(
            x=log_probs,
            xlens=xlens,
            blank=self.blank_id,
            eos=self.eos_id,
            margin=self.margin
        )

    def batch_score(
        self,
        yseqs: torch.Tensor,
        states: List[Optional[Tuple]],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple]]]:
        """Batch score prefixes using full CTC forward algorithm.

        This implements the complete CTC prefix scoring with forward variables
        r^n and r^b maintained across all time steps.

        Args:
            yseqs: Token sequences (batch, seq_len) as torch.Tensor
            states: List of CTC states [(r, log_psi, f_min, f_max), ...]
            xs: Encoder outputs (batch, enc_len, dim)

        Returns:
            Tuple of (log_probs, new_states)
                - log_probs: (batch, vocab_size) - incremental CTC scores
                - new_states: List of updated states per hypothesis
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"CTC batch_score: yseqs shape={yseqs.shape}, xs shape={xs.shape}, n_states={len(states)}")

        # Initialize scorer if needed (first call or new encoder block)
        if self.impl is None:
            logger.debug("CTC: Initializing impl with batch_init_state")
            self.batch_init_state(xs)
            logger.debug("CTC: batch_init_state complete")

        # Convert yseqs from (batch, seq_len) to List[torch.Tensor]
        # Each element is a 1D tensor of token IDs
        y_list = [yseqs[i] for i in range(yseqs.size(0))]

        # PROPER STATE BATCHING: Merge list of states into batched state
        # After select_state(), each state is (r, s, f_min, f_max) [4 elements]
        # We need to stack them into (r_batched, s_batched, f_min, f_max) for scoring
        merged_state = None
        if states and states[0] is not None:
            try:
                # Check if states are individual (4 elements) or already batched (5 elements)
                if len(states[0]) == 4:
                    # Individual states from select_state: (r, s, f_min, f_max)
                    # Stack them along hypothesis dimension
                    logger.debug(f"CTC: Batching {len(states)} individual states")

                    # Check if state is compatible with current input_length
                    state_T = states[0][0].shape[0]
                    if state_T != self.impl.input_length:
                        logger.warning(f"CTC: State time dimension {state_T} != current input_length {self.impl.input_length}, resetting state")
                        merged_state = None
                    else:
                        merged_state = (
                            torch.stack([s[0] for s in states], dim=2),  # r: (T, 2, n_bh)
                            torch.stack([s[1] for s in states]),          # s: (n_bh, vocab)
                            states[0][2],  # f_min (shared across hypotheses)
                            states[0][3],  # f_max (shared across hypotheses)
                        )
                        logger.debug(f"CTC: Batched state r shape: {merged_state[0].shape}")
                else:
                    # Already batched (5 elements) - shouldn't happen with proper select_state
                    logger.warning(f"CTC: Got batched state with {len(states[0])} elements, expected 4")
                    merged_state = states[0]
            except Exception as e:
                logger.error(f"CTC: Error batching states: {e}, resetting to None")
                logger.error(f"CTC: states[0] type: {type(states[0])}, len: {len(states[0]) if states[0] else 'None'}")
                merged_state = None

        # Call CTCPrefixScoreTH forward algorithm
        # Returns: (batch, vocab_size) scores and (r, log_psi, f_min, f_max, scoring_idmap)
        logger.debug(f"CTC: Calling impl.__call__ with {len(y_list)} sequences")
        scores, new_state = self.impl(
            y=y_list,
            state=merged_state,
            scoring_ids=None,  # Score full vocabulary (can optimize later)
            att_w=None  # No attention-based windowing (can add later)
        )
        logger.debug(f"CTC: impl.__call__ complete, scores shape={scores.shape}")

        # Unmerge state: return same state for all hypotheses
        # In beam search, we'll select the appropriate states later
        new_states = [new_state for _ in range(len(states))]

        return scores, new_states

    def batch_score_partial(
        self,
        yseqs: torch.Tensor,
        ids: torch.Tensor,
        states: List[Optional[Tuple]],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple]]]:
        """Batch score only top-K candidate tokens (partial scoring).

        This is the KEY OPTIMIZATION that makes CTC feasible!
        Instead of computing forward algorithm for all 1024 tokens, we only
        compute it for top-K (e.g., 40) selected by the decoder.
        This gives ~25x speedup in the forward algorithm computation.

        NOTE: Returns FULL vocabulary scores (batch, vocab_size), but non-selected
        tokens have -inf scores. This allows combining with other scorers.

        Following ESPnet's implementation in:
        espnet_streaming_decoder/espnet/nets/scorers/ctc.py:101-126

        Args:
            yseqs: Token sequences (batch, seq_len) as torch.Tensor
            ids: Top-K candidate token IDs to score (batch, K)
            states: List of CTC states [(r, log_psi, f_min, f_max), ...]
            xs: Encoder outputs (batch, enc_len, dim)

        Returns:
            Tuple of (log_probs, new_states)
                - log_probs: (batch, vocab_size) - FULL vocab, but only K have real scores
                - new_states: List of updated states per hypothesis
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"CTC batch_score_partial: yseqs shape={yseqs.shape}, ids shape={ids.shape}, n_states={len(states)}")

        # Initialize scorer if needed (first call or new encoder block)
        if self.impl is None:
            logger.debug("CTC: Initializing impl with batch_init_state")
            self.batch_init_state(xs)
            logger.debug("CTC: batch_init_state complete")

        # Convert yseqs from (batch, seq_len) to List[torch.Tensor]
        y_list = [yseqs[i] for i in range(yseqs.size(0))]

        # PROPER STATE BATCHING: Same as batch_score
        merged_state = None
        if states and states[0] is not None:
            try:
                if len(states[0]) == 4:
                    # Individual states from select_state: (r, s, f_min, f_max)
                    logger.debug(f"CTC: Batching {len(states)} individual states (partial)")

                    # Check if state is compatible with current input_length
                    state_T = states[0][0].shape[0]
                    if state_T != self.impl.input_length:
                        logger.warning(f"CTC: State time dimension {state_T} != current input_length {self.impl.input_length}, resetting state")
                        merged_state = None
                    else:
                        merged_state = (
                            torch.stack([s[0] for s in states], dim=2),  # r: (T, 2, n_bh)
                            torch.stack([s[1] for s in states]),          # s: (n_bh, vocab)
                            states[0][2],  # f_min
                            states[0][3],  # f_max
                        )
                        logger.debug(f"CTC: Batched state r shape: {merged_state[0].shape}")
                else:
                    logger.warning(f"CTC: Got batched state with {len(states[0])} elements in partial scoring")
                    merged_state = states[0]
            except Exception as e:
                logger.error(f"CTC: Error batching states in partial: {e}")
                merged_state = None

        # Call CTCPrefixScoreTH with PARTIAL SCORING (scoring_ids)
        # This is THE KEY OPTIMIZATION - score only top-K tokens!
        logger.debug(f"CTC: Calling impl.__call__ with partial scoring, ids shape={ids.shape}")
        scores, new_state = self.impl(
            y=y_list,
            state=merged_state,
            scoring_ids=ids,  # ← THE FIX! Pass top-K ids for partial scoring
            att_w=None
        )
        logger.debug(f"CTC: impl.__call__ complete (partial), scores shape={scores.shape}")

        # Unmerge state
        new_states = [new_state for _ in range(len(states))]

        return scores, new_states

    def extend_prob(self, x: torch.Tensor) -> None:
        """Extend CTC probability matrix with new encoder output.

        This is called in streaming mode when a new encoder block arrives.

        TEMPORARY IMPLEMENTATION: Just reinitialize with new encoder output
        instead of extending. This is less efficient but avoids complexity
        of proper state accumulation.

        Args:
            x: New encoder output (batch, enc_len_new, enc_dim)
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"CTC extend_prob: Reinitializing with encoder output shape {x.shape}")

        # TEMPORARY: Reinitialize instead of extending
        # TODO: Implement proper extend_prob that accumulates frames
        self.impl = None  # Force reinitialization in batch_score

    def extend_state(self, state: Optional[Tuple]) -> Optional[Tuple]:
        """Extend forward variables when probability matrix grows.

        When extend_prob() adds new time steps, we need to extend
        the forward variables r to cover those new time steps.

        Args:
            state: (r, log_psi, f_min, f_max, scoring_idmap) or None

        Returns:
            Extended state with r covering new time length
        """
        if self.impl is None or state is None:
            return state

        return self.impl.extend_state(state)

    def select_state(self, state: Optional[Tuple], i: int, new_id: int = None) -> Optional[Tuple]:
        """Select state for specific hypothesis and token.

        This is THE CRITICAL METHOD for proper CTC scoring!
        It extracts the forward variables for one hypothesis extending with one token.

        Following ESPnet's implementation in:
        espnet_streaming_decoder/espnet/nets/scorers/ctc.py:40-63

        Args:
            state: Batched CTC state (r, log_psi, f_min, f_max, scoring_idmap)
                   or None for initial state
            i: Hypothesis index in batch (which hypothesis to select)
            new_id: Token ID being added to this hypothesis (which token)

        Returns:
            Selected state (r_selected, s, f_min, f_max) [4 elements]
            - r_selected: (T, 2) forward variables for THIS hypothesis+token
            - s: (vocab_size,) prefix score for this hypothesis
            - f_min, f_max: frame window bounds
        """
        if state is None:
            return None

        # Unpack batched state (5 elements)
        if len(state) == 4:
            # Already selected state, just return it
            return state

        r, log_psi, f_min, f_max, scoring_idmap = state

        # Select hypothesis i's score for token new_id
        # Expand to full vocab size for compatibility
        s = log_psi[i, new_id].expand(log_psi.size(1))

        # Select forward variables for hypothesis i, token new_id
        if scoring_idmap is not None:
            # Partial scoring: map new_id to its index in the scoring subset
            token_idx = scoring_idmap[i, new_id]
            if token_idx >= 0:
                r_selected = r[:, :, i, token_idx]  # (T, 2)
            else:
                # Token not in scoring subset, use blank's forward variables
                r_selected = r[:, :, i, 0]  # (T, 2)
        else:
            # Full vocabulary scoring
            r_selected = r[:, :, i, new_id]  # (T, 2)

        # Return 4-element tuple for individual hypothesis
        return (r_selected, s, f_min, f_max)

    def score(
        self,
        yseq: torch.Tensor,
        state: Optional[Tuple],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Score prefix using CTC (single hypothesis).

        This is a wrapper around batch_score for single-hypothesis scoring.

        Args:
            yseq: Token sequence (seq_len,)
            state: CTC state tuple or None
            x: Encoder output (enc_len, dim)

        Returns:
            Tuple of (log_probs, new_state)
        """
        # Convert to batch format
        yseqs = yseq.unsqueeze(0)  # (1, seq_len)
        xs = x.unsqueeze(0)  # (1, enc_len, dim)
        states = [state]

        # Call batch score
        scores, new_states = self.batch_score(yseqs, states, xs)

        # Extract single result
        return scores[0], new_states[0]


class LengthBonus:
    """Length bonus scorer to encourage longer sequences.

    Args:
        weight: Length bonus weight
    """

    def __init__(self, weight: float = 0.0):
        self.weight = weight

    def score(self, yseq: torch.Tensor) -> float:
        """Compute length bonus for sequence.

        Args:
            yseq: Token sequence (seq_len,)

        Returns:
            Length bonus score
        """
        return self.weight * len(yseq)

    def batch_score(self, yseqs: torch.Tensor) -> torch.Tensor:
        """Compute length bonus for batch of sequences.

        Args:
            yseqs: Token sequences (batch, seq_len)

        Returns:
            Length bonuses (batch,)
        """
        lengths = (yseqs != 0).sum(dim=1).float()  # Assuming 0 is padding
        return self.weight * lengths


class CoverageScorer:
    """Coverage scorer to avoid repetition (optional).

    This is a placeholder for more advanced coverage scoring.

    Args:
        weight: Coverage penalty weight
    """

    def __init__(self, weight: float = 0.0):
        self.weight = weight

    def score(self, yseq: torch.Tensor) -> float:
        """Compute coverage score (penalty for repetition).

        Args:
            yseq: Token sequence (seq_len,)

        Returns:
            Coverage score (negative penalty)
        """
        if len(yseq) <= 1:
            return 0.0

        # Simple penalty for repeated tokens
        unique_tokens = len(set(yseq.tolist()))
        total_tokens = len(yseq)
        repetition_ratio = 1.0 - (unique_tokens / total_tokens)

        return -self.weight * repetition_ratio


class CompositeScorer:
    """Composite scorer that combines multiple scorers.

    Args:
        scorers: Dictionary of scorers {name: (scorer, weight)}
    """

    def __init__(self, scorers: dict):
        self.scorers = scorers

    def batch_score(
        self,
        yseqs: torch.Tensor,
        states: dict,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Score batch using all scorers.

        Args:
            yseqs: Token sequences (batch, seq_len)
            states: Dictionary of states {scorer_name: state}
            xs: Encoder outputs (batch, enc_len, dim)

        Returns:
            Tuple of (combined_scores, new_states)
                - combined_scores: (batch, vocab_size)
                - new_states: Dictionary of updated states
        """
        batch_size = yseqs.size(0)
        vocab_size = None
        combined_scores = None
        new_states = {}

        for name, (scorer, weight) in self.scorers.items():
            if weight == 0.0:
                continue

            state = states.get(name)

            # Score based on scorer type
            if hasattr(scorer, "batch_score"):
                if isinstance(state, dict):
                    state = state.get("states")

                scores, new_state = scorer.batch_score(yseqs, state, xs)
                new_states[name] = new_state

                # Initialize combined scores
                if combined_scores is None:
                    vocab_size = scores.size(-1)
                    combined_scores = torch.zeros(batch_size, vocab_size, device=scores.device)

                combined_scores += weight * scores

        return combined_scores, new_states
