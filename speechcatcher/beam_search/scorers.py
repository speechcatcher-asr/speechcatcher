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
    """CTC prefix scorer with incremental computation.

    This implements efficient CTC prefix scoring for beam search by
    incrementally computing forward probabilities. This avoids the O(n²)
    complexity of recomputing from scratch at each step.

    Args:
        ctc: CTC module
        blank_id: Blank token ID (default: 0)
        eos_id: End-of-sentence token ID
    """

    def __init__(self, ctc: nn.Module, blank_id: int = 0, eos_id: int = 2):
        self.ctc = ctc
        self.blank_id = blank_id
        self.eos_id = eos_id

    def score(
        self,
        yseq: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Score prefix using CTC.

        This is a simplified version. Full implementation would use
        incremental forward algorithm.

        Args:
            yseq: Token sequence (seq_len,)
            state: CTC state (r, log_psi) where:
                - r: (vocab_size,) - non-blank ending scores
                - log_psi: (vocab_size,) - blank ending scores
            x: Encoder output (enc_len, dim)

        Returns:
            Tuple of (log_probs, new_state)
        """
        # Get CTC log probabilities
        with torch.no_grad():
            logits = self.ctc.ctc_lo(x)  # (enc_len, dim) -> (enc_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

        # For simplicity, return the mean log prob across time
        # Full implementation would use forward-backward algorithm
        mean_log_probs = log_probs.mean(dim=0)  # (vocab_size,)

        return mean_log_probs, state

    def batch_score(
        self,
        yseqs: torch.Tensor,
        states: List[Optional[List[torch.Tensor]]],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Optional[List[torch.Tensor]]]]:
        """Batch score prefixes using CTC.

        Args:
            yseqs: Token sequences (batch, seq_len)
            states: List of CTC states
            xs: Encoder outputs (batch, enc_len, dim)

        Returns:
            Tuple of (log_probs, new_states)
                - log_probs: (batch, vocab_size)
        """
        batch_size = yseqs.size(0)

        with torch.no_grad():
            logits = self.ctc.ctc_lo(xs)  # (batch, enc_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

        # Simplified: return mean log probs across time
        mean_log_probs = log_probs.mean(dim=1)  # (batch, vocab_size)

        # States unchanged for simplified version
        new_states = states

        return mean_log_probs, new_states


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
