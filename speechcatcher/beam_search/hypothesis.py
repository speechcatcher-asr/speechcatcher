"""Hypothesis and beam state classes for beam search."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Hypothesis:
    """Single hypothesis in beam search.

    Matches ESPnet's Hypothesis structure for compatibility.

    Attributes:
        yseq: Token sequence (torch.Tensor of token IDs)
        score: Total log probability score
        scores: Score breakdown by component (e.g., 'decoder', 'ctc', 'lm')
        states: Per-scorer states dict {scorer_name: state}
        xpos: Encoder frame positions for each token (torch.Tensor)
    """

    yseq: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    states: Dict[str, Any] = field(default_factory=dict)
    xpos: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary (for compatibility with ESPnet)."""
        return {
            "yseq": self.yseq,
            "score": self.score,
            "scores": self.scores,
        }

    def __repr__(self) -> str:
        yseq_list = self.yseq.tolist() if len(self.yseq) > 0 else []
        yseq_str = str(yseq_list[:10]) + ('...' if len(yseq_list) > 10 else '')
        return f"Hypothesis(yseq={yseq_str}, score={self.score:.2f})"


@dataclass
class BeamState:
    """Beam state for blockwise synchronous beam search.

    Attributes:
        hypotheses: List of active hypotheses
        encoder_states: Encoder streaming states
        encoder_out: Current encoder output (batch, time, dim)
        encoder_out_lens: Encoder output lengths
        processed_frames: Number of frames processed so far
        is_final: Whether this is the final block
    """

    hypotheses: List[Hypothesis] = field(default_factory=list)
    encoder_states: Optional[Dict] = None
    encoder_out: Optional[torch.Tensor] = None
    encoder_out_lens: Optional[torch.Tensor] = None
    processed_frames: int = 0
    is_final: bool = False

    def __repr__(self) -> str:
        return (
            f"BeamState(n_hyps={len(self.hypotheses)}, "
            f"processed_frames={self.processed_frames}, "
            f"is_final={self.is_final})"
        )


def create_initial_hypothesis(sos_id: int = 1, device: str = "cpu") -> Hypothesis:
    """Create initial hypothesis with SOS token.

    Args:
        sos_id: Start-of-sentence token ID
        device: Device to place tensors on

    Returns:
        Initial hypothesis
    """
    return Hypothesis(
        yseq=torch.tensor([sos_id], dtype=torch.long, device=device),
        score=0.0,
        scores={},
        states={},
        xpos=torch.tensor([0], dtype=torch.long, device=device),
    )


def batch_hypotheses(hypotheses: List[Hypothesis], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Batch multiple hypotheses for parallel scoring.

    Args:
        hypotheses: List of hypotheses to batch
        device: Device to place tensors on

    Returns:
        Dictionary with batched tensors
    """
    if not hypotheses:
        return {}

    # Batch yseq (already torch.Tensor in each hypothesis)
    max_len = max(len(h.yseq) for h in hypotheses)
    yseq_batch = torch.zeros(len(hypotheses), max_len, dtype=torch.long, device=device)

    for i, h in enumerate(hypotheses):
        yseq_batch[i, : len(h.yseq)] = h.yseq.to(device)

    # Batch states - now dict-based
    # States is Dict[str, Any] where each scorer has its own state
    # We need to reorganize this for batch processing
    states_batch = {}
    if hypotheses[0].states:
        # Get all scorer names
        scorer_names = hypotheses[0].states.keys()
        for scorer_name in scorer_names:
            # Each scorer's state is a list of layer states
            states_batch[scorer_name] = [h.states[scorer_name] for h in hypotheses]

    return {
        "yseq": yseq_batch,
        "states": states_batch,
        "scores": [h.score for h in hypotheses],
    }


def top_k_hypotheses(hypotheses: List[Hypothesis], k: int) -> List[Hypothesis]:
    """Select top-k hypotheses by score.

    Args:
        hypotheses: List of hypotheses
        k: Number of hypotheses to keep

    Returns:
        Top-k hypotheses sorted by score (descending)
    """
    return sorted(hypotheses, key=lambda h: h.score, reverse=True)[:k]


def append_token(tensor: torch.Tensor, token_id: int) -> torch.Tensor:
    """Append a token to a tensor sequence.

    Args:
        tensor: Original sequence tensor
        token_id: Token ID to append

    Returns:
        New tensor with token appended
    """
    return torch.cat([tensor, torch.tensor([token_id], dtype=torch.long, device=tensor.device)])


def append_position(xpos: torch.Tensor, position: int) -> torch.Tensor:
    """Append an encoder position to xpos tensor.

    Args:
        xpos: Original position tensor
        position: Encoder frame position to append

    Returns:
        New tensor with position appended
    """
    return torch.cat([xpos, torch.tensor([position], dtype=torch.long, device=xpos.device)])
