"""Hypothesis and beam state classes for beam search."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Hypothesis:
    """Single hypothesis in beam search.

    Attributes:
        yseq: Token sequence (list of token IDs)
        score: Total log probability score
        scores: Score breakdown by component (e.g., 'decoder', 'ctc', 'lm')
        states: Decoder states (list of tensors for each layer)
        yseq_tensor: Cached tensor version of yseq
    """

    yseq: List[int] = field(default_factory=list)
    score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    states: Optional[List[torch.Tensor]] = None
    yseq_tensor: Optional[torch.Tensor] = None

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary (for compatibility with ESPnet)."""
        return {
            "yseq": torch.tensor(self.yseq, dtype=torch.long),
            "score": self.score,
            "scores": self.scores,
        }

    def __repr__(self) -> str:
        return f"Hypothesis(yseq={self.yseq[:10]}{'...' if len(self.yseq) > 10 else ''}, score={self.score:.2f})"


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


def create_initial_hypothesis(sos_id: int = 1) -> Hypothesis:
    """Create initial hypothesis with SOS token.

    Args:
        sos_id: Start-of-sentence token ID

    Returns:
        Initial hypothesis
    """
    return Hypothesis(
        yseq=[sos_id],
        score=0.0,
        scores={},
        states=None,
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

    # Batch yseq
    max_len = max(len(h.yseq) for h in hypotheses)
    yseq_batch = torch.zeros(len(hypotheses), max_len, dtype=torch.long, device=device)

    for i, h in enumerate(hypotheses):
        yseq_batch[i, : len(h.yseq)] = torch.tensor(h.yseq, dtype=torch.long)

    # Batch states if available
    states_batch = None
    if hypotheses[0].states is not None:
        # States are list of tensors (one per layer)
        n_layers = len(hypotheses[0].states)
        states_batch = [
            torch.stack([h.states[layer_idx] for h in hypotheses])
            for layer_idx in range(n_layers)
        ]

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


def merge_scores(
    hypotheses: List[Hypothesis],
    new_scores: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> List[Hypothesis]:
    """Merge new scores into hypotheses.

    Args:
        hypotheses: List of hypotheses (will be modified in place)
        new_scores: Dictionary of new scores {'scorer_name': (batch, vocab)}
        weights: Dictionary of scorer weights {'scorer_name': weight}

    Returns:
        Updated hypotheses
    """
    for i, h in enumerate(hypotheses):
        # Update individual scores
        for scorer_name, scores in new_scores.items():
            if scores is not None:
                h.scores[scorer_name] = scores[i].item() if scores.dim() > 0 else scores.item()

        # Update total score as weighted sum
        h.score = sum(h.scores.get(name, 0.0) * weight for name, weight in weights.items())

    return hypotheses
