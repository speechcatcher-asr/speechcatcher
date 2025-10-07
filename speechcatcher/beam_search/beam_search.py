"""Blockwise Synchronous Beam Search (BSBS) for streaming ASR.

Based on: "Blockwise Streaming Transformer for Spoken Language Understanding
and Simultaneous Speech Translation" (Tsunoo et al., 2021)
https://arxiv.org/abs/1910.07204
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from speechcatcher.beam_search.hypothesis import (
    BeamState,
    Hypothesis,
    create_initial_hypothesis,
    top_k_hypotheses,
)
from speechcatcher.beam_search.scorers import ScorerInterface

logger = logging.getLogger(__name__)


class BeamSearch:
    """Standard beam search decoder.

    Args:
        scorers: Dictionary of scorers {name: scorer}
        weights: Dictionary of scorer weights {name: weight}
        beam_size: Beam size (default: 10)
        vocab_size: Vocabulary size
        sos_id: Start-of-sentence token ID
        eos_id: End-of-sentence token ID
        max_length: Maximum sequence length
        device: Device to run on
    """

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int = 10,
        vocab_size: int = 1000,
        sos_id: int = 1,
        eos_id: int = 2,
        max_length: int = 500,
        device: str = "cpu",
    ):
        self.scorers = scorers
        self.weights = weights
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_length = max_length
        self.device = device

    def batch_score_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        encoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """Score all hypotheses for next token prediction.

        Args:
            hypotheses: List of hypotheses to score
            encoder_out: Encoder output (batch, enc_len, dim)

        Returns:
            Combined scores (batch, vocab_size)
        """
        if not hypotheses:
            return torch.zeros(0, self.vocab_size, device=self.device)

        batch_size = len(hypotheses)

        # Prepare batch
        max_len = max(len(h.yseq) for h in hypotheses)
        yseqs = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        for i, h in enumerate(hypotheses):
            yseqs[i, : len(h.yseq)] = torch.tensor(h.yseq, dtype=torch.long, device=self.device)

        # Expand encoder output for batch
        encoder_out_batch = encoder_out.expand(batch_size, -1, -1)

        # Initialize combined scores
        combined_scores = torch.zeros(batch_size, self.vocab_size, device=self.device)

        # Score with each scorer
        for scorer_name, scorer in self.scorers.items():
            weight = self.weights.get(scorer_name, 0.0)
            if weight == 0.0:
                continue

            # Extract states for this scorer
            states = [h.states for h in hypotheses]

            # Batch score
            scores, new_states = scorer.batch_score(yseqs, states, encoder_out_batch)

            # Update states in hypotheses
            for i, h in enumerate(hypotheses):
                h.states = new_states[i]

            # Add weighted scores
            combined_scores += weight * scores

        return combined_scores

    def search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> List[Hypothesis]:
        """Perform beam search decoding.

        Args:
            encoder_out: Encoder output (1, enc_len, dim)
            encoder_out_lens: Encoder output length (1,)

        Returns:
            List of final hypotheses sorted by score
        """
        # Initialize with SOS token
        beam = [create_initial_hypothesis(self.sos_id)]

        # Iterative decoding
        for step in range(self.max_length):
            # Score all hypotheses
            scores = self.batch_score_hypotheses(beam, encoder_out)  # (beam, vocab)

            # Expand hypotheses
            new_hypotheses = []
            for i, hyp in enumerate(beam):
                # Get top-k tokens for this hypothesis
                top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

                for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                    new_hyp = Hypothesis(
                        yseq=hyp.yseq + [token],
                        score=hyp.score + score,
                        scores=hyp.scores.copy(),
                        states=hyp.states,
                    )
                    new_hypotheses.append(new_hyp)

            # Prune to beam size
            beam = top_k_hypotheses(new_hypotheses, self.beam_size)

            # Check if all hypotheses ended with EOS
            if all(h.yseq[-1] == self.eos_id for h in beam):
                break

        return beam


class BlockwiseSynchronousBeamSearch:
    """Blockwise Synchronous Beam Search (BSBS) for streaming ASR.

    BSBS processes audio in blocks and performs beam search synchronously
    at block boundaries. This enables low-latency streaming ASR while
    maintaining high accuracy.

    Args:
        encoder: Encoder model
        scorers: Dictionary of scorers {name: scorer}
        weights: Dictionary of scorer weights {name: weight}
        beam_size: Beam size
        vocab_size: Vocabulary size
        sos_id: Start-of-sentence token ID
        eos_id: End-of-sentence token ID
        block_size: Block size in frames (after subsampling)
        hop_size: Hop size between blocks
        look_ahead: Look-ahead frames
        reliability_threshold: Threshold for block boundary detection
        device: Device to run on
    """

    def __init__(
        self,
        encoder: nn.Module,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int = 10,
        vocab_size: int = 1000,
        sos_id: int = 1,
        eos_id: int = 2,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
        reliability_threshold: float = 0.8,
        max_length: int = 500,
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.scorers = scorers
        self.weights = weights
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.reliability_threshold = reliability_threshold
        self.max_length = max_length
        self.device = device

        # Internal beam search for each block
        self.beam_search = BeamSearch(
            scorers=scorers,
            weights=weights,
            beam_size=beam_size,
            vocab_size=vocab_size,
            sos_id=sos_id,
            eos_id=eos_id,
            max_length=max_length,
            device=device,
        )

    def process_block(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        prev_state: Optional[BeamState] = None,
        is_final: bool = False,
    ) -> BeamState:
        """Process a single block of features.

        Args:
            features: Input features (1, time, feat_dim)
            feature_lens: Feature lengths (1,)
            prev_state: Previous beam state
            is_final: Whether this is the final block

        Returns:
            Updated beam state
        """
        # Initialize state if needed
        if prev_state is None:
            prev_state = BeamState(
                hypotheses=[create_initial_hypothesis(self.sos_id)],
                encoder_states=None,
                processed_frames=0,
            )

        # Encode block
        encoder_out, encoder_out_lens, encoder_states = self.encoder(
            features,
            feature_lens,
            prev_states=prev_state.encoder_states,
            is_final=is_final,
            infer_mode=True,
        )

        # Update state
        new_state = BeamState(
            hypotheses=prev_state.hypotheses,
            encoder_states=encoder_states,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            processed_frames=prev_state.processed_frames + features.size(1),
            is_final=is_final,
        )

        # Perform beam search if we have encoder output
        if encoder_out.size(1) > 0:
            # Score current hypotheses
            scores = self.beam_search.batch_score_hypotheses(
                new_state.hypotheses, encoder_out
            )

            # Expand and prune beam
            new_hypotheses = []
            for i, hyp in enumerate(new_state.hypotheses):
                top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

                for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                    new_hyp = Hypothesis(
                        yseq=hyp.yseq + [token],
                        score=hyp.score + score,
                        scores=hyp.scores.copy(),
                        states=hyp.states,
                    )
                    new_hypotheses.append(new_hyp)

            # Prune to beam size
            new_state.hypotheses = top_k_hypotheses(new_hypotheses, self.beam_size)

        return new_state

    def recognize_stream(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> List[Hypothesis]:
        """Recognize speech from streaming features (non-incremental version).

        This processes the entire utterance but simulates streaming by
        processing in blocks.

        Args:
            features: Input features (1, time, feat_dim)
            feature_lens: Feature lengths (1,)

        Returns:
            List of final hypotheses sorted by score
        """
        # Process in blocks
        state = None
        total_frames = feature_lens[0].item()
        current_frame = 0

        while current_frame < total_frames:
            # Extract block
            end_frame = min(current_frame + self.block_size, total_frames)
            block = features[:, current_frame:end_frame, :]
            block_lens = torch.tensor([block.size(1)], device=self.device)

            is_final = (end_frame >= total_frames)

            # Process block
            state = self.process_block(block, block_lens, state, is_final)

            current_frame += self.hop_size

        return state.hypotheses if state else []


def create_beam_search(
    model: nn.Module,
    beam_size: int = 10,
    ctc_weight: float = 0.3,
    decoder_weight: float = 0.7,
    device: str = "cpu",
) -> BlockwiseSynchronousBeamSearch:
    """Create BSBS beam search from model.

    Args:
        model: ESPnetASRModel instance
        beam_size: Beam size
        ctc_weight: CTC scorer weight
        decoder_weight: Decoder scorer weight
        device: Device to run on

    Returns:
        BlockwiseSynchronousBeamSearch instance
    """
    from speechcatcher.beam_search.scorers import CTCPrefixScorer, DecoderScorer

    # Create scorers
    scorers = {}
    weights = {}

    if model.decoder is not None:
        scorers["decoder"] = DecoderScorer(model.decoder)
        weights["decoder"] = decoder_weight

    if model.ctc is not None:
        scorers["ctc"] = CTCPrefixScorer(model.ctc)
        weights["ctc"] = ctc_weight

    # Create BSBS
    return BlockwiseSynchronousBeamSearch(
        encoder=model.encoder,
        scorers=scorers,
        weights=weights,
        beam_size=beam_size,
        vocab_size=model.vocab_size,
        device=device,
    )
