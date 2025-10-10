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
    append_token,
    append_position,
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
        use_bbd: bool = True,
        bbd_conservative: bool = True,
    ):
        self.scorers = scorers
        self.weights = weights
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_length = max_length
        self.device = device
        self.use_bbd = use_bbd
        self.bbd_conservative = bbd_conservative

    def batch_score_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        encoder_out: torch.Tensor,
        pre_beam_size: int = 40,
    ) -> Tuple[torch.Tensor, Dict[str, List]]:
        """Score all hypotheses for next token prediction with two-pass strategy.

        TWO-PASS SCORING STRATEGY (following ESPnet):
        1. Full scorers (decoder) score entire vocabulary
        2. Select top-K candidates (pre-beam search)
        3. Partial scorers (CTC) score only top-K candidates
        4. Combine scores

        This gives ~25x speedup for CTC by scoring only 40 tokens instead of 1024.

        Args:
            hypotheses: List of hypotheses to score
            encoder_out: Encoder output (batch, enc_len, dim)
            pre_beam_size: Number of top candidates for partial scorers (default: 40)

        Returns:
            Tuple of:
                - Combined scores (batch, vocab_size)
                - Dict of new states per scorer {scorer_name: [state_0, state_1, ...]}
        """
        if not hypotheses:
            return torch.zeros(0, self.vocab_size, device=self.device), {}

        batch_size = len(hypotheses)

        # Prepare batch - yseq is already torch.Tensor
        max_len = max(len(h.yseq) for h in hypotheses)
        yseqs = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        for i, h in enumerate(hypotheses):
            yseqs[i, : len(h.yseq)] = h.yseq.to(self.device)

        # Expand encoder output for batch
        encoder_out_batch = encoder_out.expand(batch_size, -1, -1)

        # Initialize combined scores
        combined_scores = torch.zeros(batch_size, self.vocab_size, device=self.device)

        # Collect new states from all scorers
        all_new_states = {}

        # PASS 1: Score with FULL scorers (e.g., decoder)
        # These scorers must score entire vocabulary to select top-K candidates
        full_scorer_scores = torch.zeros(batch_size, self.vocab_size, device=self.device)

        for scorer_name, scorer in self.scorers.items():
            weight = self.weights.get(scorer_name, 0.0)
            if weight == 0.0:
                continue

            # Check if scorer supports partial scoring
            has_partial = hasattr(scorer, 'batch_score_partial')

            if not has_partial:
                # Full scorer - score entire vocabulary
                states = [h.states.get(scorer_name) for h in hypotheses]
                scores, new_states = scorer.batch_score(yseqs, states, encoder_out_batch)

                # Store states
                all_new_states[scorer_name] = new_states

                # Add to both combined and full_scorer scores
                combined_scores += weight * scores
                full_scorer_scores += weight * scores

        # PRE-BEAM SEARCH: Select top-K candidates based on full scorers
        # This is THE KEY OPTIMIZATION - only score these K tokens with CTC
        top_k_scores, top_k_ids = torch.topk(
            full_scorer_scores,
            k=min(pre_beam_size, self.vocab_size),
            dim=-1
        )  # (batch, K), (batch, K)

        # PASS 2: Score ONLY top-K with partial scorers (e.g., CTC)
        for scorer_name, scorer in self.scorers.items():
            weight = self.weights.get(scorer_name, 0.0)
            if weight == 0.0:
                continue

            # Check if scorer supports partial scoring
            has_partial = hasattr(scorer, 'batch_score_partial')

            if has_partial:
                # Partial scorer - score only top-K candidates
                # NOTE: batch_score_partial computes forward algorithm for only K tokens
                # but RETURNS full vocab scores (with non-K tokens as -inf)
                states = [h.states.get(scorer_name) for h in hypotheses]

                # Score only top-K tokens (but returns full vocab size)
                scores, new_states = scorer.batch_score_partial(
                    yseqs, top_k_ids, states, encoder_out_batch
                )  # (batch, vocab_size) with only K non-inf values

                # Store states
                all_new_states[scorer_name] = new_states

                # Add weighted scores (already full vocab size)
                combined_scores += weight * scores

        return combined_scores, all_new_states

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
        beam = [create_initial_hypothesis(self.sos_id, device=self.device)]

        # Iterative decoding
        for step in range(self.max_length):
            # Score all hypotheses and get new states
            scores, new_states_dict = self.batch_score_hypotheses(beam, encoder_out)  # (beam, vocab)

            # Expand hypotheses
            new_hypotheses = []
            for i, hyp in enumerate(beam):
                # Get top-k tokens for this hypothesis
                top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

                for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                    # Merge states from ALL scorers with proper state selection
                    new_states_for_hyp = {}
                    for scorer_name in new_states_dict:
                        scorer = self.scorers[scorer_name]
                        state = new_states_dict[scorer_name][i]

                        # CRITICAL FIX: Call select_state to get correct state for this hyp+token
                        if hasattr(scorer, 'select_state'):
                            new_states_for_hyp[scorer_name] = scorer.select_state(state, i, token)
                        else:
                            new_states_for_hyp[scorer_name] = state

                    new_hyp = Hypothesis(
                        yseq=append_token(hyp.yseq, token),
                        score=hyp.score + score,
                        scores=hyp.scores.copy(),  # TODO: Update individual scores
                        states=new_states_for_hyp,  # Properly selected states!
                        xpos=hyp.xpos,  # Keep same xpos for now
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
        use_bbd: bool = True,
        bbd_conservative: bool = True,
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
        self.use_bbd = use_bbd
        self.bbd_conservative = bbd_conservative

        # Internal beam search for each block with BBD
        self.beam_search = BeamSearch(
            scorers=scorers,
            weights=weights,
            beam_size=beam_size,
            vocab_size=vocab_size,
            sos_id=sos_id,
            eos_id=eos_id,
            max_length=max_length,
            device=device,
            use_bbd=use_bbd,
            bbd_conservative=bbd_conservative,
        )

    def extend_scorers(
        self,
        encoder_out: torch.Tensor,
        hypotheses: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Extend scorers with new encoder output and update hypothesis states.

        This is called after each encoder block to:
        1. Extend probability matrices (extend_prob) for streaming scorers
        2. Extend hypothesis states (extend_state) to cover new time steps

        Following ESPnet's pattern from batch_beam_search_online.py:extend()

        Args:
            encoder_out: New encoder output block (batch, enc_len, dim)
            hypotheses: Current hypotheses with states to extend

        Returns:
            Updated hypotheses with extended states
        """
        # Extend probability matrices for scorers that support streaming
        for scorer_name, scorer in self.scorers.items():
            if hasattr(scorer, "extend_prob"):
                # Extend CTC probability matrix with new encoder output
                scorer.extend_prob(encoder_out)

        # Extend hypothesis states to match new probability matrix size
        updated_hypotheses = []
        for hyp in hypotheses:
            # Copy hypothesis and update states
            new_states = {}
            for scorer_name in hyp.states:
                scorer = self.scorers.get(scorer_name)
                if scorer and hasattr(scorer, "extend_state"):
                    # Extend this scorer's state
                    new_states[scorer_name] = scorer.extend_state(hyp.states[scorer_name])
                else:
                    # Keep state unchanged
                    new_states[scorer_name] = hyp.states[scorer_name]

            # Create updated hypothesis with extended states
            updated_hyp = Hypothesis(
                yseq=hyp.yseq,
                score=hyp.score,
                scores=hyp.scores.copy(),
                states=new_states,
                xpos=hyp.xpos,
            )
            updated_hypotheses.append(updated_hyp)

        return updated_hypotheses

    def compute_reliability_scores(
        self,
        hypotheses: List[Hypothesis],
        scores: torch.Tensor,
        encoder_out: torch.Tensor,
        evaluated_hyps: set,
    ) -> List[float]:
        """Compute reliability scores for BBD (Block Boundary Detection).

        Based on Equation (12-13) from Tsunoo et al., 2021:

        r(y_{0:i-1}, h_{1:b}) = max over j in [0, i-1]:
            log p(y_j | y_{0:i-1}, h_{1:b}) + α(y_{0:i-1}, h_{1:b})

        s(y_{0:i}, h_{1:b}) = α(y_{0:i}, h_{1:b}) - r(y_{0:i-1}, h_{1:b})

        If s ≤ 0: hypothesis is unreliable (has <eos> or repetition with higher score)

        Args:
            hypotheses: Current hypotheses (y_{0:i-1})
            scores: Scores for next tokens (batch, vocab_size)
            encoder_out: Encoder output h_{1:b}
            evaluated_hyps: Set of already evaluated hypothesis sequences (Ω_R)

        Returns:
            List of reliability scores, one per hypothesis
        """
        reliability_scores = []

        for i, hyp in enumerate(hypotheses):
            # Current hypothesis score: α(y_{0:i-1}, h_{1:b})
            current_score = hyp.score

            # Find repetitions: tokens that already exist in yseq
            # Note: <sos> and <eos> are the same token, so <eos> is always a repetition
            yseq_list = hyp.yseq.tolist()

            # Compute r(y_{0:i-1}, h_{1:b}): highest score among repetitions
            # For each token j in [0, i-1], compute score of appending y_j
            max_repetition_score = float('-inf')

            for token_id in set(yseq_list):  # Check each unique token in history
                # Skip if this repetition was already evaluated
                hyp_with_rep = tuple(yseq_list + [token_id])
                if hyp_with_rep in evaluated_hyps:
                    continue

                # Score of appending this repeated token
                # log p(y_j | y_{0:i-1}, h_{1:b}) + α(y_{0:i-1}, h_{1:b})
                repetition_score = current_score + scores[i, token_id].item()
                max_repetition_score = max(max_repetition_score, repetition_score)

            # If no unevaluated repetitions, assume reliable (no constraint)
            if max_repetition_score == float('-inf'):
                reliability_scores.append(float('inf'))
                continue

            # r(y_{0:i-1}, h_{1:b})
            r_score = max_repetition_score

            # For the next token, check reliability
            # We need to check the TOP token that would be appended
            # s(y_{0:i}, h_{1:b}) = α(y_{0:i}, h_{1:b}) - r(y_{0:i-1}, h_{1:b})

            # Get the best next token score
            best_next_score = scores[i].max().item()
            alpha_next = current_score + best_next_score  # α(y_{0:i}, h_{1:b})

            # Reliability score
            s_score = alpha_next - r_score
            reliability_scores.append(s_score)

        return reliability_scores

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
                hypotheses=[create_initial_hypothesis(self.sos_id, device=self.device)],
                encoder_states=None,
                processed_frames=0,
            )

        # Encode block
        # NOTE: Use infer_mode=False to match batch mode behavior
        # Batch mode doesn't pass infer_mode, so it defaults to False and uses forward_train
        encoder_out, encoder_out_lens, encoder_states = self.encoder(
            features,
            feature_lens,
            prev_states=prev_state.encoder_states,
            is_final=is_final,
            infer_mode=False,  # Match batch mode
        )

        # Extend scorers with new encoder output (for streaming CTC support)
        # This calls extend_prob() on scorers and extend_state() on hypotheses
        extended_hypotheses = self.extend_scorers(encoder_out, prev_state.hypotheses)

        # Update state with extended hypotheses
        new_state = BeamState(
            hypotheses=extended_hypotheses,
            encoder_states=encoder_states,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            processed_frames=prev_state.processed_frames + features.size(1),
            is_final=is_final,
        )

        # Perform beam search if we have encoder output
        # Run decoding steps with BBD (Block Boundary Detection)
        if encoder_out.size(1) > 0:
            # Track hypotheses from previous step for BBD rollback
            prev_step_hypotheses = new_state.hypotheses

            # Decode until BBD detects block boundary or max length
            # Conservative estimate: ~2 tokens per encoder frame (max)
            # This prevents decoding beyond what CTC can support
            max_decode_steps = min(encoder_out.size(1) * 2, 100)  # Cap at 100 tokens per block

            for step in range(max_decode_steps):
                new_state.output_index += 1

                # Score current hypotheses and get new states
                scores, new_states_dict = self.beam_search.batch_score_hypotheses(
                    new_state.hypotheses, encoder_out
                )

                # BBD: Check reliability scores BEFORE expanding
                if self.use_bbd and not is_final:
                    reliability_scores = self.compute_reliability_scores(
                        new_state.hypotheses, scores, encoder_out, new_state.evaluated_hyps
                    )

                    # If ANY hypothesis is unreliable (s ≤ 0), stop and wait for next block
                    has_unreliable = any(s <= 0 for s in reliability_scores)

                    if has_unreliable:
                        logger.debug(f"BBD: Unreliable hypothesis detected at step {step}, "
                                   f"min_reliability={min(reliability_scores):.4f}, stopping block")

                        # Store all current hypotheses in evaluated set (Ω_R)
                        for hyp in new_state.hypotheses:
                            new_state.evaluated_hyps.add(tuple(hyp.yseq.tolist()))

                        # Revert to previous step (conservative: i-2, non-conservative: i-1)
                        if self.bbd_conservative and len(prev_step_hypotheses) > 0:
                            # Conservative: go back 2 steps
                            new_state.hypotheses = prev_step_hypotheses
                            new_state.output_index -= 2
                        else:
                            # Non-conservative: go back 1 step (keep current)
                            new_state.output_index -= 1

                        # Stop decoding this block, wait for more encoder output
                        break

                # Expand and prune beam
                new_hypotheses = []
                for i, hyp in enumerate(new_state.hypotheses):
                    top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

                    for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                        # Merge states from ALL scorers with proper state selection
                        new_states_for_hyp = {}
                        for scorer_name in new_states_dict:
                            scorer = self.scorers[scorer_name]
                            state = new_states_dict[scorer_name][i]

                            # Call select_state to get correct state for this hyp+token
                            if hasattr(scorer, 'select_state'):
                                new_states_for_hyp[scorer_name] = scorer.select_state(state, i, token)
                            else:
                                new_states_for_hyp[scorer_name] = state

                        # Track encoder position
                        current_encoder_pos = encoder_out.size(1) - 1  # Last frame index

                        new_hyp = Hypothesis(
                            yseq=append_token(hyp.yseq, token),
                            score=hyp.score + score,
                            scores=hyp.scores.copy(),
                            states=new_states_for_hyp,
                            xpos=append_position(hyp.xpos, current_encoder_pos),
                        )
                        new_hypotheses.append(new_hyp)

                # Save current hypotheses for potential rollback
                prev_step_hypotheses = new_state.hypotheses

                # Prune to beam size
                new_state.hypotheses = top_k_hypotheses(new_hypotheses, self.beam_size)

                # Check if all hypotheses ended with EOS (only stop if is_final)
                all_eos = all(h.yseq[-1].item() == self.eos_id for h in new_state.hypotheses)
                if all_eos and is_final:
                    logger.debug(f"All hypotheses ended with EOS at step {step}")
                    break

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
    use_bbd: bool = True,
    bbd_conservative: bool = True,
) -> BlockwiseSynchronousBeamSearch:
    """Create BSBS beam search from model.

    Args:
        model: ESPnetASRModel instance
        beam_size: Beam size
        ctc_weight: CTC scorer weight
        decoder_weight: Decoder scorer weight
        device: Device to run on
        use_bbd: Use Block Boundary Detection (BBD) for streaming
        bbd_conservative: Use conservative BBD (rollback 2 steps vs 1)

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

    # CTC scoring with full forward algorithm implementation
    # Now uses batch_score_partial for top-K scoring (25x speedup!)
    if model.ctc is not None and ctc_weight > 0:
        scorers["ctc"] = CTCPrefixScorer(model.ctc, blank_id=0, eos_id=2)
        weights["ctc"] = ctc_weight

    # Create BSBS
    return BlockwiseSynchronousBeamSearch(
        encoder=model.encoder,
        scorers=scorers,
        weights=weights,
        beam_size=beam_size,
        vocab_size=model.vocab_size,
        device=device,
        use_bbd=use_bbd,
        bbd_conservative=bbd_conservative,
    )
