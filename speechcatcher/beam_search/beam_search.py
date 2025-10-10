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

        # Encoder output buffer (accumulated across chunks)
        # This is CRITICAL for streaming! Matches ESPnet's encbuffer
        self.encoder_buffer = None
        self.processed_block = 0

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

    def detect_repetition_or_eos(
        self,
        hypotheses: List[Hypothesis],
    ) -> bool:
        """Detect if any hypothesis has repetition or EOS.

        Simplified BBD from ESPnet's batch_beam_search_online.py (lines 209-218).
        Instead of complex reliability scores, just check:
        - Is the last predicted token already in the sequence? (repetition)
        - Is the last predicted token <eos>?

        Args:
            hypotheses: Current hypotheses

        Returns:
            True if repetition or EOS detected in any hypothesis
        """
        for hyp in hypotheses:
            if len(hyp.yseq) < 2:
                continue

            last_token = hyp.yseq[-1].item()

            # Check if last token appears in previous tokens (repetition)
            # Note: <sos> == <eos>, so EOS is always a repetition of SOS
            prev_tokens = hyp.yseq[:-1].tolist()
            if last_token in prev_tokens:
                logger.debug(f"BBD: Detected repetition - token {last_token} in {hyp.yseq.tolist()}")
                return True

        return False

    def process_block(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        prev_state: Optional[BeamState] = None,
        is_final: bool = False,
    ) -> BeamState:
        """Process features with encoder buffering and block extraction.

        This implements the critical buffering mechanism from ESPnet's
        batch_beam_search_online.py that we were missing!

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

        # Step 1: Encode features
        # CRITICAL: Use infer_mode=True for streaming!
        # ESPnet streaming uses infer_mode=True (asr_inference_streaming.py:330)

        # Check if features are too small for encoder (conv2d kernel is 3x3)
        # If features are too small, skip encoding and use empty output
        if features.size(1) < 3:
            logger.debug(f"Skipping encoder: features too small ({features.shape}, need >=3 frames)")
            encoder_out = torch.zeros(
                1, 0, 256,  # Empty encoder output
                dtype=features.dtype,
                device=features.device,
            )
            encoder_out_lens = torch.tensor([0], dtype=torch.long)
            encoder_states = prev_state.encoder_states  # Keep previous states
        else:
            encoder_out, encoder_out_lens, encoder_states = self.encoder(
                features,
                feature_lens,
                prev_states=prev_state.encoder_states,
                is_final=is_final,
                infer_mode=True,  # Streaming mode!
            )

        # Step 2: Accumulate encoder outputs in buffer
        # This is the CRITICAL missing piece! Matches ESPnet's encbuffer logic
        # (batch_beam_search_online.py:118-121)
        if encoder_out.size(1) > 0:  # Only accumulate if encoder produced output
            if self.encoder_buffer is None:
                self.encoder_buffer = encoder_out
                logger.debug(f"Initialized encoder buffer: {self.encoder_buffer.shape}")
            else:
                self.encoder_buffer = torch.cat([self.encoder_buffer, encoder_out], dim=1)
                logger.debug(f"Accumulated encoder buffer: {self.encoder_buffer.shape}")

        # Step 3: Extract and process blocks from buffer
        # Match ESPnet's block extraction logic (batch_beam_search_online.py:132-168)
        current_state = prev_state
        ret = None

        while True:
            # Calculate current block end frame
            # Formula from ESPnet: block_size - look_ahead + hop_size * processed_block
            cur_end_frame = (
                self.block_size - self.look_ahead +
                self.hop_size * self.processed_block
            )

            logger.debug(f"Block {self.processed_block}: cur_end_frame={cur_end_frame}, "
                        f"buffer_size={self.encoder_buffer.shape[1] if self.encoder_buffer is not None else 0}")

            # Check if we have enough frames in buffer for this block
            if self.encoder_buffer is not None and cur_end_frame <= self.encoder_buffer.shape[1]:
                # Extract block from buffer: frames [0, cur_end_frame)
                block_encoder_out = self.encoder_buffer.narrow(1, 0, cur_end_frame)
                block_is_final = False

                logger.debug(f"Processing block {self.processed_block} with {cur_end_frame} frames")

                # Process this block
                current_state = self._decode_one_block(
                    block_encoder_out,
                    current_state,
                    block_is_final
                )

                ret = current_state
                self.processed_block += 1
            elif is_final and self.encoder_buffer is not None and self.encoder_buffer.shape[1] > 0:
                # Final chunk: process remaining buffer
                logger.debug(f"Final block with {self.encoder_buffer.shape[1]} frames")

                current_state = self._decode_one_block(
                    self.encoder_buffer,
                    current_state,
                    is_final=True
                )

                ret = current_state
                break
            else:
                # Not enough frames yet, wait for more input
                logger.debug(f"Waiting for more frames (need {cur_end_frame}, have {self.encoder_buffer.shape[1] if self.encoder_buffer is not None else 0})")
                break

        # Update encoder states for next chunk
        if ret is not None:
            ret.encoder_states = encoder_states

        return ret if ret is not None else prev_state

    def _decode_one_block(
        self,
        encoder_out: torch.Tensor,
        prev_state: BeamState,
        is_final: bool = False,
    ) -> BeamState:
        """Decode one block of encoder output.

        This replaces the old inline decoding logic in process_block.

        Args:
            encoder_out: Encoder output block (1, time, dim)
            prev_state: Previous beam state
            is_final: Whether this is the final block

        Returns:
            Updated beam state
        """

        # Extend scorers with new encoder output (for streaming CTC support)
        # This calls extend_prob() on scorers and extend_state() on hypotheses
        extended_hypotheses = self.extend_scorers(encoder_out, prev_state.hypotheses)

        # Update state with extended hypotheses
        # Note: We don't update encoder_states here - that's done in process_block
        new_state = BeamState(
            hypotheses=extended_hypotheses,
            encoder_states=None,  # Will be set by process_block
            encoder_out=encoder_out,
            encoder_out_lens=torch.tensor([encoder_out.size(1)], dtype=torch.long),
            processed_frames=prev_state.processed_frames,  # Don't increment here
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

                # Expand and prune beam FIRST, then check for repetition/EOS
                # This matches ESPnet's approach (expand, then check, then potentially rollback)

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

                # Prune to beam size
                new_state.hypotheses = top_k_hypotheses(new_hypotheses, self.beam_size)

                # BBD: Check for repetition/EOS AFTER beam expansion
                # Matches ESPnet's approach (batch_beam_search_online.py:209-218)
                if self.use_bbd and not is_final:
                    has_repetition = self.detect_repetition_or_eos(new_state.hypotheses)

                    if has_repetition:
                        logger.debug(f"BBD: Repetition/EOS detected at step {step}, stopping block")

                        # Rollback to previous hypotheses (1 step, matching ESPnet)
                        if len(prev_step_hypotheses) > 0:
                            new_state.hypotheses = prev_step_hypotheses
                            new_state.output_index -= 1
                            logger.debug(f"BBD: Rolled back to output_index={new_state.output_index}")

                        # Stop decoding this block, wait for more encoder output
                        break

                # Save current hypotheses for potential rollback in next iteration
                prev_step_hypotheses = new_state.hypotheses

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
