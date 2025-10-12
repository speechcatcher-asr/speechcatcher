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
        vocab_size: int = 1024,
        sos_id: int = 1023,
        eos_id: int = 1023,
        max_length: int = 500,
        device: str = "cpu",
        use_bbd: bool = True,
        bbd_conservative: bool = True,
    ):
        """Initialize beam search.

        Note: sos_id and eos_id defaults assume vocab_size=1024.
        Use create_beam_search() factory function which automatically
        calculates these from vocab_size.
        """
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
        vocab_size: int = 1024,
        sos_id: int = 1023,
        eos_id: int = 1023,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
        reliability_threshold: float = 0.8,
        max_length: int = 500,
        device: str = "cpu",
        use_bbd: bool = True,
        bbd_conservative: bool = True,
    ):
        """Initialize BSBS decoder.

        Note: sos_id and eos_id defaults assume vocab_size=1024.
        Use create_beam_search() factory function which automatically
        calculates these from vocab_size.
        """
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

        # GLOBAL STATE TRACKING (matches ESPnet's batch_beam_search_online.py:90-99)
        # These persist across ALL blocks throughout the entire decoding session
        self.running_hyps = None      # Current active hypotheses (persists across blocks)
        self.prev_hyps = []          # Previous iteration hypotheses (for rewinding on EOS)
        self.ended_hyps = []         # Completed hypotheses (reached EOS)
        self.process_idx = 0         # Global position in beam search loop (NOT reset between blocks!)
        self.prev_output = None      # Previous streaming output

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

    def reset(self):
        """Reset streaming state between utterances.

        Matches ESPnet's reset() in batch_beam_search_online.py:90-99
        Call this before processing a new audio utterance.
        """
        self.encoder_buffer = None
        self.running_hyps = None
        self.prev_hyps = []
        self.ended_hyps = []
        self.processed_block = 0
        self.process_idx = 0
        self.prev_output = None
        logger.debug("BlockwiseSynchronousBeamSearch state reset")

    def _copy_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Deep copy hypotheses including all scorer states.

        This is CRITICAL for rewinding mechanism. When we save prev_hyps,
        we need a true deep copy so that subsequent modifications don't
        affect the saved state.

        Args:
            hypotheses: List of hypotheses to copy

        Returns:
            Deep copied list of hypotheses
        """
        import copy

        copied = []
        for hyp in hypotheses:
            # Deep copy states (most complex part)
            new_states = {}
            for scorer_name, state in hyp.states.items():
                scorer = self.scorers.get(scorer_name)

                # Try scorer-specific copy method first
                if scorer and hasattr(scorer, 'copy_state'):
                    new_states[scorer_name] = scorer.copy_state(state)
                else:
                    # Fallback: try generic deep copy
                    try:
                        new_states[scorer_name] = copy.deepcopy(state)
                    except Exception as e:
                        # Last resort: reference (risky but better than crash)
                        logger.warning(f"Could not copy state for {scorer_name}: {e}. Using reference.")
                        new_states[scorer_name] = state

            # Create new hypothesis with copied data
            copied.append(Hypothesis(
                yseq=hyp.yseq.clone(),  # Clone tensor
                score=hyp.score,  # float, immutable
                scores=hyp.scores.copy(),  # Shallow copy dict (floats inside)
                states=new_states,  # Deep copied states
                xpos=hyp.xpos.clone() if hyp.xpos is not None else None,  # Clone tensor
            ))

        return copied

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
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"extend_scorers: Extending {len(hypotheses)} hypotheses with encoder_out shape {encoder_out.shape}")

        for scorer_name, scorer in self.scorers.items():
            if hasattr(scorer, "extend_prob"):
                # Extend CTC probability matrix with new encoder output
                # print(f"[DEBUG] extend_scorers: Calling {scorer_name}.extend_prob() with shape {encoder_out.shape}")
                scorer.extend_prob(encoder_out)
                # print(f"[DEBUG] extend_scorers: {scorer_name}.extend_prob() complete")

        # Extend hypothesis states to match new probability matrix size
        updated_hypotheses = []
        for idx, hyp in enumerate(hypotheses):
            # Copy hypothesis and update states
            new_states = {}
            for scorer_name in hyp.states:
                scorer = self.scorers.get(scorer_name)
                if scorer and hasattr(scorer, "extend_state"):
                    # Extend this scorer's state
                    old_state = hyp.states[scorer_name]
                    # if old_state and len(old_state) >= 1 and hasattr(old_state[0], 'shape'):
                        # print(f"[DEBUG] extend_scorers: hyp {idx} {scorer_name} state T={old_state[0].shape[0]} before extend")
                    new_states[scorer_name] = scorer.extend_state(old_state)
                    # if new_states[scorer_name] and len(new_states[scorer_name]) >= 1 and hasattr(new_states[scorer_name][0], 'shape'):
                        # print(f"[DEBUG] extend_scorers: hyp {idx} {scorer_name} state T={new_states[scorer_name][0].shape[0]} after extend")
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

            # Special case: SOS == EOS (1023)
            # [SOS, ..., EOS] is valid - EOS at the end is expected!
            # Only detect ACTUAL repetition (same token appearing in middle of sequence)
            # Skip SOS/EOS tokens in repetition check
            if last_token == self.sos_id or last_token == self.eos_id:
                logger.debug(f"BBD: Skipping SOS/EOS token {last_token} in repetition detection")
                continue

            # Check if last token appears in MIDDLE of sequence (not start/end)
            # Exclude first token (SOS) from repetition check
            middle_tokens = hyp.yseq[1:-1].tolist()
            if last_token in middle_tokens:
                # print(f"[DEBUG] BBD: Detected repetition - token {last_token} appears in middle of sequence {hyp.yseq.tolist()}")
                logger.debug(f"BBD: Detected repetition - token {last_token} in {hyp.yseq.tolist()}")
                return True

        return False

    def process_block(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        is_final: bool = False,
    ) -> BeamState:
        """Process features with encoder buffering and block extraction.

        NOW USES GLOBAL STATE! Matches ESPnet's batch_beam_search_online.py architecture.
        - Uses self.running_hyps instead of prev_state parameter
        - Hypotheses persist across ALL blocks (not reset between blocks)
        - Enables proper rewinding mechanism when EOS detected

        Args:
            features: Input features (1, time, feat_dim)
            feature_lens: Feature lengths (1,)
            is_final: Whether this is the final block

        Returns:
            Updated beam state (for backward compatibility with Speech2TextStreaming)
        """
        # Initialize global state on first call
        # Matches ESPnet: if self.running_hyps is None, initialize
        if self.running_hyps is None:
            self.running_hyps = [create_initial_hypothesis(self.sos_id, device=self.device)]
            logger.debug("Initialized self.running_hyps with initial hypothesis")

        # Create prev_state from global state for encoder states tracking
        # NOTE: This is temporary during refactoring - eventually encoder states
        # should also be global, but that's a bigger change
        prev_state = BeamState(
            hypotheses=self.running_hyps,  # Use global hypotheses!
            encoder_states=getattr(self, '_encoder_states', None),  # Temp storage
            processed_frames=0,  # Not used anymore
        )

        # Step 1: Encode features
        # CRITICAL: Use infer_mode=True for streaming!
        # ESPnet streaming uses infer_mode=True (asr_inference_streaming.py:330)

        logger.debug(f"process_block: features={features.shape}, is_final={is_final}, prev_encoder_states={'present' if prev_state.encoder_states else 'None'}")

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
            logger.debug(f"Calling encoder with features={features.shape}, prev_states={'present' if prev_state.encoder_states else 'None'}")
            encoder_out, encoder_out_lens, encoder_states = self.encoder(
                features,
                feature_lens,
                prev_states=prev_state.encoder_states,
                is_final=is_final,
                infer_mode=True,  # Streaming mode!
            )
            logger.debug(f"Encoder returned: {encoder_out.shape}")

        # Step 2: Accumulate encoder outputs in buffer
        # This is the CRITICAL missing piece! Matches ESPnet's encbuffer logic
        # (batch_beam_search_online.py:118-121)
        logger.debug(f"Encoder produced: {encoder_out.shape}")
        if encoder_out.size(1) > 0:  # Only accumulate if encoder produced output
            if self.encoder_buffer is None:
                self.encoder_buffer = encoder_out
                logger.debug(f"Initialized encoder buffer: {self.encoder_buffer.shape}")
            else:
                self.encoder_buffer = torch.cat([self.encoder_buffer, encoder_out], dim=1)
                logger.debug(f"Accumulated encoder buffer: {self.encoder_buffer.shape}")
        else:
            logger.debug(f"Encoder produced empty output, not accumulating")

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
            # NOTE: ESPnet uses strict < comparison, not <=
            if self.encoder_buffer is not None and cur_end_frame < self.encoder_buffer.shape[1]:
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

        # Update GLOBAL state from processing results
        # CRITICAL: This is where hypotheses persist across blocks!
        if ret is not None:
            # Update global hypotheses from block processing
            self.running_hyps = ret.hypotheses
            logger.debug(f"Updated self.running_hyps to {len(self.running_hyps)} hypotheses")

        # Store encoder states globally for next call
        self._encoder_states = encoder_states

        # Return BeamState for backward compatibility with Speech2TextStreaming
        # NOTE: This return value is less important now that state is global,
        # but Speech2TextStreaming still expects it
        return BeamState(
            hypotheses=self.running_hyps,
            encoder_states=self._encoder_states,
            processed_frames=0,  # Not tracked anymore
        )

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
        # print(f"[DEBUG] _decode_one_block: Calling extend_scorers with {len(prev_state.hypotheses)} hypotheses")
        extended_hypotheses = self.extend_scorers(encoder_out, prev_state.hypotheses)
        # print(f"[DEBUG] _decode_one_block: Extended to {len(extended_hypotheses)} hypotheses")

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
        # CRITICAL: Use GLOBAL process_idx that persists across blocks!
        # This matches ESPnet: while self.process_idx < maxlen (batch_beam_search_online.py:395)
        if encoder_out.size(1) > 0:
            # Track hypotheses from previous step for BBD rollback
            prev_step_hypotheses = new_state.hypotheses

            # Decode until BBD/EOS/max length detected
            # GLOBAL loop - process_idx continues across ALL blocks!
            while self.process_idx < self.max_length:
                # print(f"[DEBUG] _decode_one_block: process_idx={self.process_idx}/{self.max_length}, is_final={is_final}")
                new_state.output_index += 1

                # Score current hypotheses and get new states
                scores, new_states_dict = self.beam_search.batch_score_hypotheses(
                    new_state.hypotheses, encoder_out
                )

                # Expand and prune beam FIRST, then check for repetition/EOS
                # This matches ESPnet's approach (expand, then check, then potentially rollback)

                # DEBUG: Log first step scores for each block
                if self.process_idx == 0:
                    logger.debug(f"Block {self.processed_block}, Step 0 scores (frames={encoder_out.size(1)}):")
                    top_10_scores, top_10_tokens = torch.topk(scores[0], 10)
                    for j, (s, t) in enumerate(zip(top_10_scores.tolist(), top_10_tokens.tolist())):
                        logger.debug(f"  {j+1}. Token {t:4d}: {s:>8.4f}")

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

                # Check for completed hypotheses (ending with EOS)
                # Break when ANY hypothesis reaches EOS (conservative approach)
                # This achieved 768/835 words (92% parity)
                completed_hyps = [h for h in new_state.hypotheses if h.yseq[-1].item() == self.eos_id]

                if len(completed_hyps) > 0:
                    if not is_final:
                        # Break immediately when EOS detected
                        # print(f"[DEBUG] EOS: {len(completed_hyps)} hyp(s) reached EOS at process_idx={self.process_idx}, breaking")
                        logger.info(f"Detected hyp(s) reaching EOS in this block, stopping.")
                        break
                    else:
                        # For final: stop when best hypothesis reaches EOS
                        best_hyp = max(new_state.hypotheses, key=lambda h: h.score)
                        best_has_eos = best_hyp.yseq[-1].item() == self.eos_id

                        if best_has_eos:
                            logger.info(f"Best hypothesis reached EOS in final block.")
                            break

                # BBD: Check for repetition AFTER beam expansion
                # Matches ESPnet's approach (batch_beam_search_online.py:209-218)
                if self.use_bbd and not is_final:
                    # print(f"[DEBUG] BBD: Checking repetition at process_idx={self.process_idx}, is_final={is_final}")
                    has_repetition = self.detect_repetition_or_eos(new_state.hypotheses)
                    # print(f"[DEBUG] BBD: has_repetition={has_repetition}")

                    if has_repetition:
                        # print(f"[DEBUG] BBD: Repetition detected at process_idx={self.process_idx}, stopping block")
                        # for hyp in new_state.hypotheses:
                            # print(f"[DEBUG] BBD: Hypothesis: {hyp.yseq.tolist()}")
                        logger.debug(f"BBD: Repetition detected at process_idx={self.process_idx}, stopping block")

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
                    logger.debug(f"All hypotheses ended with EOS at process_idx={self.process_idx}")
                    break

                # CRITICAL: Save state for potential rewind
                # This happens AFTER all break conditions (EOS, BBD), so if we break,
                # prev_hyps is NOT updated and still contains the previous good state
                # Matches ESPnet: self.prev_hyps = self.running_hyps (line 448)
                self.prev_hyps = self._copy_hypotheses(new_state.hypotheses)
                logger.debug(f"Saved prev_hyps ({len(self.prev_hyps)} hypotheses) after all checks")

                # CRITICAL: Increment global position counter!
                # This is what makes process_idx persist across blocks
                # Matches ESPnet: self.process_idx += 1 (batch_beam_search_online.py:463)
                self.process_idx += 1
                logger.debug(f"Incremented process_idx to {self.process_idx}")

        # REWINDING MECHANISM (ESPnet batch_beam_search_online.py:477-480)
        # CRITICAL FIX: Use > 1 (NOT >= 1) to match ESPnet exactly!
        # This allows process_idx to progress past 0, preventing infinite loops
        if self.process_idx > 1 and len(self.prev_hyps) > 0:
            # print(f"[DEBUG] REWIND: Rewinding from process_idx={self.process_idx} to {self.process_idx-1}")
            # print(f"[DEBUG] REWIND: Restoring {len(self.prev_hyps)} prev_hyps")
            # Restore hypotheses to state before EOS was predicted
            new_state.hypotheses = self.prev_hyps
            # Go back one step in global position
            self.process_idx -= 1
            # Clear prev_hyps for next block
            self.prev_hyps = []
            logger.info(f"REWOUND to process_idx={self.process_idx}")

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
    sos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
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
        sos_id: Start-of-sequence token ID (default: vocab_size - 1)
        eos_id: End-of-sequence token ID (default: vocab_size - 1)

    Returns:
        BlockwiseSynchronousBeamSearch instance
    """
    from speechcatcher.beam_search.scorers import CTCPrefixScorer, DecoderScorer

    # Calculate token IDs from vocab_size if not provided
    # ESPnet token list structure: [<blank>, SP[0], SP[3..N], <sos/eos>]
    # So <sos/eos> is always at position vocab_size - 1
    if sos_id is None:
        sos_id = model.vocab_size - 1
    if eos_id is None:
        eos_id = model.vocab_size - 1

    # Create scorers
    scorers = {}
    weights = {}

    if model.decoder is not None:
        scorers["decoder"] = DecoderScorer(model.decoder, sos_id=sos_id, eos_id=eos_id)
        weights["decoder"] = decoder_weight

    # CTC scoring with full forward algorithm implementation
    # Now uses batch_score_partial for top-K scoring (25x speedup!)
    if model.ctc is not None and ctc_weight > 0:
        scorers["ctc"] = CTCPrefixScorer(model.ctc, blank_id=0, eos_id=eos_id)
        weights["ctc"] = ctc_weight

    # Create BSBS
    return BlockwiseSynchronousBeamSearch(
        encoder=model.encoder,
        scorers=scorers,
        weights=weights,
        beam_size=beam_size,
        vocab_size=model.vocab_size,
        sos_id=sos_id,
        eos_id=eos_id,
        device=device,
        use_bbd=use_bbd,
        bbd_conservative=bbd_conservative,
    )
