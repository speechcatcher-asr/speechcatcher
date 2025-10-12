"""Speech2TextStreaming API - drop-in replacement for ESPnet streaming interface."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from speechcatcher.beam_search import BlockwiseSynchronousBeamSearch, create_beam_search
from speechcatcher.model import ESPnetASRModel
from speechcatcher.model.checkpoint_loader import (
    apply_feature_normalization,
    load_espnet_model_from_directory,
    load_normalization_stats,
)

logger = logging.getLogger(__name__)

# Try to import sentencepiece for BPE tokenization
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not installed - text decoding will not be available")


class Speech2TextStreaming:
    """Streaming speech recognition interface.

    This class provides a drop-in replacement for ESPnet's
    Speech2TextStreaming interface for streaming ASR.

    Args:
        model_dir: Directory containing model checkpoint and config
        beam_size: Beam size for beam search (default: 10)
        ctc_weight: CTC loss weight in joint training (default: 0.3)
        device: Device to run inference on (default: "cpu")
        dtype: Data type for model (default: "float32")
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        beam_size: int = 10,
        ctc_weight: float = 0.3,
        device: str = "cpu",
        dtype: str = "float32",
        use_bbd: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.use_amp = (dtype == "float16" and device.startswith("cuda"))
        self.use_bbd = use_bbd

        # Load model
        logger.info(f"Loading model from {self.model_dir}")
        self.model = self._load_model()

        # For AMP (automatic mixed precision), keep model in FP32
        # Autocast will automatically downcast operations to FP16 where safe
        if self.use_amp:
            self.model = self.model.to(device).to(torch.float32)
            logger.info("Using AMP: model weights in FP32, operations auto-cast to FP16")
        else:
            self.model = self.model.to(device).to(self.dtype)

        self.model.eval()

        # Load normalization stats
        # Try multiple possible locations
        stats_paths = [
            self.model_dir / "feats_stats.npz",
            self.model_dir.parent / "asr_stats_raw_de_bpe1024/train/feats_stats.npz",  # ESPnet structure (correct path)
            self.model_dir.parent.parent / "asr_stats_raw_de_bpe1024/train/feats_stats.npz",  # Alternative
            self.model_dir / "../stats/train/feats_stats.npz",
        ]

        self.mean = None
        self.std = None
        for stats_path in stats_paths:
            if stats_path.exists():
                try:
                    self.mean, self.std = load_normalization_stats(stats_path)
                    logger.info(f"Loaded stats from {stats_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load stats from {stats_path}: {e}")

        if self.mean is None:
            logger.warning(f"Normalization stats not found in any of: {[str(p) for p in stats_paths]}")

        # Load BPE tokenizer if available
        self.tokenizer = None
        self.token_list = None  # ESPnet's token vocabulary (with <blank> at position 0)
        if HAS_SENTENCEPIECE:
            bpe_paths = [
                self.model_dir / "bpe.model",
                self.model_dir.parent.parent / "data/de_token_list/bpe_unigram1024/bpe.model",
                self.model_dir / "../data/de_token_list/bpe_unigram1024/bpe.model",
            ]

            for bpe_path in bpe_paths:
                if bpe_path.exists():
                    try:
                        self.tokenizer = spm.SentencePieceProcessor()
                        self.tokenizer.Load(str(bpe_path))
                        logger.info(f"Loaded BPE tokenizer from {bpe_path}")

                        # Build ESPnet-style token list
                        # ESPnet removes <s> (SP ID 1) and </s> (SP ID 2) tokens
                        # ESPnet vocabulary = ["<blank>", SP[0], SP[3..1023], "<sos/eos>"]
                        vocab_size = self.tokenizer.GetPieceSize()
                        self.token_list = (
                            ["<blank>", self.tokenizer.IdToPiece(0)] +
                            [self.tokenizer.IdToPiece(i) for i in range(3, vocab_size)] +
                            ["<sos/eos>"]
                        )
                        logger.info(f"Built token list with {len(self.token_list)} tokens")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load BPE from {bpe_path}: {e}")

            if self.tokenizer is None:
                logger.warning(f"BPE tokenizer not found in any of: {[str(p) for p in bpe_paths]}")
        else:
            logger.warning("sentencepiece not installed - install with: pip install sentencepiece")

        # Frontend parameters for streaming (needed for apply_frontend)
        if self.model.frontend is not None:
            self.win_length = self.model.frontend.win_length
            self.hop_length = self.model.frontend.hop_length
        else:
            self.win_length = 400  # Default
            self.hop_length = 160  # Default

        # Create beam search
        logger.info(f"Creating beam search with beam_size={beam_size}, use_bbd={use_bbd}")
        self.beam_search = create_beam_search(
            model=self.model,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
            decoder_weight=1.0 - ctc_weight,
            device=device,
            use_bbd=use_bbd,
        )

        # Streaming state
        self.reset()

        logger.info("Speech2TextStreaming initialized")

    def _load_model(self) -> ESPnetASRModel:
        """Load model from directory."""
        # Build a dummy model first to get the structure
        # Then load weights

        # Try different possible checkpoint locations
        search_paths = [
            self.model_dir / "valid.acc.best.pth",
            self.model_dir / "valid.acc.ave_6best.pth",
            self.model_dir / "valid.acc.ave.pth",
            self.model_dir / "model.pth",
            self.model_dir / "checkpoint.pth",
        ]

        # Also search in exp/ subdirectories (ESPnet model structure)
        exp_dirs = list(self.model_dir.glob("exp/*/"))
        for exp_dir in exp_dirs:
            search_paths.extend([
                exp_dir / "valid.acc.best.pth",
                exp_dir / "valid.acc.ave_6best.pth",
                exp_dir / "valid.acc.ave.pth",
                exp_dir / "model.pth",
                exp_dir / "checkpoint.pth",
            ])

        checkpoint_path = None
        for path in search_paths:
            if path.exists():
                checkpoint_path = path
                break

        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in {self.model_dir}")

        # Load checkpoint using proper name mapping
        from speechcatcher.model.checkpoint_loader import load_espnet_model_from_directory

        # First, build model architecture by inferring from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # Infer vocab size
        vocab_size = None
        if "decoder.embed.0.weight" in state_dict:
            vocab_size = state_dict["decoder.embed.0.weight"].shape[0]
        elif "decoder.output_layer.weight" in state_dict:
            vocab_size = state_dict["decoder.output_layer.weight"].shape[0]

        if vocab_size is None:
            raise ValueError("Could not infer vocab_size from checkpoint")

        # Load config if available
        import yaml
        config_path = self.model_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            encoder_conf = config.get("encoder_conf", {})
            decoder_conf = config.get("decoder_conf", {})
            frontend_conf = config.get("frontend_conf", {})

            # Build model (read architecture params from config)
            model = ESPnetASRModel.build_model(
                vocab_size=vocab_size,
                input_size=80,  # Standard log-mel
                encoder_output_size=encoder_conf.get("output_size", 256),
                encoder_attention_heads=encoder_conf.get("attention_heads", 4),
                encoder_num_blocks=encoder_conf.get("num_blocks", 12),
                decoder_attention_heads=decoder_conf.get("attention_heads", 4),
                decoder_num_blocks=decoder_conf.get("num_blocks", 6),
                use_frontend=True,  # Enable STFT frontend for raw audio
                frontend_n_fft=frontend_conf.get("n_fft", 512),
                frontend_hop_length=frontend_conf.get("hop_length", 160),
                frontend_win_length=frontend_conf.get("win_length", 400),
            )
        else:
            # Fallback to default architecture
            model = ESPnetASRModel.build_model(
                vocab_size=vocab_size,
                input_size=80,
                encoder_output_size=256,
                encoder_num_blocks=12,
                decoder_num_blocks=6,
                use_frontend=True,  # Enable STFT frontend for raw audio
            )

        # Load weights using proper ESPnet -> speechcatcher name mapping
        from speechcatcher.model.checkpoint_loader import load_espnet_weights
        model, arch_info = load_espnet_weights(model, checkpoint_path, strict=False)

        logger.info(f"Loaded model with architecture: {arch_info}")

        return model

    def reset(self):
        """Reset streaming state."""
        self.beam_state = None
        self.processed_frames = 0
        self.frontend_states = None  # {"waveform_buffer": tensor}

        # Reset beam search state (encoder buffer, processed blocks, etc.)
        # This calls BlockwiseSynchronousBeamSearch.reset()
        if hasattr(self.beam_search, 'reset'):
            self.beam_search.reset()

        logger.debug("Streaming state reset")

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply feature normalization.

        Args:
            features: Input features (time, feat_dim)

        Returns:
            Normalized features
        """
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std
        return features

    def apply_frontend(
        self,
        speech: torch.Tensor,
        prev_states: Optional[Dict] = None,
        is_final: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
        """Apply frontend with waveform buffering and frame trimming.

        Based on espnet_streaming_decoder.asr_inference_streaming.apply_frontend (lines 206-300).

        Args:
            speech: Raw audio waveform (samples,)
            prev_states: Previous state dict with "waveform_buffer"
            is_final: Whether this is the final chunk

        Returns:
            - feats: Features (1, time, feat_dim) or None if not enough samples
            - feats_lengths: Feature lengths (1,) or None
            - next_states: Next state dict or None
        """
        import math

        # 1. Concatenate with buffer from previous chunk
        if prev_states is not None and "waveform_buffer" in prev_states:
            buf = prev_states["waveform_buffer"]
            speech = torch.cat([buf, speech], dim=0)

        # 2. Check if we have enough samples for STFT window
        has_enough_samples = speech.size(0) > self.win_length
        if not has_enough_samples:
            if is_final:
                # Pad with zeros to reach win_length
                pad = torch.zeros(
                    self.win_length - speech.size(0),
                    dtype=speech.dtype,
                    device=speech.device,
                )
                speech = torch.cat([speech, pad], dim=0)
            else:
                # Not enough samples yet, buffer and return None
                next_states = {"waveform_buffer": speech.clone()}
                return None, None, next_states

        # 3. Determine how much to process vs buffer for next chunk
        if is_final:
            # Process everything
            speech_to_process = speech
            waveform_buffer = None
        else:
            # Calculate number of frames we can produce
            n_frames = (speech.size(0) - (self.win_length - self.hop_length)) // self.hop_length
            n_residual = (speech.size(0) - (self.win_length - self.hop_length)) % self.hop_length

            # Process only complete frames, keep residual for next chunk
            process_length = (self.win_length - self.hop_length) + n_frames * self.hop_length
            speech_to_process = speech.narrow(0, 0, process_length)

            # Buffer includes overlap for STFT continuity
            buffer_start = speech.size(0) - (self.win_length - self.hop_length) - n_residual
            buffer_length = (self.win_length - self.hop_length) + n_residual
            waveform_buffer = speech.narrow(0, buffer_start, buffer_length).clone()

        # 4. Extract features using frontend
        # Add batch dimension: (samples,) -> (1, samples)
        speech_to_process = speech_to_process.unsqueeze(0).to(self.dtype)

        if self.model.frontend is not None:
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        feats, feats_lengths = self.model.frontend(speech_to_process)
                else:
                    feats, feats_lengths = self.model.frontend(speech_to_process)
        else:
            raise RuntimeError("Model has no frontend")

        # 5. Apply normalization
        if self.mean is not None and self.std is not None:
            feats_np = feats.squeeze(0).cpu().numpy()  # (time, feat_dim)
            feats_np = self.normalize_features(feats_np)
            feats = torch.from_numpy(feats_np).unsqueeze(0).to(self.device).to(self.dtype)

        # 6. Trim overlapping frames at chunk boundaries
        # Calculate trim amount: half of the frames needed for one STFT window
        trim_frames = math.ceil(math.ceil(self.win_length / self.hop_length) / 2)

        if is_final:
            # Final chunk: trim beginning (keep end)
            if prev_states is None:
                # First and only chunk - no trimming
                pass
            else:
                # Trim the overlapping frames from the beginning
                if feats.size(1) > trim_frames:
                    feats = feats.narrow(1, trim_frames, feats.size(1) - trim_frames)
        else:
            # Non-final chunk: trim beginning and/or end
            if prev_states is None:
                # First chunk: only trim end
                if feats.size(1) > trim_frames:
                    feats = feats.narrow(1, 0, feats.size(1) - trim_frames)
            else:
                # Middle chunk: trim both ends
                total_trim = 2 * trim_frames
                if feats.size(1) > total_trim:
                    feats = feats.narrow(1, trim_frames, feats.size(1) - total_trim)
                else:
                    # Too short after trimming, skip this chunk
                    logger.warning(f"Feature chunk too short after trimming: {feats.size(1)} frames")
                    # Return None but keep buffer for next chunk
                    next_states = {"waveform_buffer": waveform_buffer} if waveform_buffer is not None else None
                    return None, None, next_states

        # Update feature lengths after trimming
        feats_lengths = torch.tensor([feats.size(1)], dtype=torch.long, device=self.device)

        # 7. Prepare next state
        if is_final:
            next_states = None
        else:
            next_states = {"waveform_buffer": waveform_buffer}

        return feats, feats_lengths, next_states

    def __call__(
        self,
        speech: Union[np.ndarray, torch.Tensor],
        is_final: bool = False,
    ) -> List[Tuple[str, List[str], List[int]]]:
        """Process speech chunk and return recognition results.

        Args:
            speech: Input speech chunk - raw audio (samples,) or features (time, feat_dim)
            is_final: Whether this is the final chunk

        Returns:
            List of (text, tokens, token_ids) tuples for each hypothesis
        """
        # Convert to tensor if needed
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)

        speech = speech.to(self.device).to(self.dtype)

        # For raw audio (1D), apply frontend with buffering and trimming
        if speech.dim() == 1:
            # Apply frontend with proper streaming handling
            feats, feats_lengths, self.frontend_states = self.apply_frontend(
                speech, self.frontend_states, is_final=is_final
            )

            # If not enough samples yet (None returned), return empty results
            if feats is None:
                return []

            speech = feats
            speech_lengths = feats_lengths

        elif speech.dim() == 2:
            # Pre-computed features - normalize them
            speech_np = speech.cpu().numpy() if isinstance(speech, torch.Tensor) else speech
            speech_np = self.normalize_features(speech_np)
            speech = torch.from_numpy(speech_np).to(self.device).to(self.dtype)

            # Add batch dimension: (time, feat_dim) -> (1, time, feat_dim)
            speech = speech.unsqueeze(0)
            speech_lengths = torch.tensor([speech.size(1)], device=self.device)
        else:
            # Already batched (3D)
            speech_lengths = torch.tensor([speech.size(1)], device=self.device)

        # Process block with features (not raw audio)
        # Use automatic mixed precision (autocast) if FP16 is enabled
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    self.beam_state = self.beam_search.process_block(
                        speech, speech_lengths, self.beam_state, is_final
                    )
            else:
                self.beam_state = self.beam_search.process_block(
                    speech, speech_lengths, self.beam_state, is_final
                )

        # Convert hypotheses to output format
        # For streaming: only output COMPLETED hypotheses (with EOS)
        # For final: output all hypotheses
        if not is_final:
            # Filter to only completed hypotheses (ending with EOS)
            output_hyps = [h for h in self.beam_state.hypotheses if h.yseq[-1].item() == 1023]
            # If no completed hypotheses, output nothing
            if not output_hyps:
                output_hyps = []
        else:
            # For final: output all hypotheses
            output_hyps = self.beam_state.hypotheses

        results = []
        for hyp in output_hyps:
            # DEBUG: Show what we're outputting
            logger.debug(f"Output: output_index={self.beam_state.output_index}, hyp.yseq length={len(hyp.yseq)}, first 30 tokens={hyp.yseq.tolist()[:30]}")
            if is_final and len(hyp.yseq) > 50:
                print(f"[DEBUG] Final hypothesis has {len(hyp.yseq)} tokens! First 50: {hyp.yseq.tolist()[:50]}")

            # Extract committed tokens only (up to output_index)
            # yseq structure: [SOS, t1, t2, ..., tn, (EOS if final)]
            # output_index: number of tokens committed (excluding SOS)
            #
            # During streaming: output tokens 1 to output_index (inclusive)
            # After final: output tokens 1 to -1 (excluding SOS and EOS)
            if is_final:
                # Remove SOS token (always present at position 0)
                token_ids = hyp.yseq[1:]
                # Remove EOS token if present at the end
                if len(token_ids) > 0 and token_ids[-1].item() == 1023:
                    token_ids = token_ids[:-1]
            else:
                # Only output committed tokens (yseq[1:output_index+1])
                # output_index is the last committed token position
                end_idx = min(self.beam_state.output_index + 1, len(hyp.yseq))
                token_ids = hyp.yseq[1:end_idx]

                # If hypothesis ended with EOS, exclude it from output
                # This happens when a hypothesis completes during streaming
                if len(token_ids) > 0 and token_ids[-1].item() == 1023:
                    token_ids = token_ids[:-1]

            # Remove special tokens:
            # - <blank> (ID=0): CTC blank token
            # - <unk> (ID=1): Unknown token
            # - <sos/eos> (ID=1023): Start/end of sentence token
            # ESPnet filters these before converting to text
            token_ids_filtered = [tid for tid in token_ids if tid not in [0, 1, 1023]]

            # Decode to text using ESPnet's token list
            # The token IDs are in ESPnet vocabulary space (with <blank> at position 0)
            if self.token_list is not None:
                # Convert tensor token IDs to integers and look up in token_list
                tokens = [self.token_list[int(tid)] for tid in token_ids_filtered]
                # Join pieces and replace sentencepiece underscore with space
                text = "".join(tokens).replace("▁", " ").strip()
            elif self.tokenizer is not None:
                # Fallback: use raw SentencePiece (will be wrong if model uses ESPnet vocab!)
                logger.warning("Using raw SentencePiece - token IDs may not match!")
                tokens = [self.tokenizer.IdToPiece(int(tid)) for tid in token_ids_filtered]
                text = "".join(tokens).replace("▁", " ").strip()
            else:
                # Fallback: return token IDs as strings
                tokens = [str(tid) for tid in token_ids_filtered]
                text = " ".join(tokens)

            results.append((text, tokens, list(token_ids_filtered)))

        return results

    def recognize(
        self,
        speech: Union[np.ndarray, torch.Tensor],
    ) -> List[Tuple[str, List[str], List[int]]]:
        """Non-streaming recognition (process entire utterance).

        Args:
            speech: Input speech (samples,) or features (time, feat_dim)

        Returns:
            List of (text, tokens, token_ids) tuples
        """
        # Reset state
        self.reset()

        # Process as final chunk
        return self(speech, is_final=True)

    def recognize_stream(
        self,
        chunks: List[Union[np.ndarray, torch.Tensor]],
    ) -> List[Tuple[str, List[str], List[int]]]:
        """Streaming recognition with multiple chunks.

        Args:
            chunks: List of speech chunks

        Returns:
            Final recognition results
        """
        # Reset state
        self.reset()

        results = None
        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            results = self(chunk, is_final=is_final)

        return results if results is not None else []

    @property
    def n_best_hypotheses(self) -> int:
        """Return number of hypotheses (beam size)."""
        return self.beam_size

    def get_best_hypothesis(self) -> Optional[Tuple[str, List[str], List[int]]]:
        """Get the best hypothesis from current state.

        Returns:
            Best hypothesis as (text, tokens, token_ids) or None
        """
        if self.beam_state is None or not self.beam_state.hypotheses:
            return None

        results = self(torch.zeros(1, 1, 80, device=self.device), is_final=True)
        return results[0] if results else None


def create_streaming_interface(
    model_dir: Union[str, Path],
    beam_size: int = 10,
    ctc_weight: float = 0.3,
    device: str = "cpu",
) -> Speech2TextStreaming:
    """Create Speech2TextStreaming interface.

    Args:
        model_dir: Directory containing model checkpoint and config
        beam_size: Beam size for beam search
        ctc_weight: CTC weight
        device: Device to run on

    Returns:
        Speech2TextStreaming instance
    """
    return Speech2TextStreaming(
        model_dir=model_dir,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        device=device,
    )
