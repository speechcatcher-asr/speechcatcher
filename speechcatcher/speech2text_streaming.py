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
    ):
        self.model_dir = Path(model_dir)
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        self.device = device
        self.dtype = getattr(torch, dtype)

        # Load model
        logger.info(f"Loading model from {self.model_dir}")
        self.model = self._load_model()
        self.model = self.model.to(device).to(self.dtype)
        self.model.eval()

        # Load normalization stats
        stats_path = self.model_dir / "feats_stats.npz"
        if stats_path.exists():
            self.mean, self.std = load_normalization_stats(stats_path)
        else:
            logger.warning(f"Normalization stats not found: {stats_path}")
            self.mean = None
            self.std = None

        # Create beam search
        logger.info(f"Creating beam search with beam_size={beam_size}")
        self.beam_search = create_beam_search(
            model=self.model,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
            decoder_weight=1.0 - ctc_weight,
            device=device,
        )

        # Streaming state
        self.reset()

        logger.info("Speech2TextStreaming initialized")

    def _load_model(self) -> ESPnetASRModel:
        """Load model from directory."""
        # Build a dummy model first to get the structure
        # Then load weights
        checkpoint_path = self.model_dir / "valid.acc.best.pth"
        if not checkpoint_path.exists():
            # Try alternative checkpoint names
            for alt_name in ["model.pth", "checkpoint.pth"]:
                alt_path = self.model_dir / alt_name
                if alt_path.exists():
                    checkpoint_path = alt_path
                    break

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found in {self.model_dir}")

        # Load checkpoint to infer architecture
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

            # Build model
            model = ESPnetASRModel.build_model(
                vocab_size=vocab_size,
                input_size=80,  # Standard log-mel
                encoder_output_size=encoder_conf.get("output_size", 256),
                encoder_attention_heads=encoder_conf.get("attention_heads", 4),
                encoder_num_blocks=encoder_conf.get("num_blocks", 12),
                decoder_attention_heads=decoder_conf.get("attention_heads", 4),
                decoder_num_blocks=decoder_conf.get("num_blocks", 6),
                use_frontend=False,  # We'll handle features externally
            )
        else:
            # Fallback to default architecture
            model = ESPnetASRModel.build_model(
                vocab_size=vocab_size,
                input_size=80,
                encoder_output_size=256,
                encoder_num_blocks=12,
                decoder_num_blocks=6,
                use_frontend=False,
            )

        # Load weights
        model.load_state_dict(state_dict, strict=False)

        return model

    def reset(self):
        """Reset streaming state."""
        self.beam_state = None
        self.processed_frames = 0
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

    def __call__(
        self,
        speech: Union[np.ndarray, torch.Tensor],
        is_final: bool = False,
    ) -> List[Tuple[str, List[str], List[int]]]:
        """Process speech chunk and return recognition results.

        Args:
            speech: Input speech chunk (samples,) or features (time, feat_dim)
            is_final: Whether this is the final chunk

        Returns:
            List of (text, tokens, token_ids) tuples for each hypothesis
        """
        # Convert to tensor if needed
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)

        speech = speech.to(self.device).to(self.dtype)

        # Ensure 2D (time, feat_dim)
        if speech.dim() == 1:
            # Raw audio - need to extract features
            # For now, assume features are already extracted
            raise NotImplementedError("Raw audio input not yet supported")

        # Normalize features
        if isinstance(speech, torch.Tensor):
            speech_np = speech.cpu().numpy()
        else:
            speech_np = speech

        speech_np = self.normalize_features(speech_np)
        speech = torch.from_numpy(speech_np).to(self.device).to(self.dtype)

        # Add batch dimension
        if speech.dim() == 2:
            speech = speech.unsqueeze(0)  # (1, time, feat_dim)

        speech_lengths = torch.tensor([speech.size(1)], device=self.device)

        # Process block
        with torch.no_grad():
            self.beam_state = self.beam_search.process_block(
                speech, speech_lengths, self.beam_state, is_final
            )

        # Convert hypotheses to output format
        results = []
        for hyp in self.beam_state.hypotheses:
            # Remove SOS token
            token_ids = hyp.yseq[1:]  # Skip SOS

            # For now, return token IDs as strings (proper tokenizer needed)
            tokens = [str(tid) for tid in token_ids]
            text = " ".join(tokens)

            results.append((text, tokens, token_ids))

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
