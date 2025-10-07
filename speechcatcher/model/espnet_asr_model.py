"""ESPnet-compatible ASR model wrapper."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from speechcatcher.model.ctc import CTC
from speechcatcher.model.decoder import TransformerDecoder
from speechcatcher.model.encoder import ContextualBlockTransformerEncoder
from speechcatcher.model.frontend import STFTFrontend

logger = logging.getLogger(__name__)


class ESPnetASRModel(nn.Module):
    """ESPnet-compatible ASR model.

    This is the main model class that combines:
    - Frontend: STFT feature extraction
    - Encoder: Contextual block streaming Transformer
    - Decoder: Transformer decoder with attention
    - CTC: Connectionist Temporal Classification head

    Args:
        vocab_size: Size of vocabulary
        frontend: Frontend module (STFTFrontend)
        encoder: Encoder module (ContextualBlockTransformerEncoder)
        decoder: Decoder module (TransformerDecoder)
        ctc: CTC module (CTC)
        ctc_weight: Weight for CTC loss in joint training (default: 0.3)

    Shape:
        - Audio input: (batch, samples) or (batch, time, features)
        - Token input: (batch, target_len)
        - Output: Dict with 'encoder_out', 'decoder_out', 'ctc_out', 'loss'
    """

    def __init__(
        self,
        vocab_size: int,
        frontend: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        ctc: Optional[nn.Module] = None,
        ctc_weight: float = 0.3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight

        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training or inference.

        Args:
            speech: Audio input (batch, samples) or features (batch, time, feat)
            speech_lengths: Audio lengths (batch,)
            text: Target tokens (batch, target_len), optional for inference
            text_lengths: Target lengths (batch,), optional for inference

        Returns:
            Dictionary containing:
                - encoder_out: Encoder output (batch, enc_time, feat)
                - encoder_out_lens: Encoder output lengths (batch,)
                - ctc_logits: CTC logits if CTC module present
                - decoder_logits: Decoder logits if decoder present
                - loss: Combined loss if targets provided
                - ctc_loss: CTC loss component
                - att_loss: Attention (decoder) loss component
        """
        # Extract features if frontend present
        if self.frontend is not None:
            if speech.dim() == 2:  # Raw audio
                feats, feats_lengths = self.frontend(speech)
            else:  # Already features
                feats = speech
                feats_lengths = speech_lengths
        else:
            feats = speech
            feats_lengths = speech_lengths

        # Encode
        if self.encoder is not None:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, prev_states=None, is_final=True
            )
        else:
            encoder_out = feats
            encoder_out_lens = feats_lengths

        results = {
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
        }

        # CTC forward
        ctc_loss = None
        if self.ctc is not None:
            ctc_logits, ctc_loss = self.ctc(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            results["ctc_logits"] = ctc_logits
            if ctc_loss is not None:
                results["ctc_loss"] = ctc_loss

        # Decoder forward
        att_loss = None
        if self.decoder is not None and text is not None:
            # Decoder expects input tokens (text shifted right with <sos>)
            # For simplicity, we use text as-is (assuming it's already prepared)
            decoder_logits, decoder_out_lens = self.decoder(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            results["decoder_logits"] = decoder_logits
            results["decoder_out_lens"] = decoder_out_lens

            # Compute attention loss (cross-entropy)
            if text is not None:
                # Flatten for loss computation
                vocab_size = decoder_logits.size(-1)
                logits_flat = decoder_logits.view(-1, vocab_size)
                targets_flat = text.view(-1)

                # Compute cross-entropy loss (ignore padding)
                att_loss = nn.functional.cross_entropy(
                    logits_flat, targets_flat, ignore_index=0, reduction="mean"
                )
                results["att_loss"] = att_loss

        # Combined loss
        if ctc_loss is not None and att_loss is not None:
            loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * att_loss
            results["loss"] = loss
        elif ctc_loss is not None:
            results["loss"] = ctc_loss
        elif att_loss is not None:
            results["loss"] = att_loss

        return results

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        prev_states: Optional[Dict] = None,
        is_final: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Encode audio to hidden representations (for streaming inference).

        Args:
            speech: Audio input (batch, samples) or features (batch, time, feat)
            speech_lengths: Audio lengths (batch,)
            prev_states: Previous encoder states for streaming
            is_final: Whether this is the final chunk

        Returns:
            Tuple of (encoder_out, encoder_out_lens, next_states)
        """
        # Extract features
        if self.frontend is not None:
            if speech.dim() == 2:  # Raw audio
                feats, feats_lengths = self.frontend(speech)
            else:
                feats = speech
                feats_lengths = speech_lengths
        else:
            feats = speech
            feats_lengths = speech_lengths

        # Encode
        if self.encoder is not None:
            encoder_out, encoder_out_lens, next_states = self.encoder(
                feats, feats_lengths, prev_states=prev_states, is_final=is_final, infer_mode=True
            )
        else:
            encoder_out = feats
            encoder_out_lens = feats_lengths
            next_states = None

        return encoder_out, encoder_out_lens, next_states

    @classmethod
    def build_model(
        cls,
        vocab_size: int,
        input_size: int = 80,
        encoder_output_size: int = 256,
        encoder_attention_heads: int = 4,
        encoder_linear_units: int = 2048,
        encoder_num_blocks: int = 12,
        decoder_attention_heads: int = 4,
        decoder_linear_units: int = 2048,
        decoder_num_blocks: int = 6,
        use_frontend: bool = True,
        use_ctc: bool = True,
        use_decoder: bool = True,
        ctc_weight: float = 0.3,
        **kwargs,
    ) -> "ESPnetASRModel":
        """Build an ESPnet ASR model from configuration.

        Args:
            vocab_size: Vocabulary size
            input_size: Input feature dimension (e.g., 80 for log-mel)
            encoder_output_size: Encoder hidden dimension
            encoder_attention_heads: Encoder attention heads
            encoder_linear_units: Encoder FFN hidden dimension
            encoder_num_blocks: Number of encoder layers
            decoder_attention_heads: Decoder attention heads
            decoder_linear_units: Decoder FFN hidden dimension
            decoder_num_blocks: Number of decoder layers
            use_frontend: Whether to include STFT frontend
            use_ctc: Whether to include CTC head
            use_decoder: Whether to include attention decoder
            ctc_weight: CTC loss weight
            **kwargs: Additional arguments for encoder/decoder

        Returns:
            ESPnetASRModel instance
        """
        # Build frontend
        frontend = None
        if use_frontend:
            frontend = STFTFrontend(
                n_fft=512,
                hop_length=160,
                win_length=400,
                n_mels=input_size,
            )

        # Build encoder
        encoder = ContextualBlockTransformerEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            attention_heads=encoder_attention_heads,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_blocks,
            **{k: v for k, v in kwargs.items() if k.startswith("encoder_")},
        )

        # Build decoder
        decoder = None
        if use_decoder:
            decoder = TransformerDecoder(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                attention_heads=decoder_attention_heads,
                linear_units=decoder_linear_units,
                num_blocks=decoder_num_blocks,
                **{k: v for k, v in kwargs.items() if k.startswith("decoder_")},
            )

        # Build CTC
        ctc = None
        if use_ctc:
            ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)

        return cls(
            vocab_size=vocab_size,
            frontend=frontend,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_weight=ctc_weight,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Path,
        checkpoint_name: str = "valid.acc.best.pth",
        device: str = "cpu",
    ) -> "ESPnetASRModel":
        """Load a pretrained ESPnet model from directory.

        Args:
            model_dir: Directory containing checkpoint and config
            checkpoint_name: Name of checkpoint file
            device: Device to load model to

        Returns:
            ESPnetASRModel instance with loaded weights
        """
        from speechcatcher.model.checkpoint_loader import load_espnet_model_from_directory

        model_dir = Path(model_dir)

        # Load config to infer architecture
        import yaml
        config_path = model_dir / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Build model from config
        # This is simplified - real implementation should parse config more thoroughly
        encoder_conf = config.get("encoder_conf", {})
        decoder_conf = config.get("decoder_conf", {})

        # Try to infer vocab size from checkpoint first
        checkpoint_path = model_dir / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # Get vocab size from decoder embedding or output layer
        vocab_size = None
        if "decoder.embed.0.weight" in state_dict:
            vocab_size = state_dict["decoder.embed.0.weight"].shape[0]
        elif "decoder.output_layer.weight" in state_dict:
            vocab_size = state_dict["decoder.output_layer.weight"].shape[0]

        if vocab_size is None:
            raise ValueError("Could not infer vocab_size from checkpoint")

        # Build model
        model = cls.build_model(
            vocab_size=vocab_size,
            encoder_output_size=encoder_conf.get("output_size", 256),
            encoder_attention_heads=encoder_conf.get("attention_heads", 4),
            encoder_num_blocks=encoder_conf.get("num_blocks", 12),
            decoder_attention_heads=decoder_conf.get("attention_heads", 4),
            decoder_num_blocks=decoder_conf.get("num_blocks", 6),
        )

        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        logger.info(f"Loaded model from {model_dir}")

        return model
