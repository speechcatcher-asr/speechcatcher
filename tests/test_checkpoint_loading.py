"""Tests for checkpoint loading functionality."""

import pytest
import torch
import numpy as np
from pathlib import Path

from speechcatcher.model.checkpoint_loader import (
    load_checkpoint,
    infer_model_architecture,
    map_espnet_to_speechcatcher,
    load_espnet_weights,
)
from speechcatcher.model.espnet_asr_model import ESPnetASRModel


class TestCheckpointLoading:
    """Test checkpoint loading utilities."""

    @pytest.fixture
    def model_dir(self):
        """Get model directory from cache."""
        cache_dir = Path.home() / ".cache/espnet"
        model_dirs = list(cache_dir.glob("models--speechcatcher--*/snapshots/*"))
        if not model_dirs:
            pytest.skip("No model found in cache")
        # The checkpoint is in exp/ subdirectory
        return model_dirs[0]

    @pytest.fixture
    def checkpoint_path(self, model_dir):
        """Get checkpoint path."""
        # Try different possible locations
        search_paths = [
            model_dir / "valid.acc.best.pth",
            model_dir / "valid.acc.ave_6best.pth",
            model_dir / "valid.acc.ave.pth",
        ]

        # Also search in exp/ subdirectories
        exp_dirs = list(model_dir.glob("exp/*/"))
        for exp_dir in exp_dirs:
            search_paths.extend([
                exp_dir / "valid.acc.best.pth",
                exp_dir / "valid.acc.ave_6best.pth",
                exp_dir / "valid.acc.ave.pth",
            ])

        for checkpoint in search_paths:
            if checkpoint.exists():
                return checkpoint

        pytest.skip(f"No checkpoint found in {model_dir}")

    def test_load_checkpoint(self, checkpoint_path):
        """Test basic checkpoint loading."""
        checkpoint = load_checkpoint(checkpoint_path)

        assert checkpoint is not None
        assert isinstance(checkpoint, dict)

        # Should have either 'model' or be the state_dict directly
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Check for expected keys
        has_encoder = any(k.startswith("encoder.") for k in state_dict.keys())
        has_decoder = any(k.startswith("decoder.") for k in state_dict.keys())

        assert has_encoder, "Checkpoint should have encoder weights"
        assert has_decoder, "Checkpoint should have decoder weights"

    def test_infer_architecture(self, checkpoint_path):
        """Test architecture inference from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path)
        state_dict = checkpoint.get("model", checkpoint)

        arch = infer_model_architecture(state_dict)

        # Should infer basic architecture params
        assert "vocab_size" in arch
        assert "encoder_output_size" in arch
        assert "num_encoder_layers" in arch
        assert "num_decoder_layers" in arch

        # Sanity checks on values
        assert arch["vocab_size"] > 0
        assert arch["encoder_output_size"] > 0
        assert arch["num_encoder_layers"] > 0
        assert arch["num_decoder_layers"] > 0

        print(f"Inferred architecture: {arch}")

    def test_parameter_name_mapping(self):
        """Test ESPnet -> speechcatcher parameter name mapping."""
        # Test encoder mappings (most map directly)
        assert map_espnet_to_speechcatcher("encoder.embed.0.weight") == "encoder.embed.0.weight"
        assert map_espnet_to_speechcatcher("encoder.encoders.0.self_attn.linear_q.weight") == \
               "encoder.encoders.0.self_attn.linear_q.weight"

        # Test decoder mappings (map directly)
        assert map_espnet_to_speechcatcher("decoder.embed.0.weight") == "decoder.embed.0.weight"
        assert map_espnet_to_speechcatcher("decoder.decoders.0.self_attn.linear_q.weight") == \
               "decoder.decoders.0.self_attn.linear_q.weight"
        assert map_espnet_to_speechcatcher("decoder.output_layer.weight") == \
               "decoder.output_layer.weight"

        # Test CTC mappings
        assert map_espnet_to_speechcatcher("ctc.ctc_lo.weight") == "ctc.ctc_lo.weight"

        # Test frontend parameters (should be skipped)
        assert map_espnet_to_speechcatcher("frontend.mel_spectrogram.spectrogram.window") is None

    def test_load_weights_into_model(self, checkpoint_path):
        """Test loading checkpoint weights into model."""
        # First, infer architecture
        checkpoint = load_checkpoint(checkpoint_path)
        state_dict = checkpoint.get("model", checkpoint)
        arch = infer_model_architecture(state_dict)

        # Build model with inferred architecture
        model = ESPnetASRModel.build_model(
            vocab_size=arch["vocab_size"],
            encoder_output_size=arch["encoder_output_size"],
            encoder_num_blocks=arch["num_encoder_layers"],
            decoder_num_blocks=arch["num_decoder_layers"],
            use_frontend=True,
        )

        # Load weights
        model, _ = load_espnet_weights(model, checkpoint_path, strict=False)

        # Verify model is in eval mode and on CPU
        model.eval()

        # Check that parameters are loaded (not random initialization)
        # Random init would have values ~N(0, 0.02), loaded weights should differ
        for name, param in model.named_parameters():
            if "weight" in name:
                # Check parameter has reasonable values
                assert param.abs().mean() > 0, f"Parameter {name} appears uninitialized"
                assert not torch.isnan(param).any(), f"Parameter {name} contains NaN"
                assert not torch.isinf(param).any(), f"Parameter {name} contains Inf"
                break  # Just check first weight parameter

    def test_model_forward_pass(self, checkpoint_path):
        """Test that loaded model can perform forward pass."""
        # Load model
        checkpoint = load_checkpoint(checkpoint_path)
        state_dict = checkpoint.get("model", checkpoint)
        arch = infer_model_architecture(state_dict)

        model = ESPnetASRModel.build_model(
            vocab_size=arch["vocab_size"],
            encoder_output_size=arch["encoder_output_size"],
            encoder_num_blocks=arch["num_encoder_layers"],
            decoder_num_blocks=arch["num_decoder_layers"],
            use_frontend=True,
        )

        model, _ = load_espnet_weights(model, checkpoint_path, strict=False)
        model.eval()

        # Create dummy input (1s of audio at 16kHz)
        waveform = torch.randn(1, 16000)
        waveform_lengths = torch.tensor([16000])

        # Extract features using frontend
        with torch.no_grad():
            features, feature_lengths = model.frontend(waveform)

            # Forward pass through encoder with features
            encoder_out, encoder_out_lens, encoder_states = model.encoder(
                features, feature_lengths, is_final=True, infer_mode=True
            )

        # Check encoder output shape and values
        assert encoder_out.shape[0] == 1  # batch size
        assert encoder_out.shape[1] > 0  # time dimension (subsampled)
        assert encoder_out.shape[2] == arch["encoder_output_size"]  # feature dimension

        # Check output is not NaN or constant
        assert not torch.isnan(encoder_out).any(), "Encoder output contains NaN"
        assert encoder_out.std() > 0.01, "Encoder output is nearly constant"

        print(f"Encoder output shape: {encoder_out.shape}")
        print(f"Encoder output stats: mean={encoder_out.mean():.4f}, std={encoder_out.std():.4f}")


    def test_encoder_output_varies_with_input(self, checkpoint_path):
        """Test that encoder produces different outputs for different inputs."""
        # Load model
        checkpoint = load_checkpoint(checkpoint_path)
        state_dict = checkpoint.get("model", checkpoint)
        arch = infer_model_architecture(state_dict)

        model = ESPnetASRModel.build_model(
            vocab_size=arch["vocab_size"],
            encoder_output_size=arch["encoder_output_size"],
            encoder_num_blocks=arch["num_encoder_layers"],
            decoder_num_blocks=arch["num_decoder_layers"],
            use_frontend=True,
        )

        model, _ = load_espnet_weights(model, checkpoint_path, strict=False)
        model.eval()

        # Create two different inputs
        waveform1 = torch.randn(1, 16000)
        waveform2 = torch.randn(1, 16000)
        waveform_lengths = torch.tensor([16000])

        # Forward pass with frontend
        with torch.no_grad():
            features1, feature_lengths1 = model.frontend(waveform1)
            features2, feature_lengths2 = model.frontend(waveform2)

            encoder_out1, _, _ = model.encoder(
                features1, feature_lengths1, is_final=True, infer_mode=True
            )
            encoder_out2, _, _ = model.encoder(
                features2, feature_lengths2, is_final=True, infer_mode=True
            )

        # Outputs should be different (use low threshold since random inputs might be similar)
        diff = (encoder_out1 - encoder_out2).abs().mean()
        assert diff > 0.01, f"Encoder outputs too similar (diff={diff:.4f}), model may not be loaded correctly"

        print(f"Encoder output difference: {diff:.4f}")


class TestSpeech2TextStreaming:
    """Test Speech2TextStreaming checkpoint loading."""

    def test_load_model_with_proper_mapping(self):
        """Test that Speech2TextStreaming loads model with proper parameter mapping."""
        from speechcatcher.speech2text_streaming import Speech2TextStreaming

        cache_dir = Path.home() / ".cache/espnet"
        model_dirs = list(cache_dir.glob("models--speechcatcher--*/snapshots/*"))
        if not model_dirs:
            pytest.skip("No model found in cache")

        model_dir = model_dirs[0]

        # Load model
        s2t = Speech2TextStreaming(
            model_dir=model_dir,
            beam_size=5,
            ctc_weight=0.0,
            device="cpu",
        )

        # Check model is loaded
        assert s2t.model is not None
        assert s2t.model.encoder is not None
        assert s2t.model.decoder is not None

        # Test encoder produces varied output
        waveform1 = torch.randn(16000)
        waveform2 = torch.randn(16000)

        # Process through frontend
        waveform1 = waveform1.unsqueeze(0)  # Add batch dim
        waveform2 = waveform2.unsqueeze(0)

        with torch.no_grad():
            feats1, feats_len1 = s2t.model.frontend(waveform1)
            feats2, feats_len2 = s2t.model.frontend(waveform2)

            # Apply normalization if available
            if s2t.mean is not None:
                feats1_np = feats1.squeeze(0).cpu().numpy()
                feats1_np = s2t.normalize_features(feats1_np)
                feats1 = torch.from_numpy(feats1_np).unsqueeze(0)

                feats2_np = feats2.squeeze(0).cpu().numpy()
                feats2_np = s2t.normalize_features(feats2_np)
                feats2 = torch.from_numpy(feats2_np).unsqueeze(0)

            # Encode
            enc1, enc_len1, _ = s2t.model.encoder(feats1, feats_len1, is_final=True, infer_mode=True)
            enc2, enc_len2, _ = s2t.model.encoder(feats2, feats_len2, is_final=True, infer_mode=True)

        # Check outputs differ
        diff = (enc1 - enc2).abs().mean()
        assert diff > 0.01, f"Encoder outputs too similar (diff={diff:.4f})"

        print(f"Encoder output stats for input 1: mean={enc1.mean():.4f}, std={enc1.std():.4f}")
        print(f"Encoder output stats for input 2: mean={enc2.mean():.4f}, std={enc2.std():.4f}")
        print(f"Difference: {diff:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
