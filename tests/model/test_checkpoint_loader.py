"""Tests for checkpoint loading utilities."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from speechcatcher.model.checkpoint_loader import (
    apply_feature_normalization,
    infer_model_architecture,
    load_config,
    load_normalization_stats,
    map_espnet_to_speechcatcher,
)
from speechcatcher.model.decoder import TransformerDecoder
from speechcatcher.model.encoder import ContextualBlockTransformerEncoder


def test_map_espnet_keys():
    """Test ESPnet to speechcatcher key mapping."""
    print("\n=== Testing Key Mapping ===")

    test_cases = [
        ("encoder.embed.0.weight", "encoder.embed.conv.0.weight"),
        ("encoder.encoders.0.self_attn.linear_q.weight", "encoder.encoders.0.self_attn.linear_q.weight"),
        ("decoder.embed.0.weight", "decoder.embed.0.weight"),
        ("decoder.decoders.0.self_attn.linear_q.weight", "decoder.decoders.0.self_attn.linear_q.weight"),
        ("decoder.output_layer.weight", "decoder.output_layer.weight"),
        ("ctc.ctc_lo.weight", "ctc.ctc_lo.weight"),
    ]

    for espnet_key, expected in test_cases:
        mapped = map_espnet_to_speechcatcher(espnet_key)
        assert mapped == expected, f"Expected {expected}, got {mapped}"
        print(f"✓ {espnet_key} -> {mapped}")

    print("✓ Key mapping test passed")


def test_infer_architecture():
    """Test architecture inference from state_dict."""
    print("\n=== Testing Architecture Inference ===")

    # Create mock state dict
    state_dict = {
        # Encoder
        "encoder.embed.0.weight": torch.randn(256, 1, 3, 3),  # Conv2d output_dim=256
        "encoder.encoders.0.self_attn.linear_q.weight": torch.randn(256, 256),
        "encoder.encoders.1.self_attn.linear_q.weight": torch.randn(256, 256),
        "encoder.encoders.2.self_attn.linear_q.weight": torch.randn(256, 256),
        # Decoder
        "decoder.embed.0.weight": torch.randn(1000, 256),  # vocab_size=1000
        "decoder.decoders.0.self_attn.linear_q.weight": torch.randn(256, 256),
        "decoder.decoders.1.self_attn.linear_q.weight": torch.randn(256, 256),
        "decoder.output_layer.weight": torch.randn(1000, 256),
        # CTC
        "ctc.ctc_lo.weight": torch.randn(1000, 256),
    }

    arch = infer_model_architecture(state_dict)

    assert arch["num_encoder_layers"] == 3, f"Expected 3 encoder layers, got {arch['num_encoder_layers']}"
    assert arch["num_decoder_layers"] == 2, f"Expected 2 decoder layers, got {arch['num_decoder_layers']}"
    assert arch["encoder_output_size"] == 256, f"Expected encoder_output_size=256, got {arch['encoder_output_size']}"
    assert arch["vocab_size"] == 1000, f"Expected vocab_size=1000, got {arch['vocab_size']}"
    assert arch["ctc_vocab_size"] == 1000, f"Expected ctc_vocab_size=1000, got {arch['ctc_vocab_size']}"

    print(f"✓ Inferred architecture: {arch}")
    print("✓ Architecture inference test passed")


def test_load_config():
    """Test config loading."""
    print("\n=== Testing Config Loading ===")

    # Create temporary config file
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        config_data = {
            "frontend_conf": {
                "n_fft": 512,
                "hop_length": 160,
                "win_length": 400,
            },
            "encoder_conf": {
                "output_size": 256,
                "attention_heads": 4,
                "num_blocks": 6,
            },
            "decoder_conf": {
                "attention_heads": 4,
                "num_blocks": 6,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_config(config_path)

        assert config["frontend_conf"]["n_fft"] == 512
        assert config["encoder_conf"]["output_size"] == 256
        assert config["decoder_conf"]["attention_heads"] == 4

        print(f"✓ Loaded config: {len(config)} top-level keys")
        print("✓ Config loading test passed")


def test_normalization_stats():
    """Test normalization stats loading and application."""
    print("\n=== Testing Normalization Stats ===")

    # Create temporary stats file
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = Path(tmpdir) / "feats_stats.npz"

        feat_dim = 80
        mean = np.random.randn(feat_dim).astype(np.float32)
        std = np.random.rand(feat_dim).astype(np.float32) + 0.5  # Ensure positive

        np.savez(stats_path, mean=mean, std=std)

        # Load stats
        loaded_mean, loaded_std = load_normalization_stats(stats_path)

        assert np.allclose(loaded_mean, mean), "Mean mismatch"
        assert np.allclose(loaded_std, std), "Std mismatch"

        print(f"✓ Loaded stats: mean shape {loaded_mean.shape}, std shape {loaded_std.shape}")

        # Test normalization
        features = torch.randn(2, 100, feat_dim)
        normalized = apply_feature_normalization(features, loaded_mean, loaded_std)

        assert normalized.shape == features.shape, "Shape mismatch after normalization"

        # Verify normalization (approximately zero mean and unit variance)
        # Note: Not exact due to batch statistics
        print(f"✓ Normalized features: mean ≈ {normalized.mean():.4f}, std ≈ {normalized.std():.4f}")
        print("✓ Normalization test passed")


def test_checkpoint_creation_and_loading():
    """Test creating and loading a mock checkpoint."""
    print("\n=== Testing Checkpoint Creation and Loading ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a small encoder
        input_size = 80
        output_size = 128
        encoder = ContextualBlockTransformerEncoder(
            input_size=input_size,
            output_size=output_size,
            attention_heads=4,
            num_blocks=2,
            block_size=40,
            hop_size=16,
        )

        # Save encoder state dict as mock checkpoint
        checkpoint_path = tmpdir / "model.pth"
        checkpoint = {
            "model": encoder.state_dict(),
            "epoch": 100,
        }
        torch.save(checkpoint, checkpoint_path)

        print(f"✓ Created mock checkpoint with {len(encoder.state_dict())} parameters")

        # Create a new encoder and try to load weights
        encoder_new = ContextualBlockTransformerEncoder(
            input_size=input_size,
            output_size=output_size,
            attention_heads=4,
            num_blocks=2,
            block_size=40,
            hop_size=16,
        )

        # Load checkpoint directly (not using load_espnet_weights since keys match)
        loaded_checkpoint = torch.load(checkpoint_path)
        encoder_new.load_state_dict(loaded_checkpoint["model"])

        print(f"✓ Loaded checkpoint into new encoder")

        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            encoder.named_parameters(), encoder_new.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.allclose(param1, param2), f"Parameter {name1} values mismatch"

        print("✓ Weight verification passed")
        print("✓ Checkpoint creation and loading test passed")


def test_decoder_checkpoint():
    """Test decoder checkpoint handling."""
    print("\n=== Testing Decoder Checkpoint ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create decoder
        vocab_size = 1000
        encoder_output_size = 256
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            attention_heads=4,
            num_blocks=2,
        )

        # Save checkpoint
        checkpoint_path = tmpdir / "decoder.pth"
        torch.save({"model": decoder.state_dict()}, checkpoint_path)

        print(f"✓ Created decoder checkpoint with {len(decoder.state_dict())} parameters")

        # Infer architecture from state dict
        checkpoint = torch.load(checkpoint_path)

        # Add "decoder." prefix to keys to match ESPnet format
        prefixed_state_dict = {f"decoder.{k}": v for k, v in checkpoint["model"].items()}
        arch = infer_model_architecture(prefixed_state_dict)

        assert arch.get("num_decoder_layers") == 2, \
            f"Expected 2 decoder layers, got {arch.get('num_decoder_layers')}"
        assert arch.get("vocab_size") == vocab_size, \
            f"Expected vocab_size={vocab_size}, got {arch.get('vocab_size')}"

        print(f"✓ Inferred decoder architecture: {arch}")
        print("✓ Decoder checkpoint test passed")


def main():
    """Run all checkpoint loader tests."""
    print("=" * 80)
    print("Running Checkpoint Loader Tests")
    print("=" * 80)

    try:
        test_map_espnet_keys()
        test_infer_architecture()
        test_load_config()
        test_normalization_stats()
        test_checkpoint_creation_and_loading()
        test_decoder_checkpoint()

        print("\n" + "=" * 80)
        print("ALL CHECKPOINT LOADER TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
