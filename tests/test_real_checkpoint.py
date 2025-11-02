"""Test loading a real ESPnet checkpoint."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from speechcatcher.model.checkpoint_loader import (
    infer_model_architecture,
    load_config,
    load_normalization_stats,
)
from speechcatcher.model import ESPnetASRModel
from speechcatcher.speech2text_streaming import Speech2TextStreaming


# Model directory
MODEL_DIR = Path.home() / ".cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024"
STATS_DIR = Path.home() / ".cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_stats_raw_de_bpe1024/train"


def test_load_config():
    """Test loading the real config file."""
    print("\n=== Testing Config Loading ===")

    config_path = MODEL_DIR / "config.yaml"

    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        return False

    config = load_config(config_path)

    print(f"✓ Config loaded from {config_path}")
    print(f"\nKey configuration:")
    print(f"  Encoder type: {config.get('encoder', 'unknown')}")
    print(f"  Encoder conf: {config.get('encoder_conf', {})}")
    print(f"  Decoder conf: {config.get('decoder_conf', {})}")
    print(f"  Frontend conf: {config.get('frontend_conf', {})}")

    return True


def test_load_checkpoint():
    """Test loading the real checkpoint."""
    print("\n=== Testing Checkpoint Loading ===")

    checkpoint_path = MODEL_DIR / "valid.acc.ave_6best.pth"

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Checkpoint size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    print(f"✓ Checkpoint loaded")
    print(f"  Number of parameters: {len(state_dict)}")

    # Infer architecture
    arch = infer_model_architecture(state_dict)

    print(f"\nInferred architecture:")
    for key, value in arch.items():
        print(f"  {key}: {value}")

    # Check some key parameters
    print(f"\nSample parameters:")
    for key in list(state_dict.keys())[:5]:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")

    return True


def test_load_stats():
    """Test loading normalization stats."""
    print("\n=== Testing Normalization Stats ===")

    stats_path = STATS_DIR / "feats_stats.npz"

    if not stats_path.exists():
        print(f"✗ Stats not found: {stats_path}")
        return False

    mean, std = load_normalization_stats(stats_path)

    print(f"✓ Stats loaded from {stats_path}")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std range: [{std.min():.3f}, {std.max():.3f}]")

    return True


def test_build_model_from_config():
    """Test building model from config."""
    print("\n=== Testing Model Building ===")

    config_path = MODEL_DIR / "config.yaml"
    checkpoint_path = MODEL_DIR / "valid.acc.ave_6best.pth"

    if not config_path.exists() or not checkpoint_path.exists():
        print("✗ Required files not found")
        return False

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load checkpoint to get vocab size
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Infer vocab size
    vocab_size = None
    if "decoder.embed.0.weight" in state_dict:
        vocab_size = state_dict["decoder.embed.0.weight"].shape[0]
    elif "decoder.output_layer.weight" in state_dict:
        vocab_size = state_dict["decoder.output_layer.weight"].shape[0]

    print(f"Vocab size: {vocab_size}")

    encoder_conf = config.get("encoder_conf", {})
    decoder_conf = config.get("decoder_conf", {})

    print(f"\nBuilding model with:")
    print(f"  Encoder output size: {encoder_conf.get('output_size')}")
    print(f"  Encoder blocks: {encoder_conf.get('num_blocks')}")
    print(f"  Decoder blocks: {decoder_conf.get('num_blocks')}")

    try:
        model = ESPnetASRModel.build_model(
            vocab_size=vocab_size,
            input_size=80,
            encoder_output_size=encoder_conf.get("output_size", 256),
            encoder_attention_heads=encoder_conf.get("attention_heads", 4),
            encoder_num_blocks=encoder_conf.get("num_blocks", 12),
            decoder_attention_heads=decoder_conf.get("attention_heads", 4),
            decoder_num_blocks=decoder_conf.get("num_blocks", 6),
            use_frontend=False,
        )

        print(f"✓ Model built successfully")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {n_params:,}")

        return True

    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_weights():
    """Test loading weights into model."""
    print("\n=== Testing Weight Loading ===")

    config_path = MODEL_DIR / "config.yaml"
    checkpoint_path = MODEL_DIR / "valid.acc.ave_6best.pth"

    if not config_path.exists() or not checkpoint_path.exists():
        print("✗ Required files not found")
        return False

    try:
        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # Get vocab size
        vocab_size = state_dict["decoder.embed.0.weight"].shape[0]

        encoder_conf = config.get("encoder_conf", {})
        decoder_conf = config.get("decoder_conf", {})

        # Build model
        model = ESPnetASRModel.build_model(
            vocab_size=vocab_size,
            input_size=80,
            encoder_output_size=encoder_conf.get("output_size", 256),
            encoder_attention_heads=encoder_conf.get("attention_heads", 4),
            encoder_num_blocks=encoder_conf.get("num_blocks", 12),
            decoder_attention_heads=decoder_conf.get("attention_heads", 4),
            decoder_num_blocks=decoder_conf.get("num_blocks", 6),
            use_frontend=False,
        )

        # Load weights
        print("Loading weights into model...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"✓ Weights loaded")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")

        if missing_keys:
            print(f"\n  Sample missing keys:")
            for key in missing_keys[:5]:
                print(f"    - {key}")

        if unexpected_keys:
            print(f"\n  Sample unexpected keys:")
            for key in unexpected_keys[:5]:
                print(f"    - {key}")

        return True

    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speech2text_api():
    """Test Speech2TextStreaming API with real checkpoint."""
    print("\n=== Testing Speech2TextStreaming API ===")

    if not MODEL_DIR.exists():
        print(f"✗ Model directory not found: {MODEL_DIR}")
        return False

    try:
        print("Initializing Speech2TextStreaming...")
        speech2text = Speech2TextStreaming(
            model_dir=MODEL_DIR,
            beam_size=5,
            ctc_weight=0.3,
            device="cpu",
        )

        print(f"✓ API initialized successfully")
        print(f"  Vocab size: {speech2text.model.vocab_size}")
        print(f"  Beam size: {speech2text.beam_size}")

        # Test with dummy features
        import numpy as np
        features = np.random.randn(100, 80).astype(np.float32)

        print("\nTesting recognition with dummy features...")
        results = speech2text.recognize(features)

        print(f"✓ Recognition completed")
        print(f"  Number of hypotheses: {len(results)}")
        print(f"  Best hypothesis length: {len(results[0][2])} tokens")

        return True

    except Exception as e:
        print(f"✗ Failed to initialize API: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Real ESPnet Checkpoint Loading")
    print("=" * 80)

    print(f"\nModel directory: {MODEL_DIR}")
    print(f"Stats directory: {STATS_DIR}")

    if not MODEL_DIR.exists():
        print(f"\n✗ Model directory not found!")
        print(f"  Expected: {MODEL_DIR}")
        sys.exit(1)

    results = {
        "Config loading": test_load_config(),
        "Checkpoint loading": test_load_checkpoint(),
        "Stats loading": test_load_stats(),
        "Model building": test_build_model_from_config(),
        "Weight loading": test_load_weights(),
        "Speech2Text API": test_speech2text_api(),
    }

    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
