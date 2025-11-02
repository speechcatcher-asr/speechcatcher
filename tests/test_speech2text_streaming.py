"""Tests for Speech2TextStreaming API."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from speechcatcher.model import ESPnetASRModel
from speechcatcher.speech2text_streaming import Speech2TextStreaming


def create_dummy_model_dir():
    """Create a dummy model directory with checkpoint and config."""
    tmpdir = tempfile.mkdtemp()
    tmpdir = Path(tmpdir)

    # Create small model
    vocab_size = 100
    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=80,
        encoder_output_size=64,
        encoder_num_blocks=1,
        decoder_num_blocks=1,
        use_frontend=False,
    )

    # Save checkpoint
    checkpoint_path = tmpdir / "valid.acc.best.pth"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    # Save config
    config = {
        "encoder_conf": {
            "output_size": 64,
            "attention_heads": 4,
            "num_blocks": 1,
        },
        "decoder_conf": {
            "attention_heads": 4,
            "num_blocks": 1,
        },
    }
    config_path = tmpdir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Save normalization stats
    feat_dim = 80
    mean = np.zeros(feat_dim, dtype=np.float32)
    std = np.ones(feat_dim, dtype=np.float32)
    stats_path = tmpdir / "feats_stats.npz"
    np.savez(stats_path, mean=mean, std=std)

    return tmpdir


def test_speech2text_streaming_init():
    """Test Speech2TextStreaming initialization."""
    print("\n=== Testing Speech2TextStreaming Initialization ===")

    model_dir = create_dummy_model_dir()

    try:
        speech2text = Speech2TextStreaming(
            model_dir=model_dir,
            beam_size=5,
            ctc_weight=0.3,
            device="cpu",
        )

        assert speech2text.model is not None
        assert speech2text.beam_search is not None
        assert speech2text.beam_size == 5

        print(f"✓ Model loaded from {model_dir}")
        print(f"✓ Beam size: {speech2text.beam_size}")
        print("✓ Initialization test passed")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(model_dir)


def test_speech2text_recognize():
    """Test non-streaming recognition."""
    print("\n=== Testing Non-Streaming Recognition ===")

    model_dir = create_dummy_model_dir()

    try:
        speech2text = Speech2TextStreaming(
            model_dir=model_dir,
            beam_size=3,
        )

        # Create dummy features
        features = np.random.randn(100, 80).astype(np.float32)

        # Recognize
        results = speech2text.recognize(features)

        assert len(results) > 0, "Should return at least one result"
        assert len(results) <= 3, "Should not exceed beam size"

        for text, tokens, token_ids in results:
            assert isinstance(text, str)
            assert isinstance(tokens, list)
            assert isinstance(token_ids, list)

        print(f"✓ Input features shape: {features.shape}")
        print(f"✓ Number of results: {len(results)}")
        print(f"✓ Best result tokens: {results[0][1][:10]}...")
        print("✓ Recognition test passed")

    finally:
        import shutil
        shutil.rmtree(model_dir)


def test_speech2text_streaming():
    """Test streaming recognition with chunks."""
    print("\n=== Testing Streaming Recognition ===")

    model_dir = create_dummy_model_dir()

    try:
        speech2text = Speech2TextStreaming(
            model_dir=model_dir,
            beam_size=3,
        )

        # Create dummy feature chunks
        chunks = [
            np.random.randn(100, 80).astype(np.float32) for _ in range(3)
        ]

        # Streaming recognition
        results = speech2text.recognize_stream(chunks)

        assert len(results) > 0
        assert len(results) <= 3

        print(f"✓ Number of chunks: {len(chunks)}")
        print(f"✓ Number of results: {len(results)}")
        print(f"✓ Best result: {results[0][0][:50]}...")
        print("✓ Streaming test passed")

    finally:
        import shutil
        shutil.rmtree(model_dir)


def test_speech2text_incremental():
    """Test incremental streaming (calling multiple times)."""
    print("\n=== Testing Incremental Streaming ===")

    model_dir = create_dummy_model_dir()

    try:
        speech2text = Speech2TextStreaming(
            model_dir=model_dir,
            beam_size=3,
        )

        # Reset state
        speech2text.reset()

        # Process chunks incrementally
        chunks = [
            np.random.randn(100, 80).astype(np.float32) for _ in range(3)
        ]

        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            results = speech2text(chunk, is_final=is_final)

            print(f"✓ Chunk {i+1}: {len(results)} hypotheses")

        assert len(results) > 0

        print("✓ Incremental streaming test passed")

    finally:
        import shutil
        shutil.rmtree(model_dir)


def test_speech2text_reset():
    """Test state reset."""
    print("\n=== Testing State Reset ===")

    model_dir = create_dummy_model_dir()

    try:
        speech2text = Speech2TextStreaming(
            model_dir=model_dir,
            beam_size=3,
        )

        # Process some data
        features = np.random.randn(100, 80).astype(np.float32)
        speech2text(features, is_final=False)

        assert speech2text.beam_state is not None, "State should be set"

        # Reset
        speech2text.reset()

        assert speech2text.beam_state is None, "State should be cleared"
        assert speech2text.processed_frames == 0, "Frame counter should be reset"

        print("✓ State reset test passed")

    finally:
        import shutil
        shutil.rmtree(model_dir)


def main():
    """Run all Speech2TextStreaming tests."""
    print("=" * 80)
    print("Running Speech2TextStreaming Tests")
    print("=" * 80)

    try:
        test_speech2text_streaming_init()
        test_speech2text_recognize()
        test_speech2text_streaming()
        test_speech2text_incremental()
        test_speech2text_reset()

        print("\n" + "=" * 80)
        print("ALL SPEECH2TEXT STREAMING TESTS PASSED!")
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
