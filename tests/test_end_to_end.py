"""End-to-end test with real audio/video file."""

import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from speechcatcher.speech2text_streaming import Speech2TextStreaming


# Paths
MODEL_DIR = Path.home() / ".cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024"
TEST_VIDEO = Path(__file__).parent.parent / "Neujahrsansprache.mp4"
REFERENCE_TRANSCRIPT = Path(__file__).parent.parent / "Neujahrsansprache.mp4.txt"
REFERENCE_JSON = Path(__file__).parent.parent / "Neujahrsansprache.mp4.json"


def extract_audio_features(video_path: Path) -> np.ndarray:
    """Extract audio features from video file.

    For now, this is a placeholder. In production, you would:
    1. Extract audio with ffmpeg
    2. Resample to 16kHz
    3. Compute log-mel features with librosa or torchaudio

    Args:
        video_path: Path to video file

    Returns:
        Features array (time, 80)
    """
    print(f"\n⚠️  Audio feature extraction not yet implemented")
    print(f"   Would extract from: {video_path}")
    print(f"   Need to:")
    print(f"   1. Extract audio with ffmpeg: ffmpeg -i {video_path.name} -ar 16000 -ac 1 audio.wav")
    print(f"   2. Compute log-mel features with torchaudio/librosa")
    print(f"   3. Return (time, 80) feature array")

    # Return dummy features for now to test the pipeline
    # Real implementation would extract actual features
    print(f"\n   Using DUMMY features for pipeline testing...")
    return np.random.randn(1000, 80).astype(np.float32)


def load_reference_transcript(ref_path: Path) -> str:
    """Load reference transcript."""
    with open(ref_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_reference_json(json_path: Path) -> dict:
    """Load reference JSON with detailed results."""
    if not json_path.exists():
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate.

    This is a simplified version. Production would use jiwer or similar.

    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript

    Returns:
        WER as float (0.0 = perfect, 1.0 = completely wrong)
    """
    # Tokenize
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Simple edit distance (Levenshtein)
    # This is a placeholder - real WER calculation is more complex

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    # For now, just return a rough estimate based on length difference
    # Real implementation would use proper edit distance
    len_diff = abs(len(ref_words) - len(hyp_words))
    rough_wer = len_diff / len(ref_words)

    print(f"\n⚠️  WER computation is simplified")
    print(f"   Reference words: {len(ref_words)}")
    print(f"   Hypothesis words: {len(hyp_words)}")
    print(f"   Rough WER estimate: {rough_wer:.1%}")
    print(f"   (Real WER would use proper edit distance)")

    return rough_wer


def test_pipeline_structure():
    """Test the complete pipeline structure without real audio."""
    print("\n" + "=" * 80)
    print("Testing Pipeline Structure")
    print("=" * 80)

    # Check files
    print("\n1. Checking test files...")
    if not TEST_VIDEO.exists():
        print(f"   ✗ Test video not found: {TEST_VIDEO}")
        return False
    print(f"   ✓ Test video found: {TEST_VIDEO.name} ({TEST_VIDEO.stat().st_size / 1024 / 1024:.1f} MB)")

    if not REFERENCE_TRANSCRIPT.exists():
        print(f"   ✗ Reference transcript not found: {REFERENCE_TRANSCRIPT}")
        return False

    reference = load_reference_transcript(REFERENCE_TRANSCRIPT)
    print(f"   ✓ Reference transcript loaded: {len(reference)} chars, {len(reference.split())} words")

    # Initialize model
    print("\n2. Initializing Speech2TextStreaming...")
    if not MODEL_DIR.exists():
        print(f"   ✗ Model directory not found: {MODEL_DIR}")
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")

        speech2text = Speech2TextStreaming(
            model_dir=MODEL_DIR,
            beam_size=10,
            ctc_weight=0.3,
            device=device,
        )
        print(f"   ✓ Model initialized successfully")
        print(f"     - Vocab size: {speech2text.model.vocab_size}")
        print(f"     - Beam size: {speech2text.beam_size}")

    except Exception as e:
        print(f"   ✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Extract features (placeholder)
    print("\n3. Extracting audio features...")
    features = extract_audio_features(TEST_VIDEO)
    print(f"   Features shape: {features.shape}")

    # Run recognition
    print("\n4. Running recognition...")
    try:
        results = speech2text.recognize(features)

        print(f"   ✓ Recognition completed")
        print(f"     - Number of hypotheses: {len(results)}")

        # Display results
        for i, (text, tokens, token_ids) in enumerate(results[:3]):
            print(f"\n   Hypothesis {i+1}:")
            print(f"     - Tokens: {len(token_ids)}")
            print(f"     - Text preview: {text[:100]}...")

        # Compare with reference (rough)
        best_text = results[0][0]
        wer = compute_wer(reference, best_text)

        print(f"\n5. Quality metrics:")
        print(f"   - Estimated WER: {wer:.1%}")
        print(f"   ⚠️  Note: Using DUMMY features, so output is random")
        print(f"   ⚠️  With real features, WER should be <5%")

        return True

    except Exception as e:
        print(f"   ✗ Recognition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_mode():
    """Test streaming mode with chunks."""
    print("\n" + "=" * 80)
    print("Testing Streaming Mode")
    print("=" * 80)

    if not MODEL_DIR.exists():
        print(f"✗ Model directory not found")
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        speech2text = Speech2TextStreaming(
            model_dir=MODEL_DIR,
            beam_size=5,
            ctc_weight=0.3,
            device=device,
        )

        # Create dummy chunks (in reality, these would be from streaming audio)
        chunk_size = 200  # ~2 seconds of features
        n_chunks = 5
        chunks = [np.random.randn(chunk_size, 80).astype(np.float32) for _ in range(n_chunks)]

        print(f"\nProcessing {n_chunks} chunks of {chunk_size} frames each...")

        # Process chunks
        speech2text.reset()
        for i, chunk in enumerate(chunks):
            is_final = (i == n_chunks - 1)
            results = speech2text(chunk, is_final=is_final)
            print(f"  Chunk {i+1}/{n_chunks}: {len(results)} hypotheses")

        print(f"\n✓ Streaming mode test completed")
        print(f"  Final result: {len(results[0][2])} tokens")

        return True

    except Exception as e:
        print(f"✗ Streaming mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run end-to-end tests."""
    print("=" * 80)
    print("End-to-End Testing with Neujahrsansprache.mp4")
    print("=" * 80)

    results = {
        "Pipeline structure": test_pipeline_structure(),
        "Streaming mode": test_streaming_mode(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")

    # Next steps
    print("\n" + "=" * 80)
    print("Next Steps for Full End-to-End Testing")
    print("=" * 80)
    print("""
1. Implement audio feature extraction:
   - Use ffmpeg to extract audio from video
   - Resample to 16kHz mono
   - Compute log-mel features with torchaudio/librosa

2. Load BPE tokenizer:
   - Load tokens.json or vocab.txt from model directory
   - Decode token IDs to actual text

3. Run full recognition:
   - Extract real features from Neujahrsansprache.mp4
   - Run beam search with real model
   - Decode tokens to text
   - Compare with reference transcript

4. Compute WER:
   - Use jiwer library for accurate WER calculation
   - Target: WER < 5% (within ±0.3 of reference)

5. Performance benchmarking:
   - Measure Real-Time Factor (RTF)
   - Measure latency
   - Measure memory usage
    """)

    if all(results.values()):
        print("\n" + "=" * 80)
        print("✓ PIPELINE TESTS PASSED!")
        print("=" * 80)
        print("\nThe decoder implementation is ready for real audio testing.")
        print("Install audio dependencies: pip install ffmpeg-python librosa jiwer")
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
