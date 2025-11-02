"""Tests for beam search modules."""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from speechcatcher.beam_search import (
    BeamSearch,
    BlockwiseSynchronousBeamSearch,
    Hypothesis,
    create_beam_search,
    create_initial_hypothesis,
)
from speechcatcher.beam_search.scorers import CTCPrefixScorer, DecoderScorer
from speechcatcher.model import ESPnetASRModel


def test_hypothesis():
    """Test Hypothesis class."""
    print("\n=== Testing Hypothesis ===")

    hyp = create_initial_hypothesis(sos_id=1)

    assert hyp.yseq == [1], f"Expected yseq=[1], got {hyp.yseq}"
    assert hyp.score == 0.0, f"Expected score=0.0, got {hyp.score}"

    # Add tokens
    hyp.yseq.append(5)
    hyp.score += 0.5

    assert hyp.yseq == [1, 5]
    assert hyp.score == 0.5

    print(f"✓ Initial hypothesis: {hyp}")
    print("✓ Hypothesis test passed")


def test_decoder_scorer():
    """Test DecoderScorer."""
    print("\n=== Testing DecoderScorer ===")

    from speechcatcher.model.decoder import TransformerDecoder

    vocab_size = 100
    encoder_output_size = 128

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        num_blocks=2,
    )
    decoder.eval()

    scorer = DecoderScorer(decoder, sos_id=1, eos_id=2)

    # Test single hypothesis scoring
    yseq = torch.tensor([1, 5, 10])  # SOS + 2 tokens
    encoder_out = torch.randn(20, encoder_output_size)  # (enc_len, dim)
    state = None

    with torch.no_grad():
        log_probs, new_state = scorer.score(yseq, state, encoder_out)

    assert log_probs.shape == (vocab_size,), f"Expected shape ({vocab_size},), got {log_probs.shape}"
    assert new_state is not None, "State should be updated"

    print(f"✓ Single hypothesis scored, log_probs shape: {log_probs.shape}")

    # Test batch scoring
    batch_size = 4
    yseqs = torch.randint(0, vocab_size, (batch_size, 5))
    encoder_outs = torch.randn(batch_size, 20, encoder_output_size)
    states = [None] * batch_size

    with torch.no_grad():
        log_probs_batch, new_states = scorer.batch_score(yseqs, states, encoder_outs)

    assert log_probs_batch.shape == (batch_size, vocab_size), \
        f"Expected shape ({batch_size}, {vocab_size}), got {log_probs_batch.shape}"
    assert len(new_states) == batch_size

    print(f"✓ Batch scoring: {log_probs_batch.shape}")
    print("✓ DecoderScorer test passed")


def test_ctc_scorer():
    """Test CTCPrefixScorer."""
    print("\n=== Testing CTCPrefixScorer ===")

    from speechcatcher.model.ctc import CTC

    vocab_size = 100
    encoder_output_size = 128

    ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)
    ctc.eval()

    scorer = CTCPrefixScorer(ctc, blank_id=0, eos_id=2)

    # Test single hypothesis scoring
    yseq = torch.tensor([1, 5, 10])
    encoder_out = torch.randn(20, encoder_output_size)
    state = None

    with torch.no_grad():
        log_probs, new_state = scorer.score(yseq, state, encoder_out)

    assert log_probs.shape == (vocab_size,), f"Expected shape ({vocab_size},), got {log_probs.shape}"

    print(f"✓ Single hypothesis scored, log_probs shape: {log_probs.shape}")

    # Test batch scoring
    batch_size = 4
    yseqs = torch.randint(0, vocab_size, (batch_size, 5))
    encoder_outs = torch.randn(batch_size, 20, encoder_output_size)
    states = [None] * batch_size

    with torch.no_grad():
        log_probs_batch, new_states = scorer.batch_score(yseqs, states, encoder_outs)

    assert log_probs_batch.shape == (batch_size, vocab_size), \
        f"Expected shape ({batch_size}, {vocab_size}), got {log_probs_batch.shape}"

    print(f"✓ Batch scoring: {log_probs_batch.shape}")
    print("✓ CTCPrefixScorer test passed")


def test_beam_search():
    """Test basic BeamSearch."""
    print("\n=== Testing BeamSearch ===")

    from speechcatcher.model.decoder import TransformerDecoder
    from speechcatcher.model.ctc import CTC

    vocab_size = 50
    encoder_output_size = 64
    beam_size = 5

    # Create small models
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        num_blocks=1,
    )
    decoder.eval()

    ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)
    ctc.eval()

    # Create scorers
    scorers = {
        "decoder": DecoderScorer(decoder),
        "ctc": CTCPrefixScorer(ctc),
    }
    weights = {
        "decoder": 0.7,
        "ctc": 0.3,
    }

    # Create beam search
    beam_search = BeamSearch(
        scorers=scorers,
        weights=weights,
        beam_size=beam_size,
        vocab_size=vocab_size,
    )

    # Test search
    encoder_out = torch.randn(1, 30, encoder_output_size)
    encoder_out_lens = torch.tensor([30])

    with torch.no_grad():
        hypotheses = beam_search.search(encoder_out, encoder_out_lens)

    assert len(hypotheses) <= beam_size, f"Expected at most {beam_size} hypotheses, got {len(hypotheses)}"
    assert all(isinstance(h, Hypothesis) for h in hypotheses)

    print(f"✓ Beam search returned {len(hypotheses)} hypotheses")
    print(f"✓ Best hypothesis: yseq length={len(hypotheses[0].yseq)}, score={hypotheses[0].score:.2f}")
    print("✓ BeamSearch test passed")


def test_blockwise_beam_search():
    """Test BlockwiseSynchronousBeamSearch."""
    print("\n=== Testing BlockwiseSynchronousBeamSearch ===")

    vocab_size = 50
    beam_size = 5

    # Create small model
    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=80,
        encoder_output_size=64,
        encoder_num_blocks=1,
        decoder_num_blocks=1,
        use_frontend=False,
    )
    model.eval()

    # Create BSBS
    bsbs = create_beam_search(
        model=model,
        beam_size=beam_size,
        ctc_weight=0.3,
        decoder_weight=0.7,
    )

    # Test recognition (need enough samples for conv2d subsampling)
    features = torch.randn(1, 200, 80)
    feature_lens = torch.tensor([200])

    with torch.no_grad():
        hypotheses = bsbs.recognize_stream(features, feature_lens)

    assert len(hypotheses) > 0, "Should return at least one hypothesis"
    assert len(hypotheses) <= beam_size, f"Expected at most {beam_size} hypotheses, got {len(hypotheses)}"

    print(f"✓ BSBS returned {len(hypotheses)} hypotheses")
    print(f"✓ Best hypothesis: yseq length={len(hypotheses[0].yseq)}, score={hypotheses[0].score:.2f}")
    print("✓ BlockwiseSynchronousBeamSearch test passed")


def test_streaming_blocks():
    """Test BSBS with multiple blocks."""
    print("\n=== Testing Streaming Blocks ===")

    vocab_size = 50
    beam_size = 3

    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=80,
        encoder_output_size=64,
        encoder_num_blocks=1,
        decoder_num_blocks=1,
        use_frontend=False,
    )
    model.eval()

    bsbs = create_beam_search(model=model, beam_size=beam_size)

    # Simulate streaming with 3 blocks (need enough samples for conv2d)
    blocks = [torch.randn(1, 100, 80) for _ in range(3)]
    block_lens = torch.tensor([100])

    state = None
    with torch.no_grad():
        for i, block in enumerate(blocks):
            is_final = (i == len(blocks) - 1)
            state = bsbs.process_block(block, block_lens, state, is_final)
            print(f"✓ Processed block {i+1}/{len(blocks)}, n_hyps={len(state.hypotheses)}")

    assert state is not None
    assert len(state.hypotheses) > 0

    print(f"✓ Final hypotheses: {len(state.hypotheses)}")
    print("✓ Streaming blocks test passed")


def main():
    """Run all beam search tests."""
    print("=" * 80)
    print("Running Beam Search Tests")
    print("=" * 80)

    try:
        test_hypothesis()
        test_decoder_scorer()
        test_ctc_scorer()
        test_beam_search()
        test_blockwise_beam_search()
        test_streaming_blocks()

        print("\n" + "=" * 80)
        print("ALL BEAM SEARCH TESTS PASSED!")
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
