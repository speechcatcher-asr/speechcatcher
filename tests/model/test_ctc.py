"""Tests for CTC module."""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from speechcatcher.model.ctc import CTC, ctc_greedy_decode, ctc_prefix_beam_search


def test_ctc_forward():
    """Test CTC forward pass."""
    print("\n=== Testing CTC Forward ===")

    batch_size = 2
    time_steps = 50
    encoder_output_size = 256
    vocab_size = 100

    ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)
    ctc.eval()

    # Encoder output
    hs_pad = torch.randn(batch_size, time_steps, encoder_output_size)
    hlens = torch.tensor([time_steps, time_steps // 2])

    # Forward without targets (inference mode)
    logits, loss = ctc(hs_pad, hlens)

    assert logits.shape == (batch_size, time_steps, vocab_size), \
        f"Expected shape ({batch_size}, {time_steps}, {vocab_size}), got {logits.shape}"
    assert loss is None, "Loss should be None without targets"

    print(f"✓ Encoder output shape: {hs_pad.shape}")
    print(f"✓ Logits shape: {logits.shape}")
    print("✓ CTC forward (inference) passed")


def test_ctc_loss():
    """Test CTC loss computation."""
    print("\n=== Testing CTC Loss ===")

    batch_size = 2
    time_steps = 50
    encoder_output_size = 256
    vocab_size = 100
    target_len = 10

    ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)
    ctc.train()  # Training mode

    # Encoder output
    hs_pad = torch.randn(batch_size, time_steps, encoder_output_size)
    hlens = torch.tensor([time_steps, time_steps // 2])

    # Targets
    ys_pad = torch.randint(1, vocab_size, (batch_size, target_len))  # Avoid blank (0)
    ys_lens = torch.tensor([target_len, target_len // 2])

    # Forward with targets
    logits, loss = ctc(hs_pad, hlens, ys_pad, ys_lens)

    assert logits.shape == (batch_size, time_steps, vocab_size)
    assert loss is not None, "Loss should not be None with targets"
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"✓ Encoder output shape: {hs_pad.shape}")
    print(f"✓ Target shape: {ys_pad.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    print("✓ CTC loss computation passed")


def test_ctc_log_softmax():
    """Test CTC log softmax."""
    print("\n=== Testing CTC Log Softmax ===")

    batch_size = 2
    time_steps = 50
    encoder_output_size = 256
    vocab_size = 100

    ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)
    ctc.eval()

    # Encoder output
    hs_pad = torch.randn(batch_size, time_steps, encoder_output_size)

    # Get log probabilities
    log_probs = ctc.log_softmax(hs_pad)

    assert log_probs.shape == (batch_size, time_steps, vocab_size)

    # Check that probabilities sum to 1 (in log space: log(sum(exp(log_probs))) ≈ 0)
    probs_sum = torch.exp(log_probs).sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), \
        "Probabilities should sum to 1"

    print(f"✓ Log probs shape: {log_probs.shape}")
    print(f"✓ Probs sum check: {probs_sum[0, 0].item():.6f} ≈ 1.0")
    print("✓ CTC log softmax passed")


def test_ctc_argmax():
    """Test CTC argmax (greedy decoding)."""
    print("\n=== Testing CTC Argmax ===")

    batch_size = 2
    time_steps = 50
    encoder_output_size = 256
    vocab_size = 100

    ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)
    ctc.eval()

    # Encoder output
    hs_pad = torch.randn(batch_size, time_steps, encoder_output_size)

    # Argmax decoding
    predictions = ctc.argmax(hs_pad)

    assert predictions.shape == (batch_size, time_steps)
    assert predictions.min() >= 0 and predictions.max() < vocab_size, \
        "Predictions should be in valid token range"

    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Prediction range: [{predictions.min().item()}, {predictions.max().item()}]")
    print("✓ CTC argmax passed")


def test_greedy_decode():
    """Test greedy CTC decoding with blank removal."""
    print("\n=== Testing Greedy Decode ===")

    batch_size = 2
    time_steps = 20
    vocab_size = 10
    blank = 0

    # Create simple log probabilities
    log_probs = torch.randn(batch_size, time_steps, vocab_size)
    log_probs = torch.log_softmax(log_probs, dim=-1)
    lengths = torch.tensor([time_steps, time_steps // 2])

    # Decode
    decoded = ctc_greedy_decode(log_probs, lengths, blank=blank)

    assert len(decoded) == batch_size, f"Expected {batch_size} sequences, got {len(decoded)}"

    # Check that blanks are removed
    for seq in decoded:
        assert blank not in seq, "Decoded sequence should not contain blanks"
        print(f"✓ Decoded sequence: {seq[:10]}{'...' if len(seq) > 10 else ''} (len={len(seq)})")

    print("✓ Greedy decode passed")


def test_greedy_decode_deduplication():
    """Test that greedy decode removes consecutive duplicates."""
    print("\n=== Testing Greedy Decode Deduplication ===")

    batch_size = 1
    time_steps = 10
    vocab_size = 5
    blank = 0

    # Create deterministic sequence with duplicates: [1, 1, 0, 2, 2, 0, 3, 3, 3, 0]
    log_probs = torch.full((batch_size, time_steps, vocab_size), -100.0)
    log_probs[0, 0, 1] = 0.0  # token 1
    log_probs[0, 1, 1] = 0.0  # token 1 (duplicate)
    log_probs[0, 2, 0] = 0.0  # blank
    log_probs[0, 3, 2] = 0.0  # token 2
    log_probs[0, 4, 2] = 0.0  # token 2 (duplicate)
    log_probs[0, 5, 0] = 0.0  # blank
    log_probs[0, 6, 3] = 0.0  # token 3
    log_probs[0, 7, 3] = 0.0  # token 3 (duplicate)
    log_probs[0, 8, 3] = 0.0  # token 3 (duplicate)
    log_probs[0, 9, 0] = 0.0  # blank

    lengths = torch.tensor([time_steps])

    # Decode
    decoded = ctc_greedy_decode(log_probs, lengths, blank=blank)

    # Expected: [1, 2, 3] (duplicates and blanks removed)
    expected = [1, 2, 3]
    assert decoded[0] == expected, f"Expected {expected}, got {decoded[0]}"

    print(f"✓ Input pattern: [1, 1, 0, 2, 2, 0, 3, 3, 3, 0]")
    print(f"✓ Decoded: {decoded[0]}")
    print("✓ Deduplication passed")


def test_prefix_beam_search():
    """Test CTC prefix beam search."""
    print("\n=== Testing Prefix Beam Search ===")

    batch_size = 1
    time_steps = 20
    vocab_size = 10
    beam_size = 5
    blank = 0

    # Create log probabilities
    log_probs = torch.randn(batch_size, time_steps, vocab_size)
    log_probs = torch.log_softmax(log_probs, dim=-1)
    lengths = torch.tensor([time_steps])

    # Beam search decode
    decoded = ctc_prefix_beam_search(log_probs, lengths, beam_size=beam_size, blank=blank)

    assert len(decoded) == batch_size
    assert blank not in decoded[0], "Decoded sequence should not contain blanks"

    print(f"✓ Beam size: {beam_size}")
    print(f"✓ Decoded sequence: {decoded[0][:10]}{'...' if len(decoded[0]) > 10 else ''} (len={len(decoded[0])})")
    print("✓ Prefix beam search passed")


def test_ctc_loss_gradient():
    """Test that CTC loss produces valid gradients."""
    print("\n=== Testing CTC Loss Gradient ===")

    batch_size = 2
    time_steps = 50
    encoder_output_size = 128
    vocab_size = 50
    target_len = 10

    ctc = CTC(vocab_size=vocab_size, encoder_output_size=encoder_output_size)

    # Encoder output (requires grad)
    hs_pad = torch.randn(batch_size, time_steps, encoder_output_size, requires_grad=True)
    hlens = torch.tensor([time_steps, time_steps // 2])

    # Targets
    ys_pad = torch.randint(1, vocab_size, (batch_size, target_len))
    ys_lens = torch.tensor([target_len, target_len // 2])

    # Forward
    logits, loss = ctc(hs_pad, hlens, ys_pad, ys_lens)

    # Backward
    loss.backward()

    # Check gradients
    assert hs_pad.grad is not None, "Gradients should be computed"
    assert not torch.isnan(hs_pad.grad).any(), "Gradients should not be NaN"
    assert not torch.isinf(hs_pad.grad).any(), "Gradients should not be Inf"

    grad_norm = hs_pad.grad.norm().item()
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Gradient norm: {grad_norm:.4f}")
    print("✓ CTC loss gradient passed")


def main():
    """Run all CTC tests."""
    print("=" * 80)
    print("Running CTC Tests")
    print("=" * 80)

    try:
        test_ctc_forward()
        test_ctc_loss()
        test_ctc_log_softmax()
        test_ctc_argmax()
        test_greedy_decode()
        test_greedy_decode_deduplication()
        test_prefix_beam_search()
        test_ctc_loss_gradient()

        print("\n" + "=" * 80)
        print("ALL CTC TESTS PASSED!")
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
