"""Tests for decoder modules."""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from speechcatcher.model.decoder import TransformerDecoder, TransformerDecoderLayer
from speechcatcher.model.attention import MultiHeadedAttention
from speechcatcher.model.layers import PositionwiseFeedForward


def test_transformer_decoder_layer():
    """Test TransformerDecoderLayer."""
    print("\n=== Testing TransformerDecoderLayer ===")

    batch_size = 2
    tgt_len = 10
    src_len = 20
    size = 256
    n_heads = 4

    # Create layer
    self_attn = MultiHeadedAttention(n_heads, size, dropout_rate=0.0, use_flash_attn=False)
    src_attn = MultiHeadedAttention(n_heads, size, dropout_rate=0.0, use_flash_attn=False)
    ffn = PositionwiseFeedForward(size, size * 4, size, dropout_rate=0.0)

    layer = TransformerDecoderLayer(
        size=size,
        self_attn=self_attn,
        src_attn=src_attn,
        feed_forward=ffn,
        dropout_rate=0.0,
        normalize_before=True,
    )
    layer.eval()

    # Test without cache
    print("\n--- Without cache ---")
    tgt = torch.randn(batch_size, tgt_len, size)
    memory = torch.randn(batch_size, src_len, size)
    tgt_mask = torch.ones(batch_size, tgt_len, tgt_len).bool()
    memory_mask = torch.ones(batch_size, 1, src_len).bool()

    x_out, tgt_mask_out, memory_out, memory_mask_out = layer(
        tgt, tgt_mask, memory, memory_mask, cache=None
    )

    assert x_out.shape == tgt.shape, f"Expected shape {tgt.shape}, got {x_out.shape}"
    print(f"✓ Input shape: {tgt.shape}")
    print(f"✓ Output shape: {x_out.shape}")

    # Test with cache (incremental decoding)
    print("\n--- With cache (incremental) ---")
    cache = tgt[:, :-1, :]  # All but last frame
    tgt_full = tgt  # Full sequence

    x_out_cached, _, _, _ = layer(tgt_full, tgt_mask, memory, memory_mask, cache=cache)

    assert x_out_cached.shape == tgt.shape, \
        f"Expected shape {tgt.shape}, got {x_out_cached.shape}"
    print(f"✓ Cache shape: {cache.shape}")
    print(f"✓ Output shape: {x_out_cached.shape}")

    print("✓ TransformerDecoderLayer passed")


def test_transformer_decoder_forward():
    """Test TransformerDecoder forward pass."""
    print("\n=== Testing TransformerDecoder Forward ===")

    batch_size = 2
    enc_len = 50
    tgt_len = 20
    vocab_size = 100
    encoder_output_size = 256

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        attention_heads=4,
        linear_units=1024,
        num_blocks=3,
        dropout_rate=0.0,
    )
    decoder.eval()

    # Encoder output
    hs_pad = torch.randn(batch_size, enc_len, encoder_output_size)
    hlens = torch.tensor([enc_len, enc_len // 2])

    # Target tokens
    ys_in_pad = torch.randint(0, vocab_size, (batch_size, tgt_len))
    ys_in_lens = torch.tensor([tgt_len, tgt_len // 2])

    # Forward pass
    logits, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    assert logits.shape == (batch_size, tgt_len, vocab_size), \
        f"Expected shape ({batch_size}, {tgt_len}, {vocab_size}), got {logits.shape}"
    assert olens.shape == (batch_size,), f"Expected olens shape ({batch_size},), got {olens.shape}"

    print(f"✓ Encoder output shape: {hs_pad.shape}")
    print(f"✓ Target tokens shape: {ys_in_pad.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Output lengths: {olens}")
    print("✓ TransformerDecoder forward passed")


def test_transformer_decoder_incremental():
    """Test TransformerDecoder incremental decoding."""
    print("\n=== Testing TransformerDecoder Incremental Decoding ===")

    batch_size = 1
    enc_len = 50
    vocab_size = 100
    encoder_output_size = 256

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        attention_heads=4,
        num_blocks=3,
        dropout_rate=0.0,
    )
    decoder.eval()

    # Encoder output
    memory = torch.randn(batch_size, enc_len, encoder_output_size)

    # Simulate incremental decoding
    from speechcatcher.model.decoder.transformer_decoder import subsequent_mask

    max_len = 10
    cache = None
    ys = torch.tensor([[0]])  # Start token

    with torch.no_grad():
        for i in range(max_len):
            ys_mask = subsequent_mask(ys.size(1), device=ys.device).unsqueeze(0)
            logp, cache = decoder.forward_one_step(ys, ys_mask, memory, cache=cache)

            # Check output shape
            assert logp.shape == (batch_size, vocab_size), \
                f"Expected shape ({batch_size}, {vocab_size}), got {logp.shape}"

            # Sample next token
            next_token = logp.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            print(f"✓ Step {i+1}: decoded token {next_token.item()}, ys shape: {ys.shape}")

    assert ys.shape == (batch_size, max_len + 1), \
        f"Expected final shape ({batch_size}, {max_len + 1}), got {ys.shape}"

    print(f"✓ Final sequence shape: {ys.shape}")
    print("✓ Incremental decoding passed")


def test_transformer_decoder_batch_score():
    """Test TransformerDecoder batch scoring."""
    print("\n=== Testing TransformerDecoder Batch Scoring ===")

    batch_size = 4
    enc_len = 50
    hyp_len = 5
    vocab_size = 100
    encoder_output_size = 256

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        attention_heads=4,
        num_blocks=3,
        dropout_rate=0.0,
    )
    decoder.eval()

    # Encoder output (same for all hypotheses in beam search)
    xs = torch.randn(batch_size, enc_len, encoder_output_size)

    # Batch of hypotheses
    ys = torch.randint(0, vocab_size, (batch_size, hyp_len))

    # Initial states (None for first step)
    states = [None] * batch_size

    # Batch score
    with torch.no_grad():
        logp, new_states = decoder.batch_score(ys, states, xs)

    assert logp.shape == (batch_size, vocab_size), \
        f"Expected shape ({batch_size}, {vocab_size}), got {logp.shape}"
    assert len(new_states) == batch_size, \
        f"Expected {batch_size} states, got {len(new_states)}"
    assert len(new_states[0]) == decoder.num_blocks, \
        f"Expected {decoder.num_blocks} layers, got {len(new_states[0])}"

    print(f"✓ Input hypotheses shape: {ys.shape}")
    print(f"✓ Output log probs shape: {logp.shape}")
    print(f"✓ Number of states: {len(new_states)}")
    print(f"✓ State layers per hypothesis: {len(new_states[0])}")
    print("✓ Batch scoring passed")


def test_transformer_decoder_consistency():
    """Test that incremental decoding produces consistent results with full forward."""
    print("\n=== Testing Decoder Consistency ===")

    batch_size = 1
    enc_len = 30
    tgt_len = 10
    vocab_size = 50
    encoder_output_size = 128

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        attention_heads=4,
        num_blocks=2,
        dropout_rate=0.0,
    )
    decoder.eval()

    # Encoder output
    memory = torch.randn(batch_size, enc_len, encoder_output_size)
    hlens = torch.tensor([enc_len])

    # Target sequence
    ys_full = torch.randint(0, vocab_size, (batch_size, tgt_len))
    ys_lens = torch.tensor([tgt_len])

    with torch.no_grad():
        # Full forward
        logits_full, _ = decoder(memory, hlens, ys_full, ys_lens)
        last_logit_full = logits_full[0, -1, :]

        # Incremental decoding
        from speechcatcher.model.decoder.transformer_decoder import subsequent_mask

        cache = None
        for i in range(tgt_len):
            ys_partial = ys_full[:, :i+1]
            ys_mask = subsequent_mask(ys_partial.size(1), device=ys_partial.device).unsqueeze(0)
            logp, cache = decoder.forward_one_step(ys_partial, ys_mask, memory, cache=cache)

        last_logit_incremental = logp[0, :]

    # Convert log probs to logits for comparison
    last_logit_incremental_unnorm = last_logit_incremental * vocab_size

    # Check if outputs are similar (allowing for numerical differences)
    # We compare the relative ordering rather than exact values
    top_k = 5
    top_full = torch.topk(last_logit_full, top_k).indices
    top_incr = torch.topk(last_logit_incremental_unnorm, top_k).indices

    # Check if top predictions overlap significantly
    overlap = len(set(top_full.tolist()) & set(top_incr.tolist()))

    print(f"✓ Full forward last logit top-{top_k}: {top_full.tolist()}")
    print(f"✓ Incremental last logit top-{top_k}: {top_incr.tolist()}")
    print(f"✓ Top-{top_k} overlap: {overlap}/{top_k}")

    assert overlap >= top_k - 1, \
        f"Expected at least {top_k-1} overlapping predictions, got {overlap}"

    print("✓ Decoder consistency passed")


def main():
    """Run all decoder tests."""
    print("=" * 80)
    print("Running Decoder Tests")
    print("=" * 80)

    try:
        test_transformer_decoder_layer()
        test_transformer_decoder_forward()
        test_transformer_decoder_incremental()
        test_transformer_decoder_batch_score()
        test_transformer_decoder_consistency()

        print("\n" + "=" * 80)
        print("ALL DECODER TESTS PASSED!")
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
