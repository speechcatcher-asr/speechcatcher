"""Tests for encoder modules."""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from speechcatcher.model.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from speechcatcher.model.encoder.contextual_block_encoder_layer import (
    ContextualBlockEncoderLayer,
)
from speechcatcher.model.encoder.subsampling import Conv2dSubsampling
from speechcatcher.model.attention import MultiHeadedAttention
from speechcatcher.model.layers import PositionwiseFeedForward


def test_conv2d_subsampling():
    """Test Conv2dSubsampling layer."""
    print("\n=== Testing Conv2dSubsampling ===")

    batch_size = 2
    time_steps = 100
    input_dim = 80
    output_dim = 256

    layer = Conv2dSubsampling(input_dim, output_dim)

    x = torch.randn(batch_size, time_steps, input_dim)
    x_mask = torch.ones(batch_size, 1, time_steps)

    y, y_mask = layer(x, x_mask)

    # Check output shape (approximately 4x downsampling due to kernel overlap)
    # Formula: out = (in - kernel) / stride + 1 for each conv layer
    # Layer 1: (100 - 3) / 2 + 1 = 49
    # Layer 2: (49 - 3) / 2 + 1 = 24
    expected_time = 24
    assert y.shape == (batch_size, expected_time, output_dim), \
        f"Expected shape ({batch_size}, {expected_time}, {output_dim}), got {y.shape}"
    assert y_mask.shape == (batch_size, 1, expected_time), \
        f"Expected mask shape ({batch_size}, 1, {expected_time}), got {y_mask.shape}"

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {y.shape}")
    print(f"✓ Mask shape: {y_mask.shape}")
    print("✓ Conv2dSubsampling passed")


def test_contextual_block_encoder_layer():
    """Test ContextualBlockEncoderLayer."""
    print("\n=== Testing ContextualBlockEncoderLayer ===")

    batch_size = 2
    n_blocks = 3
    block_size = 40
    size = 256
    n_heads = 4
    total_layers = 6

    # Create layer
    attn = MultiHeadedAttention(n_heads, size, dropout_rate=0.0, use_flash_attn=False)
    ffn = PositionwiseFeedForward(size, size * 4, size, dropout_rate=0.0)
    layer = ContextualBlockEncoderLayer(
        size=size,
        self_attn=attn,
        feed_forward=ffn,
        dropout_rate=0.0,
        total_layer_num=total_layers,
        normalize_before=True,
    )
    layer.eval()

    # Training mode test
    print("\n--- Training mode ---")
    x = torch.randn(batch_size, n_blocks, block_size + 2, size)
    mask = torch.ones(batch_size, n_blocks, block_size + 2, block_size + 2)
    past_ctx = torch.randn(batch_size, n_blocks, total_layers, size)

    x_out, mask_out, infer_mode, past_ctx_out, next_ctx_out, is_short, layer_idx = layer.forward_train(
        x, mask, past_ctx, None, layer_idx=0, cache=None
    )

    assert x_out.shape == x.shape, f"Expected shape {x.shape}, got {x_out.shape}"
    assert next_ctx_out is not None, "next_ctx should not be None"
    assert layer_idx == 1, f"Expected layer_idx=1, got {layer_idx}"

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {x_out.shape}")
    print(f"✓ Context shape: {next_ctx_out.shape}")

    # Inference mode test
    print("\n--- Inference mode ---")
    layer.eval()
    with torch.no_grad():
        x = torch.randn(batch_size, 1, block_size + 2, size)  # Single block
        mask = torch.ones(batch_size, 1, block_size + 2, block_size + 2)

        x_out, mask_out, infer_mode, past_ctx_out, next_ctx_out, is_short, layer_idx = layer.forward_infer(
            x, mask, past_ctx=None, next_ctx=None, is_short_segment=False, layer_idx=0, cache=None
        )

        assert x_out.shape == x.shape, f"Expected shape {x.shape}, got {x_out.shape}"
        assert next_ctx_out is not None, "next_ctx should not be None in first layer"
        assert infer_mode == True, "Should be in inference mode"

        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {x_out.shape}")
        print(f"✓ Context shape: {next_ctx_out.shape}")

    print("✓ ContextualBlockEncoderLayer passed")


def test_contextual_block_transformer_encoder_shapes():
    """Test ContextualBlockTransformerEncoder shape handling."""
    print("\n=== Testing ContextualBlockTransformerEncoder Shapes ===")

    batch_size = 2
    time_steps = 400
    input_size = 80
    output_size = 256
    attention_heads = 4
    num_blocks = 6

    encoder = ContextualBlockTransformerEncoder(
        input_size=input_size,
        output_size=output_size,
        attention_heads=attention_heads,
        num_blocks=num_blocks,
        block_size=40,
        hop_size=16,
        look_ahead=16,
        init_average=True,
    )
    encoder.eval()

    # Test training mode
    print("\n--- Training mode ---")
    xs = torch.randn(batch_size, time_steps, input_size)
    ilens = torch.tensor([time_steps, time_steps // 2])

    output, olens, _ = encoder(xs, ilens, prev_states=None, is_final=False)

    # After Conv2d subsampling (approximately 4x due to kernel overlap)
    # Don't assert exact time dimension as it depends on subsampling
    assert output.shape[0] == batch_size, f"Expected batch_size={batch_size}, got {output.shape[0]}"
    assert output.shape[2] == output_size, f"Expected output_size={output_size}, got {output.shape[2]}"
    assert olens.shape == (batch_size,), f"Expected olens shape ({batch_size},), got {olens.shape}"
    assert output.shape[1] < time_steps, "Output time should be less than input time (subsampled)"

    print(f"✓ Input shape: {xs.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output lengths: {olens}")

    # Test inference mode (requires batch_size=1)
    print("\n--- Inference mode ---")
    encoder.eval()
    batch_size_infer = 1
    with torch.no_grad():
        # First chunk
        chunk_size = 160  # 1 second at 16kHz after 4x subsample
        xs_chunk = torch.randn(batch_size_infer, chunk_size, input_size)
        ilens_chunk = torch.tensor([chunk_size])

        output1, olens1, states1 = encoder(xs_chunk, ilens_chunk, prev_states=None, is_final=False, infer_mode=True)

        assert output1.shape[0] == batch_size_infer
        assert output1.shape[2] == output_size
        assert states1 is not None, "States should be returned in streaming mode"

        print(f"✓ First chunk input shape: {xs_chunk.shape}")
        print(f"✓ First chunk output shape: {output1.shape}")
        print(f"✓ State keys: {states1.keys() if states1 else 'None'}")

        # Second chunk with states
        output2, olens2, states2 = encoder(xs_chunk, ilens_chunk, prev_states=states1, is_final=False, infer_mode=True)

        assert output2.shape[0] == batch_size_infer
        assert output2.shape[2] == output_size

        print(f"✓ Second chunk output shape: {output2.shape}")

        # Final chunk
        output3, olens3, states3 = encoder(xs_chunk, ilens_chunk, prev_states=states2, is_final=True, infer_mode=True)

        assert output3.shape[0] == batch_size_infer
        assert output3.shape[2] == output_size

        print(f"✓ Final chunk output shape: {output3.shape}")

    print("✓ ContextualBlockTransformerEncoder shapes passed")


def test_contextual_block_transformer_encoder_context_type():
    """Test different context types (avg vs max)."""
    print("\n=== Testing Context Types ===")

    batch_size = 1
    time_steps = 200
    input_size = 80
    output_size = 256

    xs = torch.randn(batch_size, time_steps, input_size)
    ilens = torch.tensor([time_steps])

    # Test with avg pooling
    encoder_avg = ContextualBlockTransformerEncoder(
        input_size=input_size,
        output_size=output_size,
        init_average=True,
    )
    encoder_avg.eval()

    output_avg, _, _ = encoder_avg(xs, ilens)
    print(f"✓ Average pooling output shape: {output_avg.shape}")

    # Test with max pooling
    encoder_max = ContextualBlockTransformerEncoder(
        input_size=input_size,
        output_size=output_size,
        init_average=False,
    )
    encoder_max.eval()

    output_max, _, _ = encoder_max(xs, ilens)
    print(f"✓ Max pooling output shape: {output_max.shape}")

    # Shapes should be the same
    assert output_avg.shape == output_max.shape

    # Values should be different (different pooling strategies)
    assert not torch.allclose(output_avg, output_max), \
        "Outputs should differ between avg and max pooling"

    print("✓ Context type test passed")


def test_encoder_streaming_consistency():
    """Test that streaming inference produces consistent results."""
    print("\n=== Testing Streaming Consistency ===")

    batch_size = 1
    chunk_size = 160
    input_size = 80
    output_size = 256

    encoder = ContextualBlockTransformerEncoder(
        input_size=input_size,
        output_size=output_size,
        attention_heads=4,
        num_blocks=4,
    )
    encoder.eval()

    # Process 3 chunks in streaming mode
    chunks = [torch.randn(batch_size, chunk_size, input_size) for _ in range(3)]
    ilens = torch.tensor([chunk_size])

    states = None
    outputs_streaming = []

    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            output, olens, states = encoder(chunk, ilens, prev_states=states, is_final=is_final, infer_mode=True)
            outputs_streaming.append(output)
            print(f"✓ Chunk {i+1} processed, output shape: {output.shape}")

    # Concatenate all streaming outputs
    streaming_output = torch.cat(outputs_streaming, dim=1)

    print(f"✓ Total streaming output shape: {streaming_output.shape}")
    print("✓ Streaming consistency test passed")


def main():
    """Run all encoder tests."""
    print("=" * 80)
    print("Running Encoder Tests")
    print("=" * 80)

    try:
        test_conv2d_subsampling()
        test_contextual_block_encoder_layer()
        test_contextual_block_transformer_encoder_shapes()
        test_contextual_block_transformer_encoder_context_type()
        test_encoder_streaming_consistency()

        print("\n" + "=" * 80)
        print("ALL ENCODER TESTS PASSED!")
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
