"""Manual test to verify layers work correctly."""

import sys
import torch

# Add parent directory to path
sys.path.insert(0, '/home/ben/speechcatcher')

from speechcatcher.model.layers import (
    PositionwiseFeedForward,
    PositionalEncoding,
    RelPositionalEncoding,
    StreamPositionalEncoding,
    ConvolutionModule,
    LayerNorm,
)
from speechcatcher.model.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from speechcatcher.model.frontend import STFTFrontend


def test_feed_forward():
    """Test PositionwiseFeedForward."""
    print("Testing PositionwiseFeedForward...")
    layer = PositionwiseFeedForward(256, 1024, 256)
    x = torch.randn(2, 10, 256)
    out = layer(x)
    assert out.shape == (2, 10, 256), f"Expected (2, 10, 256), got {out.shape}"
    print("✓ PositionwiseFeedForward passed")


def test_positional_encoding():
    """Test positional encoding variants."""
    print("\nTesting PositionalEncoding...")
    pe = PositionalEncoding(256)
    x = torch.randn(2, 100, 256)
    out = pe(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✓ PositionalEncoding passed")

    print("Testing RelPositionalEncoding...")
    rel_pe = RelPositionalEncoding(256)
    out, pos_emb = rel_pe(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert pos_emb.shape == (1, 100, 256), f"Expected (1, 100, 256), got {pos_emb.shape}"
    print("✓ RelPositionalEncoding passed")

    print("Testing StreamPositionalEncoding...")
    spe = StreamPositionalEncoding(256)
    x1 = torch.randn(1, 10, 256)
    x2 = torch.randn(1, 10, 256)
    out1 = spe(x1)
    out2 = spe(x2)
    assert out1.shape == x1.shape
    assert out2.shape == x2.shape
    spe.reset()
    print("✓ StreamPositionalEncoding passed")


def test_convolution():
    """Test ConvolutionModule."""
    print("\nTesting ConvolutionModule...")
    conv = ConvolutionModule(256, kernel_size=31)
    x = torch.randn(2, 100, 256)
    out = conv(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✓ ConvolutionModule passed")


def test_layer_norm():
    """Test LayerNorm."""
    print("\nTesting LayerNorm...")
    ln = LayerNorm(256)
    x = torch.randn(2, 100, 256)
    out = ln(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✓ LayerNorm passed")


def test_multi_head_attention():
    """Test MultiHeadedAttention."""
    print("\nTesting MultiHeadedAttention...")
    attn = MultiHeadedAttention(n_head=4, n_feat=256)
    x = torch.randn(2, 10, 256)
    out = attn(x, x, x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✓ MultiHeadedAttention passed")

    print("Testing MultiHeadedAttention with cache...")
    q = torch.randn(1, 1, 256)
    k = torch.randn(1, 1, 256)
    v = torch.randn(1, 1, 256)
    out, cache = attn.forward_with_cache(q, k, v)
    assert out.shape == (1, 1, 256)
    assert cache[0].shape[2] == 1  # K cache time dimension
    assert cache[1].shape[2] == 1  # V cache time dimension
    print("✓ MultiHeadedAttention with cache passed")


def test_rel_position_attention():
    """Test RelPositionMultiHeadedAttention."""
    print("\nTesting RelPositionMultiHeadedAttention...")
    attn = RelPositionMultiHeadedAttention(n_head=4, n_feat=256)
    x = torch.randn(2, 10, 256)
    pos_emb = torch.randn(1, 10, 256)
    out = attn(x, x, x, pos_emb)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✓ RelPositionMultiHeadedAttention passed")


def test_stft_frontend():
    """Test STFTFrontend."""
    print("\nTesting STFTFrontend...")
    frontend = STFTFrontend(n_fft=512, hop_length=128, n_mels=80)

    # Generate 1 second of audio at 16kHz
    waveform = torch.randn(2, 16000)
    features, lengths = frontend(waveform)

    print(f"  Input shape: {waveform.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Lengths: {lengths}")

    assert features.shape[0] == 2  # Batch size
    assert features.shape[2] == 80  # n_mels
    assert lengths.shape == (2,)
    print("✓ STFTFrontend passed")


def main():
    """Run all tests."""
    print("="*60)
    print("Running manual tests for model layers")
    print("="*60)

    try:
        test_feed_forward()
        test_positional_encoding()
        test_convolution()
        test_layer_norm()
        test_multi_head_attention()
        test_rel_position_attention()
        test_stft_frontend()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
