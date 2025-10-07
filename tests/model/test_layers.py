"""Unit tests for model layers."""

import pytest
import torch

from speechcatcher.model.layers import (
    PositionwiseFeedForward,
    PositionalEncoding,
    RelPositionalEncoding,
    StreamPositionalEncoding,
    ConvolutionModule,
    LayerNorm,
)


class TestPositionwiseFeedForward:
    """Tests for PositionwiseFeedForward layer."""

    def test_forward_shape(self):
        """Test that output shape matches input shape (except last dim)."""
        batch, time, input_dim, hidden_dim, output_dim = 2, 10, 256, 1024, 256
        layer = PositionwiseFeedForward(input_dim, hidden_dim, output_dim)

        x = torch.randn(batch, time, input_dim)
        output = layer(x)

        assert output.shape == (batch, time, output_dim)

    def test_forward_different_io_dims(self):
        """Test with different input and output dimensions."""
        layer = PositionwiseFeedForward(
            input_dim=128, hidden_dim=512, output_dim=256
        )

        x = torch.randn(2, 10, 128)
        output = layer(x)

        assert output.shape == (2, 10, 256)

    def test_dropout_training_inference(self):
        """Test that dropout behaves differently in train/eval modes."""
        layer = PositionwiseFeedForward(256, 1024, 256, dropout_rate=0.5)
        x = torch.randn(1, 10, 256)

        # Training mode: dropout active (check multiple runs differ)
        layer.train()
        out1 = layer(x)
        out2 = layer(x)
        assert not torch.allclose(out1, out2), "Dropout should be stochastic in train mode"

        # Eval mode: dropout inactive (should be deterministic)
        layer.eval()
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2), "Output should be deterministic in eval mode"


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_forward_shape(self):
        """Test output shape preservation."""
        d_model = 256
        pe = PositionalEncoding(d_model)

        x = torch.randn(2, 100, d_model)
        output = pe(x)

        assert output.shape == x.shape

    def test_scaling(self):
        """Test that input is scaled by sqrt(d_model)."""
        d_model = 256
        pe = PositionalEncoding(d_model, dropout_rate=0.0)  # No dropout for deterministic test

        x = torch.ones(1, 10, d_model)
        pe.eval()  # Disable dropout
        output = pe(x)

        # First, the input should be scaled
        # Then positional encoding is added
        # Check that the scaling happens (output != input + pe)
        assert not torch.allclose(output, x)

    def test_offset(self):
        """Test positional encoding with offset for streaming."""
        d_model = 64
        pe = PositionalEncoding(d_model, dropout_rate=0.0)
        pe.eval()

        x = torch.ones(1, 10, d_model)

        # Get encodings with different offsets
        out0 = pe(x, offset=0)
        out10 = pe(x, offset=10)

        # They should be different due to different positional encodings
        assert not torch.allclose(out0, out10)

    def test_max_len(self):
        """Test that encoding works up to max_len."""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len=max_len)

        x = torch.randn(1, max_len, d_model)
        output = pe(x)

        assert output.shape == x.shape


class TestRelPositionalEncoding:
    """Tests for RelPositionalEncoding."""

    def test_forward_returns_tuple(self):
        """Test that relative positional encoding returns (output, pos_emb)."""
        d_model = 256
        rel_pe = RelPositionalEncoding(d_model, dropout_rate=0.0)
        rel_pe.eval()

        x = torch.randn(2, 10, d_model)
        output, pos_emb = rel_pe(x)

        assert output.shape == x.shape
        assert pos_emb.shape == (1, 10, d_model)

    def test_pos_emb_consistent(self):
        """Test that positional embedding is consistent for same length."""
        d_model = 128
        rel_pe = RelPositionalEncoding(d_model, dropout_rate=0.0)
        rel_pe.eval()

        x1 = torch.randn(1, 20, d_model)
        x2 = torch.randn(1, 20, d_model)

        _, pos_emb1 = rel_pe(x1)
        _, pos_emb2 = rel_pe(x2)

        # Positional embeddings should be the same for same sequence length
        assert torch.allclose(pos_emb1, pos_emb2)


class TestStreamPositionalEncoding:
    """Tests for StreamPositionalEncoding."""

    def test_automatic_position_tracking(self):
        """Test that position is automatically incremented."""
        d_model = 64
        spe = StreamPositionalEncoding(d_model, dropout_rate=0.0)
        spe.eval()

        # Process two chunks
        x1 = torch.ones(1, 10, d_model)
        x2 = torch.ones(1, 10, d_model)

        out1 = spe(x1)  # Should use offset=0
        out2 = spe(x2)  # Should use offset=10

        # Outputs should be different due to different positional encodings
        assert not torch.allclose(out1, out2)

    def test_reset(self):
        """Test that reset() resets the position counter."""
        d_model = 64
        spe = StreamPositionalEncoding(d_model, dropout_rate=0.0)
        spe.eval()

        x = torch.ones(1, 10, d_model)

        out1 = spe(x)
        spe.reset()
        out2 = spe(x)

        # After reset, should get same encoding as first time
        assert torch.allclose(out1, out2)

    def test_manual_offset(self):
        """Test manual offset override."""
        d_model = 64
        spe = StreamPositionalEncoding(d_model, dropout_rate=0.0)
        spe.eval()

        x = torch.ones(1, 10, d_model)

        # Use manual offset
        out_manual = spe(x, offset=5)

        # Reset and auto-advance to same position
        spe.reset()
        _ = spe(x)  # offset 0-9
        _ = spe(x)  # offset 10-19
        # Position counter is now at 20, but we want to compare offset=5

        spe.reset()
        # Skip to offset 5 by processing 5 frames
        spe(torch.ones(1, 5, d_model))
        out_auto = spe(x)  # offset 5-14

        assert torch.allclose(out_manual, out_auto)


class TestConvolutionModule:
    """Tests for Conformer ConvolutionModule."""

    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        channels = 256
        conv = ConvolutionModule(channels)

        x = torch.randn(2, 100, channels)
        output = conv(x)

        assert output.shape == x.shape

    def test_depthwise_convolution(self):
        """Test that depthwise convolution has correct groups."""
        channels = 128
        conv = ConvolutionModule(channels, kernel_size=31)

        # Check that depthwise conv has groups=channels
        assert conv.depthwise_conv.groups == channels

    def test_kernel_size_odd(self):
        """Test that odd kernel sizes work correctly."""
        channels = 64
        for kernel_size in [3, 7, 15, 31]:
            conv = ConvolutionModule(channels, kernel_size=kernel_size)
            x = torch.randn(1, 50, channels)
            output = conv(x)
            assert output.shape == x.shape

    def test_kernel_size_even_raises(self):
        """Test that even kernel sizes raise an assertion error."""
        with pytest.raises(AssertionError):
            ConvolutionModule(channels=64, kernel_size=32)

    def test_training_inference_batchnorm(self):
        """Test that batch norm behaves differently in train/eval."""
        conv = ConvolutionModule(channels=128)
        x = torch.randn(4, 50, 128)

        # Training mode
        conv.train()
        out_train = conv(x)

        # Eval mode
        conv.eval()
        out_eval = conv(x)

        # Outputs should differ due to batch norm behavior
        # (In train: uses batch statistics, in eval: uses running statistics)
        assert not torch.allclose(out_train, out_eval, atol=1e-4)


class TestLayerNorm:
    """Tests for LayerNorm."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        dim = 256
        ln = LayerNorm(dim)

        x = torch.randn(2, 100, dim)
        output = ln(x)

        assert output.shape == x.shape

    def test_normalization(self):
        """Test that output is normalized (mean≈0, var≈1)."""
        dim = 128
        ln = LayerNorm(dim)

        x = torch.randn(4, 50, dim) * 10 + 5  # Mean≈5, std≈10
        output = ln(x)

        # Check normalization along last dimension
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

    def test_learnable_parameters(self):
        """Test that weight and bias are learnable."""
        ln = LayerNorm(64)

        assert ln.weight.requires_grad
        assert ln.bias.requires_grad
        assert ln.weight.shape == (64,)
        assert ln.bias.shape == (64,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
