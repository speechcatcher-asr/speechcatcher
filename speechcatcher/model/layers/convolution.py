"""Convolution module for Conformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer.

    This implements the convolution module from "Conformer: Convolution-augmented
    Transformer for Speech Recognition" (Gulati et al., 2020).

    Architecture:
        LayerNorm -> Pointwise Conv (expansion) -> GLU -> Depthwise Conv
        -> BatchNorm -> Swish -> Pointwise Conv (projection) -> Dropout

    Args:
        channels: Number of input/output channels
        kernel_size: Kernel size for depthwise convolution (default: 31)
        dropout_rate: Dropout rate (default: 0.1)
        bias: Whether to use bias in convolutions (default: True)

    Shape:
        - Input: (batch, time, channels)
        - Output: (batch, time, channels)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 31,
        dropout_rate: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding"

        self.layernorm = nn.LayerNorm(channels)

        # Pointwise expansion (2x channels for GLU)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,  # Depthwise
            bias=bias,
        )

        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = Swish()

        # Pointwise projection
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, time, channels)

        Returns:
            Output tensor (batch, time, channels)
        """
        # LayerNorm
        x = self.layernorm(x)

        # Transpose to (batch, channels, time) for Conv1d
        x = x.transpose(1, 2)

        # Pointwise expansion
        x = self.pointwise_conv1(x)

        # GLU (Gated Linear Unit): split into two halves and apply gate
        x_a, x_b = x.chunk(2, dim=1)
        x = x_a * torch.sigmoid(x_b)

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # BatchNorm + Swish
        x = self.batch_norm(x)
        x = self.activation(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)

        # Transpose back to (batch, time, channels)
        x = x.transpose(1, 2)

        # Dropout
        return self.dropout(x)
