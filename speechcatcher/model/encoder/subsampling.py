"""Subsampling layers for encoder."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling without positional encoding.

    This module reduces the time dimension by applying strided convolutions.
    Based on ESPnet's Conv2dSubsamplingWOPosEnc.

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (model dimension)
        dropout_rate: Dropout rate
        kernels: List of kernel sizes for each conv layer (default: [3, 3])
        strides: List of strides for each conv layer (default: [2, 2])

    Shape:
        - Input: (batch, time, input_dim)
        - Output: (batch, time', output_dim) where time' = time // prod(strides)

    Example:
        With kernels=[3, 3], strides=[2, 2], subsample factor is 4x
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        kernels: list = None,
        strides: list = None,
    ):
        super().__init__()

        if kernels is None:
            kernels = [3, 3]
        if strides is None:
            strides = [2, 2]

        assert len(kernels) == len(strides), "kernels and strides must have same length"

        self.kernels = kernels
        self.strides = strides

        # Build conv layers
        conv_layers = []
        for i, (kernel, stride) in enumerate(zip(kernels, strides)):
            in_channels = 1 if i == 0 else output_dim
            conv_layers.extend([
                nn.Conv2d(in_channels, output_dim, kernel, stride),
                nn.ReLU(),
            ])

        self.conv = nn.Sequential(*conv_layers)

        # Calculate output feature dimension after convolutions
        # Each conv reduces: out = (in - kernel) / stride + 1
        out_len = input_dim
        for kernel, stride in zip(kernels, strides):
            out_len = math.floor((out_len - kernel) / stride + 1)

        # Linear projection to output_dim
        self.out = nn.Linear(output_dim * out_len, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (batch, time, input_dim)
            x_mask: Optional mask tensor (batch, 1, time)

        Returns:
            Tuple of:
                - Subsampled output (batch, time', output_dim)
                - Subsampled mask (batch, 1, time') or None
        """
        # Add channel dimension: (batch, time, input_dim) -> (batch, 1, time, input_dim)
        x = x.unsqueeze(1)

        # Apply convolutions: (batch, 1, time, input_dim) -> (batch, output_dim, time', feat')
        x = self.conv(x)

        # Reshape and project
        batch, channels, time, feat = x.size()
        # (batch, channels, time, feat) -> (batch, time, channels * feat)
        x = x.transpose(1, 2).contiguous().view(batch, time, channels * feat)
        # (batch, time, channels * feat) -> (batch, time, output_dim)
        x = self.out(x)

        # Subsample mask if provided
        if x_mask is not None:
            for kernel, stride in zip(self.kernels, self.strides):
                # Subsample mask: take every stride-th element, accounting for kernel
                x_mask = x_mask[:, :, :-kernel+1:stride]

        return x, x_mask
