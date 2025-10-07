"""Position-wise feed-forward network for Transformer/Conformer."""

from typing import Optional

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed-forward network.

    This implements the FFN layer from "Attention is All You Need":
        FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension (typically 4x input_dim for Transformers)
        output_dim: Output dimension (typically same as input_dim)
        dropout_rate: Dropout rate
        activation: Activation function (default: ReLU)

    Shape:
        - Input: (batch, time, input_dim)
        - Output: (batch, time, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, time, input_dim)

        Returns:
            Output tensor (batch, time, output_dim)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
