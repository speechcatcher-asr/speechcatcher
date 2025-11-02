"""Positional encoding modules for Transformer/Conformer."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Absolute positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Embedding dimension
        dropout_rate: Dropout rate
        max_len: Maximum sequence length (default: 5000)
        reverse: Whether to reverse the positional encoding (default: False)

    Shape:
        - Input: (batch, time, d_model)
        - Output: (batch, time, d_model)
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
        reverse: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, time, d_model)
            offset: Positional offset for streaming (default: 0)

        Returns:
            Output with positional encoding (batch, time, d_model)
        """
        # Scale input by sqrt(d_model) as in original Transformer
        x = x * math.sqrt(self.d_model)

        # Add positional encoding
        seq_len = x.size(1)
        if self.reverse:
            # For right-to-left processing
            pe = self.pe[:, offset : offset + seq_len].flip(1)
        else:
            pe = self.pe[:, offset : offset + seq_len]

        x = x + pe
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for Conformer.

    This extends absolute positional encoding by providing both
    the positionally encoded input and the positional encoding itself
    for use in relative positional multi-head attention.

    Args:
        d_model: Embedding dimension
        dropout_rate: Dropout rate
        max_len: Maximum sequence length (default: 5000)

    Shape:
        - Input: (batch, time, d_model)
        - Output: Tuple of
            - (batch, time, d_model): Positionally encoded input
            - (1, time, d_model): Positional encoding for relative attention
    """

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch, time, d_model)
            offset: Positional offset for streaming (default: 0)

        Returns:
            Tuple of:
                - Output with positional encoding (batch, time, d_model)
                - Positional encoding tensor (1, time, d_model)
        """
        # Scale input by sqrt(d_model)
        x = x * math.sqrt(self.d_model)

        # Get positional encoding
        seq_len = x.size(1)
        pe = self.pe[:, offset : offset + seq_len]

        # Add positional encoding and apply dropout
        x = self.dropout(x + pe)

        # Return both the encoded input and the positional encoding
        return x, pe


class StreamPositionalEncoding(PositionalEncoding):
    """Streaming positional encoding with state management.

    This variant maintains an internal position counter for streaming
    applications, allowing proper positional encoding across chunks.

    Args:
        d_model: Embedding dimension
        dropout_rate: Dropout rate
        max_len: Maximum sequence length (default: 5000)

    Shape:
        - Input: (batch, time, d_model)
        - Output: (batch, time, d_model)
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__(d_model, dropout_rate, max_len)
        self.register_buffer("_current_position", torch.tensor(0, dtype=torch.long))

    def forward(
        self, x: torch.Tensor, offset: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass with automatic position tracking.

        Args:
            x: Input tensor (batch, time, d_model)
            offset: Manual positional offset (if None, uses internal counter)

        Returns:
            Output with positional encoding (batch, time, d_model)
        """
        if offset is None:
            offset = self._current_position.item()
            self._current_position += x.size(1)

        return super().forward(x, offset)

    def reset(self):
        """Reset the internal position counter."""
        self._current_position.zero_()
