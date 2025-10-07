"""Layer normalization variants."""

import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """Layer normalization with optional dimension parameter.

    This is a thin wrapper around torch.nn.LayerNorm that provides
    compatibility with the ESPnet interface and allows for easier
    pre-norm / post-norm switching.

    Args:
        dim: Normalization dimension
        eps: Epsilon for numerical stability (default: 1e-12)

    Shape:
        - Input: (*, dim)
        - Output: (*, dim)
    """

    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (*, dim)

        Returns:
            Normalized tensor (*, dim)
        """
        return super().forward(x)
