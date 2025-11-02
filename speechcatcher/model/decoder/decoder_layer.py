"""Transformer decoder layer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from speechcatcher.model.layers import LayerNorm


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer.

    This layer implements a standard Transformer decoder block with:
    1. Masked self-attention on target sequence
    2. Cross-attention to encoder output
    3. Position-wise feed-forward network

    Args:
        size: Model dimension
        self_attn: Self-attention module (MultiHeadedAttention)
        src_attn: Source attention module (MultiHeadedAttention)
        feed_forward: Feed-forward module (PositionwiseFeedForward)
        dropout_rate: Dropout rate
        normalize_before: Whether to apply layer norm before attention/FFN
        concat_after: Whether to concat attention input and output (adds linear layer)

    Shape:
        - Target input: (batch, target_len, size)
        - Memory input: (batch, source_len, size)
        - Output: (batch, target_len, size)
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            tgt: Target sequence (batch, target_len, size)
            tgt_mask: Target mask (batch, target_len, target_len) - causal mask
            memory: Encoder output (batch, source_len, size)
            memory_mask: Memory mask (batch, 1, source_len) or (batch, target_len, source_len)
            cache: Cached previous output for incremental decoding (batch, target_len-1, size)

        Returns:
            Tuple of (output, tgt_mask, memory, memory_mask)
        """
        # Self-attention block
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            # Full sequence processing
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # Incremental decoding: compute only the last frame
            assert cache.shape == (tgt.shape[0], tgt.shape[1] - 1, self.size), \
                f"Cache shape {cache.shape} != expected {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None if tgt_mask is None else tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))

        if not self.normalize_before:
            x = self.norm1(x)

        # Cross-attention block
        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        if self.concat_after:
            x_concat = torch.cat((x, self.src_attn(x, memory, memory, memory_mask)), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))

        if not self.normalize_before:
            x = self.norm2(x)

        # Feed-forward block
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        # Concatenate cache for incremental decoding
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask
