"""Contextual Block Encoder Layer for streaming Transformer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from speechcatcher.model.layers import LayerNorm


class ContextualBlockEncoderLayer(nn.Module):
    """Contextual Block Encoder Layer.

    This layer implements the core of contextual block processing for streaming
    Transformer ASR. It handles context vector propagation between blocks.

    Args:
        size: Model dimension
        self_attn: Self-attention module (MultiHeadedAttention)
        feed_forward: Feed-forward module (PositionwiseFeedForward)
        dropout_rate: Dropout rate
        total_layer_num: Total number of encoder layers (for context storage)
        normalize_before: Whether to apply layer norm before attention/FFN
        concat_after: Whether to concat attention input and output (adds linear layer)

    Shape:
        - Input: (batch, n_blocks, block_size+2, size)
        - Output: (batch, n_blocks, block_size+2, size)

    Note:
        The block_size+2 includes:
        - Position 0: previous context vector
        - Positions 1 to block_size: actual block frames
        - Position block_size+1: current context vector (for next block)
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
        total_layer_num: int,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.total_layer_num = total_layer_num

        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        infer_mode: bool = False,
        past_ctx: Optional[torch.Tensor] = None,
        next_ctx: Optional[torch.Tensor] = None,
        is_short_segment: bool = False,
        layer_idx: int = 0,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor], Optional[torch.Tensor], bool, int]:
        """Forward pass with mode selection.

        Args:
            x: Input tensor (batch, n_blocks, block_size+2, size)
            mask: Attention mask (batch, n_blocks, block_size+2, block_size+2)
            infer_mode: Whether in inference mode (streaming)
            past_ctx: Previous context vectors from previous blocks
            next_ctx: Storage for next context vectors
            is_short_segment: Whether processing a short segment (no context needed)
            layer_idx: Current layer index
            cache: Optional cache for incremental decoding

        Returns:
            Tuple of (x, mask, infer_mode, past_ctx, next_ctx, is_short_segment, layer_idx+1)
        """
        if self.training or not infer_mode:
            return self.forward_train(x, mask, past_ctx, next_ctx, layer_idx, cache)
        else:
            return self.forward_infer(x, mask, past_ctx, next_ctx, is_short_segment, layer_idx, cache)

    def forward_train(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        past_ctx: Optional[torch.Tensor],
        next_ctx: Optional[torch.Tensor],
        layer_idx: int,
        cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor], Optional[torch.Tensor], bool, int]:
        """Forward for training (simulates streaming with blocks).

        Args:
            x: Input (batch, n_blocks, block_size+2, size)
            mask: Mask (batch, n_blocks, block_size+2, block_size+2)
            past_ctx: Context from input (batch, n_blocks, total_layer_num, size)
            next_ctx: Context storage (batch, n_blocks, total_layer_num, size)
            layer_idx: Current layer
            cache: Unused in training

        Returns:
            Updated x, mask, False, next_ctx, next_ctx, False, layer_idx+1
        """
        nbatch = x.size(0)
        nblock = x.size(1)

        # Initialize or update context storage
        if past_ctx is not None:
            if next_ctx is None:
                next_ctx = past_ctx.new_zeros(nbatch, nblock, self.total_layer_num, x.size(-1))
            else:
                # Set first frame of each block from previous layer's context
                x[:, :, 0] = past_ctx[:, :, layer_idx]

        # Reshape from (nbatch, nblock, block_size+2, dim) to (nbatch*nblock, block_size+2, dim)
        x = x.view(-1, x.size(-2), x.size(-1))
        if mask is not None:
            mask = mask.view(-1, mask.size(-2), mask.size(-1))

        # Self-attention block
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            # Incremental mode (not used in training typically)
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        # Feed-forward block
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        layer_idx += 1

        # Reshape back to (nbatch, nblock, block_size+2, dim)
        x = x.view(nbatch, -1, x.size(-2), x.size(-1)).squeeze(1)
        if mask is not None:
            mask = mask.view(nbatch, -1, mask.size(-2), mask.size(-1)).squeeze(1)

        # Store context for next layer (last frame of each block)
        if next_ctx is not None and layer_idx < self.total_layer_num:
            next_ctx[:, 0, layer_idx, :] = x[:, 0, -1, :]
            next_ctx[:, 1:, layer_idx, :] = x[:, 0:-1, -1, :]

        return x, mask, False, next_ctx, next_ctx, False, layer_idx

    def forward_infer(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        past_ctx: Optional[torch.Tensor],
        next_ctx: Optional[torch.Tensor],
        is_short_segment: bool,
        layer_idx: int,
        cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor], Optional[torch.Tensor], bool, int]:
        """Forward for inference (true streaming).

        Args:
            x: Input (batch, n_blocks, block_size+2, size)
            mask: Mask (batch, n_blocks, block_size+2, block_size+2)
            past_ctx: Context from previous blocks (batch, total_layer_num, size)
            next_ctx: Context storage (batch, total_layer_num, size)
            is_short_segment: Whether this is a short segment
            layer_idx: Current layer
            cache: Unused

        Returns:
            Updated x, mask, True, past_ctx, next_ctx, is_short_segment, layer_idx+1
        """
        nbatch = x.size(0)
        nblock = x.size(1)

        # Initialize context storage at layer 0
        if layer_idx == 0:
            assert next_ctx is None
            next_ctx = x.new_zeros(nbatch, self.total_layer_num, x.size(-1))

        # Reshape for processing
        x = x.view(-1, x.size(-2), x.size(-1))
        if mask is not None:
            mask = mask.view(-1, mask.size(-2), mask.size(-1))

        # Self-attention block
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        # Feed-forward block
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        # Reshape back
        x = x.view(nbatch, nblock, x.size(-2), x.size(-1))
        if mask is not None:
            mask = mask.view(nbatch, nblock, mask.size(-2), mask.size(-1))

        # Propagate context information (last frame) to first frame of next block
        if not is_short_segment:
            if past_ctx is None:
                # First block of utterance: copy last frame to first
                x[:, 0, 0, :] = x[:, 0, -1, :]
            else:
                # Use context from previous block
                x[:, 0, 0, :] = past_ctx[:, layer_idx, :]

            if nblock > 1:
                # Inter-block context: copy last frame of block i-1 to first frame of block i
                x[:, 1:, 0, :] = x[:, 0:-1, -1, :]

            # Store context for next blocks
            next_ctx[:, layer_idx, :] = x[:, -1, -1, :]
        else:
            next_ctx = None

        return x, mask, True, past_ctx, next_ctx, is_short_segment, layer_idx + 1
