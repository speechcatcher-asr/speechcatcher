"""Multi-head attention with Flash Attention 2 support."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class MultiHeadedAttention(nn.Module):
    """Multi-head attention with optional Flash Attention 2.

    This implements standard scaled dot-product attention with multiple heads.
    If Flash Attention 2 is available and the input is on CUDA with appropriate
    compute capability, it will use the optimized kernel. Otherwise, it falls
    back to vanilla PyTorch implementation.

    Args:
        n_head: Number of attention heads
        n_feat: Model dimension
        dropout_rate: Dropout rate (default: 0.0)
        use_flash_attn: Whether to use Flash Attention if available (default: True)

    Shape:
        - Input: query (batch, time_q, n_feat), key (batch, time_k, n_feat),
                 value (batch, time_v, n_feat)
        - Output: (batch, time_q, n_feat)
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        assert n_feat % n_head == 0, "n_feat must be divisible by n_head"

        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
        self.use_flash_attn = use_flash_attn and FLASH_ATTENTION_AVAILABLE

        # Store attention weights for analysis (only in vanilla mode)
        self.attn = None

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query: Query tensor (batch, time_q, n_feat)
            key: Key tensor (batch, time_k, n_feat)
            value: Value tensor (batch, time_v, n_feat)

        Returns:
            Tuple of transformed tensors:
                - q: (batch, n_head, time_q, d_k)
                - k: (batch, n_head, time_k, d_k)
                - v: (batch, n_head, time_v, d_k)
        """
        n_batch = query.size(0)

        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)

        # Transpose to (batch, n_head, time, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention_vanilla(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vanilla scaled dot-product attention.

        Args:
            query: Query tensor (batch, n_head, time_q, d_k)
            key: Key tensor (batch, n_head, time_k, d_k)
            value: Value tensor (batch, n_head, time_v, d_k)
            mask: Attention mask (batch, 1, time_k) or (batch, time_q, time_k)

        Returns:
            Output tensor (batch, time_q, n_feat)
        """
        n_batch = value.size(0)

        # Compute attention scores: (batch, n_head, time_q, time_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, *, time_k)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask == 0, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        # Apply dropout
        p_attn = self.dropout(self.attn)

        # Apply attention to values: (batch, n_head, time_q, d_k)
        x = torch.matmul(p_attn, value)

        # Reshape: (batch, time_q, n_feat)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)

    def forward_attention_flash(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Flash Attention 2 implementation.

        Args:
            query: Query tensor (batch, n_head, time_q, d_k)
            key: Key tensor (batch, n_head, time_k, d_k)
            value: Value tensor (batch, n_head, time_v, d_k)
            mask: Not used in Flash Attention (causal flag instead)
            causal: Whether to use causal masking

        Returns:
            Output tensor (batch, time_q, n_feat)
        """
        n_batch = value.size(0)

        # Flash Attention expects (batch, seqlen, nheads, headdim)
        # Current: (batch, n_head, time, d_k)
        q = query.transpose(1, 2)  # (batch, time_q, n_head, d_k)
        k = key.transpose(1, 2)  # (batch, time_k, n_head, d_k)
        v = value.transpose(1, 2)  # (batch, time_v, n_head, d_k)

        # Apply Flash Attention
        # Note: flash_attn_func handles scaling internally
        x = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_rate if self.training else 0.0,
            softmax_scale=1.0 / math.sqrt(self.d_k),
            causal=causal,
        )

        # Reshape: (batch, time_q, n_feat)
        x = x.reshape(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query tensor (batch, time_q, n_feat)
            key: Key tensor (batch, time_k, n_feat)
            value: Value tensor (batch, time_v, n_feat)
            mask: Attention mask (batch, 1, time_k) or (batch, time_q, time_k)
            cache: Tuple of cached (key, value) for incremental decoding

        Returns:
            Output tensor (batch, time_q, n_feat)
        """
        # Transform Q, K, V
        q, k, v = self.forward_qkv(query, key, value)

        # Handle cached K, V for incremental decoding
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)  # Concat along time dimension
            v = torch.cat([v_cache, v], dim=2)

        # Choose attention implementation
        use_flash = (
            self.use_flash_attn
            and query.is_cuda
            and mask is None  # Flash Attention doesn't support arbitrary masks
            and cache is None  # Flash Attention path for training/full decoding
        )

        if use_flash:
            output = self.forward_attention_flash(q, k, v, mask=mask, causal=False)
        else:
            output = self.forward_attention_vanilla(q, k, v, mask=mask)

        return output

    def forward_with_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with cache return for incremental decoding.

        Args:
            query: Query tensor (batch, time_q, n_feat)
            key: Key tensor (batch, time_k, n_feat)
            value: Value tensor (batch, time_v, n_feat)
            cache: Tuple of cached (key, value) from previous step
            mask: Attention mask

        Returns:
            Tuple of:
                - Output tensor (batch, time_q, n_feat)
                - Updated cache (k, v) in (batch, n_head, time, d_k) format
        """
        # Transform Q, K, V
        q, k, v = self.forward_qkv(query, key, value)

        # Handle cached K, V
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Compute attention (always use vanilla for incremental decoding)
        output = self.forward_attention_vanilla(q, k, v, mask=mask)

        # Return output and updated cache
        return output, (k, v)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-head attention with relative positional encoding.

    This implements relative positional encoding as described in
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    and used in Conformer.

    Args:
        n_head: Number of attention heads
        n_feat: Model dimension
        dropout_rate: Dropout rate (default: 0.0)
        use_flash_attn: Whether to use Flash Attention if available (default: False)
                       Note: Flash Attention doesn't support relative positions yet

    Shape:
        - Input: query (batch, time_q, n_feat), key (batch, time_k, n_feat),
                 value (batch, time_v, n_feat), pos_emb (1, time_k, n_feat)
        - Output: (batch, time_q, n_feat)
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        use_flash_attn: bool = False,  # Disabled by default for rel pos
    ):
        super().__init__(n_head, n_feat, dropout_rate, use_flash_attn=False)

        # Linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        # Learnable biases for relative position
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))

        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative shift operation.

        Args:
            x: Input tensor (batch, n_head, time_q, time_k)

        Returns:
            Shifted tensor (batch, n_head, time_q, time_k)
        """
        batch, n_head, time_q, time_k = x.size()
        zero_pad = torch.zeros((batch, n_head, time_q, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(batch, n_head, time_k + 1, time_q)
        x = x_padded[:, :, 1:].view_as(x)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with relative positional encoding.

        Args:
            query: Query tensor (batch, time_q, n_feat)
            key: Key tensor (batch, time_k, n_feat)
            value: Value tensor (batch, time_v, n_feat)
            pos_emb: Positional embedding (1, time_k, n_feat)
            mask: Attention mask (batch, 1, time_k) or (batch, time_q, time_k)

        Returns:
            Output tensor (batch, time_q, n_feat)
        """
        n_batch = query.size(0)

        # Transform Q, K, V
        q, k, v = self.forward_qkv(query, key, value)

        # Transform positional embedding
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (1, n_head, time_k, d_k)

        # Add biases to queries
        q_with_bias_u = q + self.pos_bias_u.view(1, self.h, 1, self.d_k)
        q_with_bias_v = q + self.pos_bias_v.view(1, self.h, 1, self.d_k)

        # Compute attention scores with content and position
        # Content-based attention
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # Position-based attention
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        # Combine and scale
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask == 0, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        # Apply dropout
        p_attn = self.dropout(self.attn)

        # Apply attention to values
        x = torch.matmul(p_attn, v)

        # Reshape
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)
