"""Transformer decoder for ASR."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from speechcatcher.model.attention import MultiHeadedAttention
from speechcatcher.model.decoder.decoder_layer import TransformerDecoderLayer
from speechcatcher.model.layers import (
    LayerNorm,
    PositionalEncoding,
    PositionwiseFeedForward,
)


def subsequent_mask(size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Create causal mask for decoder self-attention.

    Args:
        size: Sequence length
        device: Target device

    Returns:
        Causal mask (1, size, size) with lower triangular True values
    """
    mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return ~mask


def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create padding mask from lengths.

    Args:
        lengths: Sequence lengths (batch,)
        max_len: Maximum length (if None, use max of lengths)

    Returns:
        Padding mask (batch, max_len) where True indicates padding
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = int(lengths.max())

    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)

    return seq_range_expand >= seq_length_expand


class TransformerDecoder(nn.Module):
    """Transformer decoder for ASR.

    This implements a standard Transformer decoder with:
    - Token embedding layer
    - Positional encoding
    - Stack of decoder layers (self-attention + cross-attention + FFN)
    - Output projection to vocabulary

    Supports both full-sequence decoding (training) and incremental decoding
    (beam search inference) with KV caching.

    Args:
        vocab_size: Size of vocabulary
        encoder_output_size: Dimension of encoder output (= model dimension)
        attention_heads: Number of attention heads
        linear_units: FFN hidden dimension
        num_blocks: Number of decoder layers
        dropout_rate: Dropout rate
        positional_dropout_rate: Positional encoding dropout
        self_attention_dropout_rate: Self-attention dropout
        src_attention_dropout_rate: Cross-attention dropout
        input_layer: Input layer type ('embed' or 'linear')
        use_output_layer: Whether to use output projection layer
        normalize_before: Pre-norm (True) or post-norm (False)
        concat_after: Whether to concat attention input/output

    Shape:
        - Encoder output: (batch, enc_len, encoder_output_size)
        - Target tokens: (batch, tgt_len) - token IDs
        - Output: (batch, tgt_len, vocab_size) - logits
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        attention_dim = encoder_output_size

        # Input layer
        if input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(vocab_size, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"input_layer must be 'embed' or 'linear', got {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        # Decoder layers
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(
                size=attention_dim,
                self_attn=MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                src_attn=MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                feed_forward=PositionwiseFeedForward(
                    attention_dim, linear_units, attention_dim, dropout_rate
                ),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
            )
            for _ in range(num_blocks)
        ])

        # Output layer
        if use_output_layer:
            self.output_layer = nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            hs_pad: Encoder output (batch, enc_len, feat)
            hlens: Encoder output lengths (batch,)
            ys_in_pad: Target token IDs (batch, tgt_len)
            ys_in_lens: Target lengths (batch,)

        Returns:
            Tuple of:
                - logits: Output logits (batch, tgt_len, vocab_size)
                - olens: Output lengths (batch,)
        """
        tgt = ys_in_pad

        # Create target mask: (batch, tgt_len, tgt_len)
        # Combines padding mask and causal mask
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)  # (batch, 1, tgt_len)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)  # (1, tgt_len, tgt_len)
        tgt_mask = tgt_mask & m  # (batch, tgt_len, tgt_len)

        # Create memory mask: (batch, 1, enc_len)
        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, max_len=memory.size(1)))[:, None, :].to(memory.device)

        # Padding adjustment for Longformer (if needed)
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = nn.functional.pad(memory_mask, (0, padlen), "constant", False)

        # Embed target tokens
        x = self.embed(tgt)

        # Apply decoder layers
        for decoder in self.decoders:
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask)

        # Final normalization
        if self.normalize_before:
            x = self.after_norm(x)

        # Output projection
        if self.output_layer is not None:
            x = self.output_layer(x)

        # Output lengths (reduce along last dimension of tgt_mask)
        olens = tgt_mask.sum(dim=2).sum(dim=1)
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step for incremental decoding (beam search).

        Args:
            tgt: Input token ids (batch, current_len)
            tgt_mask: Input token mask (batch, current_len, current_len)
            memory: Encoder output (batch, enc_len, feat)
            cache: Cached outputs from previous steps
                   List of (batch, current_len-1, size) for each layer

        Returns:
            Tuple of:
                - y: Log probabilities for next token (batch, vocab_size)
                - new_cache: Updated cache for next step
        """
        x = self.embed(tgt)

        if cache is None:
            cache = [None] * len(self.decoders)

        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, None, cache=c)
            new_cache.append(x)

        # Take only the last frame for output
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        # Project to vocabulary and compute log probabilities
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def score(
        self,
        ys: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Score interface for beam search (single hypothesis).

        Args:
            ys: Prefix tokens (hyp_len,)
            state: Previous decoder states
            x: Encoder output (enc_len, feat)

        Returns:
            Tuple of (log_prob, new_state)
        """
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Optional[List[torch.Tensor]]],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """Batch scoring interface for beam search.

        Args:
            ys: Batch of prefix tokens (batch, hyp_len)
            states: List of decoder states for each hypothesis
            xs: Encoder output (batch, enc_len, feat)

        Returns:
            Tuple of:
                - logp: Log probabilities for next token (batch, vocab_size)
                - state_list: Updated states for each hypothesis
        """
        n_batch = len(ys)
        n_layers = len(self.decoders)

        # Merge states: transpose [batch, layer] -> [layer, batch]
        if states[0] is None:
            batch_state = None
        else:
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # Batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # Transpose state: [layer, batch] -> [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]

        return logp, state_list
