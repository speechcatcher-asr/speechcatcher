"""Contextual Block Transformer Encoder for streaming ASR."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from speechcatcher.model.attention import MultiHeadedAttention
from speechcatcher.model.encoder.contextual_block_encoder_layer import ContextualBlockEncoderLayer
from speechcatcher.model.encoder.subsampling import Conv2dSubsampling
from speechcatcher.model.layers import (
    LayerNorm,
    PositionwiseFeedForward,
    StreamPositionalEncoding,
)


class ContextualBlockTransformerEncoder(nn.Module):
    """Contextual Block Transformer Encoder.

    This implements the streaming Transformer encoder from:
    "Transformer ASR with contextual block processing" (Tsunoo et al., 2019)
    https://arxiv.org/abs/1910.07204

    The encoder processes audio in blocks with context vector inheritance,
    enabling streaming ASR with bounded latency.

    Args:
        input_size: Input feature dimension (e.g., 80 for log-mel)
        output_size: Model dimension (e.g., 256)
        attention_heads: Number of attention heads
        linear_units: FFN hidden dimension
        num_blocks: Number of encoder layers
        dropout_rate: Dropout rate
        positional_dropout_rate: Positional encoding dropout
        attention_dropout_rate: Attention dropout
        input_layer: Input layer type ('conv2d', 'conv2d6', 'conv2d8', 'linear', None)
        normalize_before: Pre-norm (True) or post-norm (False)
        concat_after: Whether to concat attention input/output
        positionwise_layer_type: FFN type ('linear', 'conv1d', 'conv1d-linear')
        positionwise_conv_kernel_size: Kernel size for conv1d FFN
        block_size: Frames per block (after subsampling)
        hop_size: Frames to advance per block
        look_ahead: Right context frames
        init_average: Use average (True) or max (False) for context initialization
        ctx_pos_enc: Apply positional encoding to context vectors

    Shape:
        - Input: (batch, time, input_size)
        - Output: (batch, time', output_size) where time' depends on subsampling
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
        init_average: bool = True,
        ctx_pos_enc: bool = True,
    ):
        super().__init__()

        self._output_size = output_size
        self.pos_enc = StreamPositionalEncoding(output_size, positional_dropout_rate)

        # Input layer (subsampling)
        if input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.LayerNorm(output_size),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
            )
            self.subsample = 1
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size, output_size, dropout_rate, kernels=[3, 3], strides=[2, 2]
            )
            self.subsample = 4
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling(
                input_size, output_size, dropout_rate, kernels=[3, 5], strides=[2, 3]
            )
            self.subsample = 6
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling(
                input_size, output_size, dropout_rate, kernels=[3, 3, 3], strides=[2, 2, 2]
            )
            self.subsample = 8
        elif input_layer is None:
            self.embed = None
            self.subsample = 1
        else:
            raise ValueError(f"Unknown input_layer: {input_layer}")

        self.normalize_before = normalize_before

        # Build encoder layers
        self.encoders = nn.ModuleList([
            ContextualBlockEncoderLayer(
                size=output_size,
                self_attn=MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                feed_forward=PositionwiseFeedForward(
                    output_size, linear_units, output_size, dropout_rate
                ),
                dropout_rate=dropout_rate,
                total_layer_num=num_blocks,
                normalize_before=normalize_before,
                concat_after=concat_after,
            )
            for _ in range(num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        # Block processing parameters
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.init_average = init_average
        self.ctx_pos_enc = ctx_pos_enc

    def output_size(self) -> int:
        """Return output dimension."""
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[dict] = None,
        is_final: bool = True,
        infer_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """Forward pass with mode selection.

        Args:
            xs_pad: Input tensor (batch, time, input_size)
            ilens: Input lengths (batch,)
            prev_states: Previous states for streaming
            is_final: Whether this is the final chunk
            infer_mode: Whether in inference mode

        Returns:
            Tuple of (output, output_lengths, next_states)
        """
        if self.training or not infer_mode:
            return self.forward_train(xs_pad, ilens, prev_states)
        else:
            return self.forward_infer(xs_pad, ilens, prev_states, is_final)

    def forward_train(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """Forward pass for training (simulates streaming).

        Processes the entire utterance in blocks to simulate streaming behavior.

        Args:
            xs_pad: Input (batch, time, input_size)
            ilens: Lengths (batch,)
            prev_states: Unused in training

        Returns:
            Tuple of (output, lengths, None)
        """
        # Create padding mask
        masks = self._make_pad_mask(ilens, xs_pad)

        # Apply input layer (subsampling)
        if isinstance(self.embed, Conv2dSubsampling):
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        total_frame_num = xs_pad.size(1)
        ys_pad = xs_pad.new_zeros(xs_pad.size())

        past_size = self.block_size - self.hop_size - self.look_ahead

        # Short sequence: process without blocking
        if self.block_size == 0 or total_frame_num <= self.block_size:
            xs_pad = self.pos_enc(xs_pad)
            xs_pad, masks = self._apply_encoder_layers(xs_pad, masks, infer_mode=False)

            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)

            olens = masks.squeeze(1).sum(1) if masks is not None else ilens.new_full([xs_pad.size(0)], xs_pad.size(1))
            return xs_pad, olens, None

        # Block processing for training
        block_num = math.ceil(float(total_frame_num - past_size - self.look_ahead) / float(self.hop_size))
        bsize = xs_pad.size(0)

        # Initialize context vectors
        addin = self._initialize_context_vectors(xs_pad, block_num)

        if self.ctx_pos_enc:
            addin = self.pos_enc(addin)

        xs_pad = self.pos_enc(xs_pad)

        # Prepare block-wise input
        xs_chunk, mask_online = self._prepare_block_input_train(xs_pad, addin, block_num, masks)

        # Apply encoder layers
        ys_chunk, _, _, _, _, _, _ = self._apply_encoder_layers_with_ctx(
            xs_chunk, mask_online, infer_mode=False, past_ctx=xs_chunk
        )

        # Extract output from blocks
        ys_pad = self._extract_output_from_blocks_train(ys_chunk, ys_pad, block_num, total_frame_num)

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        olens = masks.squeeze(1).sum(1) if masks is not None else ilens.new_full([ys_pad.size(0)], ys_pad.size(1))
        return ys_pad, olens, None

    def forward_infer(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[dict],
        is_final: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """Forward pass for streaming inference.

        Args:
            xs_pad: Input chunk (batch, time, input_size)
            ilens: Lengths (batch,)
            prev_states: Previous streaming states
            is_final: Whether this is the final chunk

        Returns:
            Tuple of (output, None, next_states or None)
        """
        # Extract previous states
        if prev_states is None:
            prev_addin = None
            buffer_before_downsampling = None
            ilens_buffer = None
            buffer_after_downsampling = None
            n_processed_blocks = 0
            past_encoder_ctx = None
        else:
            prev_addin = prev_states["prev_addin"]
            buffer_before_downsampling = prev_states["buffer_before_downsampling"]
            ilens_buffer = prev_states["ilens_buffer"]
            buffer_after_downsampling = prev_states["buffer_after_downsampling"]
            n_processed_blocks = prev_states["n_processed_blocks"]
            past_encoder_ctx = prev_states["past_encoder_ctx"]

        bsize = xs_pad.size(0)
        assert bsize == 1, "Inference only supports batch_size=1"

        # Concatenate with buffered input
        if prev_states is not None:
            xs_pad = torch.cat([buffer_before_downsampling, xs_pad], dim=1)
            ilens += ilens_buffer

        # Handle buffering before downsampling
        if is_final:
            buffer_before_downsampling = None
        else:
            n_samples = xs_pad.size(1) // self.subsample - 1
            if n_samples < 2:
                # Not enough samples, buffer and return empty
                next_states = {
                    "prev_addin": prev_addin,
                    "buffer_before_downsampling": xs_pad,
                    "ilens_buffer": ilens,
                    "buffer_after_downsampling": buffer_after_downsampling,
                    "n_processed_blocks": n_processed_blocks,
                    "past_encoder_ctx": past_encoder_ctx,
                }
                return xs_pad.new_zeros(bsize, 0, self._output_size), xs_pad.new_zeros(bsize), next_states

            # Keep residual for next chunk
            n_res_samples = xs_pad.size(1) % self.subsample + self.subsample * 2
            buffer_before_downsampling = xs_pad.narrow(1, xs_pad.size(1) - n_res_samples, n_res_samples)
            xs_pad = xs_pad.narrow(1, 0, n_samples * self.subsample)
            ilens_buffer = ilens.new_full([1], dtype=torch.long, fill_value=n_res_samples)
            ilens = ilens.new_full([1], dtype=torch.long, fill_value=n_samples * self.subsample)

        # Apply subsampling
        if isinstance(self.embed, Conv2dSubsampling):
            xs_pad, _ = self.embed(xs_pad, None)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        # Concatenate with buffered features
        if buffer_after_downsampling is not None:
            xs_pad = torch.cat([buffer_after_downsampling, xs_pad], dim=1)

        total_frame_num = xs_pad.size(1)

        # Calculate block number and buffering
        if is_final:
            past_size = self.block_size - self.hop_size - self.look_ahead
            block_num = math.ceil(float(total_frame_num - past_size - self.look_ahead) / float(self.hop_size))
            buffer_after_downsampling = None
        else:
            if total_frame_num <= self.block_size:
                # Buffer until we have enough frames
                next_states = {
                    "prev_addin": prev_addin,
                    "buffer_before_downsampling": buffer_before_downsampling,
                    "ilens_buffer": ilens_buffer,
                    "buffer_after_downsampling": xs_pad,
                    "n_processed_blocks": n_processed_blocks,
                    "past_encoder_ctx": past_encoder_ctx,
                }
                return xs_pad.new_zeros(bsize, 0, self._output_size), xs_pad.new_zeros(bsize), next_states

            overlap_size = self.block_size - self.hop_size
            block_num = max(0, xs_pad.size(1) - overlap_size) // self.hop_size
            res_frame_num = xs_pad.size(1) - self.hop_size * block_num
            buffer_after_downsampling = xs_pad.narrow(1, xs_pad.size(1) - res_frame_num, res_frame_num)
            xs_pad = xs_pad.narrow(1, 0, block_num * self.hop_size + overlap_size)

        # Short segment special case
        assert self.block_size > 0
        if n_processed_blocks == 0 and total_frame_num <= self.block_size and is_final:
            xs_chunk = self.pos_enc(xs_pad).unsqueeze(1)
            xs_pad, _ = self._apply_encoder_layers(xs_chunk, None, infer_mode=True, is_short=True)
            xs_pad = xs_pad.squeeze(0)
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)
            return xs_pad, None, None

        # Prepare blocks for processing
        xs_chunk = xs_pad.new_zeros(bsize, block_num, self.block_size + 2, xs_pad.size(-1))

        for i in range(block_num):
            cur_hop = i * self.hop_size
            chunk_length = min(self.block_size, total_frame_num - cur_hop)
            chunk_data = xs_pad.narrow(1, cur_hop, chunk_length)

            # Initialize context vector
            if self.init_average:
                addin = chunk_data.mean(1, keepdim=True)
            else:
                addin = chunk_data.max(1, keepdim=True)[0]

            if self.ctx_pos_enc:
                addin = self.pos_enc(addin, offset=i + n_processed_blocks)

            if prev_addin is None:
                prev_addin = addin

            xs_chunk[:, i, 0] = prev_addin
            xs_chunk[:, i, -1] = addin

            # Position encode the chunk
            chunk = self.pos_enc(chunk_data, offset=cur_hop + self.hop_size * n_processed_blocks)
            xs_chunk[:, i, 1:chunk_length + 1] = chunk

            prev_addin = addin

        # Create masks
        mask_online = self._create_block_mask(xs_pad, block_num)

        # Apply encoder
        ys_chunk, _, _, _, past_encoder_ctx, _, _ = self._apply_encoder_layers_with_ctx(
            xs_chunk, mask_online, infer_mode=True, past_ctx=past_encoder_ctx
        )

        # Remove context positions
        ys_chunk = ys_chunk.narrow(2, 1, self.block_size)

        # Extract output
        offset = self.block_size - self.look_ahead - self.hop_size
        if is_final:
            y_length = xs_pad.size(1) if n_processed_blocks == 0 else xs_pad.size(1) - offset
        else:
            y_length = block_num * self.hop_size
            if n_processed_blocks == 0:
                y_length += offset

        ys_pad = self._extract_output_from_blocks_infer(ys_chunk, y_length, offset, block_num, n_processed_blocks, is_final)

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        if is_final:
            next_states = None
        else:
            next_states = {
                "prev_addin": prev_addin,
                "buffer_before_downsampling": buffer_before_downsampling,
                "ilens_buffer": ilens_buffer,
                "buffer_after_downsampling": buffer_after_downsampling,
                "n_processed_blocks": n_processed_blocks + block_num,
                "past_encoder_ctx": past_encoder_ctx,
            }

        return ys_pad, None, next_states

    def _make_pad_mask(self, lengths: torch.Tensor, xs: torch.Tensor) -> Optional[torch.Tensor]:
        """Create padding mask."""
        batch_size = xs.size(0)
        max_len = xs.size(1)
        seq_range = torch.arange(0, max_len, dtype=torch.long, device=xs.device)
        seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)
        mask = seq_range < lengths.unsqueeze(1)
        return mask.unsqueeze(1)  # (batch, 1, time)

    def _initialize_context_vectors(self, xs_pad: torch.Tensor, block_num: int) -> torch.Tensor:
        """Initialize context vectors for all blocks."""
        bsize = xs_pad.size(0)
        total_frame_num = xs_pad.size(1)
        addin = xs_pad.new_zeros(bsize, block_num, xs_pad.size(-1))

        cur_hop = 0
        for block_idx in range(block_num):
            chunk_length = min(self.block_size, total_frame_num - cur_hop)
            chunk = xs_pad.narrow(1, cur_hop, chunk_length)
            if self.init_average:
                addin[:, block_idx, :] = chunk.mean(1)
            else:
                addin[:, block_idx, :] = chunk.max(1)[0]
            cur_hop += self.hop_size

        return addin

    def _prepare_block_input_train(
        self, xs_pad: torch.Tensor, addin: torch.Tensor, block_num: int, masks: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare block-wise input for training."""
        bsize = xs_pad.size(0)
        total_frame_num = xs_pad.size(1)

        # Create mask
        mask_online = xs_pad.new_zeros(bsize, block_num, self.block_size + 2, self.block_size + 2)
        mask_online.narrow(2, 1, self.block_size + 1).narrow(3, 0, self.block_size + 1).fill_(1)

        # Create chunk container
        xs_chunk = xs_pad.new_zeros(bsize, block_num, self.block_size + 2, xs_pad.size(-1))

        # Fill chunks
        left_idx = 0
        for block_idx in range(block_num):
            chunk_length = min(self.block_size, total_frame_num - left_idx)
            xs_chunk[:, block_idx, 1:chunk_length + 1] = xs_pad.narrow(1, left_idx, chunk_length)
            left_idx += self.hop_size

        # Fill context vectors
        xs_chunk[:, 0, 0] = addin[:, 0]
        xs_chunk[:, 1:, 0] = addin[:, 0:block_num - 1]
        xs_chunk[:, :, self.block_size + 1] = addin

        return xs_chunk, mask_online

    def _extract_output_from_blocks_train(
        self, ys_chunk: torch.Tensor, ys_pad: torch.Tensor, block_num: int, total_frame_num: int
    ) -> torch.Tensor:
        """Extract output from block-wise processing (training)."""
        offset = self.block_size - self.look_ahead - self.hop_size + 1
        cur_hop = self.block_size - self.look_ahead

        # First block
        ys_pad[:, 0:cur_hop] = ys_chunk[:, 0, 1:cur_hop + 1]

        # Middle blocks
        left_idx = self.hop_size
        for block_idx in range(1, block_num - 1):
            ys_pad[:, cur_hop:cur_hop + self.hop_size] = ys_chunk[:, block_idx, offset:offset + self.hop_size]
            cur_hop += self.hop_size
            left_idx += self.hop_size

        # Last block
        last_size = total_frame_num - left_idx
        if block_num > 1:
            ys_pad[:, cur_hop:total_frame_num] = ys_chunk[:, block_num - 1, offset:last_size + 1]

        return ys_pad

    def _extract_output_from_blocks_infer(
        self, ys_chunk: torch.Tensor, y_length: int, offset: int, block_num: int, n_processed_blocks: int, is_final: bool
    ) -> torch.Tensor:
        """Extract output from block-wise processing (inference)."""
        bsize = ys_chunk.size(0)
        ys_pad = ys_chunk.new_zeros(bsize, y_length, ys_chunk.size(-1))

        if n_processed_blocks == 0:
            ys_pad[:, 0:offset] = ys_chunk[:, 0, 0:offset]

        for i in range(block_num):
            cur_hop = i * self.hop_size
            if n_processed_blocks == 0:
                cur_hop += offset

            if i == block_num - 1 and is_final:
                chunk_length = min(self.block_size - offset, ys_pad.size(1) - cur_hop)
            else:
                chunk_length = self.hop_size

            ys_pad[:, cur_hop:cur_hop + chunk_length] = ys_chunk[:, i, offset:offset + chunk_length]

        return ys_pad

    def _create_block_mask(self, xs_pad: torch.Tensor, block_num: int) -> torch.Tensor:
        """Create block-wise attention mask."""
        mask_online = xs_pad.new_zeros(xs_pad.size(0), block_num, self.block_size + 2, self.block_size + 2)
        mask_online.narrow(2, 1, self.block_size + 1).narrow(3, 0, self.block_size + 1).fill_(1)
        return mask_online

    def _apply_encoder_layers(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], infer_mode: bool, is_short: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply encoder layers without context (simple mode)."""
        for layer in self.encoders:
            x, mask, _, _, _, _, _ = layer(x, mask, infer_mode=infer_mode, is_short_segment=is_short)
        return x, mask

    def _apply_encoder_layers_with_ctx(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], infer_mode: bool, past_ctx: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor], Optional[torch.Tensor], bool, int]:
        """Apply encoder layers with context."""
        next_ctx = None
        layer_idx = 0
        for layer in self.encoders:
            x, mask, infer_mode, past_ctx, next_ctx, is_short, layer_idx = layer(
                x, mask, infer_mode=infer_mode, past_ctx=past_ctx, next_ctx=next_ctx, layer_idx=layer_idx
            )
        return x, mask, infer_mode, past_ctx, next_ctx, is_short, layer_idx
