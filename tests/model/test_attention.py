"""Unit tests for attention mechanisms."""

import pytest
import torch

from speechcatcher.model.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)


class TestMultiHeadedAttention:
    """Tests for MultiHeadedAttention."""

    def test_forward_shape(self):
        """Test output shape with self-attention."""
        n_head, n_feat = 4, 256
        attn = MultiHeadedAttention(n_head, n_feat)

        batch, time = 2, 10
        x = torch.randn(batch, time, n_feat)

        # Self-attention
        output = attn(x, x, x)

        assert output.shape == (batch, time, n_feat)

    def test_forward_cross_attention_shape(self):
        """Test output shape with cross-attention."""
        n_head, n_feat = 8, 512
        attn = MultiHeadedAttention(n_head, n_feat)

        batch = 2
        query = torch.randn(batch, 5, n_feat)
        key = torch.randn(batch, 10, n_feat)
        value = torch.randn(batch, 10, n_feat)

        output = attn(query, key, value)

        # Output time dimension should match query
        assert output.shape == (batch, 5, n_feat)

    def test_mask_application(self):
        """Test that masking prevents attention to masked positions."""
        n_head, n_feat = 4, 64
        attn = MultiHeadedAttention(n_head, n_feat, dropout_rate=0.0)
        attn.eval()

        batch, time = 1, 5
        x = torch.randn(batch, time, n_feat)

        # Create mask: only attend to first 3 positions
        mask = torch.zeros(batch, 1, time)
        mask[:, :, :3] = 1  # Attend to first 3, mask last 2

        output = attn(x, x, x, mask=mask)

        # Check that attention weights are zero for masked positions
        # (attn.attn is stored in vanilla mode)
        assert attn.attn is not None
        # Last 2 positions should have zero attention
        assert torch.allclose(
            attn.attn[:, :, :, 3:], torch.zeros_like(attn.attn[:, :, :, 3:])
        )

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 (softmax property)."""
        n_head, n_feat = 2, 128
        attn = MultiHeadedAttention(n_head, n_feat, dropout_rate=0.0)
        attn.eval()

        x = torch.randn(1, 10, n_feat)
        _ = attn(x, x, x)

        # Check attention weights sum to 1 along last dimension
        assert attn.attn is not None
        attn_sum = attn.attn.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum))

    def test_cache_incremental_decoding(self):
        """Test incremental decoding with K/V cache."""
        n_head, n_feat = 4, 256
        attn = MultiHeadedAttention(n_head, n_feat)

        batch = 1

        # First step: no cache
        q1 = torch.randn(batch, 1, n_feat)
        k1 = torch.randn(batch, 1, n_feat)
        v1 = torch.randn(batch, 1, n_feat)

        out1, cache1 = attn.forward_with_cache(q1, k1, v1, cache=None)
        assert out1.shape == (batch, 1, n_feat)

        k_cache1, v_cache1 = cache1
        assert k_cache1.shape[2] == 1  # Time dimension = 1
        assert v_cache1.shape[2] == 1

        # Second step: use cache
        q2 = torch.randn(batch, 1, n_feat)
        k2 = torch.randn(batch, 1, n_feat)
        v2 = torch.randn(batch, 1, n_feat)

        out2, cache2 = attn.forward_with_cache(q2, k2, v2, cache=cache1)
        assert out2.shape == (batch, 1, n_feat)

        k_cache2, v_cache2 = cache2
        assert k_cache2.shape[2] == 2  # Time dimension = 2 (cached + new)
        assert v_cache2.shape[2] == 2

    def test_cache_equivalence_full_vs_incremental(self):
        """Test that incremental decoding matches full decoding."""
        n_head, n_feat = 4, 128
        attn = MultiHeadedAttention(n_head, n_feat, dropout_rate=0.0)
        attn.eval()

        batch, seq_len = 1, 5

        # Generate full sequence
        q_full = torch.randn(batch, seq_len, n_feat)
        k_full = torch.randn(batch, seq_len, n_feat)
        v_full = torch.randn(batch, seq_len, n_feat)

        # Full forward pass
        out_full = attn(q_full, k_full, v_full)

        # Incremental forward pass
        outputs_incremental = []
        cache = None
        for t in range(seq_len):
            q_t = q_full[:, t:t+1, :]
            k_t = k_full[:, t:t+1, :]
            v_t = v_full[:, t:t+1, :]

            out_t, cache = attn.forward_with_cache(q_t, k_t, v_t, cache=cache)
            outputs_incremental.append(out_t)

        out_incremental = torch.cat(outputs_incremental, dim=1)

        # Should match (approximately, due to numerical precision)
        assert torch.allclose(out_full, out_incremental, atol=1e-5)

    def test_deterministic_eval_mode(self):
        """Test that eval mode produces deterministic outputs."""
        n_head, n_feat = 4, 256
        attn = MultiHeadedAttention(n_head, n_feat)
        attn.eval()

        x = torch.randn(2, 10, n_feat)

        out1 = attn(x, x, x)
        out2 = attn(x, x, x)

        assert torch.allclose(out1, out2)


class TestRelPositionMultiHeadedAttention:
    """Tests for RelPositionMultiHeadedAttention."""

    def test_forward_shape(self):
        """Test output shape with relative positional encoding."""
        n_head, n_feat = 4, 256
        attn = RelPositionMultiHeadedAttention(n_head, n_feat)

        batch, time = 2, 10
        x = torch.randn(batch, time, n_feat)
        pos_emb = torch.randn(1, time, n_feat)

        output = attn(x, x, x, pos_emb)

        assert output.shape == (batch, time, n_feat)

    def test_relative_vs_absolute_difference(self):
        """Test that relative attention differs from absolute."""
        n_head, n_feat = 4, 128

        attn_abs = MultiHeadedAttention(n_head, n_feat, dropout_rate=0.0)
        attn_rel = RelPositionMultiHeadedAttention(n_head, n_feat, dropout_rate=0.0)

        attn_abs.eval()
        attn_rel.eval()

        batch, time = 1, 20
        x = torch.randn(batch, time, n_feat)
        pos_emb = torch.randn(1, time, n_feat)

        out_abs = attn_abs(x, x, x)
        out_rel = attn_rel(x, x, x, pos_emb)

        # Outputs should be different (relative position affects attention)
        assert not torch.allclose(out_abs, out_rel, atol=1e-3)

    def test_pos_emb_influence(self):
        """Test that different positional embeddings produce different outputs."""
        n_head, n_feat = 4, 128
        attn = RelPositionMultiHeadedAttention(n_head, n_feat, dropout_rate=0.0)
        attn.eval()

        batch, time = 1, 10
        x = torch.randn(batch, time, n_feat)
        pos_emb1 = torch.randn(1, time, n_feat)
        pos_emb2 = torch.randn(1, time, n_feat)

        out1 = attn(x, x, x, pos_emb1)
        out2 = attn(x, x, x, pos_emb2)

        # Different positional embeddings should produce different outputs
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_rel_shift_shape(self):
        """Test that rel_shift maintains tensor shape."""
        n_head, n_feat = 4, 64
        attn = RelPositionMultiHeadedAttention(n_head, n_feat)

        batch, time_q, time_k = 2, 10, 10
        x = torch.randn(batch, n_head, time_q, time_k)

        shifted = attn.rel_shift(x)

        assert shifted.shape == x.shape

    def test_mask_with_relative_position(self):
        """Test masking with relative positional attention."""
        n_head, n_feat = 4, 64
        attn = RelPositionMultiHeadedAttention(n_head, n_feat, dropout_rate=0.0)
        attn.eval()

        batch, time = 1, 5
        x = torch.randn(batch, time, n_feat)
        pos_emb = torch.randn(1, time, n_feat)

        # Create mask
        mask = torch.zeros(batch, 1, time)
        mask[:, :, :3] = 1

        output = attn(x, x, x, pos_emb, mask=mask)

        # Check attention weights
        assert attn.attn is not None
        assert torch.allclose(
            attn.attn[:, :, :, 3:], torch.zeros_like(attn.attn[:, :, :, 3:])
        )

    def test_learnable_bias_parameters(self):
        """Test that position biases are learnable parameters."""
        n_head, n_feat = 4, 256
        attn = RelPositionMultiHeadedAttention(n_head, n_feat)

        assert attn.pos_bias_u.requires_grad
        assert attn.pos_bias_v.requires_grad
        assert attn.pos_bias_u.shape == (n_head, n_feat // n_head)
        assert attn.pos_bias_v.shape == (n_head, n_feat // n_head)


class TestFlashAttentionFallback:
    """Tests for Flash Attention availability and fallback."""

    def test_flash_attention_import(self):
        """Test Flash Attention import status."""
        from speechcatcher.model.attention.multi_head_attention import (
            FLASH_ATTENTION_AVAILABLE,
        )

        # Just check the flag exists (may be True or False depending on environment)
        assert isinstance(FLASH_ATTENTION_AVAILABLE, bool)

    def test_vanilla_fallback_cpu(self):
        """Test that vanilla attention is used on CPU."""
        n_head, n_feat = 4, 128
        attn = MultiHeadedAttention(n_head, n_feat, use_flash_attn=True)

        x = torch.randn(2, 10, n_feat)  # CPU tensor

        # Should work and use vanilla (Flash Attention requires CUDA)
        output = attn(x, x, x)

        assert output.shape == x.shape
        # Attention weights should be stored (vanilla mode)
        assert attn.attn is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_attention(self):
        """Test attention on CUDA device."""
        n_head, n_feat = 4, 128
        attn = MultiHeadedAttention(n_head, n_feat).cuda()

        x = torch.randn(2, 10, n_feat).cuda()

        output = attn(x, x, x)

        assert output.shape == x.shape
        assert output.is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
