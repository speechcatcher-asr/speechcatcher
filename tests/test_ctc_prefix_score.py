"""Tests for CTC prefix scoring."""

import pytest
import numpy as np
import torch

from speechcatcher.beam_search.ctc_prefix_score import CTCPrefixScore, CTCPrefixScoreTH, log_add


class TestCTCPrefixScore:
    """Test CTC prefix scoring algorithm."""

    def test_log_add(self):
        """Test numerically stable log addition."""
        # Test normal case
        a, b = 1.0, 2.0
        result = log_add(a, b)
        expected = np.log(np.exp(a) + np.exp(b))
        assert np.isclose(result, expected)

        # Test with -inf
        assert log_add(-np.inf, 1.0) == 1.0
        assert log_add(1.0, -np.inf) == 1.0

        # Test with large difference (numerical stability)
        a, b = 100.0, 1.0
        result = log_add(a, b)
        # Should be approximately a since exp(1-100) is tiny
        assert np.isclose(result, a, rtol=1e-5)

    def test_initial_state(self):
        """Test CTC initial state for empty prefix."""
        # Create dummy CTC log probs
        T, vocab_size = 10, 100
        x = np.random.randn(T, vocab_size).astype(np.float32)
        x = x - np.log(np.sum(np.exp(x), axis=1, keepdims=True))  # log softmax

        scorer = CTCPrefixScore(x, blank_id=0, eos_id=2)
        r, log_psi = scorer.initial_state()

        # Check shapes
        assert r.shape == (vocab_size,)
        assert log_psi.shape == (vocab_size,)

        # Empty prefix should be in blank state
        assert log_psi[0] == 0.0  # Blank has prob 1.0 (log prob 0.0)
        assert np.all(r == -np.inf)  # No non-blank endings for empty prefix

    def test_score_single_token(self):
        """Test scoring a single token extension."""
        # Simple test case: 3 time steps, 4 tokens (blank=0, a=1, b=2, eos=3)
        T, V = 3, 4
        x = np.array([
            [-0.1, -2.0, -2.0, -3.0],  # t=0: mostly blank
            [-2.0, -0.1, -2.0, -3.0],  # t=1: mostly 'a'
            [-2.0, -2.0, -0.1, -3.0],  # t=2: mostly 'b'
        ], dtype=np.float32)

        # Normalize to log probs
        x = x - np.log(np.sum(np.exp(x), axis=1, keepdims=True))

        scorer = CTCPrefixScore(x, blank_id=0, eos_id=3)

        # Score extending empty prefix with 'a' (token 1)
        y = torch.tensor([])  # Empty prefix
        cs = torch.tensor([0, 1, 2, 3])  # All possible next tokens
        state = None

        scores, new_state = scorer(y, cs, state)

        # Check we get scores for all candidates
        assert scores.shape == (4,)

        # Scores should be log probabilities (negative)
        assert torch.all(scores <= 0.0)

        # State should be returned
        assert new_state is not None
        r, log_psi = new_state
        assert r.shape == (V,)
        assert log_psi.shape == (V,)

    def test_score_prefix_extension(self):
        """Test extending an existing prefix."""
        T, V = 5, 4
        # Create simple uniform-ish log probs
        x = np.full((T, V), -np.log(V), dtype=np.float32)

        scorer = CTCPrefixScore(x, blank_id=0, eos_id=3)

        # Start with empty prefix
        y = torch.tensor([])
        cs = torch.tensor([1])  # Extend with token 1
        state = None

        scores1, state1 = scorer(y, cs, state)

        # Now extend the prefix [1] with another token
        y2 = torch.tensor([1])
        cs2 = torch.tensor([1, 2])  # Same token or different token

        scores2, state2 = scorer(y2, cs2, state1)

        # Should get scores for both candidates
        assert scores2.shape == (2,)

        # Scores should not be NaN
        assert not torch.isnan(scores2).any()

        # At least one score should be finite (not impossible path)
        assert not torch.isinf(scores2).all()

    def test_blank_scoring(self):
        """Test that blank tokens are handled correctly."""
        T, V = 3, 4
        x = np.full((T, V), -np.log(V), dtype=np.float32)

        scorer = CTCPrefixScore(x, blank_id=0, eos_id=3)

        # Extend with blank
        y = torch.tensor([])
        cs = torch.tensor([0])  # Blank
        state = None

        scores, new_state = scorer(y, cs, state)

        # Should get a valid score
        assert scores.shape == (1,)
        assert not torch.isnan(scores[0])
        assert not torch.isinf(scores[0])


class TestCTCPrefixScoreTH:
    """Test torch-based batched CTC prefix scorer."""

    def test_initial_state(self):
        """Test initial state creation."""
        T, V = 10, 100
        x = torch.randn(1, T, V)
        x = torch.log_softmax(x, dim=-1)
        xlens = torch.tensor([T])

        scorer = CTCPrefixScoreTH(x, xlens, blank_id=0, eos_id=2)
        r, log_psi = scorer.initial_state()

        # Check shapes
        assert r.shape == (V,)
        assert log_psi.shape == (V,)

        # Empty prefix should be in blank state
        assert log_psi[0] == 0.0
        assert torch.all(r == -float('inf'))

    def test_batch_scoring(self):
        """Test batch scoring of multiple prefixes."""
        T, V = 5, 10
        batch_size = 3

        x = torch.randn(1, T, V)
        x = torch.log_softmax(x, dim=-1)
        xlens = torch.tensor([T])

        scorer = CTCPrefixScoreTH(x, xlens, blank_id=0, eos_id=2)

        # Create batch of prefixes
        y = torch.tensor([
            [1, 2, 0],  # Prefix: [1, 2]
            [1, 3, 0],  # Prefix: [1, 3]
            [2, 0, 0],  # Prefix: [2]
        ])

        scores, state = scorer(y, None, cs=None)

        # Should get scores for all tokens in batch
        assert scores.shape == (batch_size, V)

        # All scores should be valid (not NaN)
        # Note: -inf is expected for impossible CTC paths
        assert not torch.isnan(scores).any()

        # State should be returned
        assert state is not None

    def test_incremental_scoring(self):
        """Test that incremental scoring produces consistent results."""
        T, V = 5, 10
        x = torch.randn(1, T, V)
        x = torch.log_softmax(x, dim=-1)
        xlens = torch.tensor([T])

        scorer = CTCPrefixScoreTH(x, xlens, blank_id=0, eos_id=2)

        # Score empty prefix
        y1 = torch.tensor([[0]])  # Just padding
        scores1, state1 = scorer(y1, None)

        # Score with one token
        y2 = torch.tensor([[1]])
        scores2, state2 = scorer(y2, state1)

        # Scores should change from first to second call
        assert not torch.allclose(scores1[0], scores2[0])


class TestCTCIntegration:
    """Test CTC scorer integration with beam search."""

    @pytest.fixture
    def simple_ctc_model(self):
        """Create a simple CTC model for testing."""
        class SimpleCTC(torch.nn.Module):
            def __init__(self, input_size, vocab_size):
                super().__init__()
                self.ctc_lo = torch.nn.Linear(input_size, vocab_size)

            def forward(self, x):
                return self.ctc_lo(x)

        return SimpleCTC(input_size=256, vocab_size=100)

    def test_ctc_scorer_creation(self, simple_ctc_model):
        """Test creating CTC prefix scorer."""
        from speechcatcher.beam_search.scorers import CTCPrefixScorer

        scorer = CTCPrefixScorer(simple_ctc_model, blank_id=0, eos_id=2)

        assert scorer is not None
        assert scorer.blank_id == 0
        assert scorer.eos_id == 2

    def test_ctc_scorer_batch_score(self, simple_ctc_model):
        """Test batch scoring with CTC scorer."""
        from speechcatcher.beam_search.scorers import CTCPrefixScorer

        scorer = CTCPrefixScorer(simple_ctc_model, blank_id=0, eos_id=2)

        # Create dummy encoder output
        batch_size, enc_len, dim = 2, 10, 256
        xs = torch.randn(batch_size, enc_len, dim)

        # Create dummy prefix sequences
        yseqs = torch.tensor([
            [1, 2, 3],
            [1, 4, 0],
        ])

        # Initial states
        states = [None, None]

        # Score
        scores, new_states = scorer.batch_score(yseqs, states, xs)

        # Check outputs
        assert scores.shape == (batch_size, 100)  # vocab_size = 100
        assert len(new_states) == batch_size
        assert not torch.isnan(scores).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
