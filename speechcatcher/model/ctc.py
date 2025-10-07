"""CTC (Connectionist Temporal Classification) module for ASR."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTC(nn.Module):
    """CTC module for ASR.

    This module implements CTC loss computation and provides the CTC output layer.
    CTC enables training without frame-level alignments by marginalizing over all
    possible alignments.

    Args:
        vocab_size: Number of output classes (including blank)
        encoder_output_size: Dimension of encoder output
        dropout_rate: Dropout rate before output projection
        reduce: Whether to reduce the loss (default: True)

    Shape:
        - Encoder output: (batch, time, encoder_output_size)
        - Targets: (batch, target_len) - token IDs
        - Output logits: (batch, time, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout_rate)
        self.ctc_lo = nn.Linear(encoder_output_size, vocab_size)
        self.reduce = reduce

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_pad: Optional[torch.Tensor] = None,
        ys_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional CTC loss computation.

        Args:
            hs_pad: Encoder output (batch, time, encoder_output_size)
            hlens: Encoder output lengths (batch,)
            ys_pad: Target labels (batch, target_len), optional for inference
            ys_lens: Target lengths (batch,), optional for inference

        Returns:
            Tuple of:
                - logits: CTC output logits (batch, time, vocab_size)
                - loss: CTC loss (scalar) if targets provided, else None
        """
        # Apply dropout and projection
        hs_pad = self.dropout(hs_pad)
        logits = self.ctc_lo(hs_pad)  # (batch, time, vocab_size)

        # Compute loss if targets provided
        if ys_pad is not None and ys_lens is not None:
            loss = self.loss(logits, hlens, ys_pad, ys_lens)
        else:
            loss = None

        return logits, loss

    def loss(
        self,
        logits: torch.Tensor,
        logit_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            logits: CTC logits (batch, time, vocab_size)
            logit_lens: Logit lengths (batch,)
            targets: Target labels (batch, target_len)
            target_lens: Target lengths (batch,)

        Returns:
            CTC loss (scalar if reduce=True, else (batch,))
        """
        # Log softmax over vocabulary dimension
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose for CTC loss: (time, batch, vocab_size)
        log_probs = log_probs.transpose(0, 1)

        # CTC expects int32 lengths
        input_lengths = logit_lens.int()
        target_lengths = target_lens.int()

        # Compute CTC loss
        # blank=0 (assuming blank is the first token in vocabulary)
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="mean" if self.reduce else "none",
            zero_infinity=True,
        )

        return loss

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """Apply log softmax to encoder output for CTC decoding.

        Args:
            hs_pad: Encoder output (batch, time, encoder_output_size)

        Returns:
            Log probabilities (batch, time, vocab_size)
        """
        logits = self.ctc_lo(hs_pad)
        return F.log_softmax(logits, dim=-1)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """Greedy CTC decoding (argmax).

        Args:
            hs_pad: Encoder output (batch, time, encoder_output_size)

        Returns:
            Predicted labels (batch, time)
        """
        logits = self.ctc_lo(hs_pad)
        return torch.argmax(logits, dim=-1)


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    blank: int = 0,
) -> list:
    """Greedy CTC decoding with blank removal and deduplication.

    Args:
        log_probs: Log probabilities (batch, time, vocab_size)
        lengths: Sequence lengths (batch,)
        blank: Blank token ID (default: 0)

    Returns:
        List of decoded sequences (batch,) where each sequence is a list of token IDs
    """
    # Argmax decoding
    predictions = torch.argmax(log_probs, dim=-1)  # (batch, time)

    batch_size = predictions.size(0)
    results = []

    for b in range(batch_size):
        # Get valid frames for this utterance
        length = lengths[b].item()
        pred = predictions[b, :length].tolist()

        # Remove consecutive duplicates and blanks
        decoded = []
        prev_token = None
        for token in pred:
            if token != blank and token != prev_token:
                decoded.append(token)
            prev_token = token

        results.append(decoded)

    return results


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    beam_size: int = 10,
    blank: int = 0,
) -> list:
    """CTC prefix beam search decoding.

    This is a simplified version. For full BSBS integration, use the CTCScorer
    class with the beam search module.

    Args:
        log_probs: Log probabilities (batch, time, vocab_size)
        lengths: Sequence lengths (batch,)
        beam_size: Beam size
        blank: Blank token ID (default: 0)

    Returns:
        List of best decoded sequences (batch,)
    """
    batch_size = log_probs.size(0)
    vocab_size = log_probs.size(2)

    results = []

    for b in range(batch_size):
        length = lengths[b].item()
        log_prob = log_probs[b, :length, :]  # (time, vocab_size)

        # Initialize beam with empty prefix
        # beam format: {prefix_tuple: (score, has_ended_with_blank)}
        beam = {(): (0.0, True)}  # (prefix): (log_prob, ended_with_blank)

        for t in range(length):
            new_beam = {}

            for prefix, (score, _) in beam.items():
                # Blank extension
                blank_score = score + log_prob[t, blank].item()
                new_beam[prefix] = (
                    max(new_beam.get(prefix, (float("-inf"), False))[0], blank_score),
                    True,
                )

                # Non-blank extensions
                for c in range(vocab_size):
                    if c == blank:
                        continue

                    new_prefix = prefix + (c,)
                    new_score = score + log_prob[t, c].item()

                    # Handle repeated characters
                    if len(prefix) > 0 and prefix[-1] == c:
                        # Can only extend with same character if previous ended with blank
                        # For simplicity, we'll allow it here
                        pass

                    if new_prefix not in new_beam:
                        new_beam[new_prefix] = (new_score, False)
                    else:
                        new_beam[new_prefix] = (
                            max(new_beam[new_prefix][0], new_score),
                            False,
                        )

            # Prune beam
            beam = dict(
                sorted(new_beam.items(), key=lambda x: x[1][0], reverse=True)[:beam_size]
            )

        # Get best hypothesis
        best_prefix = max(beam.items(), key=lambda x: x[1][0])[0]
        results.append(list(best_prefix))

    return results
