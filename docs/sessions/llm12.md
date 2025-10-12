# LLM Session 12: Default Decoder Switch & Code Quality Improvements

**Date**: 2025-10-12
**Branch**: `feat/decoder-rewrite-bsbs`

## Session Overview

This session focused on three main areas:
1. Switching the default decoder from native to ESPnet (since ESPnet has 100% parity)
2. Code review to eliminate hardcoded values and improve configurability
3. Implementing individual scorer score tracking in hypotheses

## 1. Default Decoder Changed to ESPnet

### Changes Made

**File**: `speechcatcher/speechcatcher.py`

Changed the `--decoder` argument default from `'native'` to `'espnet'`:

```python
# Before:
parser.add_argument('--decoder', dest='decoder', choices=['native', 'espnet'],
                    default='native', ...)

# After:
parser.add_argument('--decoder', dest='decoder', choices=['native', 'espnet'],
                    default='espnet', ...)
```

**Rationale**: The ESPnet decoder achieves 100% parity with the reference implementation, while the native decoder is at 92% parity. The native decoder remains available via `--decoder native` for experimentation.

**Help Text**: Simplified to remove percentage mentions per user request:
- `"espnet" (default) or "native" (experimental)`

**Commit**: `e46d8cc - Change default decoder to ESPnet (from native)`

## 2. Code Review: Fixed Hardcoded Values

### Problem Identified

The native decoder had hardcoded token IDs that only worked with vocab_size=1024:
- `sos_id = 1023` (hardcoded)
- `eos_id = 1023` (hardcoded)

This would break with models using different vocabulary sizes (512, 2048, etc.).

### Solution: Dynamic Token ID Calculation

**Discovery**: Analyzed ESPnet token list structure in `speech2text_streaming.py:118-122`:

```python
# ESPnet token list structure: [<blank>, SP[0], SP[3..N], <sos/eos>]
vocab_size = self.tokenizer.GetPieceSize()
self.token_list = (
    ["<blank>", self.tokenizer.IdToPiece(0)] +
    [self.tokenizer.IdToPiece(i) for i in range(3, vocab_size)] +
    ["<sos/eos>"]
)
```

**Key Insight**: `<sos/eos>` is always at position `vocab_size - 1`

### Implementation

**File**: `speechcatcher/beam_search/beam_search.py`

Modified `create_beam_search()` factory function to calculate token IDs dynamically:

```python
def create_beam_search(
    model: nn.Module,
    beam_size: int = 10,
    ctc_weight: float = 0.3,
    decoder_weight: float = 0.7,
    device: str = "cpu",
    use_bbd: bool = True,
    bbd_conservative: bool = True,
    sos_id: Optional[int] = None,  # Changed from int = 1023
    eos_id: Optional[int] = None,  # Changed from int = 1023
) -> BlockwiseSynchronousBeamSearch:
    """Create BSBS beam search from model.

    Args:
        ...
        sos_id: Start-of-sequence token ID (default: vocab_size - 1)
        eos_id: End-of-sequence token ID (default: vocab_size - 1)
    """
    # Calculate token IDs from vocab_size if not provided
    # ESPnet token list structure: [<blank>, SP[0], SP[3..N], <sos/eos>]
    # So <sos/eos> is always at position vocab_size - 1
    if sos_id is None:
        sos_id = model.vocab_size - 1
    if eos_id is None:
        eos_id = model.vocab_size - 1

    # ... rest of function
```

### Vocab Size Inference

Confirmed that `vocab_size` is inferred from the model checkpoint weights (not config):

**File**: `speechcatcher/speech2text_streaming.py:198-206`

```python
# Infer vocab size from checkpoint
vocab_size = None
if "decoder.embed.0.weight" in state_dict:
    vocab_size = state_dict["decoder.embed.0.weight"].shape[0]
elif "decoder.output_layer.weight" in state_dict:
    vocab_size = state_dict["decoder.output_layer.weight"].shape[0]
```

This is more reliable than reading from config files, as the tensor dimensions are the ground truth.

### Testing

Verified the solution works with multiple vocab sizes:

```python
for vocab_size in [512, 1024, 2048]:
    model = ESPnetASRModel(vocab_size=vocab_size)
    beam_search = create_beam_search(model, device='cpu')
    assert beam_search.sos_id == vocab_size - 1  # ✓
    assert beam_search.eos_id == vocab_size - 1  # ✓
```

**Commit**: `00b73e6 - Fix hardcoded token IDs - calculate from vocab_size`

## 3. Individual Scorer Score Tracking

### Problem

The `Hypothesis.scores` dictionary wasn't being populated with individual scorer contributions. Line 221 in `beam_search.py` had a TODO comment:

```python
scores=hyp.scores.copy(),  # TODO: Update individual scores
```

This made it impossible to analyze how decoder vs CTC components contributed to final scores.

### Solution

Modified the beam search scoring pipeline to track individual scorer scores:

#### Step 1: Update `batch_score_hypotheses()` Return Type

**File**: `speechcatcher/beam_search/beam_search.py:71-185`

```python
def batch_score_hypotheses(
    self,
    hypotheses: List[Hypothesis],
    encoder_out: torch.Tensor,
    pre_beam_size: int = 40,
) -> Tuple[torch.Tensor, Dict[str, List], Dict[str, torch.Tensor]]:
    """Score all hypotheses for next token prediction with two-pass strategy.

    Returns:
        Tuple of:
            - Combined scores (batch, vocab_size)
            - Dict of new states per scorer {scorer_name: [state_0, state_1, ...]}
            - Dict of individual scorer scores {scorer_name: (batch, vocab_size)}  # NEW!
    """
    # ... existing scoring logic ...

    # Collect individual scorer scores (unweighted)
    individual_scores = {}

    for scorer_name, scorer in self.scorers.items():
        # ... scoring ...
        individual_scores[scorer_name] = scores  # Store unweighted scores

    return combined_scores, all_new_states, individual_scores
```

#### Step 2: Update Hypothesis Creation

Updated hypothesis creation in both `BeamSearch.search()` and `BlockwiseSynchronousBeamSearch._decode_one_block()`:

```python
# Score hypotheses
scores, new_states_dict, individual_scores = self.beam_search.batch_score_hypotheses(
    hypotheses, encoder_out
)

# Expand beam
for i, hyp in enumerate(hypotheses):
    top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

    for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
        # ... state selection ...

        # Update individual scores dict with incremental scores for this token
        new_scores = hyp.scores.copy()
        for scorer_name, scorer_scores in individual_scores.items():
            # Add incremental score for this specific token
            token_score = scorer_scores[i, token].item()
            new_scores[scorer_name] = new_scores.get(scorer_name, 0.0) + token_score

        new_hyp = Hypothesis(
            yseq=append_token(hyp.yseq, token),
            score=hyp.score + score,
            scores=new_scores,  # Updated with individual scorer contributions
            states=new_states_for_hyp,
            xpos=append_position(hyp.xpos, current_encoder_pos),
        )
```

### Testing

Created unit test to verify individual score tracking:

```python
# Test output:
Initial hypothesis scores: {}

Individual scores returned:
  decoder: shape=torch.Size([1, 10])
  ctc: shape=torch.Size([1, 10])

After adding token 4:
  Combined score: 0.6782
  Individual scores dict: {'decoder': 1.8850626945495605, 'ctc': -2.1376638412475586}

✓ Individual scores successfully tracked!
```

**Commit**: `d100635 - Track individual scorer scores in hypotheses`

## 4. Documentation Quality Review

Reviewed all model code for TODOs and documentation quality:

### TODOs Found

1. **Line 144** in `asr_inference_streaming.py` - ESPnet compatibility file (not our code)
2. **Line 458** in `asr_inference_streaming.py` - ESPnet compatibility file (not our code)
3. **Line 221** in `beam_search.py` - **RESOLVED** (this session)

### Documentation Assessment

**Excellent coverage** throughout the codebase:

- **BlockwiseSynchronousBeamSearch**: Comprehensive class docstrings with all parameters
- **BBD mechanism**: Clear explanation of repetition detection and rollback logic
- **Rewinding mechanism**: Well-commented with ESPnet reference lines
- **Global state management**: Critical persistence logic clearly documented
- **Token ID calculation**: Formula explained with ESPnet token structure
- **CTC scoring**: Full forward algorithm documented with paper references
- **Hypothesis & BeamState**: All dataclasses have clear field descriptions
- **Factory functions**: Well-documented with parameter explanations

All critical sections are marked with `CRITICAL:` comments for easy identification.

## Summary of Changes

### Commits Made

1. **e46d8cc** - Change default decoder to ESPnet (from native)
2. **00b73e6** - Fix hardcoded token IDs - calculate from vocab_size
3. **d100635** - Track individual scorer scores in hypotheses

### Files Modified

- `speechcatcher/speechcatcher.py` - Default decoder argument
- `speechcatcher/beam_search/beam_search.py` - Token ID calculation, individual scores

### Key Improvements

1. **Better Defaults**: ESPnet decoder (100% parity) is now default
2. **Dynamic Configuration**: Token IDs calculated from model vocab_size, not hardcoded
3. **Better Observability**: Hypotheses track individual scorer contributions
4. **Maintainability**: Code works with any vocab size (512, 1024, 2048, etc.)

### Testing

- ✅ ESPnet decoder works as default
- ✅ Native decoder selectable via `--decoder native`
- ✅ Token IDs calculated correctly for vocab_size ∈ {512, 1024, 2048}
- ✅ Individual scorer scores tracked and accumulated
- ✅ All existing functionality preserved

## Architecture Notes

### Token ID Formula

```
ESPnet Token List: [<blank>, SP[0], SP[3..vocab_size-1], <sos/eos>]
                     ↑         ↑      ↑                   ↑
                     0         1      2..vocab_size-2     vocab_size-1

Therefore:
  blank_id = 0 (always)
  sos_id = eos_id = vocab_size - 1
```

### Score Tracking Flow

```
Hypothesis.scores = {}  (initially empty)
                    ↓
batch_score_hypotheses() returns individual_scores dict
                    ↓
For each token expansion:
  new_scores[scorer_name] += scorer_scores[hyp_idx, token_id]
                    ↓
Hypothesis.scores = {'decoder': X.XX, 'ctc': Y.YY}
```

## Next Steps (Future Work)

Potential improvements identified but not implemented:

1. Review other hardcoded architectural values:
   - `block_size: int = 40`
   - `hop_size: int = 16`
   - `look_ahead: int = 16`
   - `reliability_threshold: float = 0.8`

   Determine if these should be sourced from config or are algorithm parameters that should remain fixed.

2. Expose individual scorer scores through the API (currently internal to beam search)

3. Add score visualization tools for debugging decoder vs CTC contributions
