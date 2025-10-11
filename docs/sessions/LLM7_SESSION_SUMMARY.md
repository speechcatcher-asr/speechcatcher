# Session LLM7 Summary: Fixed Encoder State Preservation!

## Major Breakthrough! 🎉

**Found and fixed the critical encoder state preservation bug!**

### The Problem

Encoder states were NOT being preserved between chunks, causing the encoder to restart from scratch every time:

```
Chunk 1: prev_encoder_states=None → Encoder: torch.Size([1, 0, 256]) (empty)
Chunk 2: prev_encoder_states=None → Encoder: torch.Size([1, 0, 256]) (empty)
Chunk 3: prev_encoder_states=None → Encoder: torch.Size([1, 0, 256]) (empty)
... (loop forever)
```

### The Root Cause

In `speechcatcher/beam_search/beam_search.py:507-511`, encoder_states were ONLY set when `ret is not None`:

```python
# OLD CODE (BROKEN):
if ret is not None:
    ret.encoder_states = encoder_states

return ret if ret is not None else prev_state
```

But `ret` was `None` because encoder produced empty output, creating a circular dependency:
1. `encoder_states=None` → encoder produces empty output
2. Empty output → `encoder_buffer` stays None
3. `encoder_buffer=None` → `ret=None` (no blocks processed)
4. `ret=None` → encoder_states **NOT updated**
5. Next chunk: encoder_states=None again! ♻️

### The Fix

**ALWAYS** update encoder_states, even when `ret is None`:

```python
# NEW CODE (FIXED):
if ret is not None:
    ret.encoder_states = encoder_states
    return ret
else:
    # No blocks processed yet, but MUST preserve encoder states
    prev_state.encoder_states = encoder_states
    return prev_state
```

### Results After Fix ✅

```
Chunk 1: prev_encoder_states=None     → Encoder: torch.Size([1, 0, 256])
Chunk 2: prev_encoder_states=present  → Encoder: torch.Size([1, 0, 256])
Chunk 3: prev_encoder_states=present  → Encoder: torch.Size([1, 0, 256])
Chunk 4: prev_encoder_states=present  → Encoder: torch.Size([1, 24, 256]) ✅
Chunk 5: prev_encoder_states=present  → Encoder: torch.Size([1, 16, 256]) ✅
```

Encoder buffer accumulation:
```
Chunk 4: Initialized encoder buffer: torch.Size([1, 24, 256])
Chunk 5: Accumulated encoder buffer: torch.Size([1, 40, 256])
Chunk 7: Accumulated encoder buffer: torch.Size([1, 56, 256])
```

This **EXACTLY** matches ESPnet's behavior!

## Additional Fix: CTC Index Out of Bounds

Fixed crash in `ctc_prefix_score_full.py` when final chunk has fewer frames:

```python
# OLD: Could access r[88] when r.shape[0] = 88
start = max(r_prev.shape[0], 1)

# NEW: Clamp to current input length
start = min(max(r_prev.shape[0], 1), self.input_length)
```

## Current Status

### ✅ What Works

1. **Encoder state preservation** - States properly carried between chunks
2. **Encoder buffer accumulation** - 24 → 40 → 56 → 72 → 88 frames
3. **Block processing** - Blocks 0, 1, 2, 3, 4 processed correctly
4. **BBD (Block Boundary Detection)** - Correctly detects repetition
5. **CTC streaming** - No crashes with varying input lengths

### ⚠️  Current Issue

**Decoder predicting wrong token (1023 instead of German)**

Comparison at chunk 5 with 40 encoder frames:
- **ESPnet**: `'liebe'` (correct German!) ✅
- **Ours**: `''` (empty, BBD rolled back) ❌

Our decoder predicts token 1023 (Arabic 'م') repeatedly, which triggers BBD rollback to output_index=0, resulting in no text output.

ESPnet's decoder predicts the correct German tokens, so BBD doesn't trigger.

### Why This Is Puzzling

- ALL 369 decoder weights verified to match ESPnet exactly ✅
- Encoder outputs match ESPnet exactly ✅
- Frontend matches ESPnet exactly ✅
- BBD logic is correct ✅

**The decoder gets the SAME encoder output as ESPnet but produces different token predictions!**

## Files Modified

### speechcatcher/beam_search/beam_search.py
**Lines 507-520**: Fixed encoder state preservation

```python
# Update encoder states for next chunk
# CRITICAL: Always update encoder_states, even if ret is None!
# The encoder needs states to be passed for streaming to work.
if ret is not None:
    ret.encoder_states = encoder_states
    return ret
else:
    # No blocks processed yet, but MUST preserve encoder states
    prev_state.encoder_states = encoder_states
    return prev_state
```

**Lines 423, 437, 445**: Added debug logging for encoder calls

### speechcatcher/beam_search/ctc_prefix_score_full.py
**Lines 351-353**: Fixed CTC index out of bounds

```python
# Clamp to current input length to avoid out-of-bounds
start = min(max(r_prev.shape[0], 1), self.input_length)
r_prev_new[0:start] = r_prev[0:start]
```

## Test Files Created/Modified

- `test_state_preservation.py` - Traces encoder states through pipeline
- `test_full_transcription_debug.py` - Shows hypotheses and buffer state
- `test_decoder_scores_debug.py` - Attempts to debug decoder scoring (incomplete)

## Next Steps

### Immediate Investigation Needed

**Why does our decoder predict token 1023 instead of correct German tokens?**

Possible causes:
1. Decoder scoring logic differs from ESPnet
2. CTC + decoder score combination differs
3. Hypothesis state initialization differs
4. Decoder input preprocessing differs
5. Some subtle weight loading issue not caught by verification

### Debugging Approach

1. Compare ESPnet's beam_search.batch_score_hypotheses() with ours step-by-step
2. Log decoder logits for first prediction to see if token 1023 has highest score
3. Compare CTC logits between ESPnet and ours
4. Check if decoder embedding layer is being used correctly
5. Verify decoder forward pass is identical to ESPnet

### Expected Outcome

Once decoder predictions are fixed, we should see:
```
Chunk 5: ESPnet='liebe', Ours='liebe' ✅
Chunk 11 (final): ESPnet='Liebe Mitglieder unserer Universität Hamburg.',
                  Ours='Liebe Mitglieder unserer Universität Hamburg.' ✅
```

## Key Learnings

1. **State preservation is CRITICAL** - Without it, streaming encoders restart from scratch
2. **Circular dependencies are sneaky** - The bug created a self-perpetuating cycle
3. **Debug logging is essential** - Without `prev_encoder_states=present/None` logs, would never have found this
4. **Test in isolation vs full pipeline** - Encoder worked in `test_encoder_streaming_behavior.py` but not in full pipeline
5. **Index clamping prevents crashes** - Always check array bounds when extending state

## Commits Needed

1. Fix encoder state preservation
2. Fix CTC index out of bounds
3. Add debug tests

## Session Metrics

- **Bug severity**: Critical (blocking all streaming inference)
- **Time to identify**: ~10 tests, ~6 iterations
- **Lines changed**: ~15 lines
- **Impact**: Encoder now works correctly for streaming! ✅

---

**Next session**: Investigate decoder token 1023 prediction issue to achieve correct German output.
