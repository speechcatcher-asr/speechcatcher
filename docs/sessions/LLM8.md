# Session LLM8: Token Vocabulary Fix & Decoder Now Working! 🎉

**Date**: 2025-10-11
**Goal**: Fix decoder token predictions to output correct German text
**Status**: ✅ **MAJOR BREAKTHROUGH** - Decoder now outputs correct German!

## Summary

This session resolved the token 1023 prediction mystery from LLM7. After systematic investigation, we discovered **ESPnet uses a modified token vocabulary**, not the raw SentencePiece tokens! This was THE root cause of incorrect decoding.

### Results

**Before (LLM7):**
```
Chunk 5: ESPnet='liebe', Ours='' (empty, BBD rollback due to token 1023)
```

**After (LLM8):**
```
Chunk 4: Ours='Li' ✅
Chunk 5: Ours='Liebe Mit' ✅
```

**We're now decoding correct German text!** 🎉

## Major Discovery: ESPnet Token Vocabulary Transformation

### The Mystery

ESPnet reported `Token IDs=[738]` with output `'liebe'`, but:
- SentencePiece `IdToPiece(738)` = `'trag'` ❌
- ESPnet `token_list[738]` = `'▁liebe'` ✅

**The vocabularies were different!**

### The Root Cause

ESPnet modifies the SentencePiece vocabulary during model setup:

1. **Inserts `<blank>` at position 0** (for CTC)
2. **Removes `<s>` and `</s>`** (SentencePiece IDs 1 and 2)
3. **Adds `<sos/eos>` at position 1023** (end of vocabulary)
4. **Shifts all other tokens** to fill the gap

**ESPnet vocabulary mapping:**
```
["<blank>", SP[0], SP[3], SP[4], ..., SP[1023], "<sos/eos>"]
  ↑         ↑       ↑      ↑             ↑         ↑
  0         1       2      3           1022      1023
```

**Raw SentencePiece (what we were using):**
```
[SP[0], SP[1]="<s>", SP[2]="</s>", SP[3], ..., SP[1023]]
  ↑         ↑              ↑          ↑             ↑
  0         1              2          3           1023
```

This explains why:
- Token 738 in ESPnet = `▁liebe` (correct)
- Token 738 in raw SP = `trag` (2 positions off due to removal of SP[1] and SP[2])

### The Fix

**speechcatcher/speech2text_streaming.py:103-112**

```python
# Build ESPnet-style token list
# ESPnet removes <s> (SP ID 1) and </s> (SP ID 2) tokens
# ESPnet vocabulary = ["<blank>", SP[0], SP[3..1023], "<sos/eos>"]
vocab_size = self.tokenizer.GetPieceSize()
self.token_list = (
    ["<blank>", self.tokenizer.IdToPiece(0)] +
    [self.tokenizer.IdToPiece(i) for i in range(3, vocab_size)] +
    ["<sos/eos>"]
)
```

**speechcatcher/speech2text_streaming.py:461-476**

```python
# Decode to text using ESPnet's token list
if self.token_list is not None:
    # Convert tensor token IDs to integers and look up in token_list
    tokens = [self.token_list[int(tid)] for tid in token_ids_filtered]
    text = "".join(tokens).replace("▁", " ").strip()
elif self.tokenizer is not None:
    # Fallback: use raw SentencePiece (will be wrong!)
    logger.warning("Using raw SentencePiece - token IDs may not match!")
    tokens = [self.tokenizer.IdToPiece(int(tid)) for tid in token_ids_filtered]
    text = "".join(tokens).replace("▁", " ").strip()
```

## Additional Fixes

### 1. SOS/EOS Token IDs (1023 instead of 1/2)

Token 1023 was NOT Arabic 'م' - it's the `<sos/eos>` special token!

**Fixed in all files:**
- `speechcatcher/beam_search/beam_search.py`
  - Lines 47-48: BeamSearch defaults
  - Lines 260-261: BlockwiseSynchronousBeamSearch defaults
  - Lines 685-686: create_beam_search defaults
- `speechcatcher/beam_search/scorers.py`
  - Line 65: DecoderScorer defaults
  - Line 108: CTCPrefixScorer defaults
- `speechcatcher/beam_search/hypothesis.py`
  - Line 75: create_initial_hypothesis defaults

```python
# OLD (WRONG):
sos_id: int = 1
eos_id: int = 2

# NEW (CORRECT):
sos_id: int = 1023
eos_id: int = 1023
```

### 2. Output Slicing Fix

**speechcatcher/speech2text_streaming.py:439-456**

```python
# Extract committed tokens only (up to output_index)
if is_final:
    # Remove SOS and EOS tokens (like ESPnet: hyp.yseq[1:-1])
    token_ids = hyp.yseq[1:-1]
else:
    # Only output committed tokens (yseq[1:output_index+1])
    end_idx = min(self.beam_state.output_index + 1, len(hyp.yseq))
    token_ids = hyp.yseq[1:end_idx]
```

**Problem:** Was using `yseq[1:-1]` for streaming, incorrectly assuming EOS present.

**Fix:** Only slice up to `output_index+1` during streaming.

### 3. BBD Repetition Detection for SOS/EOS

**speechcatcher/beam_search/beam_search.py:381-386**

```python
if last_token == self.sos_id and len(hyp.yseq) == 2:
    logger.debug(f"BBD: Skipping [SOS, EOS] detection (valid empty sequence)")
    continue
```

**Problem:** `[SOS, EOS]` (both 1023) was detected as repetition and rolled back.

**Fix:** Allow `[1023, 1023]` as valid empty sequence.

## Investigation Process

### Step 1: Verified Weights (Again!)

Created `test_chunk5_decoder_comparison.py` to compare decoder outputs with **identical encoder input**:

```python
# Feed SAME 40-frame encoder output to both decoders
encoder_out = torch.cat(buffer, dim=1).narrow(1, 0, 40)

# Compare outputs
espnet_decoder_out, _ = espnet_s2t.asr_model.decoder.forward_one_step(...)
our_decoder_out, _ = our_s2t.model.decoder.forward_one_step(...)
```

**Results:**
```
CTC logits diff: 0.0000000000 ✅
Decoder log probs diff: 0.0000000000 ✅
Both predict same top token: 1023 ✅
```

Weights are **definitely correct**!

### Step 2: Traced ESPnet Beam Search

Created `test_espnet_beam_search_trace.py` to see what ESPnet outputs:

```python
ESPnet output at chunk 5:
('liebe', ['▁liebe'], [738], [24], Hypothesis(yseq=[1023, 738, 1023]))
```

Key observation: Token ID 738 → text 'liebe' but SP[738] = 'trag'!

### Step 3: Discovered Token List Mismatch

Compared ESPnet's `token_list[738]` with SentencePiece `IdToPiece(738)`:

```python
espnet_token_list[738] = '▁liebe'  # ESPnet
sp.IdToPiece(738) = 'trag'          # Raw SentencePiece
```

**This was the smoking gun!** ESPnet uses a modified vocabulary.

### Step 4: Understood the Transformation

Examined ESPnet's token list construction:
- Position 0: `<blank>` (inserted)
- Position 1: SP[0] (first real token)
- Position 2: SP[3] (skipped SP[1] and SP[2])
- ...
- Position 1022: SP[1023] (last SP token)
- Position 1023: `<sos/eos>` (inserted)

## Test Files Created

1. **test_chunk5_decoder_comparison.py** - Proved weights are correct with identical encoder input
2. **test_espnet_beam_search_trace.py** - Revealed token 738 → 'liebe' mapping
3. **test_first_token_prediction.py** - Compared first token predictions
4. **test_combined_scores.py** - Analyzed decoder + CTC combined scoring
5. **test_beam_search_trace.py** - Traced beam state evolution block-by-block
6. **test_token_scoring_comparison.py** - Compared token scores at chunk 4
7. **test_token_scoring_chunk5.py** - Compared beam states at chunk 5

## Current Status

### ✅ What Works

1. **Token vocabulary** - Correctly uses ESPnet's modified vocabulary
2. **SOS/EOS tokens** - Fixed to 1023 throughout
3. **Output slicing** - Only outputs committed tokens
4. **BBD repetition** - Allows [SOS, EOS] as valid empty sequence
5. **German text output** - Successfully outputs "Liebe Mit"! 🎉

### ⚠️ Remaining Differences

**Timing and scoring differences** (not model bugs):

**Chunk 4:**
- ESPnet: No output yet (waiting for more frames)
- Ours: Outputs "Li"

**Chunk 5:**
- ESPnet: `yseq=[1023, 738, 1023]` = `[SOS, ▁liebe, EOS]` (completed!)
- Ours: `yseq=[1023, 372, 7, 187, 538]` = `[SOS, ▁Li, e, be, ▁Mit]` (growing)

**Our beam at chunk 4:**
```
[1] score=-0.14, tokens=[1023, 372]  = [SOS, ▁Li]
[2] score=-4.15, tokens=[1023, 738]  = [SOS, ▁liebe]
```

Token 738 (▁liebe) IS in beam but scores ~4.0 worse than token 372 (▁Li).

**Possible causes:**
1. **Buffering strategy** - ESPnet waits longer before decoding
2. **Beam scoring** - Different CTC/decoder weight combination
3. **Completion strategy** - ESPnet ends hypothesis earlier with EOS

## Key Learnings

1. **Token vocabulary transformations** - Framework may modify tokenizer output for special tokens
2. **Trust but verify** - Even when weights match, vocabulary can differ
3. **SOS/EOS vary by model** - Not always 1/2, check actual token list
4. **Streaming output requires careful slicing** - Can't assume EOS is present
5. **Core decoder works when weights + vocabulary correct** - Timing differences are separate issue

## Files Modified

### speechcatcher/speech2text_streaming.py
- **Lines 103-112**: Build ESPnet-style token list with correct vocabulary mapping
- **Lines 439-456**: Fix output slicing to only emit committed tokens
- **Lines 461-476**: Decode using token_list instead of raw SentencePiece

### speechcatcher/beam_search/beam_search.py
- **Lines 47-48, 260-261, 381-386, 685-686**: Fix SOS/EOS to 1023
- **Lines 381-386**: Allow [SOS, EOS] as valid empty sequence in BBD

### speechcatcher/beam_search/scorers.py
- **Lines 65, 108**: Fix SOS/EOS defaults to 1023

### speechcatcher/beam_search/hypothesis.py
- **Line 75**: Fix SOS default to 1023

## Next Steps

### Immediate: Compare Decoding Strategy

The core decoder is working correctly! Now investigate timing/scoring differences:

1. **Block processing timing** - When does ESPnet start decoding vs us?
2. **CTC prefix scoring** - Are CTC scores weighted differently?
3. **Beam search scoring** - Compare exact scorer combination
4. **BBD settings** - Check block_size, hop_size, look_ahead parameters

### Expected Outcome

Match ESPnet's decoding strategy to produce:
```
Chunk 5: Ours='liebe' (matches ESPnet)
Final: 'Liebe Mitglieder unserer Universität Hamburg.' ✅
```

## Commits Needed

1. Fix token vocabulary mapping to match ESPnet
2. Fix SOS/EOS token IDs throughout
3. Fix output slicing for streaming
4. Fix BBD repetition detection for [SOS, EOS]

## Session Metrics

- **Bug severity**: Critical (blocking correct output)
- **Root cause**: Token vocabulary transformation not implemented
- **Lines changed**: ~40 lines across 4 files
- **Impact**: Decoder now produces correct German text! ✅
- **Tests created**: 7 diagnostic tests

---

**Next session**: Compare decoding strategy (timing, buffering, scoring weights) to match ESPnet's output exactly.
