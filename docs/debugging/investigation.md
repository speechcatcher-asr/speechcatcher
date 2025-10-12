# Investigation Summary: Native vs ESPnet Decoder Comparison

## Objective
Investigate why native decoder produces only ~35% of ESPnet output (238 vs 544 words on full video).

## Methodology
Step-by-step investigation of segment_1 (worst offender: 37 native words vs 96 ESPnet words).

## Key Findings

### 1. BBD Implementation is Correct ✓
- **Both decoders show identical BBD triggering patterns**
- Step 7 → Step 1 → Step 0 repeatedly
- BBD logic matches ESPnet perfectly
- **Conclusion**: BBD is NOT the problem

### 2. Encoder Outputs are Identical ✓
- Added debug logging to capture encoder statistics
- Native decoder block 0: mean=0.056936, std=0.493884, shape=[1, 16, 256]
- Encoder is deterministic and produces identical outputs
- **Conclusion**: Encoder is NOT the problem

### 3. Root Cause: Premature EOS Prediction ✗
Added hypothesis selection debugging, discovered:

**Block 0, Step 5**:
```
best_yseq=[1023, 72, 104, 260, 66, 3, 1023]
                                        ^EOS at position 6!
```

**Block 2, Step 2**:
```
best_yseq=[..., 41, 1023, 1023]
                    ^TWO EOS tokens back-to-back!
```

**Why this is catastrophic**:
1. Decoder assigns high probability to EOS (token ID 1023) prematurely
2. Hypotheses contain multiple EOS tokens in middle of sequence
3. BBD detects token 1023 repetition and triggers immediately
4. Block decoding stops before meaningful output is generated
5. Only 5-37 words produced instead of 96+

### 4. Secondary Issue: Special Token Leaking ✓ FIXED
**Before fix**: Native output contained `<sos/eos>` tokens in text:
```
Forderung, die<sos/eos><sos/eos><sos/eos>...<sos/eos> unsere...
```

**Fix applied**: Filter tokens 0, 1, 1023 from output (`speech2text_streaming.py:511`)
```python
# Before:
token_ids_filtered = [tid for tid in token_ids if tid != 0]

# After:
token_ids_filtered = [tid for tid in token_ids if tid not in [0, 1, 1023]]
```

**Result**: Output is now clean, but still very short (5 words vs 96)

## Why ESPnet Doesn't Have This Problem

The original ESPnet decoder handles EOS differently, likely through:
1. **Different hypothesis pruning**: Discards premature EOS hypotheses
2. **Different EOS scoring**: Penalizes premature EOS predictions
3. **Different state management**: Prevents EOS prediction loops
4. **Different beam selection**: Prefers hypotheses that haven't reached EOS yet

## Impact Summary

### Segments 0-3 (First 4 minutes)
| Segment | Native | ESPnet | Native % | Issue |
|---------|--------|--------|----------|-------|
| 0       | 70     | 105    | 66.7%    | Premature EOS |
| 1       | 5*     | 96     | 5.2%     | Severe premature EOS |
| 2       | 4      | 96     | 4.2%     | Critical premature EOS |
| 3       | 6      | 111    | 5.4%     | Critical premature EOS |

*After special token filtering fix

### Segments 4-7 (Last 4 minutes)
All four segments produce **identical output** (100% match), suggesting:
- No premature EOS in these segments
- Audio content dependency
- Possible acoustic differences that trigger/prevent EOS

## Files Modified

1. **`speechcatcher/speech2text_streaming.py:511`**
   - Fixed special token filtering
   - Now removes tokens 0, 1, 1023 from output

2. **`speechcatcher/beam_search/beam_search.py:475-489`**
   - Added encoder output debugging
   - Logs mean/std/min/max statistics
   - Saves first block to `/tmp/encoder_debug/block0_native.pt`

3. **`speechcatcher/beam_search/beam_search.py:666-671`**
   - Added hypothesis selection debugging
   - Logs best hypothesis score and token sequence after each step

4. **`speechcatcher/beam_search/beam_search.py:758-769`**
   - Added `decoder_name` parameter to `create_beam_search()`
   - Enables distinguishing ESPnet vs native in debug logs

## Next Steps

### Option 1: Match ESPnet's EOS Handling (Recommended)
Investigate how ESPnet's original decoder prevents premature EOS:
1. Check ESPnet's `batch_beam_search_online.py` for EOS handling
2. Look for hypothesis filtering that discards premature EOS
3. Check if ESPnet uses different EOS scoring/penalties
4. Implement similar logic in native decoder

### Option 2: Post-Filter EOS Tokens
Add logic to prevent decoder from predicting EOS until minimum length reached:
- Block EOS token in beam search until min_length tokens generated
- Similar to length penalty in standard beam search
- Risk: May not match ESPnet's exact behavior

### Option 3: Investigate Acoustic Dependency
Understand why segments 4-7 work perfectly but 0-3 fail:
- Analyze audio characteristics (volume, speech rate, silence)
- Check if encoder outputs differ in problematic segments
- May reveal model limitations or training data bias

## Recommendation

**Start with Option 1**: Study ESPnet's EOS handling and match it exactly. This is the most principled approach and will ensure our decoder behaves identically to ESPnet.

The investigation infrastructure is now in place:
- Encoder debugging
- Hypothesis selection logging
- Special token filtering
- All tools needed to compare behaviors in detail
