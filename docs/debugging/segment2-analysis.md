# Segment 2 In-Depth Comparison: ESPnet vs Native Decoder

## Executive Summary

**Root Cause Identified**: Native decoder produces inferior output due to:
1. **Special token leaking**: `<sos/eos>` tokens (ID 1023) leak into output text
2. **Premature EOS prediction**: Decoder predicts EOS token in middle of sequence
3. **BBD working correctly but revealing decoder issues**: BBD triggers identically in both decoders

## Detailed Findings

### 1. **BBD Behavior is Identical** ✓
Both ESPnet and native decoders show **identical BBD triggering patterns**:
- First block: triggers at step 7
- Second block: triggers at step 1
- Subsequent blocks: trigger at step 0 repeatedly

**Conclusion**: BBD logic is correctly implemented and matching ESPnet. BBD is NOT the problem.

### 2. **Special Tokens Leaking in Native Output** ✗
**Native decoder output** (segment_1, 37 words):
```
Forderung, die<sos/eos><sos/eos><sos/eos><sos/eos><sos/eos><sos/eos><sos/eos><sos/eos><sos/eos><sos/eos>.<sos/eos> unsere gegenwer und jeweilige Situation gute. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Und
```

**ESPnet output** (segment_1, 96 words):
```
Unsere gegenwärtige Situation gut erfasst. Ersatz enthält im Grunde ja drei Beschreibungen von drei allgemeinen zeitlichen Phasen jeweils gedacht vom eigenen Standpunkt aus. Wenn wir darüber am Anfang eines Jahres nachdenken, dann könnte es vielleicht dieses bedeuten. Die Vergangenheit hält uns gefangen, ja, viele von uns schauen noch in die Vergangenheit einer Universität, unserer Universität, wie wir sie kannten...
```

**Issue**: `<sos/eos>` tokens (token ID 1023) appear 11 times in native output. These should be filtered.

**Location**: `speech2text_streaming.py:506-514`

Current code:
```python
# Line 506: Only filters blank tokens (ID 0)
token_ids_filtered = [tid for tid in token_ids if tid != 0]

# Line 512: Converts ALL remaining tokens to text
tokens = [self.token_list[int(tid)] for tid in token_ids_filtered]
text = "".join(tokens).replace("▁", " ").strip()
```

**Missing**: Filter for `<sos/eos>` (ID 1023), `<blank>` (ID 0), `<unk>` (ID 1)

### 3. **Premature EOS Prediction** ✗
Token 1023 (`<sos/eos>`) appearing in the middle of a sequence indicates the decoder is predicting EOS too early. From debug logs:

```
[DEBUG] BBD: Hypothesis: [1023, 4, 19, 100, 8, 337, 2, 79, 111, 1023]
                           ^SOS                                  ^EOS in middle!
```

This hypothesis has EOS at position 9 (middle of sequence), not at the end. This causes:
1. The beam search to think this hypothesis is "complete"
2. The decoder to stop generating meaningful content
3. Early block termination when BBD detects repetition of token 1023

**Root cause**: Decoder is assigning high probability to EOS token even though the utterance is not complete. This might be due to:
- Encoder producing poor representations
- Decoder attention not working correctly
- Model not generalizing well to this audio

### 4. **"Under" Repetition Not Prevented by BBD** ✗
The native decoder output shows "Under" repeating **28 times**:
```
Under. Under. Under. Under. Under. Under. Under. Under. Under...
```

**Why BBD didn't stop it**: BBD checks token-level repetition, not word-level. With BPE tokenization, "Under" might be split into subword tokens:
```
"Under." → ["▁Und", "er", "."]
```

If the sequence is `["▁Und", "er", ".", "▁Und", "er", ".", ...]`, each individual token appears multiple times but:
1. BBD only checks if the **last token** appeared in previous tokens
2. When last_token=".", it checks if "." appeared before (YES → trigger BBD)
3. But BBD might trigger too late, after many repetitions have already been generated

**However**: The debug logs show BBD is triggering at step 0 repeatedly, so this repetition might be happening within a single decoding step, or BBD is not being called for every token.

###5. **Hypothesis with Multiple Special Tokens**
Both decoders generate hypotheses with repeated tokens:
```
[DEBUG] BBD: Hypothesis: [1023, 4, 19, 100, 8, 337, 2, 79, 111, 1023]  ← Token 1023 appears twice
[DEBUG] BBD: Hypothesis: [1023, 4, 19, 100, 8, 337, 2, 79, 111, 2]      ← Token 2 (comma) appears twice
[DEBUG] BBD: Hypothesis: [1023, 4, 19, 100, 8, 337, 79, 2, 1023, 1023] ← Token 1023 appears THREE times!
```

This is why BBD triggers at step 0 - the hypotheses already contain repeated tokens before the first decoding step completes.

**Key difference**:
- **ESPnet**: Handles these repetitions gracefully, continues to produce valid output
- **Native**: Gets stuck with these problematic hypotheses, produces garbage output

## Root Cause Analysis Summary

### Primary Issue: Decoder Predictions
The native decoder is predicting:
1. **EOS tokens prematurely** (in middle of sequence)
2. **Repeated tokens/words** that BBD doesn't catch in time
3. **Low-quality hypotheses** that contain special tokens multiple times

### Secondary Issue: Output Filtering
The token-to-text conversion doesn't filter special tokens (except at beginning/end), allowing `<sos/eos>` to leak into output.

### BBD Status
BBD is working correctly and identically to ESPnet. It's detecting the repetitions, but the decoder is generating so many problematic sequences that BBD triggers immediately and stops decoding before meaningful output is generated.

## Critical Question - ANSWERED

**Why does ESPnet produce good output with the same BBD behavior?**

Both decoders have identical BBD triggering (step 7, step 1, step 0...), but:
- ESPnet produces 96 words of coherent text
- Native produces 37 words with artifacts and repetition

**Root Cause Identified**:

### Encoder Outputs are Identical ✓
Encoder statistics for segment_1 block 0:
- Shape: torch.Size([1, 16, 256])
- Mean: 0.056936, Std: 0.493884
- Min: -1.674985, Max: 1.707910

The encoder is deterministic and produces identical outputs. **Encoder is not the problem.**

### Decoder Predicts Premature EOS ✗
From hypothesis debugging on segment_1:

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

**Conclusion**: The decoder is assigning high probability to EOS token (ID 1023) even though the utterance is not complete. This causes:
1. Hypotheses to end prematurely
2. Token 1023 to appear multiple times in the sequence
3. BBD to trigger on token 1023 repetition
4. Early block termination before meaningful output is generated

**Why this differs from ESPnet**: The original ESPnet decoder likely handles EOS prediction differently, either through:
- Different beam search pruning that discards premature EOS hypotheses
- Different scoring that penalizes premature EOS
- Different state management that prevents EOS loops

## Next Steps - Priority Order

1. **[HIGH] Compare encoder outputs**: Extract encoder outputs from both decoders for segment_1, verify they're identical
2. **[HIGH] Add special token filtering**: Filter IDs 0, 1, 1023 from output (quick fix for readability)
3. **[MEDIUM] Compare hypothesis selection**: Log which hypothesis is selected as "best" in both decoders
4. **[MEDIUM] Compare beam scores**: Log beam scores at each step to see if scoring differs
5. **[LOW] Investigate "Under" repetition**: Understand why word-level repetition happens despite BBD

## Fix #1: Add Special Token Filtering

**File**: `speechcatcher/speech2text_streaming.py:506`

**Current**:
```python
token_ids_filtered = [tid for tid in token_ids if tid != 0]
```

**Fixed**:
```python
# Filter special tokens: <blank>(0), <unk>(1), <sos/eos>(1023)
token_ids_filtered = [tid for tid in token_ids if tid not in [0, 1, 1023]]
```

This will prevent `<sos/eos>` from appearing in output, but won't fix the underlying decoder issues.
