# LLM10 Session: BBD Investigation and ESPnet Behavior Matching

## Session Overview

This session investigated Block Boundary Detection (BBD) behavior and discovered critical differences between our native decoder and ESPnet's implementation. We identified the root cause of output quality differences and updated our defaults to match ESPnet.

## Critical Findings

### 1. Repetition Source: Decoder, Not CTC

**Question**: Are we correctly handling CTC blank/repetition removal?

**Answer**: Yes. The massive repetitions we observed are from the **transformer decoder**, not CTC.

**Evidence**:
```
Hypothesis without BBD: [SOS, ▁Li, e, be, ▁Li, e, be, ▁Li, e, be, ...] (434 tokens!)
Decoded output: "Liebe Liebe Liebe Liebe..." (massive repetition)
```

**CTC's role**:
- Both ESPnet and our implementation only filter **blank tokens (ID=0)**
- Neither does CTC consecutive duplicate removal in final output
- CTC prefix scoring happens during beam search, but final `yseq` comes from decoder
- Code: `token_ids_filtered = [tid for tid in token_ids if tid != 0]`

### 2. ESPnet Has BBD Enabled By Default

**Critical discovery**: ESPnet sets `disable_repetition_detection=False` (i.e., BBD **enabled**) by default.

**Source**: `espnet_streaming_decoder/asr_inference_streaming.py:73`
```python
def __init__(
    self,
    ...
    disable_repetition_detection=False,  # BBD ENABLED by default
    ...
):
```

**Our previous default**: BBD **disabled** (`use_bbd=False`)

**Impact**: This explains why ESPnet produces clean output without repetition while our decoder produces 440+ token sequences of "Liebe Liebe Liebe..."

### 3. BBD Behavior Comparison

#### Test Results on 5s Audio Sample

| Implementation | BBD Setting | Result | Token Count |
|---|---|---|---|
| ESPnet | Enabled (default) | "Liebe Mitglieder unserer Universität Hamburg." | 20 tokens |
| Native (old default) | Disabled | "Liebe Liebe Liebe..." (massive repetition) | 440+ tokens |
| Native (new default) | Enabled | "Liebe Mitglieder" | 8 tokens |

#### Observation

Both ESPnet and our native decoder with BBD enabled produce clean output without repetition, though ESPnet's is slightly longer. BBD stops early in both cases due to **legitimate subword token repetition**.

### 4. Why BBD Causes Early Stopping

**BBD Logic**: Detects when the last predicted token appears anywhere in previous tokens.

**Problem with subword tokenization**:
```
Sequence: [SOS, ▁Li, e, be, ▁Mit, g, li, e, der]
                   ^^^        ^^^         ^^^
Token "e" appears multiple times legitimately in "Liebe Mitglieder"
```

When the decoder predicts "e" for the second time, BBD detects:
- Last token: `e` (ID=7)
- Previous tokens: `[SOS, ▁Li, e, be, ▁Mit, g, li]`
- Detection: `e in [SOS, ▁Li, e, be, ▁Mit, g, li]` → **TRUE**
- Action: Stop decoding this block

**Result**: Early termination despite legitimate repetition.

**ESPnet developer's comment** (batch_beam_search_online.py:206-208):
> "too sensitive that the beam search starts only after the entire inputs are available. Empirically, this flag didn't affect performance."

Despite this sensitivity, ESPnet keeps it enabled by default to prevent decoder runaway repetition.

## Changes Made

### Updated Default to Match ESPnet

**Files modified**:
1. `speechcatcher/beam_search/beam_search.py:739`
   - Changed: `use_bbd: bool = False` → `use_bbd: bool = True`
   - Comment: "Default enabled to match ESPnet behavior"

2. `speechcatcher/speechcatcher.py:703-705`
   - Renamed: `--enable-bbd` → `--disable-bbd`
   - Inverted logic: `use_bbd=not args.disable_bbd`
   - Help text: "BBD prevents repetition but may cause early stopping... (default: enabled to match ESPnet)"

3. `speechcatcher/speech2text_streaming.py:50, 142-149`
   - Added `use_bbd` parameter to `__init__`
   - Passed through to `create_beam_search()`

### Command-Line Interface

**New behavior**:
```bash
# Default: BBD enabled (matches ESPnet)
python3 -m speechcatcher.speechcatcher audio.mp4 --decoder native
→ Clean output, no repetition

# Disable BBD if experiencing early stopping
python3 -m speechcatcher.speechcatcher audio.mp4 --decoder native --disable-bbd
→ May produce repetition, but continues longer
```

## Technical Details

### BBD Implementation (ESPnet's approach)

Location: `speechcatcher/beam_search/beam_search.py:386-407`

```python
def detect_repetition_or_eos(self, hypotheses):
    """Detect repetition in hypotheses (BBD mechanism)."""
    for i, hyp in enumerate(hypotheses):
        if len(hyp.yseq) < 2:
            continue

        last_token = hyp.yseq[-1].item()

        # Skip EOS tokens (don't check hypotheses that ended)
        if last_token == self.eos_id:
            logger.debug(f"BBD: Hyp {i}: Skipping EOS hypothesis")
            continue

        # ESPnet's check: last token appears anywhere in previous tokens
        # This is: best.yseq[i, -1] in best.yseq[i, :-1]
        prev_tokens = hyp.yseq[:-1].tolist()
        if last_token in prev_tokens:
            logger.debug(f"BBD: Detected repetition - token {last_token}")
            return True

    return False
```

**Called from**: `beam_search.py:665-683` in `_decode_one_block()`
- Only checked for non-final blocks (`if self.use_bbd and not is_final`)
- Breaks decoding when repetition detected
- Rolls back to previous hypotheses (1 step)

### ESPnet's BBD Location

**File**: `espnet_streaming_decoder/batch_beam_search_online.py:200-218`

**Key logic** (lines 209-214):
```python
elif (
    not self.disable_repetition_detection
    and not prev_repeat
    and best.yseq[i, -1] in best.yseq[i, :-1]  # Same check as ours
    and not is_final
):
    prev_repeat = True
```

**Behavior**: If ANY hypothesis in the beam has repetition, set `prev_repeat=True` and break (line 216-218).

**Difference**: ESPnet uses a `prev_repeat` flag instead of directly returning from function, but effect is the same.

## Tradeoffs

### With BBD Enabled (New Default)

**Pros**:
- ✅ Prevents decoder runaway repetition
- ✅ Matches ESPnet's default behavior
- ✅ Clean output without "Liebe Liebe Liebe..." issues

**Cons**:
- ⚠️ May stop early due to legitimate subword repetition
- ⚠️ Shorter output than ESPnet in some cases
- ⚠️ Particularly sensitive with BPE/subword tokenization

### With BBD Disabled (--disable-bbd)

**Pros**:
- ✅ Decoder continues longer
- ✅ May produce more complete transcriptions

**Cons**:
- ❌ High risk of massive repetition (440+ tokens observed)
- ❌ Output becomes unusable: "word word word word..."
- ❌ Not recommended without another repetition prevention mechanism

## Remaining Questions

### Why Does ESPnet Produce Longer Output?

**Observation**: With BBD enabled:
- ESPnet: "Liebe Mitglieder unserer Universität Hamburg." (20 tokens)
- Native: "Liebe Mitglieder" (8 tokens)

**Both use identical BBD logic**, so why the difference?

**Possible explanations** (not yet investigated):
1. **Beam pruning differences**: ESPnet may prune low-scoring hypotheses differently
2. **Score calculation differences**: Slight numerical differences in CTC/decoder scoring
3. **State management**: Differences in how states are extended/maintained
4. **Timing of BBD check**: When in the beam search loop BBD is evaluated
5. **Hypothesis selection**: Which hypothesis from the beam is chosen for output

**Next steps**: Would need detailed logging of beam scores and hypothesis rankings to diagnose.

### Final Block Early Termination

**Observation**: Final block (is_final=True) stops after only 3-7 steps instead of continuing until EOS or max_steps.

**Current behavior**:
```
[DEBUG] _decode_one_block: Step 0/100, is_final=True
[DEBUG] _decode_one_block: Step 1/100, is_final=True
[DEBUG] _decode_one_block: Step 2/100, is_final=True
[DEBUG] _decode_one_block: Step 3/100, is_final=True
→ Stops here
```

**Not investigated yet**: Need to check:
- EOS detection logic in final block (lines 654-662 in beam_search.py)
- Hypothesis pruning in final block
- Whether best hypothesis reaches EOS early

## Testing Notes

### Test Commands

```bash
# Default (BBD enabled)
python3 -m speechcatcher.speechcatcher Neujahrsansprache_5s.mp4 \
    -m de_streaming_transformer_xl -n 1 --decoder native --quiet

# BBD disabled (for testing)
python3 -m speechcatcher.speechcatcher Neujahrsansprache_5s.mp4 \
    -m de_streaming_transformer_xl -n 1 --decoder native --disable-bbd --quiet

# ESPnet comparison
python3 -m speechcatcher.speechcatcher Neujahrsansprache_5s.mp4 \
    -m de_streaming_transformer_xl -n 1 --decoder espnet --quiet
```

### Test Files

- `Neujahrsansprache_5s.mp4` - 5 second German speech sample
- `Neujahrsansprache_20s.mp4` - 20 second German speech sample
- Expected: "Liebe Mitglieder, unsere Universität Hamburg. Die Vergangenheit hält uns gefangen..."

## Key Takeaways

1. **Decoder repetition is the primary issue**, not CTC
2. **ESPnet has BBD enabled by default** - we now match this
3. **BBD is essential** to prevent runaway repetition (440+ tokens)
4. **BBD causes early stopping** with subword tokenization (sensitivity tradeoff)
5. **Users can disable BBD** with `--disable-bbd` if needed
6. **Our implementation correctly matches ESPnet's BBD logic** (same repetition check)
7. **Output length differences remain unexplained** - requires further investigation

## References

- ESPnet streaming decoder: `espnet_streaming_decoder/batch_beam_search_online.py`
- ESPnet developer comment: Lines 206-208 about BBD sensitivity
- Our BBD implementation: `speechcatcher/beam_search/beam_search.py:386-407`
- Blockwise Synchronous Beam Search paper: https://arxiv.org/abs/2006.14941
