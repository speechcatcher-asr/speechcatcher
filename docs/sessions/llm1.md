# Custom Decoder Implementation Status

**Session Date:** 2025-10-07
**Goal:** Replace ESPnet's espnet_streaming_decoder with custom BSBS implementation

## Current State: BROKEN ⚠️

The decoder runs but produces repetitive/nonsensical output. Main issue: decoder generates repetitive token 1023 (Arabic character م) instead of proper German text.

---

## What Works ✅

### 1. Checkpoint Loading
- **Files:** `speechcatcher/model/checkpoint_loader.py`
- **Status:** ✅ Working - proper ESPnet → speechcatcher name mapping
- **Tests:** `tests/model/test_checkpoint_loader.py` (7 tests passing)
- **Key fix:** Maps ESPnet layer names to our custom architecture

### 2. CTC Prefix Scoring Algorithm
- **Files:**
  - `speechcatcher/beam_search/ctc_prefix_score.py` - Core algorithm
  - `speechcatcher/beam_search/scorers.py` - Integration with beam search
- **Status:** ✅ Algorithm implemented correctly
- **Tests:** `tests/test_ctc_prefix_score.py` (10 tests passing)
- **Implementation:**
  - `CTCPrefixScore` - Numpy-based forward algorithm (Graves 2012)
  - `CTCPrefixScoreTH` - Torch batched version
  - Tracks `r` (non-blank ending) and `log_psi` (blank ending) probabilities

### 3. Beam Search Infrastructure
- **Files:**
  - `speechcatcher/beam_search/beam_search.py` - Main BSBS implementation
  - `speechcatcher/beam_search/hypothesis.py` - Hypothesis dataclass
  - `speechcatcher/beam_search/scorers.py` - Scorer interfaces
- **Status:** ✅ Infrastructure complete, ❌ output quality broken
- **Key components:**
  - `BeamSearch` - Standard beam search
  - `BlockwiseSynchronousBeamSearch` - Streaming BSBS
  - `DecoderScorer` - Attention decoder scoring
  - `CTCPrefixScorer` - CTC prefix scoring

---

## What's Broken ❌

### 1. Decoder Generates Repetitive Token 1023
**Symptom:**
```
Output: ممممممم Pou ممممممممmmم wir Jo ممممungung ممممmm nur nur ممممmmmmmmmmمm
```

**Token 1023:** Arabic character "م" (meem)

**Root Cause:** Unknown - decoder gets stuck generating this token repeatedly

**Current Workaround:** None effective. Repetition detection (line 321-328 in beam_search.py) stops after 4 repeats per block, but across many blocks we still get many repetitions.

### 2. CTC Integration Issues

#### Issue A: Proper CTC Prefix Scoring Times Out
- **Location:** `scorers.py:149-194` - `CTCPrefixScorer.batch_score()`
- **Problem:** Creating `CTCPrefixScoreTH` instance for each batch is O(B×T×V), too slow
- **Symptom:** Timeout after 2 minutes on 20s audio
- **Status:** Currently DISABLED (line 398-402 in beam_search.py)

#### Issue B: Simplified CTC (Max-Pooling) Doesn't Help
- **Location:** `scorers.py:180-183`
- **Implementation:**
  ```python
  # Simplified: max-pool across time
  max_log_probs, _ = log_probs.max(dim=1)  # (batch, vocab_size)
  ```
- **Problem:** Gives same scores to different hypotheses, doesn't constrain beam search effectively
- **Result:** Different output but still repetitive/wrong

---

## File Structure

### Core Implementation Files
```
speechcatcher/
├── beam_search/
│   ├── beam_search.py          # BSBS implementation (407 lines)
│   ├── ctc_prefix_score.py     # CTC forward algorithm (346 lines)
│   ├── hypothesis.py            # Hypothesis dataclass (108 lines)
│   └── scorers.py              # Scorer interfaces (305 lines)
├── model/
│   ├── checkpoint_loader.py    # ESPnet checkpoint loading (165 lines)
│   ├── espnet_asr_model.py     # Main model class (200 lines)
│   ├── encoder.py              # Streaming encoder (482 lines)
│   ├── decoder.py              # Transformer decoder (318 lines)
│   └── ctc.py                  # CTC module (56 lines)
└── asr_inference_streaming.py  # Inference pipeline (390 lines)
```

### Test Files
```
tests/
├── beam_search/
│   └── test_beam_search.py
├── model/
│   ├── test_checkpoint_loader.py  # 7 tests ✅
│   ├── test_encoder.py
│   └── test_decoder.py
└── test_ctc_prefix_score.py       # 10 tests ✅
```

---

## Key Code Locations

### 1. Where Decoding Happens
**File:** `speechcatcher/beam_search/beam_search.py`

**Entry point:** `BlockwiseSynchronousBeamSearch.process_block()` (line 233-329)
- Encodes audio block
- Runs beam search for multiple steps (line 280-328)
- Expands hypotheses (line 292-311)
- Prunes to beam size (line 314)

**Repetition detection:** Line 321-328
```python
# Repetition detection: stop if top hypothesis has same token repeated 4+ times
top_hyp = new_state.hypotheses[0]
if len(top_hyp.yseq) >= 5:
    last_4 = top_hyp.yseq[-4:]
    if len(set(last_4)) == 1 and last_4[0] != self.sos_id:
        logger.warning(f"Repetition detected: token {last_4[0]} repeated 4 times, stopping block")
        break
```

### 2. Where CTC is Integrated
**File:** `speechcatcher/beam_search/beam_search.py`

**CTC initialization:** Line 398-402 (CURRENTLY COMMENTED OUT)
```python
# CTC prefix scoring with proper forward algorithm
# Disabled temporarily to debug timeout
# if model.ctc is not None and ctc_weight > 0:
#     scorers["ctc"] = CTCPrefixScorer(model.ctc, blank_id=0, eos_id=2)
#     weights["ctc"] = ctc_weight
```

**To re-enable CTC:** Uncomment lines 400-402

### 3. Where Scores are Combined
**File:** `speechcatcher/beam_search/beam_search.py`

**Batch scoring:** Line 59-114 - `BeamSearch.batch_score_hypotheses()`
- Scores each hypothesis with all scorers
- Combines scores with weights (line 112)
- Returns combined scores and new states

### 4. CTC Batch Scoring (BROKEN)
**File:** `speechcatcher/beam_search/scorers.py`

**Current implementation:** Line 152-188 - `CTCPrefixScorer.batch_score()`
```python
# Simplified CTC scoring: max-pool across time
# This is not proper CTC prefix scoring but is much faster
max_log_probs, _ = log_probs.max(dim=1)  # (batch, vocab_size)
```

**Problem:** Too simplified, doesn't provide useful gradient for beam search

### 5. Where Hypotheses are Converted to Text
**File:** `speechcatcher/asr_inference_streaming.py`

**Assembly function:** Line 341-367 - `assemble_hyps()`
- Removes SOS/EOS tokens (line 348)
- Removes blank tokens (token ID 0) (line 355)
- Converts to text tokens (line 358)
- **NOTE:** No CTC collapse - this is correct for hybrid CTC+attention decoding

---

## Test Commands

### Quick Tests (5s audio)
```bash
# Decoder-only (current state, produces repetitive output)
speechcatcher Neujahrsansprache_5s.mp4
cat Neujahrsansprache_5s.mp4.txt
# Expected output: ممممممم (repetitive token 1023)

# Check JSON output
cat Neujahrsansprache_5s.mp4.json | jq '.paragraphs[0].tokens'
```

### Longer Test (20s audio)
```bash
# With timeout protection
timeout 120 speechcatcher Neujahrsansprache_20s.mp4
cat Neujahrsansprache_20s.mp4.txt
# Expected: Some German words mixed with repetitive م tokens
```

### Run Tests
```bash
# Checkpoint loading tests
pytest tests/model/test_checkpoint_loader.py -v

# CTC prefix scoring tests
pytest tests/test_ctc_prefix_score.py -v

# End-to-end pipeline test
pytest tests/model/test_checkpoint_loader.py::TestEndToEnd::test_pipeline_integration -v -s
```

---

## Problem Analysis

### Hypothesis 1: Decoder State Management Issue
The decoder states might not be updating correctly between steps.

**Evidence:**
- Token 1023 appears repeatedly
- Some correct German words appear ("wir", "werden", "nur") suggesting the model isn't completely broken

**Check:** `beam_search.py` line 299-309 - state handling in hypothesis expansion

### Hypothesis 2: CTC Weight Too Low (when enabled)
Default: `decoder_weight=0.7`, `ctc_weight=0.3`

Without CTC constraint, decoder might be stuck in a local optimum.

**Test:** Re-enable CTC with different weights

### Hypothesis 3: Encoder Output Issue
Maybe the encoder is producing repetitive/incorrect features.

**Evidence against:** `test_checkpoint_loader.py::test_encoder_different_inputs_different_outputs` passes

### Hypothesis 4: Vocabulary/Token Mapping Issue
Token 1023 might be getting mapped incorrectly.

**Check:** `speechcatcher/model/checkpoint_loader.py` line 44-76 - name mapping

---

## Next Steps (Priority Order)

### Priority 1: Debug Why Token 1023 Repeats
1. Add logging to see decoder scores for each step
2. Check what token 1023 actually is in the vocabulary
3. Examine decoder attention weights - is it attending to the same position?

**Files to modify:**
- `beam_search.py` - add logging around line 294 (top_scores)
- `decoder.py` - add logging to see attention patterns

**Command to investigate:**
```bash
# Run with debug logging
PYTHONPATH=/home/ben/speechcatcher python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from speechcatcher.speechcatcher import transcribe
results = transcribe('Neujahrsansprache_5s.mp4')
print('Results:', results)
" 2>&1 | less
```

### Priority 2: Fix CTC Timeout Issue
The proper CTC prefix scoring is too slow. Options:

**Option A: Optimize CTCPrefixScoreTH**
- Reuse scorer instance across batches
- Track time step state persistently
- **File:** `scorers.py` line 152-188

**Option B: Use Simpler CTC Approximation**
- Sum log probs instead of max (closer to CTC expectation)
- Use top-k tokens instead of full vocab
- **File:** `scorers.py` line 180-183

**Option C: Use ESPnet's CTCPrefixScorer**
- Import from `espnet2.asr.decoder.s4_decoder`
- Might be already optimized
- **Risk:** May have dependencies on ESPnet internals

### Priority 3: Verify Against Original ESPnet
Compare our implementation with original `espnet_streaming_decoder`:

```bash
# Test with original decoder (for comparison)
# File: speechcatcher/asr_inference_streaming.py
# Temporarily switch back to ESPnet's decoder around line 200-250
```

### Priority 4: Add Diagnostic Outputs
Create a debug mode that shows:
- Top-5 token scores at each step
- Decoder attention weights
- Encoder output statistics
- State changes between steps

---

## Known Issues Summary

| Issue | Location | Status | Priority |
|-------|----------|--------|----------|
| Repetitive token 1023 | beam_search.py:280-328 | ❌ Blocking | P1 |
| CTC timeout | scorers.py:152-188 | ❌ Disabled | P2 |
| Max-pool CTC ineffective | scorers.py:180-183 | ❌ Broken | P2 |
| No diagnostic logging | beam_search.py (entire) | ⚠️ Needed | P3 |

---

## Related Documentation

- **Original implementation:** `docs/sessions/espnet_streaming_decoder.md`
- **Roadmap:** `docs/sessions/ROADMAP.md`
- **Decoder design doc:** `docs/sessions/decoder_implementation.md`
- **End-to-end test:** `docs/sessions/end_to_end_test_summary.md`

---

## Quick Reference: Reverting to ESPnet Decoder

If you need to revert to the working ESPnet decoder:

1. **Edit:** `speechcatcher/asr_inference_streaming.py`
2. **Find:** The `__call__` method (around line 200-300)
3. **Replace:** Custom BSBS calls with original `espnet_streaming_decoder` calls
4. **Import:** Re-add ESPnet imports at top of file

**Note:** This will lose all the custom streaming functionality we built.

---

## Architecture Diagram

```
Audio Input (20s)
    ↓
[Blockwise Processing]
    ↓
┌─────────────────────────────────────┐
│ BlockwiseSynchronousBeamSearch      │
│ (beam_search.py:169-367)            │
│                                     │
│  For each block:                    │
│  1. Encode block → encoder_out      │
│  2. Beam search (max 20 tokens):    │
│     ┌───────────────────────────┐  │
│     │ batch_score_hypotheses    │  │
│     │ ┌─────────────────────┐   │  │
│     │ │ DecoderScorer       │   │  │ ← PRODUCES REPETITIVE SCORES
│     │ │ (weight: 0.7)       │   │  │
│     │ └─────────────────────┘   │  │
│     │ ┌─────────────────────┐   │  │
│     │ │ CTCPrefixScorer     │   │  │ ← CURRENTLY DISABLED
│     │ │ (weight: 0.3)       │   │  │
│     │ └─────────────────────┘   │  │
│     └───────────────────────────┘  │
│     ↓                               │
│  3. Expand hypotheses               │
│  4. Prune to beam_size=10           │
│  5. Check repetition (4+ tokens)    │ ← STOPS PER BLOCK BUT NOT GLOBAL
└─────────────────────────────────────┘
    ↓
Best hypothesis → Text output
```

---

## Recent Changes Log

**2025-10-07:**
- ✅ Implemented CTC prefix scoring algorithm
- ✅ Added 10 tests for CTC prefix scorer (all passing)
- ❌ Attempted to integrate CTC - caused timeout
- ❌ Tried simplified max-pooling CTC - ineffective
- ❌ Attempted to add CTC collapse post-processing - incorrect approach (not needed for hybrid decoding)
- ❌ Tried adding repetition penalties in beam search - reverted (hack, not root cause fix)

**Key Insight from Session:**
> "You're just trying to hack and patch. Repeating tokens are normal to a degree in CTC, usually these are collapsed and there is a blank token _ to make sure that tokens aren't just repeated."

This led to the realization that:
1. CTC collapse is only for pure CTC decoding, not hybrid CTC+attention
2. The repetitive output is a decoder issue, not a post-processing issue
3. Need to fix the root cause in beam search scoring, not patch symptoms

---

## End of Document

**Last Updated:** 2025-10-07
**Status:** Broken - produces repetitive output
**Next Session:** Start with Priority 1 - debug why token 1023 repeats
