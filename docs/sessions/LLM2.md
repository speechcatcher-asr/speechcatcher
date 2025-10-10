# Decoder State Structure Fix & CTC Integration Attempt

**Session Date:** 2025-10-10
**Goal:** Fix state management to match ESPnet and re-enable CTC scoring
**Previous Session:** [LLM.md](LLM.md) - Identified repetitive token 1023 issue

---

## Session Overview

This session focused on fixing the fundamental state management issues identified in the comparison analysis. We successfully restructured the Hypothesis class and beam search to match ESPnet's architecture, then attempted to integrate CTC scoring with a simplified approach.

---

## Problems Identified (From compare.md)

### Critical Issues Found
1. **State Structure Wrong** - `Hypothesis.states` was `List[torch.Tensor]` but ESPnet uses `Dict[str, Any]`
2. **State Extraction Broken** - Passing all states to every scorer instead of per-scorer states
3. **Missing xpos Tracking** - Not tracking encoder frame positions
4. **No CTC Constraint** - CTC disabled, decoder drifts into repetitive local optima
5. **Score Tracking Incomplete** - Individual scorer scores never updated after initialization

### Root Cause
Without proper state management, multiple scorers (decoder + CTC) couldn't coexist. Only one scorer's state could be stored at a time, causing state confusion and preventing CTC integration.

---

## Phase 1: State Structure Fix ✅

### 1.1 Updated Hypothesis Class

**File:** `speechcatcher/beam_search/hypothesis.py`

**Before:**
```python
@dataclass
class Hypothesis:
    yseq: List[int]                          # Token sequence as list
    score: float
    scores: Dict[str, float]
    states: Optional[List[torch.Tensor]]     # Single list - can't handle multiple scorers!
    yseq_tensor: Optional[torch.Tensor]      # Redundant field
```

**After:**
```python
@dataclass
class Hypothesis:
    yseq: torch.Tensor                       # Token sequence as tensor (matches ESPnet)
    score: float
    scores: Dict[str, float]
    states: Dict[str, Any]                   # Dict per scorer - FIXED!
    xpos: torch.Tensor                       # Encoder positions - NEW!
```

**Key Changes:**
- `states` now a dict: `{"decoder": [...], "ctc": {...}}`
- `yseq` now `torch.Tensor` for consistency
- Added `xpos` tracking for encoder frame positions
- Removed redundant `yseq_tensor` field

### 1.2 Added Helper Functions

**File:** `speechcatcher/beam_search/hypothesis.py:141-164`

```python
def append_token(tensor: torch.Tensor, token_id: int) -> torch.Tensor:
    """Append a token to a tensor sequence."""
    return torch.cat([tensor, torch.tensor([token_id], ...)])

def append_position(xpos: torch.Tensor, position: int) -> torch.Tensor:
    """Append an encoder position to xpos tensor."""
    return torch.cat([xpos, torch.tensor([position], ...)])
```

### 1.3 Fixed Beam Search State Extraction

**File:** `speechcatcher/beam_search/beam_search.py:59-114`

**Before:**
```python
# Extract states for this scorer
states = [h.states for h in hypotheses]  # Wrong! Gets entire state list
```

**After:**
```python
# Extract states for THIS SPECIFIC scorer from each hypothesis
states = [h.states.get(scorer_name) for h in hypotheses]  # Correct!
```

**Impact:** Each scorer now gets its own state, preventing state confusion.

### 1.4 Fixed Hypothesis Expansion

**File:** `speechcatcher/beam_search/beam_search.py:298-314`

**Before:**
```python
# Only stored first scorer's state
new_states_for_hyp = None
if new_states_dict:
    scorer_name = list(new_states_dict.keys())[0]
    new_states_for_hyp = new_states_dict[scorer_name][i]

new_hyp = Hypothesis(
    yseq=hyp.yseq + [token],
    states=new_states_for_hyp,  # Only one scorer!
)
```

**After:**
```python
# Merge states from ALL scorers
new_states_for_hyp = {}
for scorer_name in new_states_dict:
    new_states_for_hyp[scorer_name] = new_states_dict[scorer_name][i]

# Track encoder position
current_encoder_pos = encoder_out.size(1) - 1

new_hyp = Hypothesis(
    yseq=append_token(hyp.yseq, token),
    states=new_states_for_hyp,  # All scorers!
    xpos=append_position(hyp.xpos, current_encoder_pos),
)
```

**Impact:** All scorer states now preserved across decoding steps.

### 1.5 Updated Initialization

**File:** `speechcatcher/beam_search/hypothesis.py:71-87`

```python
def create_initial_hypothesis(sos_id: int = 1, device: str = "cpu") -> Hypothesis:
    """Create initial hypothesis with SOS token."""
    return Hypothesis(
        yseq=torch.tensor([sos_id], dtype=torch.long, device=device),
        score=0.0,
        scores={},
        states={},  # Empty dict, will be populated by scorers
        xpos=torch.tensor([0], dtype=torch.long, device=device),
    )
```

---

## Phase 2: CTC Scorer Implementation 🔄

### 2.1 Simplified CTC Scorer

**File:** `speechcatcher/beam_search/scorers.py:152-190`

**Approach:** Frame-averaged CTC instead of full prefix scoring

```python
def batch_score(self, yseqs, states, xs):
    """Batch score prefixes using simplified CTC."""
    # Get CTC log probabilities
    logits = self.ctc.ctc_lo(xs)  # (batch, enc_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)

    # IMPROVED: Use mean across time instead of max
    # More stable and represents average acoustic evidence
    mean_log_probs = log_probs.mean(dim=1)  # (batch, vocab_size)

    # Apply penalty to blank token to encourage non-blank predictions
    mean_log_probs[:, self.blank_id] -= 2.0

    return mean_log_probs, states
```

**Rationale:**
- Full CTC prefix scoring is O(B×T×V) - too slow
- Frame-averaging is O(T×V) - much faster
- Still provides acoustic grounding
- Blank penalty prevents getting stuck on blank token

### 2.2 CTC Integration

**File:** `speechcatcher/beam_search/beam_search.py:402-406`

```python
# CTC scoring with simplified frame-averaged approach (fast!)
if model.ctc is not None and ctc_weight > 0:
    scorers["ctc"] = CTCPrefixScorer(model.ctc, blank_id=0, eos_id=2)
    weights["ctc"] = ctc_weight
```

### 2.3 Weight Tuning

**File:** `speechcatcher/speechcatcher.py:99-105`

Adjusted CTC weight from 0.3 → 0.1 to give decoder more influence.

---

## Testing Results

### Test 1: Decoder-Only (No CTC)
```bash
speechcatcher Neujahrsansprache_5s.mp4
```

**Output:** `ممممممم` (repetitive token 1023)

**Analysis:**
- ✅ No errors - state management works!
- ❌ Still repetitive - decoder needs CTC constraint

### Test 2: CTC Enabled (Weight 0.3)
```bash
speechcatcher Neujahrsansprache_5s.mp4
```

**Output:** `,,,` (token 0 - blank token)

**Analysis:**
- ✅ Different output - CTC is working!
- ❌ Generating blanks instead of text
- Repetition detection triggered on blank token

### Test 3: Stronger Blank Penalty
Increased blank penalty from 0.5 → 2.0

**Output:** `,,,,,` (more blanks)

**Analysis:**
- ❌ Blank penalty not strong enough
- CTC overwhelming decoder scores

### Test 4: Reduced CTC Weight (0.3 → 0.1)
```bash
speechcatcher Neujahrsansprache_5s.mp4
```

**Result:** ⏱️ **TIMEOUT after 2 minutes**

**Analysis:**
- ❌ CTC scorer causing decoder to hang
- Something wrong with CTC integration
- Not just a performance issue - it's blocking

---

## Current Status

### What Works ✅
1. **State Management** - Dict-based states properly implemented
2. **Multiple Scorers** - Infrastructure supports decoder + CTC + more
3. **Hypothesis Expansion** - All scorer states merged correctly
4. **xpos Tracking** - Encoder positions tracked
5. **Decoder Alone** - Runs without errors (but repetitive output)

### What's Broken ❌
1. **CTC Integration** - Causes timeout/hang
2. **Blank Token Dominance** - When CTC runs, it generates blanks
3. **Repetitive Output** - Without CTC, decoder gets stuck on token 1023

### Current Configuration
**File:** `speechcatcher/beam_search/beam_search.py:402-406`

```python
# CTC scoring - temporarily disabled, needs further optimization
# Issue: Causes timeout with current implementation
# if model.ctc is not None and ctc_weight > 0:
#     scorers["ctc"] = CTCPrefixScorer(model.ctc, blank_id=0, eos_id=2)
#     weights["ctc"] = ctc_weight
```

**Status:** CTC disabled pending investigation

---

## Problem Analysis: Why CTC Hangs

### Hypothesis 1: CTC Module Call Issue
The CTC module might not be compatible with our batched calling pattern:
```python
logits = self.ctc.ctc_lo(xs)  # This call might be problematic
```

**Evidence:**
- Works with low weight (generates blanks)
- Hangs with very low weight (0.1)
- Suggests it's not just a performance issue

### Hypothesis 2: State Accumulation
We're not tracking states in simplified CTC:
```python
new_states = states  # Just passing through
```

This might cause issues if CTC module expects state updates.

### Hypothesis 3: Tensor Shape Mismatch
The encoder output shape might not match what CTC expects:
```python
encoder_out_batch = encoder_out.expand(batch_size, -1, -1)
```

### Hypothesis 4: Blank Penalty Interaction
The blank penalty might be interfering with score computation:
```python
mean_log_probs[:, self.blank_id] -= 2.0
```

This happens AFTER softmax, which might create numerical issues.

---

## Comparison: What ESPnet Does Differently

### ESPnet's CTC Integration
**File:** `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py:303-318`

```python
def extend(self, x: torch.Tensor, hyps: Hypothesis):
    """Extend probabilities and states with more encoded chunks."""
    for k, d in self.scorers.items():
        if hasattr(d, "extend_prob"):
            d.extend_prob(x)  # Extend CTC probability matrix
        if hasattr(d, "extend_state"):
            hyps.states[k] = d.extend_state(hyps.states[k])
```

**Key Differences:**
1. ESPnet calls `extend_prob()` BEFORE each decoding block
2. ESPnet calls `extend_state()` to update CTC state
3. We don't have these methods in our simplified scorer

### ESPnet's CTC Scorer Structure
ESPnet uses `CTCPrefixScorer` from their library, which:
- Maintains probability matrix across all frames
- Uses incremental forward algorithm
- Has `extend_prob()` and `extend_state()` methods
- Properly handles streaming/blockwise processing

---

## Documentation Updates

### Files Reorganized
```
Root → docs/
├── compare.md (renamed from compare.MD)
├── DECODER_README.md
├── ESPNET_DECODER_ROADMAP.md
├── IMPLEMENTATION_SUMMARY.md
├── WEIGHT_LOADING_NOTES.md
└── decoder.md

Root → docs/sessions/
├── LLM.md (Oct 7 session)
├── LLM2.md (Oct 10 session - this file)
└── PHASE_0_SUMMARY.md
```

### New Documentation
- **`docs/README.md`** - Documentation index
- **`docs/compare.md`** - Detailed ESPnet comparison with fix recommendations

---

## Next Steps (Priority Order)

### Priority 1: Investigate CTC Hang 🔴
**Goal:** Understand why CTC scorer causes timeout

**Approach A: Add Debug Logging**
```python
# In scorers.py batch_score()
logger.debug(f"CTC input shape: {xs.shape}")
logger.debug(f"CTC logits shape: {logits.shape}")
logger.debug(f"Mean log probs: {mean_log_probs[0, :10]}")
```

**Approach B: Try Minimal CTC Test**
Create isolated test that just calls CTC scorer without beam search:
```python
# Test script
ctc_scorer = CTCPrefixScorer(model.ctc, blank_id=0, eos_id=2)
scores, states = ctc_scorer.batch_score(test_yseqs, test_states, test_xs)
print("CTC scores:", scores)
```

**Approach C: Use ESPnet's CTCPrefixScorer**
Instead of our simplified version, import and use ESPnet's implementation:
```python
from espnet2.asr.decoder.s4_decoder import CTCPrefixScorer as ESPnetCTCScorer
```

### Priority 2: Implement extend_prob/extend_state 🟡
**Goal:** Match ESPnet's CTC integration pattern

Add to `CTCPrefixScorer`:
```python
def extend_prob(self, x: torch.Tensor):
    """Extend CTC probability matrix with new encoder output."""
    # Accumulate CTC logits across blocks
    pass

def extend_state(self, states):
    """Extend CTC states with new time steps."""
    # Update forward variables
    pass
```

### Priority 3: Compare Step-by-Step with ESPnet 🟢
**Goal:** Find exact divergence point

Run both decoders side-by-side:
1. Same input audio
2. Log scores at each step
3. Compare token selections
4. Find first mismatch

### Priority 4: Optimize CTC Scoring 🟢
**Goal:** Make CTC fast enough for real-time

Options:
- Use sparse CTC (only top-K tokens)
- Cache CTC computations
- Use batch-optimized implementation
- Consider CUDA optimization

---

## Code Changes Summary

### Files Modified
```
speechcatcher/beam_search/hypothesis.py      ✏️ Major refactor
speechcatcher/beam_search/beam_search.py     ✏️ State handling updated
speechcatcher/beam_search/scorers.py         ✏️ CTC scorer added
speechcatcher/speechcatcher.py               ✏️ CTC weight reduced
```

### Lines of Code Changed
- **hypothesis.py**: ~60 lines changed (state structure)
- **beam_search.py**: ~50 lines changed (state extraction/merging)
- **scorers.py**: ~40 lines changed (CTC implementation)
- **Total**: ~150 lines modified

### Test Coverage
- ✅ Decoder-only tested (runs but repetitive)
- ✅ CTC weight 0.3 tested (generates blanks)
- ✅ CTC weight 0.1 tested (hangs)
- ❌ No unit tests added yet

---

## Lessons Learned

### What Worked Well
1. **Systematic Comparison** - `compare.md` analysis was invaluable
2. **Step-by-Step Fixes** - Fixing state structure first was correct approach
3. **Testing After Each Change** - Caught issues early

### What Needs Improvement
1. **CTC Understanding** - Need deeper understanding of CTC implementation
2. **Debugging Tools** - Need better logging/visualization
3. **Unit Tests** - Should have tests for each component
4. **Incremental Integration** - Should have tested CTC in isolation first

### Key Insights
1. **State Management is Critical** - Wrong structure blocks everything else
2. **CTC is Complex** - Simplified version not working, may need full implementation
3. **ESPnet Compatibility** - Matching ESPnet's structure exactly is important
4. **Scoring Balance** - CTC/decoder weight balance is delicate

---

## References

### ESPnet Source Code
- `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py` - Online beam search
- `espnet_streaming_decoder/espnet/nets/batch_beam_search.py` - Base beam search
- `espnet_streaming_decoder/espnet/nets/ctc_prefix_score.py` - CTC prefix scorer

### Our Implementation
- `speechcatcher/beam_search/beam_search.py:169-333` - BSBS process_block()
- `speechcatcher/beam_search/scorers.py:152-190` - Simplified CTC scorer
- `speechcatcher/beam_search/hypothesis.py:9-40` - Hypothesis class

### Documentation
- `docs/compare.md` - Full comparison analysis
- `docs/DECODER_README.md` - Architecture documentation
- `docs/ESPNET_DECODER_ROADMAP.md` - Development roadmap

---

## Session Conclusion

**Status:** ✅ State structure fixed, ⚠️ CTC integration blocked

We successfully fixed the fundamental state management issues that were preventing proper multi-scorer integration. The Hypothesis class now matches ESPnet's structure with dict-based states, xpos tracking, and tensor-based sequences. However, CTC integration revealed a new blocking issue - the simplified CTC scorer causes the decoder to hang with certain weight configurations.

The next session should focus on **debugging the CTC hang** with detailed logging, potentially switching to ESPnet's CTC implementation, or implementing proper `extend_prob`/`extend_state` methods to match ESPnet's integration pattern.

**Branch Status:** `feat/decoder-rewrite-bsbs`
**Commit Ready:** No (CTC disabled, needs fix before commit)

---

**End of Session - 2025-10-10**
