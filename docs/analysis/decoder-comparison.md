# Deep Dive Comparison 2: Remaining Gaps Analysis

**Date:** 2025-10-10
**Previous Comparison:** [compare.md](compare.md) - Identified 7 critical issues
**Status After LLM3:** CTC runs but output is garbled (`ممم Pou</s>Vwe heißtSe,م`)

---

## Executive Summary

After implementing the full CTC algorithm and fixing state structure, CTC runs without timeout but produces incorrect output. This document identifies **5 CRITICAL GAPS** between our implementation and ESPnet that are causing the poor recognition quality.

### Root Cause

**We're not properly selecting and managing CTC states during hypothesis expansion.** ESPnet calls `select_state(state, i, new_id)` to extract the specific forward variables for each hypothesis being extended. We just copy the same state for all expansions, causing state confusion and incorrect CTC scores.

---

## Critical Gap #1: Missing select_state() 🔴

### ESPnet Implementation

**File:** `espnet_streaming_decoder/espnet/nets/scorers/ctc.py:40-63`

```python
def select_state(self, state, i, new_id=None):
    """Select state with relative ids in the main beam search.

    Args:
        state: Decoder state for prefix tokens
        i (int): Index to select a state in the main beam search
        new_id (int): New label id to select a state if necessary

    Returns:
        state: pruned state
    """
    if type(state) == tuple:
        if len(state) == 2:  # for CTCPrefixScore
            sc, st = state
            return sc[i], st[i]
        else:  # for CTCPrefixScoreTH (need new_id > 0)
            r, log_psi, f_min, f_max, scoring_idmap = state
            s = log_psi[i, new_id].expand(log_psi.size(1))
            if scoring_idmap is not None:
                return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
            else:
                return r[:, :, i, new_id], s, f_min, f_max
    return None if state is None else state[i]
```

**Key Logic:**
1. Unpack state tuple: `(r, log_psi, f_min, f_max, scoring_idmap)`
2. **Select hypothesis i:** `r[:, :, i, scoring_idmap[i, new_id]]`
3. **Select token new_id:** Extract the forward variables for this specific token
4. **Create new score:** `s = log_psi[i, new_id].expand(log_psi.size(1))`

**What this does:**
- From `r` with shape `(T, 2, n_bh, snum)`, select `[:, :, i, token_idx]` → `(T, 2)`
- This gives the forward variables for **one specific hypothesis extending with one specific token**
- The selected `r` becomes the `r_prev` for the next scoring call

### Our Implementation

**File:** `speechcatcher/beam_search/scorers.py:237-253`

```python
def select_state(self, state: Tuple, best_ids: torch.Tensor) -> Tuple:
    """Select CTC states according to beam pruning.

    Args:
        state: (r, s, f_min, f_max, scoring_idmap)
        best_ids: (batch, beam_width) indices of best hypotheses

    Returns:
        Pruned state with only selected hypotheses
    """
    if self.impl is None or state is None:
        return state

    return self.impl.index_select_state(state, best_ids)
```

**Problems:**
1. ❌ Not called during hypothesis expansion!
2. ❌ Takes `best_ids` (2D tensor) instead of `(i, new_id)` pair
3. ❌ index_select_state() is for beam pruning, not single hypothesis selection
4. ❌ Different interface than ESPnet's select_state

### Impact

**Severe:** Without selecting the correct state, ALL hypotheses get the SAME forward variables, causing:
- Incorrect CTC scores
- State confusion across hypotheses
- Wrong token selection
- Garbled output

---

## Critical Gap #2: Hypothesis Expansion Without State Selection 🔴

### ESPnet Implementation

**File:** `espnet_streaming_decoder/espnet/nets/batch_beam_search.py:294-306`

```python
# When creating new hypotheses after beam search
states=self.merge_states(
    {
        k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
        #                       ^^^^^^^^^^^^ CRITICAL CALL!
        for k, v in states.items()
    },
    {
        k: self.part_scorers[k].select_state(
            v, part_prev_hyp_id, part_new_token_id
        )
        for k, v in part_states.items()
    },
    part_new_token_id,
)
```

**Process:**
1. After scoring, we have states for all N hypotheses
2. **select_state(v, full_prev_hyp_id)**: Select the state for the specific hypothesis being extended
3. **select_state(v, part_prev_hyp_id, part_new_token_id)**: For CTC, also select the specific token
4. Each new hypothesis gets the **correct, individualized state**

### Our Implementation

**File:** `speechcatcher/beam_search/beam_search.py:146-159`

```python
# Expand hypotheses
new_hypotheses = []
for i, hyp in enumerate(beam):
    # Get top-k tokens for this hypothesis
    top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

    for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
        # Merge states from ALL scorers
        new_states_for_hyp = {}
        for scorer_name in new_states_dict:
            new_states_for_hyp[scorer_name] = new_states_dict[scorer_name][i]
            #                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #                                 JUST COPYING STATE [i]!

        new_hyp = Hypothesis(
            yseq=append_token(hyp.yseq, token),
            score=hyp.score + score,
            scores=hyp.scores.copy(),
            states=new_states_for_hyp,  # WRONG STATE!
            xpos=append_position(hyp.xpos, current_encoder_pos),
        )
        new_hypotheses.append(new_hyp)
```

**Problems:**
1. ❌ `new_states_dict[scorer_name][i]` just gets state at index i
2. ❌ No call to `scorer.select_state(state, i, token)`
3. ❌ All K expanded hypotheses from hypothesis i get the **SAME state**
4. ❌ State doesn't account for which token was selected

### Impact

**Catastrophic:** Every expanded hypothesis gets an incorrect state:
- Hypothesis i → token A: gets state[i]
- Hypothesis i → token B: gets state[i] (WRONG! Should be different)
- Hypothesis i → token C: gets state[i] (WRONG!)

This means **all expansions from the same hypothesis are indistinguishable** to the CTC scorer!

---

## Critical Gap #3: State Structure Mismatch 🔴

### ESPnet State Structure

**After batch_score_partial:**
```python
# State is a tuple of 5 elements:
state = (r, log_psi, f_min, f_max, scoring_idmap)

# Where:
# r:              (T, 2, n_bh, snum) - Forward variables for all hyps & tokens
# log_psi:        (n_bh, vocab_size) - Prefix probabilities for all hyps
# f_min, f_max:   scalars - Frame window bounds
# scoring_idmap:  (n_bh, vocab_size) - Mapping for partial scoring
```

**After select_state(state, i, new_id):**
```python
# State becomes tuple of 4 elements:
selected_state = (r_selected, s, f_min, f_max)

# Where:
# r_selected: (T, 2) - Forward variables for ONE hypothesis extending with ONE token
# s:          (vocab_size,) - Previous prefix score for this hypothesis
# f_min, f_max: scalars - Frame window bounds
```

**State Evolution:**
```
Initial (None) → batch_score → (r, log_psi, f_min, f_max, scoring_idmap) [5 elements, batched]
                              ↓ select_state(i, new_id)
                              (r_i_token, s, f_min, f_max) [4 elements, single hyp]
                              ↓ batch_score (next step)
                              (r, log_psi, f_min, f_max, scoring_idmap) [5 elements, batched again]
```

### Our State Structure

**After batch_score:**
```python
# We return the same state for all hypotheses:
scores, new_state = self.impl(y_list, state, ...)
new_states = [new_state for _ in range(len(states))]  # Same state copied!

# State is always 5 elements:
new_state = (r, log_psi, f_min, f_max, scoring_idmap)
```

**After "expansion" (no select_state):**
```python
# We just copy state[i]:
new_states_for_hyp[scorer_name] = new_states_dict[scorer_name][i]

# This is STILL the batched state:
state = (r, log_psi, f_min, f_max, scoring_idmap)  # [5 elements, still batched!]
```

**Problems:**
1. ❌ We never convert from batched state to individual hypothesis state
2. ❌ State stays 5 elements when it should become 4 after selection
3. ❌ `r` stays shape `(T, 2, n_bh, snum)` when it should become `(T, 2)` for one hyp

### Impact

**Critical:** Our state unpacking expects 4-5 elements but the structure is wrong:
```python
# In ctc_prefix_score_full.py:131
r_prev, s_prev, f_min_prev, f_max_prev, _ = state  # Unpacking 5 elements

# But r_prev has shape (T, 2, n_bh, snum) - STILL BATCHED!
# Should have shape (T, 2) for a single hypothesis!
```

This causes the dimension mismatch we worked around by resetting state to None!

---

## Critical Gap #4: batch_score vs batch_score_partial 🟡

### ESPnet Implementation

**File:** `espnet_streaming_decoder/espnet/nets/scorers/ctc.py:101-126`

```python
def batch_score_partial(self, y, ids, state, x):
    """Score new token (PARTIAL scoring - only score ids, not full vocab).

    Args:
        y (torch.Tensor): Prefix tokens (batch, prefix_len)
        ids (torch.Tensor): Candidate tokens to score (batch, n_candidates)
        state: List of decoder states, one per hypothesis
        x (torch.Tensor): Encoder output

    Returns:
        tuple[torch.Tensor, Any]: Scores and new states
    """
    # BATCH THE STATES from list to single batched state
    batch_state = (
        (
            torch.stack([s[0] for s in state], dim=2),  # Stack r along hyp dimension
            torch.stack([s[1] for s in state]),         # Stack s along hyp dimension
            state[0][2],  # f_min (shared)
            state[0][3],  # f_max (shared)
        )
        if state[0] is not None
        else None
    )
    # Call CTCPrefixScoreTH with batched state and partial scoring
    return self.impl(y, batch_state, ids)  # ids = scoring_ids (top-K tokens)
```

**Key Points:**
1. Takes **list of states** (one per hypothesis)
2. **Stacks them** into batched state along dimension 2
3. Passes **ids** for partial scoring (only score top-K candidates)
4. Returns batched state

### Our Implementation

**File:** `speechcatcher/beam_search/scorers.py:148-205`

```python
def batch_score(self, yseqs, states, xs):
    """Batch score prefixes using full CTC forward algorithm.

    Args:
        yseqs: Token sequences (batch, seq_len)
        states: List of CTC states
        xs: Encoder outputs (batch, enc_len, dim)

    Returns:
        Tuple of (log_probs, new_states)
    """
    if self.impl is None:
        self.batch_init_state(xs)

    y_list = [yseqs[i] for i in range(yseqs.size(0))]

    # TEMPORARY: Reset state (our workaround)
    merged_state = None

    # Call CTCPrefixScoreTH
    scores, new_state = self.impl(
        y=y_list,
        state=merged_state,
        scoring_ids=None,  # FULL vocabulary scoring (inefficient!)
        att_w=None
    )

    # Return SAME state for all hypotheses
    new_states = [new_state for _ in range(len(states))]

    return scores, new_states
```

**Problems:**
1. ❌ Don't batch states from list (we reset to None instead)
2. ❌ Score full vocabulary (`scoring_ids=None`) instead of top-K
3. ❌ Return same state for all hypotheses
4. ⚠️ Called `batch_score` but interface doesn't match ESPnet's `batch_score_partial`

### Impact

**Medium:** Inefficiency and incorrect state management:
- Scoring full vocabulary is O(vocab_size) instead of O(K) where K << vocab_size
- Not batching states means we reset and lose history
- Returning same state for all means state confusion

---

## Critical Gap #5: extend_state List vs Single 🟡

### ESPnet Implementation

**File:** `espnet_streaming_decoder/espnet/nets/scorers/ctc.py:141-157`

```python
def extend_state(self, state):
    """Extend state for decoding (streaming).

    Args:
        state: The states of hyps (LIST of states, one per hypothesis)

    Returns: extended state (LIST)
    """
    new_state = []
    for s in state:  # Iterate over list of states
        new_state.append(self.impl.extend_state(s))  # Extend each one

    return new_state  # Return list
```

**Key Points:**
1. `state` is a **list of states** (one per hypothesis)
2. **Iterate** and extend each one individually
3. Return **list of extended states**

**Called from:** `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py:317-318`

```python
for k, d in self.scorers.items():
    if hasattr(d, "extend_state"):
        hyps.states[k] = d.extend_state(hyps.states[k])
        # hyps.states[k] is a LIST!
```

### Our Implementation

**File:** `speechcatcher/beam_search/scorers.py:227-235`

```python
def extend_state(self, state: Optional[Tuple]) -> Optional[Tuple]:
    """Extend forward variables when probability matrix grows.

    Args:
        state: (r, log_psi, f_min, f_max, scoring_idmap) or None

    Returns:
        Extended state with r covering new time length
    """
    if self.impl is None or state is None:
        return state

    return self.impl.extend_state(state)  # Extend single state
```

**Problems:**
1. ❌ Expects single state tuple, not list
2. ❌ Returns single state, not list
3. ❌ Interface mismatch with ESPnet

**Called from:** `speechcatcher/beam_search/beam_search.py:263-273`

```python
for hyp in hypotheses:
    new_states = {}
    for scorer_name in hyp.states:
        scorer = self.scorers.get(scorer_name)
        if scorer and hasattr(scorer, "extend_state"):
            # We extend individual hypothesis state
            new_states[scorer_name] = scorer.extend_state(hyp.states[scorer_name])
```

**Our approach:**
- We iterate over hypotheses, extract each one's state, extend individually
- ESPnet: Extend list of all states at once

### Impact

**Low:** This actually works okay, just different pattern. But inconsistent with ESPnet's interface.

---

## Additional Differences

### 6. BatchHypothesis vs List[Hypothesis] ⚪

**ESPnet:** Uses batched data structure
```python
class BatchHypothesis(NamedTuple):
    yseq: torch.Tensor    # (batch, maxlen)
    xpos: torch.Tensor    # (batch, maxlen)
    score: torch.Tensor   # (batch,)
    length: torch.Tensor  # (batch,)
    scores: Dict[str, torch.Tensor]  # values: (batch,)
    states: Dict[str, List]  # List of states per scorer
```

**Ours:** Uses list of individual hypotheses
```python
@dataclass
class Hypothesis:
    yseq: torch.Tensor       # (seq_len,) - single hypothesis
    score: float             # scalar
    scores: Dict[str, float]
    states: Dict[str, Any]   # state for this one hypothesis
    xpos: torch.Tensor       # (seq_len,)
```

**Impact:** ⚪ Different structure but functionally equivalent. ESPnet's is more efficient for GPU batching.

### 7. Pre-beam Search (Partial Scoring) ⚪

**ESPnet:** Two-stage scoring
1. Score all hypotheses with decoder/CTC
2. **Pre-beam:** Select top-K tokens (e.g., K=40) based on partial scores
3. **Full scoring:** Score only those top-K with other scorers

**Ours:** Full vocabulary scoring
- Score entire vocabulary every time
- No pre-beam optimization

**Impact:** ⚪ Performance difference but not causing incorrect output.

### 8. Repetition Detection ⚪

**ESPnet:** Line 214-223 in batch_beam_search_online.py
```python
elif (
    not self.disable_repetition_detection
    and not prev_repeat
    and best.yseq[i, -1] in best.yseq[i, :-1]  # Token appeared before
    and not is_final
):
    prev_repeat = True
```
Checks if token appeared **anywhere** in the sequence before.

**Ours:** Line 382-387 in beam_search.py
```python
if len(top_hyp.yseq) >= 5:
    last_4 = top_hyp.yseq[-4:].tolist()
    if len(set(last_4)) == 1 and last_4[0] != self.sos_id:
        # Same non-SOS token repeated 4 times
        logger.warning("Repetition detected")
        break
```
Only checks last 4 consecutive tokens.

**Impact:** ⚪ Ours is less sensitive (allows more repetition). Not the main issue.

---

## Root Cause Analysis

### Why Output is Garbled

The garbled output (`ممم Pou</s>Vwe heißtSe,م`) is caused by:

1. **State Confusion (Gap #1, #2):**
   - All expansions from hypothesis i get the same CTC state
   - CTC can't distinguish between "hypothesis extending with token A" vs "hypothesis extending with token B"
   - Scores are essentially random

2. **No State Selection (Gap #2):**
   - When expanding hypothesis i with beam_size=10 tokens, we create 10 new hypotheses
   - All 10 get `new_states_dict["ctc"][i]` (same state!)
   - Should call `scorer.select_state(state, i, token)` for each token
   - Without this, CTC forward variables don't propagate correctly

3. **State Reset Workaround (Gap #3):**
   - We reset `merged_state = None` every time to avoid dimension mismatch
   - This means CTC starts fresh every decoding step
   - No accumulation of forward variables across the sequence
   - CTC can't track prefix probabilities properly

### Why Some German Words Appear

Despite the confusion, we see some correct words (`Pou`, `heißt`) because:
- Decoder scores still contribute (weight 0.7)
- CTC provides *some* acoustic grounding even with wrong states
- When CTC and decoder agree, we get the right token

### Why Arabic Characters Appear

Token 1023 (م) is likely:
- A token the decoder learned from non-German data
- Has high probability in the decoder's softmax
- CTC can't properly constrain it due to state confusion
- Repetition detection stops it after 4 repetitions

---

## Priority Fix Order

### 🔴 Priority 1: Implement select_state() in CTCPrefixScorer

**File:** `speechcatcher/beam_search/scorers.py`

**Add method:**
```python
def select_state(self, state: Optional[Tuple], i: int, new_id: int = None) -> Optional[Tuple]:
    """Select state for specific hypothesis and token.

    Args:
        state: Batched CTC state (r, log_psi, f_min, f_max, scoring_idmap)
        i: Hypothesis index in batch
        new_id: Token ID being added to this hypothesis

    Returns:
        Selected state (r_selected, s, f_min, f_max) for this hypothesis+token
    """
    if state is None:
        return None

    r, log_psi, f_min, f_max, scoring_idmap = state

    # Select hypothesis i's score for token new_id
    s = log_psi[i, new_id].expand(log_psi.size(1))

    # Select forward variables for hypothesis i, token new_id
    if scoring_idmap is not None:
        token_idx = scoring_idmap[i, new_id]
        r_selected = r[:, :, i, token_idx]  # (T, 2)
    else:
        r_selected = r[:, :, i, new_id]  # (T, 2)

    # Return 4-element tuple for individual hypothesis
    return (r_selected, s, f_min, f_max)
```

**Importance:** This is THE critical missing piece. Without this, all other fixes are pointless.

### 🔴 Priority 2: Call select_state() During Expansion

**File:** `speechcatcher/beam_search/beam_search.py:146-159`

**Current code:**
```python
for i, hyp in enumerate(beam):
    top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

    for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
        new_states_for_hyp = {}
        for scorer_name in new_states_dict:
            new_states_for_hyp[scorer_name] = new_states_dict[scorer_name][i]  # WRONG!
```

**Fix:**
```python
for i, hyp in enumerate(beam):
    top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

    for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
        new_states_for_hyp = {}
        for scorer_name in new_states_dict:
            scorer = self.scorers[scorer_name]
            state = new_states_dict[scorer_name][i]

            # Call select_state to get correct state for this hypothesis+token
            if hasattr(scorer, 'select_state'):
                new_states_for_hyp[scorer_name] = scorer.select_state(state, i, token)
            else:
                new_states_for_hyp[scorer_name] = state
```

**Importance:** This makes the select_state() actually get called!

### 🔴 Priority 3: Remove State Reset Workaround

**File:** `speechcatcher/beam_search/scorers.py:177-188`

**Current code:**
```python
# TEMPORARY: Reset state to avoid dimension mismatch
merged_state = None
```

**Fix:**
```python
# Merge states: With proper select_state, states are now (T, 2) for each hyp
# We need to batch them back to (T, 2, n_bh) for scoring
if states[0] is not None:
    # Stack states along hypothesis dimension
    merged_state = (
        torch.stack([s[0] for s in states], dim=2),  # Stack r: (T, 2, n_bh)
        torch.stack([s[1] for s in states]),          # Stack s: (n_bh, vocab)
        states[0][2],  # f_min
        states[0][3],  # f_max
    )
    # Note: This is 4 elements, but CTCPrefixScoreTH expects None or 5 elements
    # Need to adjust unpacking in ctc_prefix_score_full.py
else:
    merged_state = None
```

**Importance:** Allows state accumulation across decoding steps.

### 🟡 Priority 4: Fix State Unpacking in ctc_prefix_score_full.py

**File:** `speechcatcher/beam_search/ctc_prefix_score_full.py:116-131`

**Current code:**
```python
if state is None:
    # Initialize
    ...
else:
    r_prev, s_prev, f_min_prev, f_max_prev, _ = state  # Expects 5 elements
```

**Fix:**
```python
if state is None:
    # Initialize
    ...
else:
    # State can be 4 or 5 elements depending on where it came from
    if len(state) == 4:
        # From select_state or batched states
        r_prev, s_prev, f_min_prev, f_max_prev = state
    else:
        # From previous __call__ (5 elements)
        r_prev, s_prev, f_min_prev, f_max_prev, _ = state
```

**Importance:** Handles both state formats correctly.

### 🟡 Priority 5: Implement True extend_prob

**File:** `speechcatcher/beam_search/scorers.py:207-225`

**Current code:**
```python
def extend_prob(self, x):
    # TEMPORARY: Reinitialize
    self.impl = None
```

**Fix:**
```python
def extend_prob(self, x):
    if self.impl is None:
        self.batch_init_state(x)
    else:
        logits = self.ctc.ctc_lo(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        self.impl.extend_prob(log_probs)  # Actually extend!
```

**Importance:** Enables true streaming with accumulation across blocks.

---

## Testing Plan

### Phase 1: Unit Test select_state

```python
def test_select_state():
    # Create mock state
    r = torch.randn(10, 2, 5, 1024)  # (T=10, 2, n_bh=5, vocab=1024)
    log_psi = torch.randn(5, 1024)   # (n_bh=5, vocab=1024)
    f_min, f_max = 0, 10
    scoring_idmap = None

    state = (r, log_psi, f_min, f_max, scoring_idmap)

    # Select hypothesis 2, token 100
    selected = ctc_scorer.select_state(state, i=2, new_id=100)

    # Check result
    assert len(selected) == 4  # (r_selected, s, f_min, f_max)
    r_selected, s, f_min_out, f_max_out = selected
    assert r_selected.shape == (10, 2)  # (T, 2) - single hypothesis
    assert s.shape == (1024,)           # (vocab,) - scores for this hyp

    # Verify correct selection
    expected_r = r[:, :, 2, 100]
    assert torch.allclose(r_selected, expected_r)
```

### Phase 2: Integration Test with 5s Audio

```bash
# Test with fixed select_state
python3 -m speechcatcher.speechcatcher Neujahrsansprache_5s.mp4

# Expected: Better output than "ممم Pou</s>Vwe heißtSe,م"
# Should see more German words, fewer Arabic characters
```

### Phase 3: Compare with ESPnet Output

```python
# Run both decoders on same audio
espnet_output = run_espnet_decoder(audio)
our_output = run_our_decoder(audio)

# Compare
print(f"ESPnet: {espnet_output}")
print(f"Ours:   {our_output}")
print(f"WER: {calculate_wer(espnet_output, our_output)}")
```

### Phase 4: Longer Audio

```bash
# Test 20s, 40s clips
python3 -m speechcatcher.speechcatcher Neujahrsansprache_20s.mp4
python3 -m speechcatcher.speechcatcher Neujahrsansprache_40s.mp4
```

---

## Expected Improvements

### After Priority 1+2 (select_state)

**Before:**
```
ممم Pou</s>Vwe heißtSe,م
```

**Expected After:**
```
Liebe Mitglieder unserer Universität Hamburg...
```
(or at least much closer to correct German)

**Why:** CTC can now properly track which hypothesis is extending with which token, giving correct scores.

### After Priority 3 (Remove Reset)

**Expected:**
- Faster recognition (no recomputation)
- Better long-range dependencies
- More coherent output across blocks

### After Priority 5 (True Streaming)

**Expected:**
- Proper accumulation across blocks
- Better context from previous blocks
- True streaming capability

---

## Verification Checklist

After implementing fixes:

- [ ] select_state() implemented and tested
- [ ] select_state() called during hypothesis expansion
- [ ] State reset workaround removed
- [ ] State unpacking handles both 4 and 5 elements
- [ ] extend_prob actually extends instead of resetting
- [ ] Unit tests pass
- [ ] 5s audio produces German text
- [ ] No Arabic characters in output
- [ ] No timeout (< 5 seconds for 5s audio)
- [ ] 20s audio works correctly
- [ ] WER similar to ESPnet
- [ ] No dimension mismatch errors
- [ ] No state confusion warnings

---

## Summary Table

| Gap | Severity | Impact | Fix Complexity | Priority |
|-----|----------|--------|----------------|----------|
| #1: Missing select_state() | 🔴 Critical | Causes state confusion, incorrect scores | Medium | P1 |
| #2: No select_state() calls | 🔴 Critical | All expansions get same state | Easy | P2 |
| #3: State structure mismatch | 🔴 Critical | Dimension errors, forced reset | Medium | P3 |
| #4: batch_score_partial | 🟡 Medium | Inefficient, interface mismatch | Medium | P4 |
| #5: extend_state list vs single | 🟡 Low | Interface inconsistency | Easy | P5 |
| #6: BatchHypothesis | ⚪ Info | Structural difference only | N/A | - |
| #7: Pre-beam search | ⚪ Info | Performance only | High | - |
| #8: Repetition detection | ⚪ Info | Different algorithm | Easy | - |

---

## Next Steps

1. **Implement select_state()** - This is THE fix
2. **Call it during expansion** - Make it actually run
3. **Remove state reset** - Enable accumulation
4. **Test and iterate** - Verify improvements
5. **Compare with ESPnet** - Ensure parity

With these fixes, we should achieve **correct German transcription** matching ESPnet's quality.

---

## References

### ESPnet Source Files

1. **espnet_streaming_decoder/espnet/nets/scorers/ctc.py:40-63**
   - select_state() implementation

2. **espnet_streaming_decoder/espnet/nets/batch_beam_search.py:294-306**
   - Calling select_state during expansion

3. **espnet_streaming_decoder/espnet/nets/batch_beam_search.py:116-126**
   - Batching states from list

4. **espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py:303-318**
   - extend() method calling extend_prob and extend_state

### Our Implementation Files

1. **speechcatcher/beam_search/scorers.py:89-282**
   - CTCPrefixScorer (needs select_state)

2. **speechcatcher/beam_search/beam_search.py:146-159**
   - Hypothesis expansion (needs select_state call)

3. **speechcatcher/beam_search/ctc_prefix_score_full.py**
   - CTCPrefixScoreTH (needs state unpacking fix)

---

**End of Comparison Document**

**Key Takeaway:** The missing `select_state()` call is causing all the output quality issues. Implementing this correctly will fix the garbled output and give us proper German transcription.
