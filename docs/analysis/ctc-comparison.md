# Decoder Implementation Comparison

**Date:** 2025-10-10
**Purpose:** Detailed comparison between custom speechcatcher decoder and ESPnet streaming decoder
**Goal:** Identify differences that may cause repetitive token generation in custom implementation

---

## Executive Summary

This document compares the working ESPnet `BatchBeamSearchOnline` implementation with the custom speechcatcher BSBS decoder. The custom decoder produces repetitive token 1023 (Arabic م) instead of correct German text.

### Key Findings

1. **✅ Decoder batch_score is identical** - Both implementations use same logic
2. **❌ CRITICAL: State management differs** - ESPnet uses dict-based states, ours uses list-based
3. **❌ CRITICAL: Score accumulation differs** - ESPnet maintains incremental scores per scorer
4. **❌ CRITICAL: Hypothesis structure differs** - ESPnet tracks `xpos` (encoder position), we don't
5. **❌ CTC integration disabled** - Our CTC scorer is commented out due to timeout
6. **⚠️ Hypothesis expansion simplified** - We simplified the state extraction logic

---

## 1. Hypothesis Data Structure

###  ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/beam_search.py:13-28`

```python
class Hypothesis(NamedTuple):
    """Hypothesis data type."""
    yseq: torch.Tensor      # Token sequence (NOT a list!)
    score: float           # Total score
    scores: Dict[str, float]  # Per-scorer scores
    states: Dict[str, Any]    # Per-scorer states (DICT!)
    xpos: torch.Tensor     # Encoder position tracking (IMPORTANT!)
```

### Speechcatcher Implementation

**Location:** `speechcatcher/beam_search/hypothesis.py:9-36`

```python
@dataclass
class Hypothesis:
    """Single hypothesis in beam search."""
    yseq: List[int]         # Token sequence (List, not tensor!)
    score: float
    scores: Dict[str, float]
    states: Optional[List[torch.Tensor]]  # Just a list, not dict!
    yseq_tensor: Optional[torch.Tensor]   # Redundant
    # MISSING: xpos - encoder position tracking!
```

### **🔴 CRITICAL DIFFERENCE #1: States Structure**

- **ESPnet:** `states: Dict[str, Any]` - Separate state per scorer
  - Example: `{"decoder": [...], "ctc": {...}}`
- **Ours:** `states: Optional[List[torch.Tensor]]` - Single list
  - Only holds ONE scorer's state!

**Impact:** When we have multiple scorers (decoder + CTC), we can only store decoder states, losing CTC state context!

### **🔴 CRITICAL DIFFERENCE #2: Missing xpos**

ESPnet tracks encoder positions with `xpos`:
```python
xpos: torch.Tensor  # Encoder frame positions for each token
```

We don't track this at all. This could affect:
- CTC alignment
- Encoder context windowing
- Block boundary detection

---

## 2. Batch Hypothesis Structure

### ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/batch_beam_search.py:15-28`

```python
class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""
    yseq: torch.Tensor      # (batch, maxlen)
    xpos: torch.Tensor      # (batch, maxlen) - encoder positions!
    score: torch.Tensor     # (batch,)
    length: torch.Tensor    # (batch,)
    scores: Dict[str, torch.Tensor]   # values: (batch,) per scorer
    states: Dict[str, Dict]           # Nested dict structure!
```

### Speechcatcher Implementation

**We don't have a BatchHypothesis class!** We just use `List[Hypothesis]` everywhere.

**Impact:**
- No vectorization of hypothesis operations
- More Python loops instead of tensor operations
- Less efficient

---

## 3. Decoder batch_score Method

### ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/pytorch_backend/transformer/decoder.py:301-336`

```python
def batch_score(self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
    """Score new token batch."""
    n_batch = len(ys)
    n_layers = len(self.decoders)

    # Merge states: [batch, layer] -> [layer, batch]
    if states[0] is None:
        batch_state = None
    else:
        batch_state = [
            torch.stack([states[b][i] for b in range(n_batch)])
            for i in range(n_layers)
        ]

    # Batch decoding
    ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
    logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

    # Transpose state: [layer, batch] -> [batch, layer]
    state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
    return logp, state_list
```

### Speechcatcher Implementation

**Location:** `speechcatcher/model/decoder/transformer_decoder.py:275-312`

```python
def batch_score(self, ys: torch.Tensor, states: List[Optional[List[torch.Tensor]]],
        xs: torch.Tensor) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """Batch scoring interface for beam search."""
    n_batch = len(ys)
    n_layers = len(self.decoders)

    # Merge states: transpose [batch, layer] -> [layer, batch]
    if states[0] is None:
        batch_state = None
    else:
        batch_state = [
            torch.stack([states[b][i] for b in range(n_batch)])
            for i in range(n_layers)
        ]

    # Batch decoding
    ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
    logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

    # Transpose state: [layer, batch] -> [batch, layer]
    state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]

    return logp, state_list
```

### **✅ IDENTICAL!**

The decoder scoring logic is the same. **This is NOT the source of the problem.**

---

## 4. Beam Search - Scoring Phase

### ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/batch_beam_search.py:150-200`

```python
def score_full(self, hyp: BatchHypothesis, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Score new hypothesis by full_scorers."""
    scores = dict()
    states = dict()

    for k, d in self.full_scorers.items():
        scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
        # ^^ Uses hyp.states[k] - gets state for SPECIFIC scorer!

    return scores, states

def search(self, running_hyps: BatchHypothesis, x: torch.Tensor):
    """Search new tokens."""
    # ...
    scores, states = self.score_full(running_hyps, x.expand(...))

    for k in self.full_scorers:
        weighted_scores += self.weights[k] * scores[k]
    # ^^ Weighted combination
```

### Speechcatcher Implementation

**Location:** `speechcatcher/beam_search/beam_search.py:59-114`

```python
def batch_score_hypotheses(self, hypotheses: List[Hypothesis],
        encoder_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List]]:
    """Score all hypotheses for next token prediction."""
    # ...

    # Score with each scorer
    for scorer_name, scorer in self.scorers.items():
        weight = self.weights.get(scorer_name, 0.0)

        # Extract states for this scorer
        states = [h.states for h in hypotheses]
        # ^^ PROBLEM: h.states is a List, not a Dict!
        #    We're passing the ENTIRE state list to each scorer!

        scores, new_states = scorer.batch_score(yseqs, states, encoder_out_batch)
        all_new_states[scorer_name] = new_states
        combined_scores += weight * scores

    return combined_scores, all_new_states
```

### **🔴 CRITICAL DIFFERENCE #3: State Extraction**

**ESPnet:**
```python
states = [h.states[k] for h in hypotheses]  # Get state for SPECIFIC scorer k
```

**Ours:**
```python
states = [h.states for h in hypotheses]  # Get ALL states (just a list!)
```

**Impact:**
- We're passing the same states to ALL scorers!
- Decoder gets decoder states (OK)
- CTC would get decoder states (WRONG!)
- State confusion between scorers!

---

## 5. Hypothesis Expansion

### ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/batch_beam_search.py:273-308`

```python
# In search() method:
for (full_prev_hyp_id, full_new_token_id,
     part_prev_hyp_id, part_new_token_id) in zip(*self.batch_beam(...)):

    prev_hyp = prev_hyps[full_prev_hyp_id]

    best_hyps.append(
        Hypothesis(
            score=weighted_scores[full_prev_hyp_id, full_new_token_id],
            yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
            xpos=self.append_token(prev_hyp.xpos, xpos_local),  # Track position!
            scores=self.merge_scores(
                prev_hyp.scores,
                {k: v[full_prev_hyp_id] for k, v in scores.items()},  # Incremental
                full_new_token_id,
                {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                part_new_token_id,
            ),
            states=self.merge_states(
                {k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                 for k, v in states.items()},  # Per-scorer state selection
                {k: self.part_scorers[k].select_state(v, part_prev_hyp_id, part_new_token_id)
                 for k, v in part_states.items()},
                part_new_token_id,
            ),
        )
    )
```

### Speechcatcher Implementation

**Location:** `speechcatcher/beam_search/beam_search.py:292-311`

```python
# In process_block() method:
for i, hyp in enumerate(new_state.hypotheses):
    top_scores, top_tokens = torch.topk(scores[i], self.beam_size)

    for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
        # Use the NEW states from scoring
        new_states_for_hyp = None
        if new_states_dict:
            # Get first scorer's states (typically 'decoder')
            scorer_name = list(new_states_dict.keys())[0]
            new_states_for_hyp = new_states_dict[scorer_name][i]
            # ^^ PROBLEM: Only uses FIRST scorer's state!

        new_hyp = Hypothesis(
            yseq=hyp.yseq + [token],
            score=hyp.score + score,  # Simple addition
            scores=hyp.scores.copy(),  # Just copy, no merge!
            states=new_states_for_hyp,  # Only one scorer's state!
        )
        new_hypotheses.append(new_hyp)
```

### **🔴 CRITICAL DIFFERENCE #4: State Selection**

**ESPnet:**
- Selects state from EACH scorer separately
- Uses `select_state(v, full_prev_hyp_id)` method
- Merges states from multiple scorers
- Tracks encoder position (`xpos`)

**Ours:**
- Only takes first scorer's state!
- No state merging
- No encoder position tracking
- **Only one scorer can have state at a time!**

---

## 6. Score Management

### ESPnet Implementation

```python
class Hypothesis(NamedTuple):
    scores: Dict[str, float]  # {"decoder": -1.2, "ctc": -0.8, "lm": -0.3}

def merge_scores(prev_scores, full_scores, full_id, part_scores, part_id):
    """Merge scores incrementally."""
    new_scores = dict(prev_scores)  # Start with previous
    for k in full_scores:
        new_scores[k] = full_scores[k]  # Update with new scores
    # ... merge partial scores
    return new_scores
```

**Score update:**
```python
score=weighted_scores[full_prev_hyp_id, full_new_token_id]  # Pre-computed total
scores=self.merge_scores(...)  # Keep individual scores
```

### Speechcatcher Implementation

```python
@dataclass
class Hypothesis:
    scores: Dict[str, float] = field(default_factory=dict)  # Same structure

# But in hypothesis expansion:
new_hyp = Hypothesis(
    score=hyp.score + score,      # Add raw combined score
    scores=hyp.scores.copy(),     # Just copy old scores, no update!
    # ...
)
```

### **🔴 CRITICAL DIFFERENCE #5: Score Tracking**

**ESPnet:**
- Maintains individual scorer scores throughout
- Updates scores incrementally with each token
- Total score recomputed from weighted sum

**Ours:**
- Scores dict never updated after initial creation!
- Just accumulate combined scores
- Individual scorer contributions lost

---

## 7. CTC Integration

### ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py:303-318`

```python
def extend(self, x: torch.Tensor, hyps: Hypothesis):
    """Extend probabilities and states with more encoded chunks."""
    for k, d in self.scorers.items():
        if hasattr(d, "extend_prob"):
            d.extend_prob(x)  # Extend CTC probability matrix
        if hasattr(d, "extend_state"):
            hyps.states[k] = d.extend_state(hyps.states[k])  # Extend CTC state
```

**Called in `process_one_block()`:**
```python
def process_one_block(self, h, is_final, maxlen, maxlenratio):
    # extend states for ctc
    self.extend(h, self.running_hyps)  # <-- Called BEFORE decoding!

    while self.process_idx < maxlen:
        best = self.search(self.running_hyps, h)
        # ...
```

### Speechcatcher Implementation

**Location:** `speechcatcher/beam_search/beam_search.py:398-403`

```python
# CTC prefix scoring with proper forward algorithm
# Disabled temporarily to debug timeout
# if model.ctc is not None and ctc_weight > 0:
#     scorers["ctc"] = CTCPrefixScorer(model.ctc, blank_id=0, eos_id=2)
#     weights["ctc"] = ctc_weight
```

**No `extend()` method!** CTC is completely disabled.

### **🔴 CRITICAL DIFFERENCE #6: CTC Missing**

**ESPnet:**
- CTC is active and properly extended with new encoder blocks
- CTC state maintained across blocks
- CTC scores guide beam search

**Ours:**
- **CTC is disabled** due to timeout issues
- Decoder runs without CTC constraint
- Beam search has NO acoustic model guidance!

**Impact:** Without CTC, the decoder can drift into repetitive local optima because there's no acoustic grounding!

---

## 8. Repetition Detection

### ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py:214-223`

```python
elif (
    not self.disable_repetition_detection
    and not prev_repeat
    and best.yseq[i, -1] in best.yseq[i, :-1]  # Check if last token repeats
    and not is_final
):
    prev_repeat = True

if prev_repeat:
    logging.info("Detected repetition.")
    break  # Stop decoding this block
```

**Rollback logic (lines 267-270):**
```python
if self.process_idx > 1 and len(self.prev_hyps) > 0:
    self.running_hyps = self.prev_hyps  # Restore previous hypotheses
    self.process_idx -= 1  # Step back
    self.prev_hyps = []
```

### Speechcatcher Implementation

**Location:** `speechcatcher/beam_search/beam_search.py:321-328`

```python
# Repetition detection: stop if top hypothesis has same token repeated 4+ times
top_hyp = new_state.hypotheses[0]
if len(top_hyp.yseq) >= 5:
    last_4 = top_hyp.yseq[-4:]
    if len(set(last_4)) == 1 and last_4[0] != self.sos_id:
        logger.warning(f"Repetition detected: token {last_4[0]} repeated 4 times")
        break  # Stop block, but NO ROLLBACK!
```

### **🔴 CRITICAL DIFFERENCE #7: No Rollback**

**ESPnet:**
- Detects ANY repetition (even 1 repeat)
- Rolls back to previous hypotheses
- Waits for next block
- More conservative

**Ours:**
- Only detects 4+ consecutive repeats
- Stops block but doesn't rollback
- No "wait for more context" mechanism
- Less effective

---

## 9. Block Processing Logic

### ESPnet Implementation

**Location:** `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py:132-177`

```python
def forward(self, x, maxlenratio=0.0, minlenratio=0.0, is_final=True):
    # Accumulate encoder output
    if self.encbuffer is None:
        self.encbuffer = x
    else:
        self.encbuffer = torch.cat([self.encbuffer, x], axis=0)

    x = self.encbuffer  # Work with full accumulated buffer

    while True:
        cur_end_frame = (
            self.block_size - self.look_ahead +
            self.hop_size * self.processed_block
        )

        if cur_end_frame < x.shape[0]:
            h = x.narrow(0, 0, cur_end_frame)  # Extract up to current block
            block_is_final = False
        else:
            if is_final:
                h = x  # Use all accumulated frames
                block_is_final = True
            else:
                break  # Wait for more data

        # Process block
        ret = self.process_one_block(h, block_is_final, maxlen, maxlenratio)
        self.processed_block += 1

        if block_is_final:
            return ret
```

### Speechcatcher Implementation

**Location:** `speechcatcher/beam_search/beam_search.py:332-367`

```python
def recognize_stream(self, features, feature_lens):
    """Recognize speech from streaming features (non-incremental version)."""
    state = None
    total_frames = feature_lens[0].item()
    current_frame = 0

    while current_frame < total_frames:
        # Extract block
        end_frame = min(current_frame + self.block_size, total_frames)
        block = features[:, current_frame:end_frame, :]
        block_lens = torch.tensor([block.size(1)], device=self.device)

        is_final = (end_frame >= total_frames)

        # Process block (encoder is called inside process_block)
        state = self.process_block(block, block_lens, state, is_final)

        current_frame += self.hop_size

    return state.hypotheses if state else []
```

### **⚠️ DIFFERENCE #8: Buffer Management**

**ESPnet:**
- Accumulates encoder output in `encbuffer`
- Decoder sees ALL encoder frames up to current block
- CTC can look at entire history

**Ours:**
- Processes raw features block-by-block
- Encoder processes each block separately with states
- Decoder only sees current block's encoder output

**Impact:** Our approach is actually more memory efficient, but requires proper encoder state management.

---

## 10. Maxlen Calculation

### ESPnet Implementation

```python
def process_one_block(self, h, is_final, maxlen, maxlenratio):
    # maxlen passed from forward():
    # maxlen = max(1, int(maxlenratio * x.size(0)))
    # or maxlen = x.shape[0] if maxlenratio == 0

    while self.process_idx < maxlen:
        # Decode up to maxlen tokens
        # ...
```

### Speechcatcher Implementation

**Location:** `speechcatcher/beam_search/beam_search.py:280-284`

```python
if encoder_out.size(1) > 0:
    # Determine max decoding length based on encoder output
    # Use conservative estimate: ~1 token per 8 encoder frames
    maxlen = min(max(encoder_out.size(1) // 8, 1), 20)  # Cap at 20 tokens

    for step in range(maxlen):
        # ...
```

### **⚠️ DIFFERENCE #9: Maxlen Estimation**

**ESPnet:**
- Uses maxlenratio (configurable ratio)
- or uses encoder length directly if maxlenratio=0
- More flexible

**Ours:**
- Hardcoded: 1 token per 8 frames
- Capped at 20 tokens per block
- May be too restrictive or too generous

---

## Summary of Critical Issues

| Issue | ESPnet | Speechcatcher | Impact |
|-------|--------|---------------|---------|
| **States structure** | Dict per scorer | Single list | **CRITICAL**: Can't handle multiple scorers |
| **State extraction** | Per-scorer selection | First scorer only | **CRITICAL**: State confusion |
| **Score tracking** | Incremental per scorer | Copy only | **HIGH**: Loses score breakdown |
| **xpos tracking** | Tracked | Missing | **MEDIUM**: No encoder position info |
| **CTC integration** | Active + extended | Disabled | **CRITICAL**: No acoustic grounding |
| **Repetition handling** | Rollback mechanism | No rollback | **HIGH**: Can't recover from repetition |
| **Hypothesis type** | BatchHypothesis | List only | **MEDIUM**: Less efficient |
| **Score merging** | Proper merge logic | No merging | **HIGH**: Accumulation errors |

---

## Root Cause Analysis

### Why Token 1023 Repeats

**Primary Cause:**
1. **No CTC constraint** - Decoder has no acoustic model guidance
2. **Improper state management** - Only first scorer's state used
3. **No score breakdown** - Can't track individual scorer contributions
4. **No rollback** - Once stuck in repetition, can't escape

**Cascade Effect:**
```
No CTC → Decoder unconstrained
    ↓
Decoder finds local optimum (token 1023 has high prior)
    ↓
No repetition rollback → Continues generating 1023
    ↓
State confusion → Decoder state doesn't properly track context
    ↓
Score accumulation errors → Wrong tokens preferred
    ↓
Repetitive output!
```

---

## Recommended Fixes (Priority Order)

### 1. **Fix State Structure (CRITICAL)**

**Change `Hypothesis.states` from `List` to `Dict`:**

```python
@dataclass
class Hypothesis:
    yseq: List[int]
    score: float
    scores: Dict[str, float]
    states: Dict[str, Any]  # Changed from List!
    xpos: List[int] = field(default_factory=list)  # Add xpos tracking
```

**Update state extraction in beam search:**
```python
# In batch_score_hypotheses:
for scorer_name, scorer in self.scorers.items():
    states = [h.states.get(scorer_name) for h in hypotheses]  # Get per-scorer state
    scores, new_states = scorer.batch_score(yseqs, states, encoder_out_batch)
    all_new_states[scorer_name] = new_states
```

### 2. **Fix Hypothesis Expansion (CRITICAL)**

**Merge ALL scorer states:**
```python
# In process_block expansion:
new_states_for_hyp = {}
for scorer_name in new_states_dict:
    new_states_for_hyp[scorer_name] = new_states_dict[scorer_name][i]

new_hyp = Hypothesis(
    yseq=hyp.yseq + [token],
    score=hyp.score + score,
    scores={scorer_name: score for scorer_name in self.scorers},  # Update all scores
    states=new_states_for_hyp,  # All scorer states
    xpos=hyp.xpos + [current_encoder_position],  # Track position
)
```

### 3. **Re-enable CTC with Optimization (CRITICAL)**

**Option A: Use simplified CTC scoring (fast)**
```python
# Use frame-averaged CTC instead of full prefix scoring
class SimplifiedCTCScorer:
    def batch_score(self, yseqs, states, xs):
        # Get CTC log probs
        logits = self.ctc.ctc_lo(xs)
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch, time, vocab)

        # Average across time (instead of max-pool)
        avg_log_probs = log_probs.mean(dim=1)  # (batch, vocab)

        return avg_log_probs, states
```

**Option B: Fix proper CTC prefix scorer**
- Reuse scorer instance (don't recreate each batch)
- Use incremental forward variables
- Cache across blocks

### 4. **Add Repetition Rollback (HIGH)**

```python
# In process_block:
prev_hypotheses = new_state.hypotheses.copy()  # Save before expansion

# After expansion, check repetition:
if repetition_detected:
    logger.warning("Repetition detected, rolling back")
    new_state.hypotheses = prev_hypotheses  # Rollback
    break  # Wait for next block
```

### 5. **Add xpos Tracking (MEDIUM)**

Track encoder frame position for each token:
```python
current_encoder_pos = encoder_out.size(1)  # Current frame
new_hyp.xpos = hyp.xpos + [current_encoder_pos]
```

### 6. **Implement BatchHypothesis (LOW)**

Create vectorized batch operations for efficiency (optional optimization).

---

## Testing Strategy

### Phase 1: Fix State Management
1. Change `Hypothesis.states` to Dict
2. Update all state extraction code
3. Test with decoder-only (no CTC)
4. **Expected:** Should still run, possibly different output

### Phase 2: Re-enable Simple CTC
1. Implement SimplifiedCTCScorer (frame-averaged)
2. Add CTC with weight 0.3
3. Test on 5s audio
4. **Expected:** Output should change, hopefully less repetitive

### Phase 3: Add Rollback
1. Implement repetition detection + rollback
2. Test on 20s audio
3. **Expected:** Should stop repetition within blocks

### Phase 4: Compare with ESPnet
1. Run both decoders on same input
2. Compare token sequences at each step
3. Identify first divergence point
4. **Expected:** Should match ESPnet output

---

## Appendix: Code Locations

### ESPnet Key Files
- **Beam search:** `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py`
- **Base batch search:** `espnet_streaming_decoder/espnet/nets/batch_beam_search.py`
- **Hypothesis:** `espnet_streaming_decoder/espnet/nets/beam_search.py:13-28`
- **Decoder:** `espnet_streaming_decoder/espnet/nets/pytorch_backend/transformer/decoder.py`

### Speechcatcher Key Files
- **Beam search:** `speechcatcher/beam_search/beam_search.py`
- **Hypothesis:** `speechcatcher/beam_search/hypothesis.py`
- **Scorers:** `speechcatcher/beam_search/scorers.py`
- **Decoder:** `speechcatcher/model/decoder/transformer_decoder.py`

---

**End of Comparison Document**
