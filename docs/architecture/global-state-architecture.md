# Global State Architecture Plan: Achieving 100% ESPnet Parity

## Current Status
- Native decoder: 654 words (78.3% of ESPnet)
- ESPnet decoder: 835 words (100%)
- **Gap: 181 words (21.7%)**

**Root Cause**: Missing global state management that persists across ALL blocks throughout entire decoding session.

## ESPnet's Architecture Analysis

### Key State Variables (from batch_beam_search_online.py:90-99)
```python
def reset(self):
    """Reset parameters."""
    self.encbuffer = None           # Encoder output buffer
    self.running_hyps = None        # Current active hypotheses
    self.prev_hyps = []            # ← KEY: Previous iteration hypotheses
    self.ended_hyps = []           # Completed hypotheses (reached EOS)
    self.processed_block = 0       # Block counter
    self.process_idx = 0           # ← KEY: Position in beam search loop
    self.prev_output = None        # Previous streaming output
    self.prev_incremental = None   # Incremental decode state
```

### ESPnet's Decoding Flow

#### 1. Main Loop Structure (batch_beam_search_online.py:391-483)
```python
def process_one_block(self, h, is_final, maxlen, minlen, maxlenratio):
    # Extend CTC states with new encoder output
    self.extend(h, self.running_hyps)

    # Beam search loop - continues ACROSS blocks!
    while self.process_idx < maxlen:
        # ONE beam search step
        best = self.search(self.running_hyps, h)

        # Check for EOS/repetition
        local_ended_hyps = [...]  # Extract EOS hypotheses

        if len(local_ended_hyps) > 0 and not is_final:
            break  # ← CRITICAL: Break WITHOUT saving prev_hyps

        # Save state BEFORE updating (for potential rewind)
        self.prev_hyps = self.running_hyps

        # Update to new hypotheses
        self.running_hyps = self.post_process(...)

        # Advance position
        self.process_idx += 1

    # End of block: Check if we need to rewind
    if self.process_idx > 1 and len(self.prev_hyps) > 0:
        self.running_hyps = self.prev_hyps   # ← REWIND
        self.process_idx -= 1                # ← REWIND
        self.prev_hyps = []
```

#### 2. Rewinding Mechanism Explanation

**Normal iteration (no EOS):**
```
Iteration N:
  1. prev_hyps = running_hyps    # Save current state
  2. running_hyps = post_process() # Update to new state
  3. process_idx += 1             # Advance

Iteration N+1:
  1. prev_hyps = running_hyps    # Save (which was new state from N)
  2. running_hyps = post_process()
  3. process_idx += 1
```

**EOS detected iteration:**
```
Iteration N:
  1. prev_hyps = running_hyps    # Save current state
  2. running_hyps = post_process()
  3. process_idx += 1

Iteration N+1:
  1. Detect EOS
  2. BREAK immediately!
  3. prev_hyps still contains state from iteration N
  4. running_hyps contains state that predicted EOS (BAD!)

End of block:
  5. running_hyps = prev_hyps    # Restore to iteration N state
  6. process_idx -= 1            # Go back one step
  7. prev_hyps = []              # Clear

Next block starts with:
  - running_hyps = last GOOD state (before EOS)
  - process_idx = continues from where we left off
```

**The key insight**: `process_idx` is NOT reset between blocks! It continues counting up across the ENTIRE decoding session. This allows the decoder to know exactly where it is in the global sequence.

### ESPnet's Block Processing

```python
# From batch_beam_search_online.py:305-348
while True:  # Process blocks until done
    cur_end_frame = self.block_size - self.look_ahead + self.hop_size * self.processed_block

    if cur_end_frame < x.shape[0]:
        h = x.narrow(0, 0, cur_end_frame)
        block_is_final = False
    else:
        if is_final:
            h = x
            block_is_final = True
        else:
            break

    if self.running_hyps is None:
        self.running_hyps = self.init_hyp(h)

    # Process one block - this modifies self.running_hyps, self.prev_hyps, self.process_idx
    ret = self.process_one_block(h, block_is_final, maxlen, minlen, maxlenratio)

    self.processed_block += 1
```

## Our Current Architecture

### What We Have
```python
class BlockwiseSynchronousBeamSearch:
    def __init__(self, ...):
        self.encoder_buffer = None     # ✓ Have (matches encbuffer)
        self.processed_block = 0       # ✓ Have
        # Missing: prev_hyps, process_idx, ended_hyps

    def process_block(self, features, prev_state, is_final):
        # Process ONE block
        # prev_state contains hypotheses, but only from PREVIOUS BLOCK
        # No global state tracking!

        new_state = self._decode_one_block(encoder_out, prev_state, is_final)
        return new_state
```

### What's Missing

1. **Global Hypothesis Tracking**
   - No `self.running_hyps` - we pass hypotheses as parameters
   - No `self.prev_hyps` - can't rewind!
   - No `self.ended_hyps` - don't track completed hypotheses globally

2. **Global Position Tracking**
   - No `self.process_idx` - can't track position across blocks
   - Each block starts loop from `step = 0` instead of continuing from previous

3. **State Persistence**
   - `prev_state` is only from previous BLOCK
   - `prev_hyps` needs to be from previous ITERATION (within same block OR from previous block)

## Required Architectural Changes

### Phase 1: Add Global State Variables

**File: `speechcatcher/beam_search/beam_search.py`**

```python
class BlockwiseSynchronousBeamSearch:
    def __init__(self, ...):
        # Existing
        self.encoder_buffer = None
        self.processed_block = 0

        # NEW: Global state tracking
        self.running_hyps = None      # Current active hypotheses (persists across blocks)
        self.prev_hyps = []          # Previous iteration hypotheses (for rewinding)
        self.ended_hyps = []         # Completed hypotheses
        self.process_idx = 0         # Global position in decoding loop
        self.prev_output = None      # Previous streaming output

    def reset(self):
        """Reset streaming state between utterances."""
        self.encoder_buffer = None
        self.running_hyps = None
        self.prev_hyps = []
        self.ended_hyps = []
        self.processed_block = 0
        self.process_idx = 0
        self.prev_output = None
```

### Phase 2: Refactor process_block() to Use Global State

**Current signature:**
```python
def process_block(self, features, prev_state, is_final):
    # Returns new_state
```

**New signature:**
```python
def process_block(self, features, is_final):
    # Uses self.running_hyps instead of prev_state parameter
    # Updates self.running_hyps, self.prev_hyps, self.process_idx IN PLACE
    # Returns output for this block (if any)
```

**Changes needed:**
1. Remove `prev_state` parameter
2. Use `self.running_hyps` instead of `prev_state.hypotheses`
3. Initialize `self.running_hyps` on first call (if None)
4. Update `self.running_hyps` in place during processing

### Phase 3: Implement Global process_idx Tracking

**In _decode_one_block():**

Current:
```python
for step in range(max_decode_steps):
    # step starts from 0 each block
```

New:
```python
while self.process_idx < max_decode_len:
    # self.process_idx continues across blocks!
    # Increment at end: self.process_idx += 1
```

**Key changes:**
- Replace `for step in range(...)` with `while self.process_idx < maxlen`
- Use `self.process_idx` instead of local `step` variable
- Increment `self.process_idx += 1` at end of each iteration
- DO NOT reset `process_idx` between blocks!

### Phase 4: Implement ESPnet-Style Rewinding

**In _decode_one_block(), before each beam search step:**

```python
while self.process_idx < maxlen:
    # SAVE state before updating (for potential rewind)
    self.prev_hyps = self.running_hyps.copy()  # Deep copy!

    # Do beam search step
    best = self.search(self.running_hyps, encoder_out)

    # Update hypotheses
    self.running_hyps = self.post_process(...)

    # Check for EOS
    local_ended_hyps = [h for h in self.running_hyps if h.yseq[-1] == self.eos_id]

    if len(local_ended_hyps) > 0 and not is_final:
        # EOS detected!
        # DON'T execute prev_hyps = running_hyps again
        # Break immediately
        break

    # Increment position
    self.process_idx += 1

# End of block: Rewind if needed
if self.process_idx > 1 and len(self.prev_hyps) > 0:
    self.running_hyps = self.prev_hyps  # Restore previous good state
    self.process_idx -= 1               # Go back one step
    self.prev_hyps = []                # Clear
```

### Phase 5: Update Speech2TextStreaming to Use New API

**File: `speechcatcher/speech2text_streaming.py`**

Current:
```python
class Speech2TextStreaming:
    def transcribe_chunk(self, features):
        # Maintains prev_state as local variable
        new_state = self.beam_search.process_block(features, self.prev_state, is_final=False)
        self.prev_state = new_state
```

New:
```python
class Speech2TextStreaming:
    def transcribe_chunk(self, features):
        # No prev_state needed! Beam search tracks globally
        output = self.beam_search.process_block(features, is_final=False)
        # Return streaming output (if any)
        return output

    def reset(self):
        """Reset for new utterance."""
        self.beam_search.reset()  # Resets all global state
```

### Phase 6: Handle Hypothesis State Properly

**Challenge**: Our `Hypothesis` class is different from ESPnet's `BatchHypothesis`.

**ESPnet uses**:
- `BatchHypothesis`: Batched representation (tensors)
- Single object containing all beam hypotheses
- Efficient for batched operations

**We use**:
- `List[Hypothesis]`: List of individual hypothesis objects
- Each hypothesis has its own states dict

**Solution**:
We need to properly deep copy hypothesis states when saving to `prev_hyps`:

```python
def _copy_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
    """Deep copy hypotheses including all scorer states."""
    copied = []
    for hyp in hypotheses:
        # Deep copy states
        new_states = {}
        for scorer_name, state in hyp.states.items():
            scorer = self.scorers[scorer_name]
            if hasattr(scorer, 'copy_state'):
                new_states[scorer_name] = scorer.copy_state(state)
            else:
                # Fallback: try to deep copy
                import copy
                try:
                    new_states[scorer_name] = copy.deepcopy(state)
                except:
                    # If can't copy, reference (risky but better than nothing)
                    new_states[scorer_name] = state

        copied.append(Hypothesis(
            yseq=hyp.yseq.clone(),
            score=hyp.score,
            scores=hyp.scores.copy(),
            states=new_states,
            xpos=hyp.xpos.clone() if hyp.xpos is not None else None,
        ))
    return copied
```

Then use it:
```python
self.prev_hyps = self._copy_hypotheses(self.running_hyps)
```

## Implementation Order

### Step 1: Add State Variables (Easy)
- Add `running_hyps`, `prev_hyps`, `ended_hyps`, `process_idx` to `__init__`
- Implement `reset()` method
- **Test**: Verify initialization and reset work

### Step 2: Implement Hypothesis Copying (Medium)
- Implement `_copy_hypotheses()` helper
- Add `copy_state()` methods to scorers (CTC, decoder)
- **Test**: Verify deep copy preserves state correctly

### Step 3: Refactor process_block() API (Medium)
- Change signature to remove `prev_state` parameter
- Use `self.running_hyps` instead
- Initialize on first call
- Update Speech2TextStreaming to match
- **Test**: Verify end-to-end still works (may be 78% still)

### Step 4: Implement Global process_idx (Medium)
- Replace `for step` with `while self.process_idx < maxlen`
- Add `self.process_idx += 1` at end of loop
- Remove any `process_idx` resets between blocks
- **Test**: Verify process_idx increments correctly across blocks

### Step 5: Implement Rewinding Mechanism (Hard)
- Add `self.prev_hyps = self._copy_hypotheses(self.running_hyps)` before beam search
- Implement rewind logic at end of block
- **Test**: Verify rewinding happens when EOS detected

### Step 6: Full Integration Testing (Critical)
- Test on Neujahrsansprache_20s.mp4
- Test on segments/segment_1.mp4
- Test on full Neujahrsansprache.mp4
- **Target**: 835 words (100% parity)

## Expected Outcomes

### After Step 3 (API Refactor)
- Should maintain 78% parity
- Proves architecture changes don't break existing functionality

### After Step 4 (Global process_idx)
- May see improvement to ~85-90%
- Decoder will have better context across blocks

### After Step 5 (Rewinding)
- **Target: 95-100% parity**
- Should match ESPnet's handling of EOS states

## Testing Strategy

### Unit Tests
1. Test `_copy_hypotheses()` preserves all state
2. Test `reset()` clears all global state
3. Test `process_idx` increments correctly
4. Test rewinding logic triggers on EOS

### Integration Tests
1. **Segment-level**: Test on segments/segment_1.mp4
   - Before: 41 words
   - Target: 96 words (ESPnet baseline)

2. **Short video**: Test on Neujahrsansprache_20s.mp4
   - Before: 8-14 words
   - Target: ~28 words (ESPnet baseline)

3. **Full video**: Test on Neujahrsansprache.mp4
   - Before: 654 words (78%)
   - Target: 835 words (100%)

### Regression Tests
- Verify no performance degradation
- Verify no new artifacts (repetition, special tokens)
- Verify output quality matches or exceeds current

## Risk Mitigation

### Risk 1: State Copying Issues
**Risk**: Improper deep copy of scorer states causes bugs
**Mitigation**:
- Implement thorough `copy_state()` methods for each scorer
- Add validation tests
- Fall back to reference if copy fails (with warning)

### Risk 2: Performance Degradation
**Risk**: Deep copying hypotheses every step is slow
**Mitigation**:
- Profile to identify bottlenecks
- Only copy when necessary (right before potential break)
- Consider copy-on-write strategies if needed

### Risk 3: API Breakage
**Risk**: Changing process_block() API breaks existing code
**Mitigation**:
- Make changes incrementally
- Keep tests passing at each step
- Add compatibility layer if needed

### Risk 4: Subtle Behavioral Differences
**Risk**: Our implementation differs subtly from ESPnet
**Mitigation**:
- Add extensive logging to match ESPnet's debug output
- Compare step-by-step with ESPnet decoder
- Use diff tools to find divergence points

## Success Metrics

### Primary Metric
- **Word count on Neujahrsansprache.mp4**: 835 words (100% of ESPnet)

### Secondary Metrics
- Segment-level accuracy matches ESPnet
- No increase in special token leaking
- No increase in repetition
- Comparable inference speed

## Timeline Estimate

- **Step 1**: 1-2 hours (straightforward)
- **Step 2**: 2-3 hours (needs careful state copying)
- **Step 3**: 2-3 hours (API changes, testing)
- **Step 4**: 2-3 hours (loop refactoring)
- **Step 5**: 3-4 hours (complex rewinding logic)
- **Step 6**: 2-4 hours (comprehensive testing, debugging)

**Total**: 12-19 hours of focused implementation

## Conclusion

This plan provides a clear path to 100% ESPnet parity by implementing proper global state management. The key insight is that ESPnet's `prev_hyps` and `process_idx` are GLOBAL variables that persist across the ENTIRE decoding session, not just within a single block.

By implementing this architecture, we will:
1. ✅ Match ESPnet's exact behavior
2. ✅ Achieve 100% word count parity
3. ✅ Enable proper rewinding on EOS detection
4. ✅ Maintain clean, artifact-free output

**No shortcuts. No excuses. Let's build it right.**
