# Streaming Implementation Comparison

## ESPnet Streaming Decoder vs Our Implementation

### Key Parameters from Paper (Tsunoo et al., 2021)

Paper configuration: `{N_l, N_c, N_r} = {16, 16, 8}` frames (after 4x subsampling)
- **block_size** = 40 frames (total context per block)
- **hop_size** = 16 frames (how much we advance per block)
- **look_ahead** = 16 frames (future context)

### ESPnet Streaming Decoder Architecture

**File**: `espnet_streaming_decoder/asr_inference_streaming.py`

```python
# Line 325-331
enc, _, self.encoder_states = self.asr_model.encoder(
    feats,
    feats_lengths,
    self.encoder_states,
    is_final=is_final,
    infer_mode=True,  # ← USES infer_mode=True!
)
```

**File**: `espnet_streaming_decoder/batch_beam_search_online.py`

```python
# Lines 133-135: Block calculation
cur_end_frame = (
    self.block_size - self.look_ahead + self.hop_size * self.processed_block
)

# Block 0: 40 - 16 + 16*0 = 24 frames
# Block 1: 40 - 16 + 16*1 = 40 frames
# Block 2: 40 - 16 + 16*2 = 56 frames
```

**BBD Implementation** (Lines 209-218):
```python
# Simple repetition detection
elif (
    not self.disable_repetition_detection
    and not prev_repeat
    and best.yseq[i, -1] in best.yseq[i, :-1]  # Is last token in history?
    and not is_final
):
    prev_repeat = True
```

**Rollback** (Lines 258-261):
```python
# NON-conservative rollback (1 step)
if self.process_idx > 1 and len(self.prev_hyps) > 0:
    self.running_hyps = self.prev_hyps
    self.process_idx -= 1  # Go back ONE step
    self.prev_hyps = []
```

**EOS Detection** (Lines 230-232):
```python
if len(local_ended_hyps) > 0 and not is_final:
    logging.info("Detected hyp(s) reaching EOS in this block.")
    break
```

### Our Implementation

**File**: `speechcatcher/beam_search/beam_search.py`

```python
# Line 372-378
encoder_out, encoder_out_lens, encoder_states = self.encoder(
    features,
    feature_lens,
    prev_states=prev_state.encoder_states,
    is_final=is_final,
    infer_mode=False,  # ← WE USE infer_mode=False!
)
```

**BBD Implementation** (Lines 347-419):
```python
# Complex reliability score calculation (Equations 12-13)
r_score = max score among repetitions
s_score = alpha_next - r_score

if s_score <= 0:
    # Unreliable hypothesis
```

**Rollback** (Lines 507-513):
```python
# CONSERVATIVE rollback (2 steps) by default
if self.bbd_conservative and len(prev_step_hypotheses) > 0:
    new_state.hypotheses = prev_step_hypotheses
    new_state.output_index -= 2  # Go back TWO steps
```

### Critical Differences

| Aspect | ESPnet Streaming | Our Implementation | Impact |
|--------|------------------|-------------------|---------|
| **Encoder mode** | `infer_mode=True` | `infer_mode=False` | ❌ WRONG MODE |
| **Block size** | 40 frames (encoder output) | Not used | ❌ NO BLOCKING |
| **Hop size** | 16 frames | Not used | ❌ NO HOPPING |
| **Look ahead** | 16 frames | Not used | ❌ NO LOOKAHEAD |
| **BBD detection** | Simple: `last_token in prev_tokens` | Complex: Eq 12-13 | ⚠️ Overly complex |
| **Rollback** | 1 step (non-conservative) | 2 steps (conservative) | ⚠️ Too conservative |
| **Block boundary** | Explicit block-by-block | Per audio chunk | ❌ WRONG GRANULARITY |

### Audio Chunk Size Analysis

**Current chunk_length**: 8192 samples (0.512s at 16kHz)

**After frontend processing**:
- STFT with hop_length=160: 8192 / 160 = **51 STFT frames**
- Subsampling by 4x: 51 / 4 = **~12 encoder frames**

**Required for first block**:
- block_size - look_ahead = 40 - 16 = **24 encoder frames minimum**

**Problem**: Our chunks give us ~12 frames, but we need at least 24!

### Required Chunk Size Calculation

To get 24 encoder frames:
- 24 encoder frames × 4 (subsampling) × 160 (hop_length) = **15,360 samples**

To get 40 encoder frames (full block):
- 40 encoder frames × 4 × 160 = **25,600 samples**

**Recommended**: `chunk_length = 25600` (1.6 seconds at 16kHz)

### Action Items

1. ✅ Change encoder to `infer_mode=True`
2. ❌ Implement proper block-based processing
   - Track `cur_end_frame` based on block_size, hop_size, look_ahead
   - Process encoder output in blocks, not raw audio chunks
3. ❌ Simplify BBD to match ESPnet
   - Replace complex reliability score with simple repetition check
   - Change rollback from 2 steps to 1 step
4. ⚠️ Increase chunk_length from 8192 to 25600
   - OR process multiple audio chunks before decoding

### Why Batch Mode Works But Streaming Doesn't

**Batch mode** (`infer_mode=False`):
- Sees ENTIRE utterance at once
- No need for block boundaries
- Can look at all future context

**Streaming mode** (`infer_mode=True`):
- Processes in LIMITED blocks
- Needs explicit block boundaries
- Has to stop when context insufficient

**Our bug**: Using `infer_mode=False` (batch mode) while trying to stream!
