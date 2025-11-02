# ROOT CAUSE: Missing Encoder Buffer in Beam Search

## Summary

**We found the root cause!** Our implementation is missing the critical encoder output buffering mechanism that ESPnet uses.

## The Problem

### What ESPnet Does (CORRECT):

```python
# batch_beam_search_online.py:118-144
def forward(self, x, is_final=True):
    # 1. ACCUMULATE encoder outputs in buffer
    if self.encbuffer is None:
        self.encbuffer = x
    else:
        self.encbuffer = torch.cat([self.encbuffer, x], axis=0)

    # 2. EXTRACT blocks from buffer
    while True:
        cur_end_frame = (
            self.block_size - self.look_ahead +
            self.hop_size * self.processed_block
        )
        if cur_end_frame < self.encbuffer.shape[0]:
            h = self.encbuffer.narrow(0, 0, cur_end_frame)  # Extract block
            self.process_one_block(h, ...)
            self.processed_block += 1
        else:
            break
```

**Block extraction formula:**
- Block 0: frames [0, 24)  (block_size - look_ahead = 40 - 16)
- Block 1: frames [0, 40)  (24 + 16*1)
- Block 2: frames [0, 56)  (24 + 16*2)
- Block 3: frames [0, 72)  (24 + 16*3)

### What We Do (WRONG):

```python
# beam_search.py:process_block()
def process_block(self, features, is_final):
    # 1. Directly encode the current chunk
    encoder_out, encoder_out_lens, encoder_states = self.encoder(
        features, ..., prev_states=prev_state.encoder_states
    )

    # 2. Directly use encoder output (NO BUFFERING!)
    # This gives us different sizes each time: 0, 24, 16, 0, 16, ...
    scores, new_states_dict = self.beam_search.batch_score_hypotheses(
        hypotheses, encoder_out  # ← WRONG! Not extracted from buffer!
    )
```

## Evidence from Test Output

```
Chunk 4/11:
  ESPnet: encbuf=24  ← First encoder output
  Ours:   enc_out=24 ← Same

Chunk 5/11:
  ESPnet: encbuf=40  ← Buffer grew! (24 + 16 hop)
  Ours:   enc_out=16 ← Wrong! Encoder output reset

  ESPnet: "liebe" (correct German!)
  Ours:   "م" (Arabic garbage)
```

**ESPnet's encbuffer grows**: 0 → 24 → 40 → 56 → 72 → 88
**Our enc_out fluctuates**: 0 → 24 → 16 → 0 → 16 → 0

## Why This Causes Token 1023 (Arabic 'م')

1. **Insufficient context**: Decoder sees only 16 frames instead of 40
2. **Truncated attention**: Can't attend to full context window
3. **Poor predictions**: Model trained on 40-frame blocks, gets 16-frame blocks
4. **Fallback behavior**: Predicts high-frequency token (1023) when confused

## The Fix

We need to implement encoder output buffering in `BlockwiseSynchronousBeamSearch`:

```python
class BlockwiseSynchronousBeamSearch:
    def __init__(self, ...):
        self.encoder_buffer = None  # Accumulated encoder outputs
        self.processed_block = 0

    def process_block(self, features, is_final):
        # 1. Encode features
        encoder_out = self.encoder(features, ...)

        # 2. ACCUMULATE in buffer
        if self.encoder_buffer is None:
            self.encoder_buffer = encoder_out
        else:
            self.encoder_buffer = torch.cat([self.encoder_buffer, encoder_out], dim=1)

        # 3. EXTRACT block(s) from buffer
        while True:
            cur_end_frame = (
                self.block_size - self.look_ahead +
                self.hop_size * self.processed_block
            )

            if cur_end_frame <= self.encoder_buffer.shape[1]:
                # Extract block: [0, cur_end_frame)
                block = self.encoder_buffer[:, :cur_end_frame, :]

                # Decode this block
                self.decode_block(block, is_final=False)
                self.processed_block += 1
            else:
                break

        # 4. If final, decode remaining buffer
        if is_final and self.encoder_buffer.shape[1] > 0:
            self.decode_block(self.encoder_buffer, is_final=True)
```

## Architecture Comparison

### ESPnet Architecture (CORRECT):

```
Audio Chunks → Encoder → [BUFFER] → Block Extractor → Decoder
    ↓              ↓         ↓             ↓              ↓
 8000 samples   Varies    Grows      Fixed blocks    Good output
 (0.5s)         0→24→16   0→24→40    24,40,56...     "Liebe..."
```

### Our Architecture (WRONG):

```
Audio Chunks → Encoder → [NO BUFFER] → Decoder
    ↓              ↓                       ↓
 25600 samples  Varies                   Bad output
 (1.6s)         0→24→16                  "م..."
```

## Next Steps

1. ✅ Identified root cause: Missing encoder buffer
2. ⏳ Implement encoder buffering in `BlockwiseSynchronousBeamSearch`
3. ⏳ Implement block extraction logic
4. ⏳ Test with multi-chunk streaming
5. ⏳ Verify output matches ESPnet: "Liebe Mitglieder..."

## Key Parameters

From config.yaml and test output:
- `block_size`: 40 frames (encoder output frames)
- `hop_size`: 16 frames (how much to advance per block)
- `look_ahead`: 16 frames (future context)
- First block: frames [0, 24)
- Second block: frames [0, 40)
- Hop between blocks: 16 frames

## Impact

This explains EVERYTHING:
- ✅ Why batch mode works (sees full context)
- ✅ Why streaming mode fails (insufficient context)
- ✅ Why token 1023 appears (model confused by short context)
- ✅ Why BBD detects repetitions (model trying to end early)
- ✅ Why ESPnet streaming works (proper buffering)
