# Streaming Flow Comparison: ESPnet vs Ours

## Overview

This document compares the execution flow when processing a single audio chunk through:
- **ESPnet**: `espnet_streaming_decoder/asr_inference_streaming.py`
- **Ours**: `speechcatcher/speech2text_streaming.py`

---

## High-Level Flow

```
Audio Chunk (numpy array)
    ↓
__call__() entry point
    ↓
Frontend (STFT + Mel + Log)
    ↓
Normalization
    ↓
Encoder
    ↓
Buffer Management
    ↓
Beam Search / Decoder
    ↓
Results (text, tokens, token_ids)
```

---

## ESPnet Flow (espnet_streaming_decoder)

### 1. Entry Point: `__call__(speech, is_final)`

**File**: `espnet_streaming_decoder/asr_inference_streaming.py`

```
Speech2TextStreaming.__call__(speech, is_final)
  ↓
[Line 316-320] Convert to tensor, move to device
  speech = torch.from_numpy(speech).to(device)
  ↓
[Line 322-326] Apply frontend with buffering
  feats, feats_len, self.frontend_states = self.apply_frontend(
      speech, self.frontend_states, is_final
  )
  ↓
[Line 328-330] Early return if not enough samples
  if feats is None:
      return []
```

### 2. Frontend: `apply_frontend(speech, prev_states, is_final)`

**File**: `espnet_streaming_decoder/asr_inference_streaming.py:206-300`

```
apply_frontend(speech, prev_states, is_final)
  ↓
[Line 208-210] Concatenate with previous waveform buffer
  if prev_states and "waveform_buffer" in prev_states:
      speech = torch.cat([prev_states["waveform_buffer"], speech])
  ↓
[Line 213-227] Check if enough samples (> win_length)
  has_enough_samples = speech.size(0) > self.win_length
  if not has_enough_samples:
      if is_final:
          pad with zeros
      else:
          buffer and return None  # ← SKIP EVERYTHING
  ↓
[Line 229-243] Calculate STFT frames and buffer residual
  n_frames = (speech.size(0) - (win_length - hop_length)) // hop_length
  process_length = (win_length - hop_length) + n_frames * hop_length
  waveform_buffer = speech[buffer_start:]  # Keep overlap
  ↓
[Line 245-249] STFT + Mel + Log (frontend)
  feats, feats_len = self.asr_model.encode(speech_to_process)
  ↓
[Line 251-262] Feature normalization
  feats = (feats - self.mean) / self.std
  ↓
[Line 264-299] Frame trimming (remove overlaps at chunk boundaries)
  trim_frames = ceil(ceil(win_length / hop_length) / 2)

  if is_final:
      if prev_states is None:
          # First and only chunk - no trimming
      else:
          # Trim beginning (keep end)
          feats = feats[:, trim_frames:]
  else:
      if prev_states is None:
          # First chunk - trim end only
          feats = feats[:, :-trim_frames]
      else:
          # Middle chunk - trim both ends
          feats = feats[:, trim_frames:-trim_frames]
  ↓
Return (feats, feats_len, next_states)
```

### 3. Encoder: `asr_model.encoder(feats, feats_len, prev_states, is_final, infer_mode=True)`

**File**: ESPnet's ContextualBlockTransformerEncoder

```
encoder(feats, feats_len, prev_states, is_final, infer_mode=True)
  ↓
[Internal] Use prev_states to maintain context
  buffer_before_downsampling (from prev_states)
  prev_addin (from prev_states)
  past_encoder_ctx (from prev_states)
  ↓
[Internal] Streaming subsampling with buffering
  Process features with context from previous chunks
  ↓
[Internal] Transformer layers with streaming
  Uses prev_states for continuity
  ↓
Return (encoder_out, encoder_out_lens, encoder_states)

Produces incremental outputs:
  Chunk 4: 24 frames
  Chunk 5: 16 frames
  Chunk 6: 16 frames
```

### 4. Beam Search: `beam_search(encoder_out, encoder_out_lens, running_hyps)`

**File**: `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py`

```
BatchBeamSearchOnline.__call__(encoder_out, encoder_out_lens, running_hyps)
  ↓
[Line 118-121] Accumulate encoder output in buffer
  if encoder_out.size(1) > 0:
      if self.encbuf is None:
          self.encbuf = encoder_out
      else:
          self.encbuf = torch.cat([self.encbuf, encoder_out], dim=1)
  ↓
[Line 132-168] Extract and decode blocks
  while True:
      cur_end_frame = block_size - look_ahead + hop_size * processed_block

      if cur_end_frame <= self.encbuf.size(1):
          # Extract block: encbuf[:, :cur_end_frame]
          block_encoder_out = self.encbuf.narrow(1, 0, cur_end_frame)
          ↓
          [Line 141-167] Decode one block
          self.running_hyps = self._decode_one_block(
              block_encoder_out, self.running_hyps, False
          )
          ↓
          self.processed_block += 1
      else:
          break  # Wait for more encoder output
  ↓
Return running_hyps
```

### 5. Decode One Block: `_decode_one_block(encoder_out, running_hyps, is_final)`

**File**: `espnet_streaming_decoder/espnet/nets/batch_beam_search_online.py:172-223`

```
_decode_one_block(encoder_out, running_hyps, is_final)
  ↓
[Line 177-186] Extend scorers (CTC, decoder)
  running_hyps = self.extend_scorer(encoder_out, running_hyps)
  ↓
[Line 188-223] Beam search loop
  for step in range(max_length):
      ↓
      [Line 191-196] Score all hypotheses
      scores = self.score_full(running_hyps, encoder_out)
      ↓
      [Line 198-203] Prune to beam size
      running_hyps = self.prune_beam(running_hyps, scores, beam_size)
      ↓
      [Line 209-218] BBD: Check for repetition or EOS
      if not is_final:
          has_repetition = self.detect_repetition_or_eos(running_hyps)
          if has_repetition:
              # Rollback 1 step
              running_hyps = self.rollback_hyps(running_hyps, 1)
              break  # Stop decoding this block
      ↓
      [Line 220-223] Check if all ended with EOS
      if all(h.yseq[-1] == eos for h in running_hyps):
          break
  ↓
Return running_hyps
```

### 6. Return Results

```
BatchBeamSearchOnline.__call__() returns running_hyps
  ↓
Speech2TextStreaming.__call__() converts to output format
  ↓
[Line 345-357] For each hypothesis:
  token_ids = hyp.yseq[1:-1]  # Remove SOS/EOS
  token_ids = filter(lambda x: x != 0, token_ids)  # Remove blanks
  text = decode_with_bpe(token_ids)
  ↓
Return [(text, tokens, token_ids), ...]
```

---

## Our Flow (speechcatcher)

### 1. Entry Point: `__call__(speech, is_final)`

**File**: `speechcatcher/speech2text_streaming.py:370-445`

```
Speech2TextStreaming.__call__(speech, is_final)
  ↓
[Line 384-389] Convert to tensor
  speech = torch.from_numpy(speech).to(device)
  ↓
[Line 390-402] Apply frontend if 1D (raw audio)
  if speech.dim() == 1:
      feats, feats_lengths, self.frontend_states = self.apply_frontend(
          speech, self.frontend_states, is_final
      )
      if feats is None:
          return []  # Not enough samples
```

### 2. Frontend: `apply_frontend(speech, prev_states, is_final)`

**File**: `speechcatcher/speech2text_streaming.py:250-368`

```
apply_frontend(speech, prev_states, is_final)
  ↓
[Line 272-275] Concatenate with buffer
  if prev_states and "waveform_buffer" in prev_states:
      speech = torch.cat([prev_states["waveform_buffer"], speech])
  ↓
[Line 277-291] Check if enough samples
  has_enough_samples = speech.size(0) > self.win_length
  if not has_enough_samples:
      if is_final:
          pad with zeros
      else:
          buffer and return None  # ← SKIP EVERYTHING
  ↓
[Line 293-310] Calculate frames and buffer
  n_frames = (speech.size(0) - (win_length - hop_length)) // hop_length
  process_length = (win_length - hop_length) + n_frames * hop_length
  waveform_buffer = speech[buffer_start:buffer_start + buffer_length]
  ↓
[Line 312-320] STFT frontend
  speech_to_process = speech_to_process.unsqueeze(0)
  feats, feats_lengths = self.model.frontend(speech_to_process)
  ↓
[Line 322-326] Normalization
  feats = (feats - self.mean) / self.std
  ↓
[Line 328-357] Frame trimming
  trim_frames = ceil(ceil(win_length / hop_length) / 2)

  if is_final:
      if prev_states is None:
          # No trimming
      else:
          # Trim beginning
          feats = feats.narrow(1, trim_frames, ...)
  else:
      if prev_states is None:
          # Trim end
          feats = feats.narrow(1, 0, feats.size(1) - trim_frames)
      else:
          # Trim both ends
          feats = feats.narrow(1, trim_frames, feats.size(1) - 2*trim_frames)
  ↓
Return (feats, feats_lengths, next_states)
```

**MATCHES ESPnet exactly** ✅

### 3. Beam Search: `beam_search.process_block(feats, feats_lengths, beam_state, is_final)`

**File**: `speechcatcher/beam_search/beam_search.py:390-520`

```
BlockwiseSynchronousBeamSearch.process_block(feats, feats_len, prev_state, is_final)
  ↓
[Line 412-417] Initialize state if needed
  if prev_state is None:
      prev_state = BeamState(
          hypotheses=[SOS],
          encoder_states=None,
      )
  ↓
[Line 423-445] Encode features
  encoder_out, encoder_out_lens, encoder_states = self.encoder(
      features, feature_lens,
      prev_states=prev_state.encoder_states,  # ← CRITICAL!
      is_final=is_final,
      infer_mode=True,
  )
  ↓
[Line 447-455] Accumulate encoder output in buffer
  if encoder_out.size(1) > 0:
      if self.encoder_buffer is None:
          self.encoder_buffer = encoder_out
      else:
          self.encoder_buffer = torch.cat([self.encoder_buffer, encoder_out], dim=1)
  ↓
[Line 457-509] Extract and decode blocks
  while True:
      cur_end_frame = block_size - look_ahead + hop_size * processed_block

      if self.encoder_buffer and cur_end_frame <= self.encoder_buffer.size(1):
          # Extract block
          block_encoder_out = self.encoder_buffer.narrow(1, 0, cur_end_frame)
          ↓
          # Decode block
          current_state = self._decode_one_block(
              block_encoder_out, current_state, False
          )
          ↓
          self.processed_block += 1
      else:
          break
  ↓
[Line 511-520] Update encoder states (CRITICAL FIX!)
  if ret is not None:
      ret.encoder_states = encoder_states
      return ret
  else:
      prev_state.encoder_states = encoder_states  # ← MUST preserve!
      return prev_state
```

**MATCHES ESPnet** ✅

### 4. Decode One Block: `_decode_one_block(encoder_out, prev_state, is_final)`

**File**: `speechcatcher/beam_search/beam_search.py:522-654`

```
_decode_one_block(encoder_out, prev_state, is_final)
  ↓
[Line 534] Extend scorers
  extended_hypotheses = self.extend_scorers(encoder_out, prev_state.hypotheses)
  ↓
[Line 536-550] Create new state
  new_state = BeamState(
      hypotheses=extended_hypotheses,
      encoder_out=encoder_out,
      encoder_out_lens=encoder_out_lens,
      ...
  )
  ↓
[Line 552-654] Beam search loop
  for step in range(max_length):
      ↓
      [Line 567-571] Score all hypotheses
      scores, new_states_dict = self.beam_search.batch_score_hypotheses(
          current_state.hypotheses, encoder_out
      )
      ↓
      [Line 573-604] Expand hypotheses
      for hyp in current_state.hypotheses:
          top_scores, top_tokens = torch.topk(scores[i], beam_size)
          for score, token in zip(top_scores, top_tokens):
              new_hyp = Hypothesis(
                  yseq=append_token(hyp.yseq, token),
                  score=hyp.score + score,
                  ...
              )
      ↓
      [Line 606-607] Prune to beam size
      beam = top_k_hypotheses(new_hypotheses, beam_size)
      ↓
      [Line 609-629] BBD: Check for repetition
      if self.use_bbd and not is_final:
          has_repetition = self.detect_repetition_or_eos(new_state.hypotheses)
          if has_repetition:
              # Rollback to prev_state
              logger.debug("BBD: Repetition/EOS detected, stopping block")
              return prev_state  # ← Rollback!
      ↓
      [Line 631-638] Check if all ended with EOS
      if all(h.yseq[-1] == eos for h in beam):
          break
  ↓
Return new_state
```

**SIMILAR to ESPnet**, but some differences in state management

### 5. Return Results

**File**: `speechcatcher/speech2text_streaming.py:423-444`

```
Speech2TextStreaming.__call__()
  ↓
beam_state = beam_search.process_block(...)
  ↓
[Line 424-444] Convert hypotheses to output
  for hyp in beam_state.hypotheses:
      token_ids = hyp.yseq[1:-1]  # Remove SOS/EOS
      token_ids_filtered = [tid for tid in token_ids if tid != 0]
      text = decode_with_bpe(token_ids_filtered)
  ↓
Return [(text, tokens, token_ids), ...]
```

**MATCHES ESPnet** ✅

---

## Key Differences Found

### ✅ MATCHING

1. **Frontend**: Identical waveform buffering, STFT, normalization, trimming
2. **Encoder buffer accumulation**: Same logic (0→24→40→56)
3. **Block extraction**: Same formula (block_size - look_ahead + hop_size * n)
4. **BBD detection**: Same repetition check
5. **Output formatting**: Same token filtering and BPE decoding

### ⚠️ POTENTIAL DIFFERENCES

#### 1. Decoder Scoring (`batch_score_hypotheses`)

**ESPnet**: `espnet/nets/batch_beam_search_online.py:score_full()`
```python
def score_full(self, running_hyps, x):
    # Uses BatchBeamSearch.score_full()
    # Batch scores all hypotheses
    # Returns combined scores (decoder + CTC)
```

**Ours**: `speechcatcher/beam_search/beam_search.py:batch_score_hypotheses()`
```python
def batch_score_hypotheses(self, hypotheses, encoder_out):
    # Our implementation
    # Scores each scorer separately
    # Combines with weights
```

**→ NEED TO VERIFY**: Does our scoring exactly match ESPnet's?

#### 2. Hypothesis State Management

**ESPnet**: Stores decoder states in hypothesis
```python
running_hyps[i].states = {
    "decoder": decoder_state,
    "ctc": ctc_state,
}
```

**Ours**: Also stores in hypothesis.states
```python
new_hyp.states = {
    "decoder": decoder_state,
    "ctc": ctc_state,
}
```

**→ SEEMS OK**, but need to verify state structure

#### 3. BBD Rollback

**ESPnet**: `batch_beam_search_online.py:rollback_hyps()`
```python
def rollback_hyps(self, running_hyps, n_steps):
    # Rolls back n_steps
    # Updates output_index
    # Returns rolled-back hypotheses
```

**Ours**: `_decode_one_block()` returns `prev_state`
```python
if has_repetition:
    return prev_state  # Rollback to previous state
```

**→ NEED TO VERIFY**: Is our rollback equivalent?

---

## Execution Trace Example

### Chunk 5 (when ESPnet outputs "liebe" but we output "")

**ESPnet**:
```
Chunk 5: 8000 samples
  ↓ apply_frontend()
  → 48 feature frames (after trimming)
  ↓ encoder()
  → 16 encoder frames
  ↓ buffer accumulation
  → encoder_buffer: 24+16 = 40 frames
  ↓ block extraction
  → cur_end_frame = 40 - 16 + 16*1 = 40
  → Process block 1 with 40 frames
  ↓ _decode_one_block()
  → Step 0: scores all tokens
  → Top token: ??? (NOT 1023!)
  → Add to hypothesis
  ↓ BBD check
  → No repetition detected
  ↓ Continue decoding
  → Eventually produces "liebe"
```

**Ours**:
```
Chunk 5: 8000 samples
  ↓ apply_frontend()
  → 48 feature frames (after trimming)
  ↓ encoder()
  → 16 encoder frames
  ↓ buffer accumulation
  → encoder_buffer: 24+16 = 40 frames
  ↓ block extraction
  → cur_end_frame = 40 - 16 + 16*1 = 40
  → Process block 1 with 40 frames
  ↓ _decode_one_block()
  → Step 0: scores all tokens
  → Top token: 1023 ❌
  → Add to hypothesis: [1, 1023]
  ↓ BBD check
  → Repetition detected! (1023 already in [1])
  ↓ Rollback
  → Return prev_state (output_index=0)
  ↓ Result
  → Empty text ""
```

**CRITICAL DIFFERENCE**: Our decoder scores token 1023 highest, ESPnet doesn't!

---

## Next Investigation

**Focus on decoder scoring at the FIRST prediction:**

1. With encoder_out of 40 frames (chunk 5)
2. Starting from hypothesis [1] (SOS only)
3. What does ESPnet's `score_full()` produce?
4. What does our `batch_score_hypotheses()` produce?
5. Why do we get token 1023 highest?

**Hypothesis**: There's a subtle difference in:
- CTC scoring logic
- Decoder scoring logic
- Score combination
- State initialization

**Action**: Create a test that directly compares the scores from both implementations for the same input.
