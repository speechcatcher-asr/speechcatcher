# Visual Streaming Flow Comparison

## Side-by-Side Flow (Chunk 5 Example)

```
┌─────────────────────────────────────────────────────────────────────┐
│ CHUNK 5: 8000 audio samples                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────┬─────────────────┐
                              │ ESPnet          │ Ours            │
                              ▼                 ▼                 │
                                                                  │
┌──────────────────────── FRONTEND ─────────────────────────────┐│
│                                                                ││
│  waveform_buffer: 320 samples                                 ││
│  + new chunk: 8000 samples                                    ││
│  = total: 8320 samples                                        ││
│                                                                ││
│  STFT (win=400, hop=160)                                      ││
│    → 51 frames (before trimming)                              ││
│                                                                ││
│  Trim both ends (trim_frames=2)                               ││
│    → 51 - 2 - 2 = 47 frames                                   ││
│                                                                ││
│  ✅ MATCH: Both produce 47-48 frames (max diff < 0.00001)     ││
└────────────────────────────────────────────────────────────────┘│
                              │                                   │
                              ├─────────────────┬─────────────────┤
                              │ ESPnet          │ Ours            │
                              ▼                 ▼                 │
                                                                  │
┌────────────────────────── ENCODER ────────────────────────────┐│
│                                                                ││
│  Input: 47 feature frames                                     ││
│  prev_states: {'buffer_before_downsampling': ...,             ││
│                'prev_addin': ...,                             ││
│                'past_encoder_ctx': ...}                       ││
│                                                                ││
│  ContextualBlockTransformerEncoder.forward()                  ││
│    with infer_mode=True                                       ││
│                                                                ││
│  Output: 16 encoder frames ✅                                 ││
│                                                                ││
│  ✅ MATCH: Both produce 16 frames (max diff < 0.000001)       ││
└────────────────────────────────────────────────────────────────┘│
                              │                                   │
                              ├─────────────────┬─────────────────┤
                              │ ESPnet          │ Ours            │
                              ▼                 ▼                 │
                                                                  │
┌──────────────────── ENCODER BUFFER ───────────────────────────┐│
│                                                                ││
│  Previous buffer: 24 frames                                   ││
│  + New output: 16 frames                                      ││
│  = Total: 40 frames                                           ││
│                                                                ││
│  ✅ MATCH: Both have 40 frames in buffer                      ││
└────────────────────────────────────────────────────────────────┘│
                              │                                   │
                              ├─────────────────┬─────────────────┤
                              │ ESPnet          │ Ours            │
                              ▼                 ▼                 │
                                                                  │
┌───────────────────── BLOCK EXTRACTION ────────────────────────┐│
│                                                                ││
│  cur_end_frame = 40 - 16 + 16 * 1 = 40                        ││
│                                                                ││
│  40 <= 40? YES ✅                                              ││
│                                                                ││
│  Extract block: encoder_buffer[:, :40]                        ││
│    → 40 frames for decoding                                   ││
│                                                                ││
│  ✅ MATCH: Both extract 40-frame block                        ││
└────────────────────────────────────────────────────────────────┘│
                              │                                   │
                              ├─────────────────┬─────────────────┤
                              │ ESPnet          │ Ours            │
                              ▼                 ▼                 │
                                                                  │
┌─────────────────── _decode_one_block() ───────────────────────┐│
│                                                                ││
│  Input: 40 encoder frames                                     ││
│  Hypotheses: [1] (SOS only)                                   ││
│                                                                ││
│  STEP 0: First prediction                                     ││
│                                                                ││
│  ┌────────────────────────────────────────────────┐           ││
│  │ THIS IS WHERE WE DIVERGE! ❌                   │           ││
│  └────────────────────────────────────────────────┘           ││
│                                                                ││
│  ESPnet:                      Ours:                           ││
│  ┌──────────────────┐         ┌──────────────────┐           ││
│  │ score_full()     │         │ batch_score_     │           ││
│  │                  │         │   hypotheses()   │           ││
│  │ CTC scoring      │         │                  │           ││
│  │ Decoder scoring  │         │ CTC scoring      │           ││
│  │ Combine          │         │ Decoder scoring  │           ││
│  │                  │         │ Combine          │           ││
│  │ Top token: ???   │         │ Top token: 1023  │           ││
│  │ (German!)        │         │ (Arabic!)        │           ││
│  └──────────────────┘         └──────────────────┘           ││
│          │                             │                      ││
│          ▼                             ▼                      ││
│                                                                ││
│  ESPnet:                      Ours:                           ││
│  Add correct token            Add token 1023                  ││
│  to hypothesis                to hypothesis                   ││
│                                                                ││
│  No repetition ✅             Repetition detected! ❌         ││
│                               (1023 in [1])                   ││
│                                                                ││
│  Continue decoding            BBD ROLLBACK →                  ││
│                               return prev_state               ││
│          │                             │                      ││
│          ▼                             ▼                      ││
│                                                                ││
│  Result: "liebe" ✅           Result: "" ❌                   ││
│                                                                ││
└────────────────────────────────────────────────────────────────┘│
                              │                                   │
                              ├─────────────────┬─────────────────┤
                              │ ESPnet          │ Ours            │
                              ▼                 ▼                 │
                                                                  │
              Text: "liebe" ✅      Text: "" ❌                   │
                                                                  │
```

## The Critical Divergence Point

```
┌─────────────────────────────────────────────────────────────────┐
│ DIVERGENCE POINT: First token prediction in _decode_one_block  │
│                                                                 │
│ Input: encoder_out.shape = (1, 40, 256)                        │
│ Current hypothesis: [1] (SOS only)                             │
│                                                                 │
│ Question: What are the scores for all 1024 tokens?             │
└─────────────────────────────────────────────────────────────────┘

ESPnet's score_full():
┌──────────────────────────────────────────────────────────┐
│ 1. Initialize scorers                                    │
│    - CTC: CTCPrefixScoreTH(encoder_out)                  │
│    - Decoder: init with encoder_out                      │
│                                                           │
│ 2. Batch score all hypotheses                            │
│    For hypothesis [1]:                                   │
│      decoder_score = decoder.score([1], state, enc_out)  │
│      ctc_score = ctc.score([1], state, enc_out)          │
│                                                           │
│ 3. Combine scores                                        │
│    combined = 0.7 * decoder + 0.3 * ctc                  │
│                                                           │
│ 4. Return top token                                      │
│    → Token ??? (German word start)                       │
└──────────────────────────────────────────────────────────┘

Our batch_score_hypotheses():
┌──────────────────────────────────────────────────────────┐
│ 1. For each scorer in scorers:                           │
│    - scorer.batch_score_partial(hypotheses, ...)         │
│                                                           │
│ 2. Decoder scorer                                        │
│    decoder.batch_score([1], states, encoder_out)         │
│      → Returns scores for all 1024 tokens                │
│                                                           │
│ 3. CTC scorer                                            │
│    ctc.batch_score_partial([1], states, encoder_out)     │
│      → Returns scores for all 1024 tokens                │
│                                                           │
│ 4. Combine with weights                                  │
│    combined = 0.7 * decoder + 0.3 * ctc                  │
│                                                           │
│ 5. Return combined scores                                │
│    → Top token: 1023 ❌                                   │
└──────────────────────────────────────────────────────────┘

HYPOTHESIS: One of these is wrong:
  a) CTC initialization/scoring
  b) Decoder initialization/scoring
  c) Score combination
  d) State structure
```

## Investigation Steps

### 1. Compare CTC Logits

```python
# ESPnet
espnet_ctc_logits = espnet_s2t.asr_model.ctc.ctc_lo(encoder_out)
# Shape: (1, 40, 1024)

# Ours
our_ctc_logits = our_s2t.model.ctc.ctc_lo(encoder_out)
# Shape: (1, 40, 1024)

# Should be identical (same weights!)
assert torch.allclose(espnet_ctc_logits, our_ctc_logits)
```

### 2. Compare CTC Prefix Scores

```python
# ESPnet
espnet_ctc = CTCPrefixScoreTH(espnet_ctc_logits, ...)
espnet_ctc_scores = espnet_ctc(hyp, ...)

# Ours
our_ctc = CTCPrefixScorer(...)
our_ctc.batch_init_state(encoder_out)
our_ctc_scores = our_ctc.batch_score_partial([hyp], ...)

# Compare
print(f"ESPnet CTC top tokens: {espnet_ctc_scores.topk(10)}")
print(f"Our CTC top tokens: {our_ctc_scores.topk(10)}")
```

### 3. Compare Decoder Logits

```python
# ESPnet
espnet_decoder_out = espnet_s2t.asr_model.decoder.forward_one_step(
    tgt=[1], memory=encoder_out, cache=None
)
espnet_decoder_logits = espnet_decoder_out[0]

# Ours
our_decoder_out = our_s2t.model.decoder.forward_one_step(
    tgt=torch.tensor([[1]]), memory=encoder_out, cache=None
)
our_decoder_logits = our_decoder_out[0]

# Compare
assert torch.allclose(espnet_decoder_logits, our_decoder_logits)
```

### 4. Compare Combined Scores

```python
# ESPnet
espnet_combined = 0.7 * espnet_decoder_logits + 0.3 * espnet_ctc_scores

# Ours
our_combined = 0.7 * our_decoder_logits + 0.3 * our_ctc_scores

# Top tokens
espnet_top = espnet_combined.topk(20)
our_top = our_combined.topk(20)

print(f"ESPnet top 20: {espnet_top}")
print(f"Our top 20: {our_top}")
```

## Expected Findings

One of these will differ:
- ❌ CTC logits (unlikely - weights match)
- ❌ Decoder logits (unlikely - weights match)
- ⚠️  CTC prefix scores (maybe - complex algorithm)
- ⚠️  Score combination (maybe - subtle difference)
- ⚠️  State initialization (maybe - wrong state structure)

**Most likely**: CTC prefix scoring or state initialization differs subtly.
