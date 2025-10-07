# Weight Loading Compatibility Notes

## Model Configuration (from config.yaml)

**Model Type:** `de_streaming_transformer_xl`
**Encoder:** `contextual_block_transformer` (NOT Conformer for this model!)
**Decoder:** `transformer`

### Encoder Config
```yaml
encoder: contextual_block_transformer
encoder_conf:
  attention_dropout_rate: 0.0
  attention_heads: 8
  block_size: 40
  ctx_pos_enc: true
  dropout_rate: 0.1
  hop_size: 16
  init_average: true
  input_layer: conv2d
  linear_units: 2048
  look_ahead: 16
  normalize_before: true
  num_blocks: 30
  output_size: 256
  positional_dropout_rate: 0.1
```

### Decoder Config
```yaml
decoder: transformer
decoder_conf:
  attention_heads: 8
  dropout_rate: 0.1
  linear_units: 2048
  num_blocks: 14
  positional_dropout_rate: 0.1
  self_attention_dropout_rate: 0.0
  src_attention_dropout_rate: 0.0
```

### Frontend Config
```yaml
frontend: default
frontend_conf:
  fs: 16k
  hop_length: 160  # Different from our default 128!
  n_fft: 512
  win_length: 400  # Different from our default 512!
```

### Model Config
```yaml
model_conf:
  ctc_weight: 0.3
  length_normalized_loss: false
  lsm_weight: 0.1
```

### Normalization
```yaml
normalize: global_mvn
normalize_conf:
  stats_file: .../feats_stats.npz
```

### Token List
- vocab_size: 1182 (including <blank>, <unk>, <sos/eos>)
- token_type: bpe
- bpemodel path available in config

## Critical Compatibility Requirements

1. **Frontend parameters must match exactly:**
   - n_fft: 512 ✓
   - hop_length: **160** (not 128!)
   - win_length: **400** (not 512!)

2. **Encoder is Transformer, NOT Conformer**
   - Must implement `ContextualBlockTransformerEncoder`
   - No ConvolutionModule needed for this model

3. **Weight loading:**
   - Checkpoint path: `~/.cache/espnet/.../valid.acc.ave_6best.pth`
   - Must map layer names correctly from ESPnet to our implementation
   - Preserve exact parameter names for compatibility

4. **Normalization stats:**
   - Load `feats_stats.npz` for global mean/variance normalization
   - Apply before encoder

## Implementation Priority

Given the model is **Transformer-based** (not Conformer):

1. ✅ Phase 0: Layers & Attention (DONE)
2. → **Phase 1a:** Implement ContextualBlockTransformerEncoder (PRIORITY)
3. → **Phase 1b:** Implement weight loading utilities
4. → Phase 2: Implement TransformerDecoder
5. → Phase 3: Implement ESPnetASRModel wrapper
6. → Phase 4: Test weight loading with real checkpoint
7. → Phase 5: Beam search & scorers
8. → Phase 6: Speech2TextStreaming API
9. → Phase 7: End-to-end test with Neujahrsansprache.mp4

## Weight Name Mapping (ESPnet → Our Implementation)

Will need to verify and document exact mapping, e.g.:
- `encoder.encoders.0.self_attn.linear_q.weight` → `encoder.layers[0].self_attn.linear_q.weight`
- etc.
