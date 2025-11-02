# ESPnet Streaming Decoder - Development Roadmap

## Current State (January 2025)

### Completed Components

#### Core Architecture (100% Complete)
- ✅ Foundation layers (FFN, LayerNorm, PositionalEncoding, ConvolutionModule)
- ✅ Multi-headed attention with Flash Attention 2 support
- ✅ STFT frontend (log-mel feature extraction)
- ✅ Conv2d subsampling (4x downsampling)
- ✅ Contextual Block Transformer Encoder (streaming-capable)
- ✅ Transformer Decoder (with incremental KV caching)
- ✅ CTC module (loss, greedy decode, prefix scoring)
- ✅ ESPnetASRModel wrapper (full integration)

#### Inference Pipeline (100% Complete)
- ✅ Blockwise Synchronous Beam Search (BSBS) implementation
- ✅ Hypothesis management and beam state tracking
- ✅ Multiple scorers (CTC + Decoder) with weighted combination
- ✅ Speech2TextStreaming API (drop-in ESPnet replacement)
- ✅ Automatic model loading and architecture inference
- ✅ Feature normalization with stats loading
- ✅ Streaming state management

#### Testing & Validation (85% Complete)
- ✅ Unit tests for all components (25+ test files)
- ✅ Real checkpoint loading (German XL model, 245 MB, 64M params)
- ✅ Pipeline structure validation
- ✅ Streaming mode validation
- ⏳ Audio feature extraction (placeholder only)
- ⏳ BPE tokenizer integration (placeholder only)
- ⏳ Full WER validation against reference

#### Documentation (100% Complete)
- ✅ DECODER_README.md - User and developer documentation
- ✅ IMPLEMENTATION_SUMMARY.md - Technical architecture overview
- ✅ Inline docstrings with type hints
- ✅ Test examples and usage patterns

### Current Limitations

1. **Audio Processing**
   - Only accepts pre-computed log-mel features
   - No raw audio input support
   - No ffmpeg integration for video extraction

2. **Tokenization**
   - Outputs token IDs only
   - No BPE vocabulary loading
   - No text decoding (returns stringified token IDs)

3. **Model Support**
   - Only Contextual Block Transformer tested
   - Conformer encoder not implemented
   - No language model (LM) scorer integration

4. **Performance**
   - No quantization support (int8/fp16 inference)
   - No ONNX export
   - No batch processing optimization

5. **Validation**
   - WER not yet validated against reference
   - No benchmark suite for RTF/latency
   - No comparison with ESPnet reference implementation

---

## Phase 1: Audio Processing Integration (High Priority)

**Goal:** Enable end-to-end recognition from video/audio files

### Tasks

1. **Audio Extraction Module** (`speechcatcher/audio/extraction.py`)
   ```python
   def extract_audio_from_video(video_path: Path, target_sr: int = 16000) -> np.ndarray:
       """Extract audio from video using ffmpeg."""

   def load_audio_file(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
       """Load audio file and resample."""
   ```
   - Use ffmpeg-python for video extraction
   - Use librosa or torchaudio for audio loading
   - Handle resampling to 16kHz mono
   - Support formats: mp4, wav, flac, mp3

2. **Feature Extraction Module** (`speechcatcher/audio/features.py`)
   ```python
   def compute_log_mel_features(
       audio: np.ndarray,
       sr: int = 16000,
       n_fft: int = 512,
       hop_length: int = 160,
       win_length: int = 400,
       n_mels: int = 80,
   ) -> np.ndarray:
       """Compute log-mel spectrogram features."""
   ```
   - Implement using torchaudio.transforms.MelSpectrogram
   - Match ESPnet feature extraction exactly
   - Support streaming feature extraction (chunk-by-chunk)

3. **Integration with Speech2TextStreaming**
   ```python
   # Update __call__ to accept raw audio
   def __call__(
       self,
       speech: Union[np.ndarray, torch.Tensor],
       is_final: bool = False,
       input_type: str = "features",  # or "audio"
   ) -> List[Tuple[str, List[str], List[int]]]:
   ```

4. **Testing**
   - Test with Neujahrsansprache.mp4
   - Validate feature dimensions match reference
   - Compare features with ESPnet extraction

**Success Criteria:**
- Can process Neujahrsansprache.mp4 end-to-end
- Feature extraction matches ESPnet output
- All tests passing

**Estimated Effort:** 4-6 hours

---

## Phase 2: BPE Tokenizer Integration (High Priority)

**Goal:** Convert token IDs to readable text

### Tasks

1. **Vocabulary Loader** (`speechcatcher/tokenizer/bpe_tokenizer.py`)
   ```python
   class BPETokenizer:
       def __init__(self, vocab_path: Path):
           """Load BPE vocabulary from tokens.json or vocab.txt."""

       def decode(self, token_ids: List[int]) -> str:
           """Decode token IDs to text."""

       def encode(self, text: str) -> List[int]:
           """Encode text to token IDs (optional)."""
   ```
   - Support sentencepiece format
   - Support ESPnet tokens.json format
   - Handle special tokens (SOS, EOS, blank)

2. **Integration with Speech2TextStreaming**
   ```python
   # Add tokenizer to __init__
   def __init__(self, model_dir, ...):
       self.tokenizer = BPETokenizer.from_model_dir(model_dir)

   # Update output format
   def __call__(self, speech, is_final=False):
       results = []
       for hyp in self.beam_state.hypotheses:
           token_ids = hyp.yseq[1:]  # Skip SOS
           text = self.tokenizer.decode(token_ids)  # Real text!
           tokens = self.tokenizer.tokens(token_ids)
           results.append((text, tokens, token_ids))
       return results
   ```

3. **Vocabulary Auto-Discovery**
   - Search for tokens.json in model directory
   - Search for vocab.txt, tokens.txt
   - Search for sentencepiece model (.model, .vocab)
   - Warn if not found

4. **Testing**
   - Test with German XL model vocabulary
   - Validate decoded text matches expected format
   - Test special character handling (umlauts, punctuation)

**Success Criteria:**
- Token IDs decode to readable German text
- Special characters rendered correctly
- Output matches ESPnet reference format

**Estimated Effort:** 3-4 hours

---

## Phase 3: WER Validation (High Priority)

**Goal:** Validate accuracy within ±0.3 WER of reference

### Tasks

1. **WER Computation** (`speechcatcher/evaluation/metrics.py`)
   ```python
   def compute_wer(reference: str, hypothesis: str) -> float:
       """Compute Word Error Rate using jiwer."""

   def compute_cer(reference: str, hypothesis: str) -> float:
       """Compute Character Error Rate."""
   ```
   - Use jiwer library for accurate WER
   - Support normalization (lowercase, punctuation removal)
   - Support CER for character-level languages

2. **Evaluation Script** (`scripts/evaluate_model.py`)
   ```python
   # Run evaluation on test set
   python scripts/evaluate_model.py \
       --model-dir /path/to/model \
       --test-video Neujahrsansprache.mp4 \
       --reference Neujahrsansprache.mp4.txt \
       --beam-size 10 \
       --ctc-weight 0.3
   ```
   - Process test file
   - Compare with reference transcript
   - Output WER, CER, RTF metrics
   - Save detailed results to JSON

3. **Comparison with ESPnet Reference**
   - Load reference JSON output
   - Compare token sequences
   - Compare scores
   - Identify discrepancies

4. **Testing**
   - Test with Neujahrsansprache.mp4
   - Compare WER with reference implementation
   - Verify ±0.3 absolute WER tolerance

**Success Criteria:**
- WER computed accurately with jiwer
- WER within ±0.3 of ESPnet reference
- Detailed error analysis available

**Estimated Effort:** 4-5 hours

---

## Phase 4: Performance Optimization (Medium Priority)

**Goal:** Improve inference speed and memory efficiency

### Tasks

1. **Benchmark Suite** (`benchmarks/benchmark_inference.py`)
   ```python
   # Measure RTF, latency, memory
   python benchmarks/benchmark_inference.py \
       --model-dir /path/to/model \
       --test-audio audio.wav \
       --device cuda
   ```
   - Real-Time Factor (RTF)
   - Algorithmic latency
   - End-to-end latency
   - Memory usage (peak, average)
   - GPU utilization

2. **Mixed Precision Inference**
   ```python
   # Add fp16 support
   speech2text = Speech2TextStreaming(
       model_dir=model_dir,
       device="cuda",
       dtype="float16",  # Enable mixed precision
   )
   ```
   - Test accuracy with fp16
   - Measure speedup
   - Add automatic mixed precision (AMP)

3. **Batch Processing**
   ```python
   # Process multiple utterances in parallel
   def recognize_batch(
       self,
       speeches: List[np.ndarray],
   ) -> List[List[Tuple[str, List[str], List[int]]]]:
   ```
   - Parallel beam search
   - Dynamic batching
   - Padding handling

4. **Quantization** (Optional)
   - INT8 quantization with PyTorch
   - Dynamic quantization for decoder
   - Measure accuracy degradation

**Success Criteria:**
- RTF < 0.3 on CPU, < 0.1 on GPU
- FP16 maintains accuracy within ±0.5 WER
- Batch processing 2-3x faster than sequential

**Estimated Effort:** 6-8 hours

---

## Phase 5: Conformer Support (Medium Priority)

**Goal:** Support Conformer-based streaming models

### Tasks

1. **Conformer Encoder Implementation** (`speechcatcher/model/encoder/conformer_encoder.py`)
   ```python
   class ContextualBlockConformerEncoderLayer(nn.Module):
       """Conformer encoder layer with streaming support."""
       def __init__(
           self,
           size: int,
           self_attn: nn.Module,
           feed_forward: nn.Module,
           feed_forward_macaron: nn.Module,
           conv_module: nn.Module,
           dropout_rate: float,
       ):
   ```
   - Implement ConformerEncoderLayer
   - Add streaming support with block processing
   - Implement context propagation
   - Handle convolution module state

2. **Architecture Auto-Detection**
   ```python
   # Detect encoder type from checkpoint
   def infer_encoder_type(state_dict: Dict) -> str:
       if "encoder.encoders.0.conv_module" in state_dict:
           return "conformer"
       else:
           return "transformer"
   ```

3. **Testing**
   - Unit tests for Conformer layers
   - Test with real Conformer checkpoint
   - Compare with Transformer performance

**Success Criteria:**
- Conformer encoder works in streaming mode
- WER matches reference implementation
- All tests passing

**Estimated Effort:** 8-10 hours

---

## Phase 6: Advanced Features (Low Priority)

**Goal:** Add production-ready features

### Tasks

1. **Language Model Integration**
   ```python
   class LanguageModelScorer(ScorerInterface):
       """N-gram or neural LM scorer."""

   # Usage
   beam_search = create_beam_search(
       model=model,
       beam_size=10,
       ctc_weight=0.3,
       decoder_weight=0.6,
       lm_weight=0.1,  # Add LM scoring
       lm_path="/path/to/lm",
   )
   ```

2. **ONNX Export**
   ```python
   # Export encoder/decoder to ONNX
   def export_to_onnx(
       model: ESPnetASRModel,
       output_dir: Path,
   ):
       torch.onnx.export(model.encoder, ...)
       torch.onnx.export(model.decoder, ...)
   ```
   - Export encoder with streaming state
   - Export decoder with KV cache
   - Test inference with onnxruntime

3. **WebRTC Integration**
   - Real-time microphone input
   - Voice Activity Detection (VAD)
   - Streaming output with partial results

4. **Multi-GPU Support**
   - Data parallel beam search
   - Model parallel for large models

**Success Criteria:**
- LM improves WER by 10-20%
- ONNX export works with minimal accuracy loss
- Real-time inference from microphone

**Estimated Effort:** 15-20 hours

---

## Testing Gaps

### Current Test Coverage

| Component | Unit Tests | Integration Tests | Real Model Tests |
|-----------|-----------|-------------------|------------------|
| Foundation layers | ✅ | ✅ | ✅ |
| Attention | ✅ | ✅ | ✅ |
| Frontend | ✅ | ✅ | ✅ |
| Encoder | ✅ | ✅ | ✅ |
| Decoder | ✅ | ✅ | ✅ |
| CTC | ✅ | ✅ | ✅ |
| Beam Search | ✅ | ✅ | ✅ |
| Speech2Text API | ✅ | ✅ | ✅ |
| Audio extraction | ❌ | ❌ | ❌ |
| BPE tokenizer | ❌ | ❌ | ❌ |
| WER validation | ❌ | ❌ | ❌ |

### Missing Tests

1. **Audio Processing Tests**
   - Video extraction correctness
   - Feature computation accuracy
   - Chunk-based streaming extraction

2. **Tokenizer Tests**
   - Vocabulary loading
   - Token ID decoding
   - Special character handling

3. **End-to-End Tests**
   - Full pipeline with real audio
   - WER measurement
   - Latency measurement

4. **Edge Cases**
   - Very short utterances (< 1 block)
   - Very long utterances (> 100 blocks)
   - Silence handling
   - Out-of-vocabulary words

---

## Known Issues & Technical Debt

### Issue 1: CTC Prefix Scoring O(n²) Complexity
**Location:** `speechcatcher/model/ctc.py:prefix_beam_search()`

**Problem:**
```python
for t in range(start, end):
    # This loop recomputes forward variables from scratch
```

**Solution:**
Implement incremental forward variable computation:
```python
class CTCPrefixScorer:
    def __init__(self):
        self.last_computed_frame = 0
        self.forward_vars = {}  # Cache

    def score_incremental(self, prefix, new_frames):
        # Only compute from last_computed_frame to end
        # Reuse cached forward_vars
```

**Priority:** Medium (works but slow for long utterances)

### Issue 2: Flash Attention Version Pinning
**Location:** `speechcatcher/model/attention/multi_head_attention.py`

**Problem:**
Flash Attention API changes between versions. Current implementation assumes v2.x.

**Solution:**
Add version detection and compatibility layer:
```python
import flash_attn
if flash_attn.__version__.startswith("2."):
    use_v2_api()
elif flash_attn.__version__.startswith("3."):
    use_v3_api()
```

**Priority:** Low (works with v2, document version requirement)

### Issue 3: Memory Leak in Long Streaming Sessions
**Location:** `speechcatcher/beam_search/beam_search.py`

**Problem:**
Beam state accumulates history without bounds.

**Solution:**
Add periodic state pruning:
```python
def prune_history(self, keep_last_n_blocks: int = 10):
    # Only keep recent encoder states
    # Discard old hypothesis history
```

**Priority:** Medium (only affects very long sessions)

### Issue 4: No Graceful Degradation for Missing Stats
**Location:** `speechcatcher/speech2text_streaming.py`

**Problem:**
Model works without normalization stats but with degraded accuracy. No warning to user.

**Solution:**
Add explicit warning and optional stats requirement:
```python
if self.mean is None:
    if require_stats:
        raise ValueError("Normalization stats required")
    else:
        logger.warning("⚠️  Stats not found, accuracy may be degraded")
```

**Priority:** Low (works, just needs better UX)

---

## Documentation Improvements

### Needed Documentation

1. **API Reference**
   - Auto-generated from docstrings
   - Sphinx or MkDocs setup
   - Hosted on GitHub Pages

2. **Tutorial Notebooks**
   - Basic usage (non-streaming)
   - Streaming usage
   - Custom model training
   - Fine-tuning on new language

3. **Architecture Deep-Dive**
   - BSBS algorithm walkthrough
   - Context propagation details
   - Beam search scoring details

4. **Deployment Guide**
   - Docker containerization
   - FastAPI REST API wrapper
   - gRPC streaming service
   - Kubernetes deployment

---

## Recommended Next Session Priority

### Immediate (Next 1-2 sessions)

1. **Audio Processing Integration** (Phase 1)
   - Highest impact: enables real testing
   - Clear scope, well-defined
   - Prerequisite for WER validation

2. **BPE Tokenizer Integration** (Phase 2)
   - Required for human-readable output
   - Straightforward implementation
   - High user value

3. **WER Validation** (Phase 3)
   - Critical success metric
   - Validates entire implementation
   - Builds confidence in accuracy

### Medium-Term (Next 3-5 sessions)

4. **Performance Benchmarking** (Phase 4)
   - Quantify improvements over baseline
   - Identify bottlenecks
   - Guide optimization efforts

5. **Conformer Support** (Phase 5)
   - Broader model compatibility
   - Significant engineering effort
   - Lower priority if Transformer sufficient

### Long-Term (Future sessions)

6. **Advanced Features** (Phase 6)
   - Production deployment features
   - Nice-to-have enhancements
   - Depends on user needs

---

## Session Preparation Checklist

Before starting next session:

- [ ] Review this roadmap
- [ ] Check latest commits on `feat/decoder-rewrite-bsbs`
- [ ] Verify test environment still works
- [ ] Confirm model checkpoint location
- [ ] Have Neujahrsansprache.mp4 accessible
- [ ] Install any new dependencies:
  ```bash
  pip install ffmpeg-python librosa jiwer sentencepiece
  ```

---

## Success Metrics

### Technical Metrics
- WER ≤ Reference + 0.3
- RTF < 0.3 (CPU), < 0.1 (GPU)
- Latency ≤ 560ms (algorithmic)
- Memory < 4GB
- All tests passing

### Code Quality Metrics
- Type hint coverage: 100%
- Test coverage: >90%
- Documentation coverage: 100%
- Ruff/black compliant: 100%

### User Experience Metrics
- API simplicity: 1 line to initialize, 1 line to recognize
- Installation time: < 5 minutes
- Error messages: Clear and actionable
- Documentation: Complete examples for all use cases

---

## Questions for Next Session

1. **Model Coverage:** Do we need to support models beyond German XL? (Other languages, sizes)

2. **Deployment Target:** What's the primary deployment scenario?
   - Research/experimentation
   - Production API service
   - Embedded/edge device
   - Real-time microphone

3. **Performance Priorities:** What matters more?
   - Accuracy (WER)
   - Speed (RTF)
   - Memory efficiency
   - Latency

4. **Feature Priorities:** Which Phase 6 features are most valuable?
   - Language model integration
   - ONNX export
   - WebRTC/real-time
   - Multi-GPU

---

## References

- [BSBS Paper (Tsunoo et al. 2019)](https://arxiv.org/abs/1910.07204)
- [ESPnet GitHub](https://github.com/espnet/espnet)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [DECODER_README.md](./DECODER_README.md)
- [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)

---

## Changelog

### 2025-01-XX - Initial Roadmap
- Documented completed work (Phases 0-7)
- Defined Phase 1-6 tasks
- Identified known issues
- Set success metrics
