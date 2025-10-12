# Decoder Rewrite - Step 1 Analysis & Plan

**Author:** Senior ASR Engineer
**Date:** 2025-10-07
**Repo:** `~/speechcatcher`
**Target:** Clean, fast O(n) streaming decoder with Blockwise Synchronous Beam Search

---

## Table of Contents

1. [Overview](#overview)
2. [Environment](#environment)
3. [Current Implementation Analysis](#current-implementation-analysis)
4. [Paper Summary: Streaming Transformer ASR with BSBS](#paper-summary)
5. [O(n²) Complexity Sources & Fixes](#on%C2%B2-complexity-sources--fixes)
6. [Proposed API Design](#proposed-api-design)
7. [Data Structures & Caching Plan](#data-structures--caching-plan)
8. [Migration Plan](#migration-plan)
9. [Next Steps](#next-steps)

---

## Overview

### Goal

Migrate and rewrite the streaming decoder from `espnet_streaming_decoder` into `speechcatcher/decoder/` as a clean, performant, well-tested module supporting **Transformer** and **Conformer** streaming models with **Blockwise Synchronous Beam Search (BSBS)**.

### Key Requirements

- **Complexity:** Fix O(n²) behavior → O(n) amortized per utterance
- **Accuracy:** WER within ±0.3 absolute of baseline
- **Latency:** Stable per-chunk decode time, RTF competitive with baseline
- **Architecture:** Support Transformer & Conformer encoders with chunking + look-ahead
- **Code Quality:** Python 3.12, PyTorch, type hints, dataclasses, ruff/black, docstrings

---

## Environment

```
Branch:          main
Python:          3.12.3
PyTorch:         2.8.0
ESPnet:          202509
espnet-streaming-decoder: 0.1.2
sentencepiece:   0.2.0
```

**Key Dependencies:**
- `espnet_streaming_decoder` (installed package to inspect)
- Paper source: `SLT2021_tsunoo_arxiv.tex`
- Current decoder: `speechcatcher/speechcatcher.py` (uses streaming decoder)

---

## Current Implementation Analysis

### Package Structure

```
espnet_streaming_decoder/
├── beam_search.py                       # Top-level streaming beam search
├── asr.py                              # ESPnet2 model/task definitions
├── contextual_block_conformer_encoder.py  # Streaming Conformer encoder
├── espnet/nets/
│   ├── beam_search.py                  # Core BeamSearch class
│   ├── batch_beam_search_online.py     # BatchBeamSearchOnline (BSBS impl)
│   ├── ctc_prefix_score.py             # CTC prefix scorer
│   └── scorers/
│       ├── ctc.py                      # CTCPrefixScorer wrapper
│       └── ...
└── espnet2/
    └── asr/encoder/
        ├── conformer_encoder.py
        ├── contextual_block_conformer_encoder.py
        └── contextual_block_transformer_encoder.py
```

### Key Classes & Data Flow

#### 1. **Hypothesis** (NamedTuple)
- **Location:** `espnet/nets/beam_search.py:13-28`
- **Fields:**
  - `yseq`: torch.Tensor (token sequence)
  - `xpos`: torch.Tensor (encoder position tracking in streaming variant)
  - `score`: float (cumulative score)
  - `scores`: Dict[str, float] (per-scorer breakdown)
  - `states`: Dict[str, Any] (per-scorer state, e.g., decoder hidden, CTC prefix state)

#### 2. **BeamSearch** (base, non-streaming)
- **Location:** `espnet/nets/beam_search.py:30-483`
- **Key Methods:**
  - `init_hyp(x)` → List[Hypothesis]
  - `search(running_hyps, x)` → List[Hypothesis]  (one step)
  - `score_full(hyp, x)` → (scores, states)  [full scorers: decoder attention]
  - `score_partial(hyp, ids, x)` → (scores, states)  [partial: CTC, pre-beam]
  - `forward(x, maxlenratio, minlenratio)` → List[Hypothesis]  (full decoding loop)

- **Scorers:**
  - `full_scorers`: Transformer decoder (needs full vocab scoring)
  - `part_scorers`: CTC prefix scorer (can pre-beam to top-k before scoring)

**Flow (non-streaming):**
1. Encode entire utterance → `h` (T, D)
2. `running_hyps = init_hyp(h)`
3. For i in range(maxlen):
   - `best = search(running_hyps, h)`  ← **all beams score against full `h`**
   - prune, update running_hyps
4. Return ended_hyps

#### 3. **BatchBeamSearchOnline** (streaming BSBS)
- **Location:** `espnet/nets/batch_beam_search_online.py:17-319`
- **Inherits:** BatchBeamSearch (vectorized beams)
- **Key Additions:**
  - `block_size`, `hop_size`, `look_ahead`: encoder chunking params
  - `encbuffer`: accumulated encoder features
  - `running_hyps`, `prev_hyps`, `ended_hyps`: hypothesis tracking
  - `processed_block`: block counter
  - `process_idx`: current decoding step index
  - `disable_repetition_detection`: flag to turn off repetition criterion

**Flow (streaming BSBS):**
1. `forward(x_chunk, is_final=False)`:
   - Append `x_chunk` to `encbuffer`
   - While `cur_end_frame < encbuffer.shape[0]`:
     - Extract block: `h = encbuffer[:cur_end_frame]`
     - `process_one_block(h, block_is_final, maxlen, maxlenratio)`
     - Increment `processed_block`
   - Return partial results or wait

2. `process_one_block(h, is_final, maxlen, maxlenratio)`:
   - Extend CTC state: `extend(h, running_hyps)` ← **grow CTC prefix score cache**
   - While `process_idx < maxlen`:
     - `best = search(running_hyps, h)`
     - Check EOS/repetition:
       - If `best.yseq[i][-1] == eos` → local_ended_hyps
       - If `best.yseq[i][-1] in best.yseq[i][:-1]` and not is_final → **repetition detected, break**
     - `running_hyps = post_process(...)`
     - Increment `process_idx`
   - If repetition detected and not is_final:
     - Rollback: `running_hyps = prev_hyps`, `process_idx -= 1` ← **wait for next block**
   - Return assembled hypotheses

3. **Block Boundary Detection (BBD):**
   - Implicit in repetition check (lines 214-223)
   - Conservative rollback: can step back 1 or 2 steps (algorithm line 525-528)

#### 4. **ContextualBlockConformerEncoder**
- **Location:** `contextual_block_conformer_encoder.py:36-591`
- **Key Features:**
  - Block processing with context inheritance (context embeddings passed between blocks)
  - `block_size`, `hop_size`, `look_ahead` params
  - `forward_train()`: processes entire utterance in blocks for training (emulates streaming)
  - `forward_infer()`: true streaming with state (`prev_addin`, buffers, `n_processed_blocks`, `past_encoder_ctx`)

**State Management (streaming):**
```python
next_states = {
    "prev_addin": prev_addin,              # context embedding from last block
    "buffer_before_downsampling": ...,     # audio buffer
    "ilens_buffer": ...,
    "buffer_after_downsampling": ...,      # feature buffer
    "n_processed_blocks": n_processed_blocks + block_num,
    "past_encoder_ctx": past_encoder_ctx,  # encoder K/V cache (per layer)
}
```

#### 5. **CTCPrefixScoreTH**
- **Location:** `espnet/nets/ctc_prefix_score.py:10-276`
- **Algorithm:** Watanabe CTC/Attention hybrid (prefix beam search)
- **Key Methods:**
  - `__call__(y, state, scoring_ids, att_w)` → (log_psi, new_state)
    - `y`: list of hypothesis token sequences (length n_bh = batch * hyps)
    - `state`: (r_prev, s_prev, f_min_prev, f_max_prev)
      - `r_prev`: forward variables (T, 2, n_bh) [non-blank, blank]
      - `s_prev`: prefix scores (n_bh, O)
    - Returns incremental CTC scores
  - `extend_prob(x)`: extend self.x with new encoder block
  - `extend_state(state)`: extend r_prev forward variables to new length

**Complexity:**
- Per call: O(T * n_bh * scoring_num) where T = input_length
- **Issue:** `for t in range(start, end)` loop (line 160-165) re-computes forward vars
- With growing T (blocks accumulated), this becomes O(n) per step → **O(n²) total**

---

### Data Flow Summary

```
Audio chunks
    ↓
ContextualBlockConformerEncoder.forward_infer(chunk, prev_states, is_final=False)
    ↓ [subsample, block processing with context inheritance]
encoder_output (partial), next_encoder_states
    ↓
BatchBeamSearchOnline.forward(encoder_output, is_final=False)
    ↓ [accumulate in encbuffer, process blocks]
BatchBeamSearchOnline.process_one_block(h, is_final, ...)
    ↓
extend(h, running_hyps)  ← extend CTC scorer with new block
    ↓
while process_idx < maxlen:
    search(running_hyps, h)
        ↓ [for each hyp in running_hyps]
        score_full(hyp, h)      ← Decoder attention (full vocab)
        score_partial(hyp, ids, h) ← CTC prefix scorer
            ↓
        CTCPrefixScoreTH.__call__(y, state, ids, att_w=None)
            ↓ [forward loop over accumulated frames]
        → scores, new_state
    → best_hyps (pruned)
    ↓
    Check EOS/repetition → BBD logic
    ↓
    Update running_hyps, process_idx++
    ↓
If repetition and not is_final: rollback, wait for next block
Else: continue or return results
```

---

## Paper Summary

**Title:** Streaming Transformer ASR with Blockwise Synchronous Beam Search
**Authors:** Emiru Tsunoo, Yosuke Kashiwagi, Shinji Watanabe
**Link:** [arXiv:2006.14941](https://arxiv.org/abs/2006.14941)

### Core Algorithm: Blockwise Synchronous Beam Search (BSBS)

#### Motivation

Standard Transformer requires entire utterance for both encoder & decoder (SAN + STA). For streaming:
- **Encoder:** Use blockwise processing with context inheritance (contextual block processing)
- **Decoder:** Cannot use full encoder output `h_{1:B}` during decoding; must decode synchronously with encoder blocks

#### Key Idea

Approximate:
```
log p(y_i | y_{0:i-1}, h_{1:B}) ≈ log p(y_i | y_{0:i-1}, h_{1:b})
```
where `b < B` (current block < total blocks).

Decoder decodes with limited blocks `h_{1:b}` until predictions become **unreliable** (BBD detects this), then waits for next block `h_{b+1}`.

#### Block Boundary Detection (BBD)

**Reliability Score:**
```
r(y_{0:i-1}, h_{1:b}) = max_{0≤j≤i-1} log p(y_j | y_{0:i-1}, h_{1:b}) + α(y_{0:i-1}, h_{1:b})
```
(highest score among hypotheses with repetition or <eos>)

```
s(y_{0:i}, h_{1:b}) = α(y_{0:i}, h_{1:b}) - r(y_{0:i-1}, h_{1:b})
```

**Decision:**
- If `s(y_{0:i}, h_{1:b}) ≤ 0` → hypothesis unreliable → set `I_b = i-1` (or `i-2` conservative), wait for block `b+1`
- If `s(y_{0:i}, h_{1:b}) > 0` → hypothesis reliable → continue decoding

**Repetition Handling:**
- Share <sos> = <eos> so <eos> is also a repetition of y_0
- Store evaluated hypotheses in `Ω_R` to avoid re-evaluating (true repetitions)
- Update: `r_{\overline{Ω}_R}(y_{0:i-1}, h_{1:b})` excludes hypotheses in `Ω_R`

#### Algorithm Pseudocode (from paper)

```
Algorithm 1: Blockwise Synchronous Beam Search

Input: encoder blocks h_b, total blocks B, beam width K
Output: Ω̂ (complete hypotheses)

Initialize: y_0 ← <sos>, Ω_0 ← {y_0}, Ω_R ← {}, b ← 1, I_* ← I_max, I_0 ← 0

while b < B:
    NextBlock ← false
    for i ← I_{b-1}+1 to I_b unless NextBlock:
        Ω_i ← Search_K(Ω_{i-1}, h_{1:b})
        for y_{0:i} ∈ Ω_i:
            if s(y_{0:i}, h_{1:b}) ≤ 0:
                NextBlock ← true
                Ω_R ← Ω_R ∪ y_{0:i}
        if NextBlock:
            if i ≥ 2:
                I_b ← i - 2    // conservative
            else:
                I_b ← i - 1
            b ← b + 1

// Ordinary decoding after b = B
for i ← I_{B-1}+1 to I_max unless EndingCriterion(Ω_{i-1}):
    Ω_i ← Search_K(Ω_{i-1}, h_{1:B})
    for y_{0:i} ∈ Ω_i:
        if y_i = <eos>:
            Ω̂ ← Ω̂ ∪ y_{0:i}

return Ω̂
```

#### CTC Integration

CTC prefix score computed incrementally:
```
p_ctc(y_{0:i} | h_{1:b}) = γ_{T_b}^(N)(y_{0:i-1}) + γ_{T_b}^(B)(y_{0:i-1})
```
where `T_b` is last frame of block b.

**On-the-fly extension:** When new block arrives, resume CTC prefix score computation (Watanabe Algorithm 2).

#### Contextual Block Encoder

- Block params: `{N_l, N_c, N_r}` = {16, 16, 8} (left context, center, right look-ahead frames, post-subsample)
- Context embedding vector `c_b^n` handed from block b-1 layer n-1 to block b layer n
- Encodes local + global (linguistic, channel, speaker) context

---

## O(n²) Complexity Sources & Fixes

### Identified O(n²) Sources

#### 1. **CTC Prefix Score Re-computation**

**Location:** `ctc_prefix_score.py:160-165`

```python
for t in range(start, end):
    rp = r[t - 1]
    rr = torch.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).view(2, 2, n_bh, snum)
    r[t] = torch.logsumexp(rr, 1) + x_[:, t]
```

**Issue:**
- Each call to `CTCPrefixScoreTH.__call__()` iterates over accumulated frames `[start, end)`
- As blocks accumulate, `end` (total frames) grows linearly with blocks
- Per-step complexity: O(T_current)
- Total over n steps: O(1 + 2 + ... + n) = **O(n²)**

**Fix:**
- **Incremental forward variables:** Cache `r[t]` (forward vars) and only compute new frames
- Track `last_computed_frame` in state
- On new block:
  - Extend `r` array from `last_computed_frame` to `new_end`
  - Update `last_computed_frame`
- **Complexity after fix:** O(new_frames) per call → **O(n) total**

**Implementation:**
```python
class CTCState:
    r: torch.Tensor         # (T_max, 2, n_bh, snum) - pre-allocated or grown
    s_prev: torch.Tensor    # (n_bh, O)
    f_min: int
    f_max: int
    last_computed_frame: int  # NEW: track where we left off
```

#### 2. **Decoder Attention K/V Rebuilds (potential)**

**Location:** Decoder forward passes in `score_full()`

**Issue:**
- Standard Transformer decoder builds K/V from scratch for each hypothesis
- With incremental decoding, this can be O(i) per step for hypothesis of length i
- If not cached, O(1 + 2 + ... + n) = **O(n²)**

**Current State (ESPnet):**
- ESPnet's Transformer decoder **does cache** incremental K/V via `state` dict
- `state['decoder']` contains cached K/V per layer
- **Likely not an O(n²) issue in current impl**, but verify in rewrite

**Fix (for rewrite):**
- Ensure decoder state caches K/V tensors per layer
- Append-only: new token → extend K/V by 1 step
- No rebuilds from scratch

#### 3. **Hypothesis List Scans**

**Location:** Beam search loops in `search()`, `post_process()`

**Issue:**
- Python for-loops over beams (small K, typically 10-30)
- Not O(n²) w.r.t. decoding steps, but can be slow
- Sorting hypotheses at every step: O(K log K) per step → O(n K log K) total (acceptable if K small)

**Fix:**
- Vectorize where possible (batch operations over beams)
- Use tensor operations instead of Python loops
- Keep beam operations O(K) or O(K log K) per step

#### 4. **State Copying / Dict Merging**

**Location:** `merge_states()`, hypothesis expansion

**Issue:**
- Copying state dicts at every beam expansion
- If states are large (nested dicts, large tensors), this can be slow
- Not strictly O(n²), but can add overhead

**Fix:**
- Use shallow copies where safe
- Share immutable state parts (encoder output, embeddings)
- Pre-allocate state tensors when size is known

---

### Summary of Fixes

| Source | Current Complexity | Fix | Target Complexity |
|--------|-------------------|-----|-------------------|
| CTC forward loop | O(n²) | Incremental forward vars + caching | O(n) |
| Decoder K/V | O(n²) if rebuilt | Cache per layer, append-only | O(n) |
| Hypothesis scans | O(n K log K) | Vectorize, batch ops | O(n K) or O(n K log K) |
| State copies | O(n * state_size) | Shallow copies, pre-alloc | O(n) |

**Overall Target:** O(n) amortized per utterance, where n = decoding steps.

---

## Proposed API Design

### Drop-in Replacement Interface

**Critical:** The rewritten decoder **must match** the `Speech2TextStreaming` interface from `speechcatcher/asr_inference_streaming.py` to be a drop-in replacement.

#### Current Integration (from speechcatcher.py)

```python
# Line 28: Import
from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming

# Lines 72-90: Initialization
speech2text = Speech2TextStreaming(
    **info,  # unpacked model files from ModelDownloader
    device=device,
    token_type=None,
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=beam_size,
    ctc_weight=0.3,
    lm_weight=0.0,
    penalty=0.0,
    nbest=1,
    disable_repetition_detection=True,
    decoder_text_length_limit=0,
    encoded_feat_length_limit=0
)

# Line 462: Streaming call
results = speech2text(speech=speech_chunk, is_final=is_final, always_assemble_hyps=not (quiet or progress))

# Line 203: Reset
speech2text.reset()

# Return format (line 364):
# List[Tuple[text: str, token: List[str], token_int: List[int], token_pos: List[int], hyp: Hypothesis]]
```

### Required Public API (Speech2TextStreaming compatible)

```python
class Speech2TextStreaming:
    """
    Drop-in replacement for espnet_streaming_decoder.asr_inference_streaming.Speech2TextStreaming
    """

    def __init__(
        self,
        asr_train_config: Union[Path, str],
        asr_model_file: Union[Path, str] = None,
        lm_train_config: Union[Path, str] = None,
        lm_file: Union[Path, str] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.5,
        lm_weight: float = 1.0,
        penalty: float = 0.0,
        nbest: int = 1,
        disable_repetition_detection=False,
        decoder_text_length_limit=0,
        encoded_feat_length_limit=0,
    ):
        """
        Initialize streaming ASR system.

        Args:
            asr_train_config: Path to model config.yaml
            asr_model_file: Path to model.pth
            lm_train_config: Optional LM config
            lm_file: Optional LM weights
            device: 'cpu' or 'cuda'
            beam_size: Beam width for search
            ctc_weight: CTC weight in joint decoding (0.0-1.0)
            disable_repetition_detection: Disable BBD repetition check
            decoder_text_length_limit: Limit decoder context (0=unlimited)
            encoded_feat_length_limit: Limit encoder context (0=unlimited)
        """
        # Build model, scorers, beam search...
        pass

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        is_final: bool = True,
        always_assemble_hyps: bool = True,
    ) -> List[Tuple[Optional[str], List[str], List[int], List[int], Hypothesis]]:
        """
        Process audio chunk and return results.

        Args:
            speech: Audio waveform (float16/32, 16kHz mono), shape (samples,)
            is_final: Whether this is the last chunk (triggers finalization)
            always_assemble_hyps: Return assembled hypotheses even if not final

        Returns:
            List of tuples (text, tokens, token_ids, token_positions, hypothesis)
            - text: str or None (decoded text if tokenizer available)
            - tokens: List[str] (BPE/char tokens, blank=0 filtered)
            - token_ids: List[int] (token IDs, blank=0 filtered)
            - token_positions: List[int] (frame positions of non-blank tokens)
            - hypothesis: Hypothesis object with yseq, score, scores, states
        """
        # Frontend (STFT) → Encoder → Decoder (BSBS)
        pass

    def reset(self):
        """Reset internal state for new utterance."""
        # Reset frontend, encoder, and beam search states
        pass

# Internal state attributes (used by __call__):
# self.frontend_states: Dict or None (waveform buffer)
# self.encoder_states: Dict or None (encoder block state)
# self.beam_search: BatchBeamSearchOnline instance
```

### High-Level Usage Example

```python
from speechcatcher.decoder import Speech2TextStreaming

# Same interface as before, drop-in replacement
speech2text = Speech2TextStreaming(
    asr_train_config="config.yaml",
    asr_model_file="model.pth",
    device='cpu',
    beam_size=10,
    ctc_weight=0.3,
    disable_repetition_detection=True,
)

# Streaming decoding
for chunk in audio_stream:
    results = speech2text(speech=chunk, is_final=False)
    if results:
        text, tokens, token_ids, positions, hyp = results[0]
        print(f"Partial: {text}")

# Finalize last chunk
results = speech2text(speech=last_chunk, is_final=True)
text, tokens, token_ids, positions, hyp = results[0]
print(f"Final: {text} (score={hyp.score:.2f})")

# Reset for next utterance
speech2text.reset()
```

### Module Structure for Drop-in Replacement

**Critical Path:** The main class must be importable as:
```python
from speechcatcher.asr_inference_streaming import Speech2TextStreaming
```

This means we create `speechcatcher/asr_inference_streaming.py` (matching the original package structure).

**Proposed structure:**

```
speechcatcher/
├── asr_inference_streaming.py  # Drop-in replacement (imports from decoder/)
├── decoder/                     # New clean implementation
│   ├── __init__.py              # Public API exports
│   ├── api.py                   # Speech2TextStreaming implementation
│   ├── base.py                  # Abstract interfaces (BeamSearch, Scorer, Hypothesis)
├── beam_search_bsbs.py      # BlockwiseSynchronousBeamSearch implementation
├── hypothesis.py            # Hypothesis, BeamState dataclasses
├── scorers.py               # AttentionScorer, CTCScorer, LMScorer
├── cache.py                 # EncoderCache, DecoderCache, CTCCache
├── streaming_encoder.py     # StreamingEncoderWrapper (Transformer/Conformer)
├── utils.py                 # Padding, masking, chunk math, logging
└── constants.py             # Special tokens, defaults
```

### Core Interfaces

```python
# base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

@dataclass
class Hypothesis:
    """Single beam hypothesis."""
    yseq: torch.Tensor              # (L,) token sequence
    score: float                    # cumulative score
    scores: Dict[str, float]        # per-scorer breakdown
    states: Dict[str, Any]          # per-scorer state
    block_idx: int                  # last block used for this hyp

    def __lt__(self, other: "Hypothesis") -> bool:
        return self.score < other.score  # for sorting

@dataclass
class BeamState:
    """Beam search state (collection of hypotheses)."""
    hyps: List[Hypothesis]          # active hypotheses
    ended: List[Hypothesis]         # completed hypotheses
    evaluated: set[tuple]           # (yseq_tuple,) for repetition tracking
    process_idx: int                # current decoding step
    block_idx: int                  # current encoder block

class ScorerInterface(ABC):
    """Base scorer interface."""

    @abstractmethod
    def init_state(self, x: torch.Tensor) -> Any:
        """Initialize scorer state given encoder output."""
        pass

    @abstractmethod
    def score(
        self,
        hyp: Hypothesis,
        x: torch.Tensor,
        candidates: Optional[torch.Tensor] = None,  # None = full vocab
    ) -> Tuple[torch.Tensor, Any]:
        """
        Score next tokens for hypothesis.

        Args:
            hyp: Current hypothesis
            x: Encoder output (T, D)
            candidates: (K,) token IDs to score (None = all vocab)

        Returns:
            scores: (K,) or (V,) log-probs
            new_state: Updated scorer state
        """
        pass

    @abstractmethod
    def extend(self, x_new: torch.Tensor, state: Any) -> Any:
        """Extend scorer with new encoder block (for streaming)."""
        pass

class BeamSearchInterface(ABC):
    """Base beam search interface."""

    @abstractmethod
    def search_step(
        self,
        beam_state: BeamState,
        x: torch.Tensor,
    ) -> BeamState:
        """Perform one beam search step."""
        pass

    @abstractmethod
    def decode(
        self,
        x: torch.Tensor,
        is_final: bool = True,
    ) -> List[Hypothesis]:
        """Decode utterance (streaming or batch)."""
        pass
```

### Streaming Encoder Wrapper

```python
# streaming_encoder.py
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class EncoderConfig:
    """Encoder configuration."""
    block_size: int = 40
    hop_size: int = 16
    look_ahead: int = 16
    subsample_factor: int = 4  # conv downsample

class StreamingEncoderWrapper:
    """
    Wrapper for Transformer/Conformer encoder with streaming support.

    Handles:
    - Blockwise processing with overlaps
    - Context embedding inheritance (Conformer)
    - State management (buffers, K/V caches)
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: EncoderConfig,
    ):
        self.encoder = encoder
        self.config = config
        self.state: Optional[Dict[str, Any]] = None

    def encode_chunk(
        self,
        chunk: torch.Tensor,  # (B, T_chunk, D_feat)
        is_final: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Encode audio chunk.

        Args:
            chunk: Audio features (B, T_chunk, D_feat)
            is_final: Whether this is the last chunk

        Returns:
            encoded: (B, T_enc, D_model) encoder output for current block(s)
            state: Updated encoder state (or None if final)
        """
        # Buffer management
        if self.state is not None:
            chunk = torch.cat([self.state['buffer'], chunk], dim=1)

        # Process complete blocks
        # ... (block extraction, context inheritance, etc.)

        return encoded, next_state

    def reset(self):
        """Reset encoder state."""
        self.state = None
```

---

## Data Structures & Caching Plan

### 1. Hypothesis & Beam State

```python
# hypothesis.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

@dataclass
class Hypothesis:
    yseq: torch.Tensor              # (L,) int64 token IDs
    score: float                    # cumulative log-prob
    scores: Dict[str, float]        # {"att": ..., "ctc": ..., "lm": ...}
    states: Dict[str, Any]          # {"att": DecoderState, "ctc": CTCState, ...}
    block_idx: int = 0              # last encoder block index used

    def __post_init__(self):
        assert self.yseq.ndim == 1
        assert self.yseq.dtype == torch.int64

    def __lt__(self, other: "Hypothesis") -> bool:
        return self.score < other.score  # for heapq/sorting

    def __hash__(self) -> int:
        return hash(tuple(self.yseq.tolist()))

    def extend(self, token_id: int, add_score: float, new_scores: Dict[str, float], new_states: Dict[str, Any]) -> "Hypothesis":
        """Create new hypothesis by appending token."""
        return Hypothesis(
            yseq=torch.cat([self.yseq, torch.tensor([token_id], dtype=torch.int64)]),
            score=self.score + add_score,
            scores={k: new_scores.get(k, v) for k, v in self.scores.items()},
            states=new_states,
            block_idx=self.block_idx,
        )

@dataclass
class BeamState:
    hyps: List[Hypothesis] = field(default_factory=list)
    ended: List[Hypothesis] = field(default_factory=list)
    evaluated: Set[Tuple[int, ...]] = field(default_factory=set)  # repetition tracking
    process_idx: int = 0
    block_idx: int = 0

    def top_k(self, k: int) -> List[Hypothesis]:
        """Return top-k hypotheses by score."""
        return sorted(self.hyps, reverse=True)[:k]  # __lt__ sorts ascending, reverse for descending
```

### 2. Cache Structures

```python
# cache.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

@dataclass
class DecoderKVCache:
    """
    Decoder self-attention K/V cache for one hypothesis.

    Per layer: K (L, D), V (L, D)
    Append-only as hypothesis grows.
    """
    k_cache: List[torch.Tensor]  # list of (L, D) per layer
    v_cache: List[torch.Tensor]  # list of (L, D) per layer

    def __post_init__(self):
        assert len(self.k_cache) == len(self.v_cache)

    def append(self, k_new: List[torch.Tensor], v_new: List[torch.Tensor]) -> "DecoderKVCache":
        """Append new K/V to cache (one token)."""
        assert len(k_new) == len(self.k_cache)
        return DecoderKVCache(
            k_cache=[torch.cat([k, k_n], dim=0) for k, k_n in zip(self.k_cache, k_new)],
            v_cache=[torch.cat([v, v_n], dim=0) for v, v_n in zip(self.v_cache, v_new)],
        )

@dataclass
class CTCState:
    """
    CTC prefix scoring state.

    Tracks forward variables r^n(h), r^b(h) and prefix scores.
    """
    r: torch.Tensor                 # (T, 2, K) forward variables [non-blank, blank]
    s: torch.Tensor                 # (K, V) prefix scores
    last_computed_frame: int        # NEW: for incremental computation
    f_min: int = 0                  # for windowing (optional)
    f_max: int = 0

    def extend_frames(self, new_T: int):
        """Pre-allocate or extend r array to new length."""
        if new_T > self.r.shape[0]:
            # Extend r
            device = self.r.device
            dtype = self.r.dtype
            K = self.r.shape[2]
            new_r = torch.full((new_T, 2, K), -1e10, dtype=dtype, device=device)
            new_r[:self.r.shape[0]] = self.r
            self.r = new_r

@dataclass
class EncoderCache:
    """
    Encoder K/V cache per block (for cross-attention in decoder).

    Optional: if encoder is contextual block, may not need explicit cache.
    """
    k_enc: torch.Tensor             # (T_enc, D)
    v_enc: torch.Tensor             # (T_enc, D)
    block_idx: int                  # which block this cache corresponds to
```

### 3. Caching Strategy

#### Encoder
- **Contextual Block Encoder:** Already stateful (context embeddings, buffers)
- Cache managed internally by encoder wrapper
- **Cross-attention K/V:** Encoder output `h` (T_enc, D) is **shared across all beams** → no per-beam cache needed, just reference

#### Decoder (per hypothesis)
- **Self-attention K/V:** Cached per hypothesis, per layer
- Append new token's K/V at each step → O(1) per token
- **No rebuilds from scratch**

#### CTC (shared across beams, then per-beam)
- **Forward variables `r`:** Incremental computation
- Track `last_computed_frame` in state
- On new block: compute `r[last_computed_frame+1 : new_end]` only
- **Prefix scores `s`:** Updated per hypothesis based on `r`

---

## Migration Plan (Updated with Model Reimplementation)

### Phase 0: Model Layers & Attention (Week 1-2)

**Tasks:**
1. **Basic building blocks** (`model/layers/`)
   - `feed_forward.py`: PositionwiseFeedForward (use `torch.nn.Linear` + activation)
   - `positional_encoding.py`: Absolute + Relative positional encoding
   - `convolution.py`: ConvolutionModule (Conformer depthwise conv)
   - `normalization.py`: LayerNorm wrapper (pre/post norm)
2. **Attention modules** (`model/attention/`)
   - `flash_attention.py`: FlashAttention2 wrapper (with compatibility checks)
   - `vanilla_attention.py`: Standard multi-head attention (fallback for CPU)
   - Auto-select based on hardware (CUDA capability, availability)
3. **Frontend** (`model/frontend/`)
   - `stft_frontend.py`: STFT → LogMel using `torchaudio.transforms`
4. Unit tests for each layer
   - Test numerical correctness (compare with espnet outputs on fixtures)
   - Test Flash Attention equivalence vs. vanilla (within epsilon)

**Deliverables:**
- `model/layers/*.py`
- `model/attention/*.py`
- `model/frontend/stft_frontend.py`
- `tests/model/test_layers.py`
- `tests/model/test_attention.py`

---

### Phase 1: Encoder Implementation (Week 2-3)

**Tasks:**
1. **Contextual Block Conformer Encoder** (`model/encoder/conformer.py`)
   - Conv2d subsampling (4x downsample)
   - ContextualBlockEncoderLayer (macaron FFN, self-attn, conv module, context inheritance)
   - Streaming inference method: `forward_infer(feats, states, is_final=False)`
   - State management: buffers, context embeddings, block boundaries
2. **Contextual Block Transformer Encoder** (`model/encoder/transformer.py`)
   - Similar structure, without conv module
3. Unit tests
   - Test block processing (chunk accumulation, state transitions)
   - Test equivalence with espnet encoder on fixed inputs
   - Test state preservation across chunks

**Deliverables:**
- `model/encoder/conformer.py`
- `model/encoder/transformer.py`
- `model/encoder/layers.py` (shared block layer logic)
- `tests/model/test_conformer.py`
- `tests/model/test_transformer.py`

---

### Phase 2: Decoder & CTC (Week 3)

**Tasks:**
1. **Transformer Decoder** (`model/decoder/transformer_decoder.py`)
   - Incremental decoding with K/V cache
   - Embedding + positional encoding
   - Self-attention + cross-attention layers
   - Output projection (vocab_size)
   - Implement `score()` method for beam search scorer interface
2. **CTC Module** (`model/ctc.py`)
   - Simple linear projection for CTC outputs
   - Use `torch.nn.CTCLoss` for training (not needed for inference)
3. **ESPnet Model** (`model/espnet_model.py`)
   - Inference-only wrapper
   - Load pretrained weights from ESPnet checkpoints
   - Forward methods for encoder + decoder
4. Unit tests
   - Test incremental decoder scoring vs. full forward
   - Test weight loading from real ESPnet checkpoint

**Deliverables:**
- `model/decoder/transformer_decoder.py`
- `model/ctc.py`
- `model/espnet_model.py`
- `tests/model/test_decoder.py`
- `tests/model/test_model_loading.py`

---

### Phase 3: Beam Search & Scorers (Week 4)

**Tasks:**
1. **Base interfaces** (`decoder/base.py`, `decoder/hypothesis.py`, `decoder/cache.py`)
   - Same as original plan
2. **Scorers** (`decoder/scorers.py`)
   - `AttentionScorer`: wraps `model/decoder/transformer_decoder.py`
   - `CTCScorer`: incremental CTC prefix scoring (fix O(n²))
   - `LMScorer`: optional shallow fusion hook
3. **BSBS Beam Search** (`decoder/beam_search_bsbs.py`)
   - Implement Algorithm 1 from paper
   - BBD (reliability score, repetition detection)
   - Vectorized beam expansion
4. Unit tests
   - Test scorer incrementality (no recomputation)
   - Test BBD triggers on synthetic examples
   - Test BSBS vs. baseline on short utterances

**Deliverables:**
- `decoder/base.py`, `decoder/hypothesis.py`, `decoder/cache.py`
- `decoder/scorers.py` (with O(n) CTC)
- `decoder/beam_search_bsbs.py`
- `tests/decoder/test_scorers.py`
- `tests/decoder/test_beam_search.py`

---

### Phase 4: Speech2TextStreaming API (Week 4-5)

**Tasks:**
1. **Speech2TextStreaming** (`decoder/api.py`)
   - Drop-in replacement for `espnet_streaming_decoder.asr_inference_streaming.Speech2TextStreaming`
   - Initialize model from config + weights
   - Frontend (STFT → LogMel)
   - Encoder streaming wrapper
   - Beam search (BSBS)
   - Return format: `List[Tuple[text, tokens, token_ids, positions, hyp]]`
2. **Top-level import** (`speechcatcher/asr_inference_streaming.py`)
   - Re-export `Speech2TextStreaming` from `decoder/api.py`
   - Drop-in replacement: `from speechcatcher.asr_inference_streaming import Speech2TextStreaming`
3. Integration test
   - Load real model checkpoint
   - Decode small audio sample
   - Compare with baseline WER

**Deliverables:**
- `decoder/api.py` (Speech2TextStreaming)
- `speechcatcher/asr_inference_streaming.py` (re-export)
- `tests/decoder/test_api.py` (end-to-end with real model)

---

### Phase 5: Benchmarking & Optimization (Week 5-6)

**Tasks:**
1. **Model benchmarks** (`benchmarks/model_bench.py`)
   - Flash Attention speedup (vs. vanilla)
   - Encoder throughput (frames/sec)
   - Memory usage (peak, per-layer breakdown)
2. **Decoder benchmarks** (`benchmarks/decoder_bench.py`)
   - Latency per chunk (ms)
   - RTF (Real-Time Factor)
   - Long-utterance scaling (confirm O(n))
3. **Accuracy validation**
   - WER on German test set (compare with baseline)
   - Target: ±0.3 absolute WER
4. Profile and optimize hotspots
   - Use `torch.profiler`
   - Vectorize loops
   - Consider `torch.compile` (PyTorch 2.0+)

**Deliverables:**
- `benchmarks/model_bench.py`
- `benchmarks/decoder_bench.py`
- Performance report in `decoder.md` (results section)

---

### Phase 6: Documentation & Polish (Week 6)

**Tasks:**
1. Comprehensive docstrings (Numpy/Google style)
2. Documentation (`docs/`)
   - `model_architecture.md`: Model components, Flash Attention, weight loading
   - `decoder_rewrite.md`: BSBS algorithm, usage, config params
   - Migration guide: How to switch from `espnet_streaming_decoder` to `speechcatcher`
3. Example scripts
   - `examples/streaming_decode.py`: Minimal working example
   - `examples/benchmark_models.py`: Compare baseline vs. new implementation
4. Code quality
   - Run `ruff` + `black`
   - Type checking with `mypy`
   - Coverage report (target: >85%)

**Deliverables:**
- `docs/model_architecture.md`
- `docs/decoder_rewrite.md`
- `examples/streaming_decode.py`
- Clean, formatted, type-checked code

---

### Phase 7: Integration & Cleanup (Week 6-7)

**Tasks:**
1. **Update speechcatcher.py**
   - Change import: `from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming`
   - To: `from speechcatcher.asr_inference_streaming import Speech2TextStreaming`
   - Verify no other changes needed (drop-in replacement)
2. **Remove espnet_streaming_decoder dependency**
   - Update `requirements.txt` / `pyproject.toml`
   - Remove `espnet-streaming-decoder` from dependencies
   - Add new dependencies: `flash-attn` (optional), `torchaudio`
3. **CI/CD**
   - Add tests to GitHub Actions (if applicable)
   - Test on CPU + GPU (if available)
4. **Final validation**
   - Run full speechcatcher workflow (live + file transcription)
   - Compare WER on multiple test files
   - Measure end-to-end RTF

**Deliverables:**
- Updated `speechcatcher/speechcatcher.py`
- Updated `requirements.txt` / `pyproject.toml`
- Removed `espnet-streaming-decoder` dependency
- Final WER + RTF report

---

## Model Architecture Analysis

### Complete Dependency Chain (What Needs Reimplementation)

Based on code inspection, here's the **full stack** that needs to be reimplemented to eliminate the `espnet_streaming_decoder` dependency:

#### 1. **Core Model Components** (Must Reimplement)

**Location in espnet_streaming_decoder:**
```
espnet2/asr/
├── espnet_model.py                          # ESPnetASRModel (CTC/Attention hybrid)
├── ctc.py                                   # CTC module
├── encoder/
│   ├── contextual_block_conformer_encoder.py  # ← CRITICAL: Streaming Conformer
│   ├── contextual_block_transformer_encoder.py # ← CRITICAL: Streaming Transformer
│   ├── conformer_encoder.py                 # Non-streaming Conformer (reference)
│   └── transformer_encoder.py               # Non-streaming Transformer (reference)
├── decoder/
│   └── transformer_decoder.py               # ← CRITICAL: Attention decoder
├── frontend/
│   └── default.py                           # STFT frontend (LogMel, etc.)
└── layers/
    └── normalize.py                         # Global mean/var normalization
```

**Low-level building blocks:**
```
espnet/nets/pytorch_backend/
├── transformer/
│   ├── attention.py                         # ← MultiHeadedAttention (vanilla, no flash)
│   ├── encoder_layer.py                     # Transformer encoder layer
│   ├── decoder_layer.py                     # Transformer decoder layer
│   ├── embedding.py                         # Positional encoding
│   ├── positionwise_feed_forward.py         # FFN
│   ├── layer_norm.py                        # Pre-norm/Post-norm LayerNorm
│   ├── subsampling.py                       # Conv2d subsampling (4x downsample)
│   └── mask.py                              # Causal masks
├── conformer/
│   ├── contextual_block_encoder_layer.py    # ← CRITICAL: Contextual block layer
│   ├── encoder_layer.py                     # Conformer layer (standard)
│   ├── convolution.py                       # ConvolutionModule (depthwise conv)
│   └── swish.py                             # Swish activation
└── nets_utils.py                            # Padding, masking utils
```

#### 2. **What to Reimplement vs. Reuse**

| Component | Action | Reason |
|-----------|--------|--------|
| **ESPnetASRModel** | Reimplement (simplified) | Drop transducer, training code; keep inference essentials |
| **ContextualBlockConformerEncoder** | Reimplement (modernized) | Core streaming encoder; add **Flash Attention 2** |
| **ContextualBlockTransformerEncoder** | Reimplement (modernized) | Same, Transformer variant |
| **TransformerDecoder** | Reimplement (simplified) | Keep incremental decoding; add Flash Attention |
| **MultiHeadedAttention** | **Replace with FlashAttention2** | 🚀 Major perf win (2-4x faster, less memory) |
| **ConvolutionModule** | Reimplement | Conformer-specific; straightforward |
| **Positional Encoding** | Reimplement | Small, easy |
| **Layer Norm / FFN** | **Use `torch.nn` directly** | No need to reimplement |
| **STFT Frontend** | **Use `torchaudio.transforms.MelSpectrogram`** | Standard, well-optimized |
| **CTC** | **Use `torch.nn.CTCLoss` + custom prefix scorer** | Already in PyTorch |

#### 3. **Key Architecture Details**

**ESPnetASRModel structure:**
```python
ESPnetASRModel:
    frontend: AbsFrontend               # STFT → LogMel (80-dim)
    normalize: AbsNormalize             # Global mean/var norm
    encoder: AbsEncoder                 # ContextualBlockConformerEncoder or Transformer
        - Conv2d subsampling (4x)
        - N encoder layers (12-18)
        - Contextual block processing
    decoder: AbsDecoder                 # TransformerDecoder
        - Embedding layer
        - M decoder layers (6)
        - Output projection (vocab_size)
    ctc: CTC                            # CTC head on encoder output
```

**Contextual Block Conformer Layer:**
```python
ContextualBlockEncoderLayer:
    # Macaron-style FFN
    feed_forward_macaron (optional)
    # Self-attention with context inheritance
    self_attn: MultiHeadedAttention (RelPos variant)
    # Convolution module
    conv_module: ConvolutionModule
        - LayerNorm
        - Pointwise conv (expansion)
        - GLU activation
        - Depthwise conv (kernel=31)
        - BatchNorm
        - Swish
        - Pointwise conv (projection)
    # FFN
    feed_forward: PositionwiseFeedForward
    # Context embedding (for block processing)
    feed_forward_ctx: FFN for context vector
```

**Streaming Inference Flow:**
```
Audio chunk (raw waveform)
    ↓
Frontend (STFT → LogMel) → features (B, T, 80)
    ↓
Normalize (global mean/var) → normalized features
    ↓
Encoder.forward_infer(feats, encoder_states, is_final=False)
    - Buffer management (left context, look-ahead)
    - Conv2d subsampling (4x)
    - Process complete blocks with context inheritance
    - Return: encoded (B, T_enc, D), next_states
    ↓
BeamSearch(encoded, is_final=False)
    - Extend CTC scorer with new frames
    - Decode with BSBS
    - Return: partial hypotheses
```

#### 4. **Flash Attention 2 Upgrade Path**

**Current (espnet):**
```python
# espnet/nets/pytorch_backend/transformer/attention.py:109
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
attn = torch.softmax(scores, dim=-1)
output = torch.matmul(attn, v)  # O(N²) memory for attention matrix
```

**Target (Flash Attention 2):**
```python
from flash_attn import flash_attn_func

# flash_attn_func automatically handles:
# - Fused kernel (no materialized attention matrix)
# - O(N) memory instead of O(N²)
# - 2-4x faster on A100/H100
output = flash_attn_func(q, k, v, causal=False, softmax_scale=1/sqrt(d_k))
```

**Benefits:**
- **Memory:** O(N) vs O(N²) → support longer sequences
- **Speed:** 2-4x faster on modern GPUs (A100, H100, RTX 4090)
- **Throughput:** Higher batch sizes fit in memory

**Compatibility:**
- Requires: `pip install flash-attn` (optional dep, fallback to torch)
- GPU: CUDA compute capability ≥ 8.0 (A100, RTX 30XX+)
- Fallback: Keep vanilla attention for CPU / older GPUs

---

## Proposed Repository Structure (Complete)

```
speechcatcher/
├── asr_inference_streaming.py       # Drop-in replacement (imports from decoder/)
├── model/                            # ← NEW: Model implementations
│   ├── __init__.py
│   ├── espnet_model.py               # ESPnetASRModel (inference-only)
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── conformer.py              # ContextualBlockConformerEncoder
│   │   ├── transformer.py            # ContextualBlockTransformerEncoder
│   │   └── layers.py                 # Shared layers (ContextualBlockLayer, Conv2dSubsampling)
│   ├── decoder/
│   │   ├── __init__.py
│   │   └── transformer_decoder.py    # TransformerDecoder (incremental)
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── flash_attention.py        # FlashAttention2 wrapper
│   │   └── vanilla_attention.py      # Fallback attention
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── convolution.py            # ConvolutionModule (Conformer)
│   │   ├── positional_encoding.py    # Abs/Rel positional encoding
│   │   ├── feed_forward.py           # PositionwiseFeedForward
│   │   └── normalization.py          # LayerNorm variants
│   ├── frontend/
│   │   ├── __init__.py
│   │   └── stft_frontend.py          # STFT → LogMel
│   └── ctc.py                        # CTC head
├── decoder/                          # Beam search & scoring (from previous plan)
│   ├── __init__.py
│   ├── api.py                        # Speech2TextStreaming (uses model/)
│   ├── base.py
│   ├── beam_search_bsbs.py
│   ├── hypothesis.py
│   ├── scorers.py
│   ├── cache.py
│   └── utils.py
├── tests/
│   ├── model/                        # ← NEW
│   │   ├── test_conformer.py
│   │   ├── test_transformer.py
│   │   ├── test_attention.py
│   │   └── test_layers.py
│   └── decoder/
│       └── ...
├── benchmarks/
│   ├── model_bench.py                # ← NEW: Flash Attention speedup
│   └── decoder_bench.py
└── docs/
    ├── model_architecture.md         # ← NEW: Model docs
    └── decoder_rewrite.md
```

---

## Next Steps

### Immediate Actions (Step 2 Start)

1. **Create branch:**
   ```bash
   git checkout -b feat/decoder-rewrite-bsbs
   ```

2. **Scaffold directories:**
   ```bash
   mkdir -p speechcatcher/model/{encoder,decoder,attention,layers,frontend}
   mkdir -p speechcatcher/decoder
   mkdir -p tests/{model,decoder}
   mkdir -p benchmarks
   touch speechcatcher/model/__init__.py
   touch speechcatcher/decoder/__init__.py
   ```

3. **Start with Phase 1 (updated):**
   - Write `model/layers/` building blocks (FFN, LayerNorm, PositionalEncoding, Convolution)
   - Write `model/attention/` with Flash Attention 2 + vanilla fallback
   - Write unit tests for each layer
   - Write `decoder/base.py`, `hypothesis.py`, `cache.py` (beam search foundations)

### Success Criteria (Recap)

- **Accuracy:** WER within ±0.3 absolute of baseline
- **Complexity:** No O(n²) hotspots; step time stable as history grows
- **Latency:** Stable per-chunk decode time; RTF ≤ baseline
- **Tests:** >90% coverage, property tests for invariants
- **Docs:** Complete API docs, migration guide, examples

---

## Appendix

### A. Key Equations from Paper

#### Reliability Score
```
r(y_{0:i-1}, h_{1:b}) = max_{0≤j≤i-1} log p(y_j | y_{0:i-1}, h_{1:b}) + α(y_{0:i-1}, h_{1:b})

s(y_{0:i}, h_{1:b}) = α(y_{0:i}, h_{1:b}) - r(y_{0:i-1}, h_{1:b})
```

#### CTC Prefix Score
```
p_ctc(y_{0:i} | h_{1:b}) = γ_{T_b}^(N)(y_{0:i-1}) + γ_{T_b}^(B)(y_{0:i-1})
```

#### Attention Score Approximation
```
log p(y_i | y_{0:i-1}, h_{1:B}) ≈ log p(y_i | y_{0:i-1}, h_{1:b})
```

### B. References

- **Paper:** Tsunoo et al., "Streaming Transformer ASR with Blockwise Synchronous Beam Search", SLT 2021
  - ArXiv: [2006.14941](https://arxiv.org/abs/2006.14941)
- **ESPnet:** [github.com/espnet/espnet](https://github.com/espnet/espnet)
- **espnet_streaming_decoder:** `pip install espnet-streaming-decoder==0.1.2`

### C. Glossary

- **BSBS:** Blockwise Synchronous Beam Search
- **BBD:** Block Boundary Detection
- **CBP-ENC:** Contextual Block Processing Encoder
- **CTC:** Connectionist Temporal Classification
- **SAN:** Self-Attention Network
- **STA:** Source-Target Attention
- **RTF:** Real-Time Factor (processing_time / audio_duration)
- **WER/CER:** Word/Character Error Rate

---

**End of Step 1 Analysis**

*Next: Implement Phase 1 (Scaffolding & Interfaces) in `speechcatcher/decoder/`*
