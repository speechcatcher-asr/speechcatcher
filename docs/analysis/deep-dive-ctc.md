# Deep Dive: CTC Implementation Comparison

**Date:** 2025-10-10
**Purpose:** Comprehensive technical comparison of CTC prefix scoring implementations
**Status:** Post State-Structure-Fix Analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [CTC Prefix Scoring Mathematics](#ctc-prefix-scoring-mathematics)
3. [ESPnet Implementation Analysis](#espnet-implementation-analysis)
4. [Our Implementation Analysis](#our-implementation-analysis)
5. [Critical Differences](#critical-differences)
6. [Numerical Stability](#numerical-stability)
7. [State Management Deep Dive](#state-management-deep-dive)
8. [Implementation Plan](#implementation-plan)

---

## Executive Summary

###  Current Status

**What Works:**
- ✅ Dict-based state structure matches ESPnet
- ✅ Hypothesis expansion merges all scorer states
- ✅ xpos tracking implemented
- ✅ Basic CTC algorithm implemented

**What's Broken:**
- ❌ CTC causes timeout/hang
- ❌ Missing `extend_prob()` and `extend_state()` methods
- ❌ Incorrect time-step tracking
- ❌ State not properly accumulated across blocks
- ❌ Simplified averaging instead of proper prefix algorithm

### Root Cause

Our CTC implementation is **fundamentally incomplete**. We implemented a time-step-by-time-step version, but ESPnet uses:
1. **Full probability matrix** accumulated across all blocks
2. **Forward algorithm over entire time axis** for each prefix
3. **extend_prob/extend_state** to handle streaming

We tried to shortcut this with frame-averaging, which doesn't work.

---

## CTC Prefix Scoring Mathematics

### The Problem

Given:
- Encoder output $\mathbf{X} = \{x_1, ..., x_T\}$ (T time steps)
- CTC posterior $p(k|x_t)$ for token $k$ at time $t$
- Current prefix $\mathbf{y} = [y_1, ..., y_u]$

**Goal:** Compute $P(\mathbf{y} \cdot k | \mathbf{X})$ for each next token $k$

### CTC Output Space

CTC output includes blank token ($\varnothing$). Multiple alignments map to same label sequence:
```
"cat" can be: c-a-t, cc-aa-tt, c-∅-a-∅-t-∅, etc.
```

### Forward Variables

**Definition (Watanabe et al. 2017):**

$$r_t^n(\mathbf{y}) = \sum_{\pi: \mathcal{B}(\pi_{1:t}) = \mathbf{y}, \pi_t \neq \varnothing} P(\pi_{1:t}|\mathbf{X})$$

$$r_t^b(\mathbf{y}) = \sum_{\pi: \mathcal{B}(\pi_{1:t}) = \mathbf{y}, \pi_t = \varnothing} P(\pi_{1:t}|\mathbf{X})$$

Where:
- $r_t^n(\mathbf{y})$: probability of prefix $\mathbf{y}$ ending with non-blank at time $t$
- $r_t^b(\mathbf{y})$: probability of prefix $\mathbf{y}$ ending with blank at time $t$
- $\mathcal{B}(\cdot)$: CTC collapse function (removes blanks & repeats)

### Recursive Computation

**For empty prefix ($\mathbf{y} = []$):**
$$r_0^b([]) = 1, \quad r_t^b([]) = r_{t-1}^b([]) \cdot p(\varnothing|x_t)$$

**For non-empty prefix ($\mathbf{y} = [y_1, ..., y_u]$):**

Let $y_u$ be the last token in prefix.

**Non-blank extension (adding token $k \neq \varnothing, k \neq y_u$):**
$$r_t^n(\mathbf{y} \cdot k) = [r_{t-1}^n(\mathbf{y}) + r_{t-1}^b(\mathbf{y})] \cdot p(k|x_t)$$

**Same token extension (adding $k = y_u$):**
$$r_t^n(\mathbf{y} \cdot k) = r_{t-1}^n(\mathbf{y}) \cdot p(k|x_t)$$
(Must have blank between same tokens)

**Blank extension:**
$$r_t^b(\mathbf{y}) = [r_{t-1}^n(\mathbf{y}) + r_{t-1}^b(\mathbf{y})] \cdot p(\varnothing|x_t)$$

### Prefix Probability

Total probability of prefix $\mathbf{y}$ at time $T$:
$$P(\mathbf{y}|\mathbf{X}) = r_T^n(\mathbf{y}) + r_T^b(\mathbf{y})$$

### Log Space

All computations done in log space:
$$\log(a + b) = \log a + \log(1 + \exp(\log b - \log a))$$

Using `torch.logsumexp` or `np.logaddexp` for numerical stability.

---

## ESPnet Implementation Analysis

### File Structure

```
espnet_streaming_decoder/espnet/nets/
├── ctc_prefix_score.py           # Core CTC algorithm
│   ├── CTCPrefixScoreTH          # Batched torch implementation
│   └── CTCPrefixScore            # Single-hyp numpy implementation
├── scorers/ctc.py                # Scorer interface wrapper
│   └── CTCPrefixScorer           # Wraps ctc_prefix_score for beam search
└── batch_beam_search_online.py  # Integration point
    └── extend() method           # Calls extend_prob/extend_state
```

### CTCPrefixScoreTH Architecture

**Initialization (`__init__`):**
```python
def __init__(self, x, xlens, blank, eos, margin=0):
    # x: (B, T, O) - batch CTC posteriors
    # Store FULL probability matrix
    self.x = torch.stack([xn, xb])  # (2, T, B, O)
    #  dim 0: [0]=non-blank probs, [1]=blank probs
    #  dim 1: time T
    #  dim 2: batch B
    #  dim 3: vocab O
```

**Key Insight:** ESPnet stores the **entire probability matrix** for all time steps!

**Forward Computation (`__call__`):**
```python
def __call__(self, y, state, scoring_ids=None, att_w=None):
    # y: list of prefixes (batch * hyps)
    # state: (r_prev, s_prev, f_min, f_max) or None

    # Initialize r (forward variables)
    r = torch.full((T, 2, n_bh, snum), logzero)  # (T, 2, batch*hyps, scoring_num)

    # Get previous cumulative prefix probs
    r_sum = torch.logsumexp(r_prev, 1)  # Sum over non-blank/blank

    # Setup log_phi (transition probabilities)
    log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)
    for idx in range(n_bh):
        # Special case: if extending with same token, use r_prev[:, 1] (blank-ending)
        log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

    # Forward recursion over time
    for t in range(start, end):
        rp = r[t - 1]
        # Stack previous: [r^n, log_phi, r^n, r^b]
        rr = torch.stack([rp[0], log_phi[t-1], rp[0], rp[1]]).view(2, 2, n_bh, snum)
        # Compute: r[t] = logsumexp(rr, dim=1) + x[t]
        r[t] = torch.logsumexp(rr, 1) + x_[:, t]

    # Compute final prefix probabilities
    log_psi = torch.logsumexp(
        torch.cat((log_phi_x[start:end], r[start-1, 0].unsqueeze(0)), dim=0),
        dim=0
    )

    # Return incremental scores
    return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)
```

**State Structure:**
```python
state = (r, s_prev, f_min, f_max, scoring_idmap)
# r: (T, 2, n_bh) - forward variables for all time steps!
# s_prev: (n_bh, vocab_size) - previous prefix scores
# f_min/f_max: windowing for attention-based pruning
# scoring_idmap: mapping for partial scoring
```

### extend_prob Method

**Purpose:** Accumulate new encoder output into probability matrix

```python
def extend_prob(self, x: torch.Tensor):
    """Extend CTC prob matrix with new encoder output.

    x: (B, T_new, O) - new encoder output
    """
    if self.x.shape[1] < x.shape[1]:  # Need to extend
        # Pad new probs
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        xn = logp.transpose(0, 1)  # (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)

        # Concatenate to existing
        tmp_x = self.x
        self.x = torch.stack([xn, xb])
        self.x[:, :tmp_x.shape[1], :, :] = tmp_x  # Keep old values

        self.input_length = x.size(1)  # Update length
```

**Key:** Probability matrix grows as new encoder blocks arrive!

### extend_state Method

**Purpose:** Extend forward variables when probability matrix grows

```python
def extend_state(self, state):
    """Extend r_prev to new time length."""
    if state is None:
        return state

    r_prev, s_prev, f_min_prev, f_max_prev = state

    # Create new r with extended time dimension
    r_prev_new = torch.full((self.input_length, 2), logzero)
    start = max(r_prev.shape[0], 1)
    r_prev_new[0:start] = r_prev  # Copy old values

    # Fill new time steps with cumulative blank probs
    for t in range(start, self.input_length):
        r_prev_new[t, 1] = r_prev_new[t-1, 1] + self.x[0, t, :, self.blank]

    return (r_prev_new, s_prev, f_min_prev, f_max_prev)
```

### Wrapper Integration

**File:** `espnet/nets/scorers/ctc.py`

```python
class CTCPrefixScorer(BatchPartialScorerInterface):
    def batch_init_state(self, x: torch.Tensor):
        """Initialize with encoder output."""
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None  # State managed inside impl

    def extend_prob(self, x: torch.Tensor):
        """Called when new encoder block arrives."""
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend all hypothesis states."""
        return [self.impl.extend_state(s) for s in state]

    def batch_score_partial(self, y, ids, state, x):
        """Score batch of hypotheses."""
        # Merge states into batch
        batch_state = (
            torch.stack([s[0] for s in state], dim=2),  # r
            torch.stack([s[1] for s in state]),          # s_prev
            state[0][2],  # f_min
            state[0][3],  # f_max
        ) if state[0] is not None else None

        return self.impl(y, batch_state, ids)
```

### Integration in Beam Search

**File:** `espnet/nets/batch_beam_search_online.py`

```python
def process_one_block(self, h, is_final, maxlen, maxlenratio):
    # BEFORE decoding: extend CTC with new encoder output
    self.extend(h, self.running_hyps)  # <-- Critical!

    # Then run beam search
    while self.process_idx < maxlen:
        best = self.search(self.running_hyps, h)
        # ...

def extend(self, x: torch.Tensor, hyps: Hypothesis):
    """Extend scorers with new encoder output."""
    for k, d in self.scorers.items():
        if hasattr(d, "extend_prob"):
            d.extend_prob(x)  # Grow probability matrix
        if hasattr(d, "extend_state"):
            hyps.states[k] = d.extend_state(hyps.states[k])  # Grow states
```

**Flow:**
1. New encoder block arrives → `extend_prob()` adds to CTC matrix
2. Existing hypothesis states extended → `extend_state()`
3. Beam search runs with updated CTC scorer
4. CTC scores computed using **full forward algorithm** over all accumulated time

---

## Our Implementation Analysis

### Our CTCPrefixScore

**File:** `speechcatcher/beam_search/ctc_prefix_score.py`

**Key Issues:**

#### 1. Time Tracking Wrong
```python
def __call__(self, y, cs, state):
    # Forward to current time step (one step per prefix token)
    t = min(self.t + 1, self.x.shape[0] - 1)  # ❌ WRONG!
    self.t = t
```

**Problem:** Incrementing `self.t` globally doesn't make sense. Different prefixes are at different positions in the encoder output!

#### 2. Single Time Step Scoring
```python
# Get log probs at time t
log_probs = self.x[t]  # (vocab_size,)  # ❌ Only one frame!

for i, c in enumerate(cs):
    # Score using single time step
    scores[i] = ... + log_probs[c]  # ❌ Missing full forward computation
```

**Problem:** We only look at one time step, not the full forward algorithm!

#### 3. State Not Accumulated
```python
# Update state for this prefix
new_r = np.full(self.vocab_size, -np.inf)
new_log_psi = np.full(self.vocab_size, -np.inf)
```

**Problem:** State is per-prefix, but we're not maintaining it properly across time!

### Our Simplified CTC Scorer

**File:** `speechcatcher/beam_search/scorers.py:152-190`

```python
def batch_score(self, yseqs, states, xs):
    # Get CTC log probabilities
    logits = self.ctc.ctc_lo(xs)  # (batch, enc_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)

    # SIMPLIFIED: Use mean across time
    mean_log_probs = log_probs.mean(dim=1)  # (batch, vocab_size)

    # Apply penalty to blank
    mean_log_probs[:, self.blank_id] -= 2.0

    return mean_log_probs, states
```

**Problems:**
1. ❌ Not using prefix algorithm at all!
2. ❌ Just averaging probabilities (not correct)
3. ❌ Ignoring prefix history
4. ❌ No forward variables
5. ❌ No extend_prob/extend_state

---

## Critical Differences

### Difference 1: Probability Matrix Storage

| Aspect | ESPnet | Ours |
|--------|--------|------|
| **Storage** | Full matrix (2, T, B, O) | Per-call only (simplified) |
| **Growth** | Accumulates via `extend_prob()` | N/A |
| **Time axis** | All T frames stored | Only current frame |
| **Memory** | O(T × V) | O(V) |

**Impact:** We can't do proper CTC prefix scoring without the full matrix!

### Difference 2: Forward Algorithm

| Aspect | ESPnet | Ours |
|--------|--------|------|
| **Algorithm** | Full forward over time axis | Single time step |
| **Variables** | $r_t^n, r_t^b$ for all $t$ | Only current |
| **Recursion** | `for t in range(start, end)` | None |
| **Correctness** | ✅ Mathematically correct | ❌ Fundamentally wrong |

### Difference 3: State Management

**ESPnet State:**
```python
state = (
    r,           # (T, 2, n_bh) - Forward variables ALL time steps
    s_prev,      # (n_bh, vocab) - Previous prefix scores
    f_min/f_max, # Windowing bounds
    scoring_idmap # Partial scoring map
)
```

**Our State:**
```python
state = (
    r,        # (vocab_size,) - Single vector (wrong!)
    log_psi,  # (vocab_size,) - Single vector
)
```

**Problem:** We don't track forward variables across time!

### Difference 4: extend Methods

| Feature | ESPnet | Ours |
|---------|--------|------|
| `extend_prob()` | ✅ Implemented | ❌ Missing |
| `extend_state()` | ✅ Implemented | ❌ Missing |
| Streaming support | ✅ Yes | ❌ No |
| Block accumulation | ✅ Yes | ❌ No |

### Difference 5: Numerical Stability

**ESPnet:**
```python
self.logzero = -10000000000.0  # Very negative but not -inf
torch.logsumexp(...)  # Stable log-sum-exp
```

**Ours:**
```python
-np.inf  # Can cause NaN issues
# Custom log_add() function - okay but not batched
```

### Difference 6: Batching

**ESPnet:**
- Processes batch × hypotheses together
- Vectorized operations: `r[t] = logsumexp(rr, 1) + x_[:, t]`
- Efficient GPU

**Ours:**
- Loop over candidates: `for i, c in enumerate(cs):`
- Not vectorized
- Slow

---

## Numerical Stability

### Log-Space Operations

**Why log space?**

Probabilities multiply: $P(A \cap B) = P(A) \cdot P(B)$

In log space: $\log P(A \cap B) = \log P(A) + \log P(B)$ ✅ More stable!

**Problem:** Addition in log space

$$\log(a + b) = \log(e^{\log a} + e^{\log b})$$

Naive implementation causes overflow/underflow.

### LogSumExp Trick

**Stable computation:**
```python
def logsumexp(a, b):
    # Ensure a >= b
    if a < b:
        a, b = b, a

    # log(exp(a) + exp(b)) = a + log(1 + exp(b - a))
    return a + np.log1p(np.exp(b - a))
```

**PyTorch provides:** `torch.logsumexp(tensor, dim)`

### Log-Zero Value

**ESPnet approach:**
```python
logzero = -10000000000.0  # Very negative, not -inf
```

**Why not -inf?**
- `-inf + x = -inf` (correct)
- `exp(-inf) = 0` (correct)
- But: `-inf - (-inf) = NaN` ❌

Using large negative number avoids NaN in edge cases.

### Float Precision

**ESPnet:**
```python
dtype=torch.float32  # Single precision sufficient
```

CTC scores are probabilities (range [0, 1]). After log: range [-inf, 0].

Float32 has ~7 decimal digits precision, sufficient for log probabilities.

---

## State Management Deep Dive

### ESPnet State Evolution

**Block 1 arrives:**
```python
# extend_prob() called with encoder_out_1
scorer.impl.x = log_probs_1  # Shape: (2, T1, B, O)

# Hypotheses initialized
hyp.states["ctc"] = None

# First scoring call
state = None
r_prev = init  # (T1, 2) initialized with cumulative blanks
scores, state = scorer(hyp.yseq, state, ...)
# state = (r, s_prev, 0, T1-1)

hyp.states["ctc"] = state
```

**Block 2 arrives:**
```python
# extend_prob() called with encoder_out_2
scorer.impl.x = cat([log_probs_1, log_probs_2])  # (2, T1+T2, B, O)

# extend_state() called for each hypothesis
for hyp in hypotheses:
    old_state = (r, s, f_min, f_max)  # r is (T1, 2)
    new_r = torch.full((T1+T2, 2), logzero)
    new_r[:T1] = r  # Keep old
    # Fill T1:T1+T2 with cumulative blanks
    for t in range(T1, T1+T2):
        new_r[t, 1] = new_r[t-1, 1] + x[0, t, blank]
    new_state = (new_r, s, f_min, f_max)
    hyp.states["ctc"] = new_state

# Scoring now uses full matrix (T1+T2)
scores, state = scorer(hyp.yseq, state, ...)
```

**Key Points:**
1. Probability matrix grows monotonically
2. Forward variables extended for new time steps
3. Old values preserved
4. Forward algorithm runs over ALL time (T1+T2)

### Our State (Current - Broken)

```python
# We don't have extend_prob/extend_state!
# Each block is scored independently
# States don't accumulate
```

**Why it fails:**
- Each block sees only its own time steps
- No continuity between blocks
- Forward algorithm can't look at full history

---

## Implementation Plan

### Phase 1: Full ESPnet-Compatible CTC ✅ Priority 1

**Goal:** Implement complete CTC prefix scorer with extend methods

#### 1.1 Rewrite CTCPrefixScoreTH

**File:** `speechcatcher/beam_search/ctc_prefix_score.py`

```python
class CTCPrefixScoreTH:
    """Full implementation matching ESPnet."""

    def __init__(self, x, xlens, blank, eos, margin=0):
        """
        Args:
            x: (B, T, O) - CTC log probabilities
            xlens: (B,) - sequence lengths
            blank: blank token ID
            eos: EOS token ID
            margin: windowing margin (0=disabled)
        """
        self.logzero = -10000000000.0  # Not -inf!
        self.blank = blank
        self.eos = eos
        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.device = x.device
        self.dtype = x.dtype

        # Pad beyond xlens
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0  # log(1) = 0 for blank

        # Store full matrix: (2, T, B, O)
        # [0] = non-blank probs, [1] = blank probs
        xn = x.transpose(0, 1)  # (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x = torch.stack([xn, xb])  # (2, T, B, O)

        self.end_frames = torch.as_tensor(xlens) - 1
        self.margin = margin

        # Indices for batch operations
        self.idx_b = torch.arange(self.batch, device=self.device)
        self.idx_bo = (self.idx_b * self.odim).unsqueeze(1)

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """
        Compute CTC prefix scores for next labels.

        Args:
            y: List of prefix sequences (batch*hyps), each (prefix_len,)
            state: Tuple (r_prev, s_prev, f_min, f_max) or None
            scoring_ids: (batch*hyps, n_candidates) partial scoring indices
            att_w: Attention weights for windowing (optional)

        Returns:
            (log_psi - s_prev, new_state)
            - log_psi: (batch*hyps, vocab) prefix scores
            - new_state: (r, log_psi, f_min, f_max, scoring_idmap)
        """
        output_length = len(y[0]) - 1  # Ignore SOS
        last_ids = [yi[-1] for yi in y]  # Last token in each prefix
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch

        # Initialize state
        if state is None:
            # Initial: r^b = cumsum(blank probs)
            r_prev = torch.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero, dtype=self.dtype, device=self.device
            )
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # Select scoring dimensions (partial scoring optimization)
        if scoring_ids is not None:
            snum = scoring_ids.size(-1)
            # Index mapping for partial scores
            # ... (implement partial scoring)
        else:
            snum = self.odim
            x_ = self.x.unsqueeze(3).repeat(1, 1, 1, n_hyps, 1).view(2, -1, n_bh, snum)

        # Allocate forward variables: (T, 2, n_bh, snum)
        r = torch.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero, dtype=self.dtype, device=self.device
        )

        if output_length == 0:
            r[0, 0] = x_[0, 0]  # Initial non-blank

        # Compute r_sum = log(r^n + r^b)
        r_sum = torch.logsumexp(r_prev, 1)  # (T, n_bh)

        # Setup log_phi (transition matrix)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)  # (T, n_bh, snum)

        # Special case: same token requires blank (use r^b only)
        for idx in range(n_bh):
            if scoring_ids is not None:
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]  # blank-ending only
            else:
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # Windowing (if attention weights provided)
        if att_w is not None and self.margin > 0:
            # Compute window boundaries from attention
            # ... (implement windowing)
            start = ...
            end = ...
        else:
            start = max(output_length, 1)
            end = self.input_length

        # Forward recursion (THE CORE ALGORITHM!)
        for t in range(start, end):
            rp = r[t - 1]  # Previous: (2, n_bh, snum)

            # Transition matrix:
            # r^n[t] can come from: r^n[t-1] or log_phi[t-1]
            # r^b[t] can come from: r^n[t-1] or r^b[t-1]
            rr = torch.stack([
                rp[0],           # r^n[t-1] -> r^n[t]
                log_phi[t-1],    # log_phi[t-1] -> r^n[t]
                rp[0],           # r^n[t-1] -> r^b[t]
                rp[1]            # r^b[t-1] -> r^b[t]
            ]).view(2, 2, n_bh, snum)

            # Update: r[t] = logsumexp over transitions + emission prob
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]
            # r[t, 0] = logsumexp([rp[0], log_phi[t-1]]) + x_[0, t]  # r^n
            # r[t, 1] = logsumexp([rp[0], rp[1]]) + x_[1, t]          # r^b

        # Compute prefix probabilities log_psi
        log_phi_x = torch.cat((
            log_phi[0].unsqueeze(0),
            log_phi[:-1]
        ), dim=0) + x_[0]  # (T, n_bh, snum)

        log_psi = torch.logsumexp(
            torch.cat((
                log_phi_x[start:end],
                r[start-1, 0].unsqueeze(0)
            ), dim=0),
            dim=0
        )  # (n_bh, snum)

        # Handle EOS specially
        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # Exclude blank
        log_psi[:, self.blank] = self.logzero

        # Return incremental scores (relative to previous)
        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def extend_prob(self, x):
        """Extend probability matrix with new encoder output.

        Args:
            x: (B, T_new, O) - new CTC posteriors
        """
        if self.x.shape[1] < x.shape[1]:
            # Pad beyond lengths
            xlens = [x.size(1)]
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0

            # Reshape new probs
            xn = x.transpose(0, 1)  # (T_new, B, O)
            xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
            x_new = torch.stack([xn, xb])  # (2, T_new, B, O)

            # Concatenate (keep old, add new)
            tmp_x = self.x
            self.x = x_new
            self.x[:, :tmp_x.shape[1], :, :] = tmp_x

            self.input_length = x.size(1)
            self.end_frames = torch.as_tensor(xlens) - 1

    def extend_state(self, state):
        """Extend state forward variables to new time length.

        Args:
            state: (r_prev, s_prev, f_min, f_max) or None

        Returns:
            Extended state
        """
        if state is None:
            return state

        r_prev, s_prev, f_min_prev, f_max_prev = state

        # Extend r_prev to new time length
        r_prev_new = torch.full(
            (self.input_length, 2),
            self.logzero, dtype=self.dtype, device=self.device
        )
        start = max(r_prev.shape[0], 1)
        r_prev_new[0:start] = r_prev  # Copy existing

        # Fill new time steps with cumulative blank
        for t in range(start, self.input_length):
            r_prev_new[t, 1] = r_prev_new[t-1, 1] + self.x[0, t, :, self.blank]

        return (r_prev_new, s_prev, f_min_prev, f_max_prev)

    def index_select_state(self, state, best_ids):
        """Select states according to beam pruning.

        Args:
            state: CTC state
            best_ids: (B, W) selected hypothesis indices

        Returns:
            Pruned state
        """
        r, s, f_min, f_max, scoring_idmap = state
        n_bh = len(s)
        n_hyps = n_bh // self.batch

        # Convert ids to batch-hyp-output space
        vidx = (best_ids + (self.idx_b * (n_hyps * self.odim)).view(-1, 1)).view(-1)

        # Select hypothesis scores
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(n_bh, self.odim)

        # Select forward probabilities
        if scoring_idmap is not None:
            # ... (implement partial scoring selection)
            pass
        else:
            snum = self.odim

        r_new = torch.index_select(
            r.view(-1, 2, n_bh * snum), 2, vidx
        ).view(-1, 2, n_bh)

        return r_new, s_new, f_min, f_max
```

#### 1.2 Update CTCPrefixScorer Wrapper

**File:** `speechcatcher/beam_search/scorers.py`

Replace simplified version with full ESPnet-compatible wrapper:

```python
class CTCPrefixScorer(ScorerInterface):
    """Full CTC prefix scorer with extend support."""

    def __init__(self, ctc: nn.Module, blank_id: int = 0, eos_id: int = 2):
        self.ctc = ctc
        self.blank_id = blank_id
        self.eos_id = eos_id
        self.impl = None  # CTCPrefixScoreTH instance

    def batch_init_state(self, x: torch.Tensor):
        """Initialize with encoder output."""
        # Get CTC posteriors
        with torch.no_grad():
            logits = self.ctc.ctc_lo(x.unsqueeze(0))  # (1, T, vocab)
            logp = torch.log_softmax(logits, dim=-1)

        # Create scorer instance
        xlen = torch.tensor([logp.size(1)], device=x.device)
        self.impl = CTCPrefixScoreTH(logp, xlen, self.blank_id, self.eos_id)

        return None  # State managed internally

    def batch_score(self, yseqs, states, xs):
        """Batch score hypotheses.

        Args:
            yseqs: (batch, max_len) prefix sequences
            states: List of states per hypothesis
            xs: (batch, enc_len, feat_dim) encoder output

        Returns:
            (scores, new_states)
            - scores: (batch, vocab_size)
            - new_states: List of updated states
        """
        # Convert yseqs to list of sequences
        y_list = []
        for b in range(yseqs.size(0)):
            # Get non-padding tokens
            mask = yseqs[b] != 0  # Assume 0 is padding
            if mask.any():
                y_list.append(yseqs[b][mask])
            else:
                y_list.append(torch.tensor([self.eos_id], device=yseqs.device))

        # Merge states for batch processing
        if states[0] is not None:
            batch_state = (
                torch.stack([s[0] for s in states], dim=2),  # r
                torch.stack([s[1] for s in states]),          # s_prev
                states[0][2],  # f_min
                states[0][3],  # f_max
            )
        else:
            batch_state = None

        # Call scorer
        scores, new_state = self.impl(y_list, batch_state, scoring_ids=None)

        # Unpack states
        r, s_prev, f_min, f_max, scoring_idmap = new_state
        new_states = [
            (r[:, :, i], s_prev[i], f_min, f_max)
            for i in range(len(states))
        ]

        return scores, new_states

    def extend_prob(self, x: torch.Tensor):
        """Extend CTC probability matrix."""
        with torch.no_grad():
            logits = self.ctc.ctc_lo(x.unsqueeze(0))
            logp = torch.log_softmax(logits, dim=-1)
        self.impl.extend_prob(logp)

    def extend_state(self, states):
        """Extend hypothesis states."""
        if not states:
            return states
        return [self.impl.extend_state(s) for s in states]
```

#### 1.3 Integrate extend in Beam Search

**File:** `speechcatcher/beam_search/beam_search.py`

Add extend call before each block:

```python
def process_block(self, features, feature_lens, prev_state=None, is_final=False):
    """Process a single block with CTC extension."""

    # ... (initialization)

    # Encode block
    encoder_out, encoder_out_lens, encoder_states = self.encoder(
        features, feature_lens,
        prev_states=prev_state.encoder_states,
        is_final=is_final, infer_mode=True
    )

    # **CRITICAL: Extend CTC scorers BEFORE decoding**
    self.extend_scorers(encoder_out, new_state.hypotheses)

    # Now perform beam search
    if encoder_out.size(1) > 0:
        maxlen = min(max(encoder_out.size(1) // 8, 1), 20)

        for step in range(maxlen):
            # ... (beam search as before)

    return new_state

def extend_scorers(self, encoder_out, hypotheses):
    """Extend all scorers with new encoder output."""
    for scorer_name, scorer in self.scorers.items():
        # Extend probability matrices
        if hasattr(scorer, 'extend_prob'):
            scorer.extend_prob(encoder_out)

        # Extend states
        if hasattr(scorer, 'extend_state'):
            for hyp in hypotheses:
                if scorer_name in hyp.states and hyp.states[scorer_name] is not None:
                    hyp.states[scorer_name] = scorer.extend_state([hyp.states[scorer_name]])[0]
```

### Phase 2: Optimization 🟡 Priority 2

Once Phase 1 works, optimize:

#### 2.1 Partial Scoring

Implement `scoring_ids` parameter to only score top-K tokens (not full vocab).

#### 2.2 Windowing

Use attention weights to limit CTC forward range (margin parameter).

#### 2.3 GPU Optimization

- Minimize CPU-GPU transfers
- Use in-place operations where possible
- Profile with `torch.profiler`

### Phase 3: Testing ✅ Priority 1

#### 3.1 Unit Tests

```python
def test_ctc_prefix_scorer_single_frame():
    """Test CTC scorer on single time step."""
    # Create dummy CTC probs
    log_probs = torch.randn(1, 1, 100)  # (B, T, vocab)
    # ... test scoring

def test_ctc_extend_prob():
    """Test extend_prob accumulation."""
    # Block 1
    log_probs_1 = torch.randn(1, 10, 100)
    scorer = CTCPrefixScoreTH(log_probs_1, ...)
    assert scorer.x.shape[1] == 10

    # Block 2
    log_probs_2 = torch.randn(1, 20, 100)
    scorer.extend_prob(log_probs_2)
    assert scorer.x.shape[1] == 20  # Extended!

def test_ctc_extend_state():
    """Test extend_state for hypotheses."""
    # ...
```

#### 3.2 Integration Test

```python
def test_ctc_with_beam_search():
    """Test full beam search with CTC."""
    # Load real model
    model = load_model(...)

    # Create beam search with CTC
    bsbs = create_beam_search(model, ctc_weight=0.3)

    # Process multiple blocks
    for block in audio_blocks:
        state = bsbs.process_block(block, ..., state)

    # Check output is reasonable
    assert len(state.hypotheses[0].yseq) > 0
    assert state.hypotheses[0].yseq[0] != 1023  # Not repetitive!
```

#### 3.3 Comparison Test

```python
def test_compare_with_espnet():
    """Compare our CTC with ESPnet's on same input."""
    # Load same audio
    audio = load_audio("test.wav")

    # Run ESPnet decoder
    espnet_result = espnet_decode(audio)

    # Run our decoder
    our_result = our_decode(audio)

    # Compare token sequences
    # (May not be identical, but should be similar)
    assert levenshtein(espnet_result, our_result) < 10
```

---

## Summary

### What We Learned

1. **CTC is complex** - Can't be simplified to frame-averaging
2. **State accumulation is critical** - Must maintain forward variables across blocks
3. **extend_prob/extend_state are required** - Not optional for streaming
4. **Full forward algorithm needed** - Can't shortcut with single time step
5. **Numerical stability matters** - Log space, logsumexp, logzero value

### Implementation Checklist

- [ ] Implement full `CTCPrefixScoreTH.__init__()`
- [ ] Implement full `CTCPrefixScoreTH.__call__()` with forward recursion
- [ ] Implement `extend_prob()`
- [ ] Implement `extend_state()`
- [ ] Implement `index_select_state()`
- [ ] Update `CTCPrefixScorer` wrapper
- [ ] Add `extend_scorers()` to beam search
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Test on real audio
- [ ] Compare with ESPnet
- [ ] Optimize (partial scoring, windowing)
- [ ] Profile performance

### Expected Outcome

With full CTC implementation:
- ✅ Decoder will have acoustic grounding
- ✅ Won't get stuck on repetitive tokens
- ✅ Output quality should match ESPnet
- ✅ Proper streaming support

### Estimated Effort

- **Phase 1 (Full CTC):** 4-6 hours coding + 2-3 hours testing
- **Phase 2 (Optimization):** 2-3 hours
- **Phase 3 (Testing):** 2-4 hours

**Total:** ~10-15 hours of focused work

---

**End of Deep Dive Analysis - 2025-10-10**
