# Documentation Index

This directory contains all technical documentation for the speechcatcher native decoder implementation.

## Table of Contents

- [Architecture](#architecture)
- [Implementation](#implementation)
- [Analysis & Comparisons](#analysis--comparisons)
- [Debugging & Fixes](#debugging--fixes)
- [Planning & Roadmap](#planning--roadmap)
- [Development Sessions](#development-sessions)

---

## Architecture

Core architectural designs and streaming decoder implementation details.

- **[decoder.md](architecture/decoder.md)** *(48K)* - Comprehensive decoder implementation documentation
  - Blockwise Synchronous Beam Search (BSBS) algorithm
  - Streaming architecture components
  - State management and hypothesis tracking

- **[streaming-flow-comparison.md](architecture/streaming-flow-comparison.md)** *(16K)* - Streaming flow comparison with ESPnet
  - Side-by-side architecture diagrams
  - Block processing flow
  - State persistence patterns

- **[streaming-flow-visual.md](architecture/streaming-flow-visual.md)** *(17K)* - Visual streaming flow documentation
  - ASCII diagrams of streaming pipeline
  - Data flow visualizations

- **[global-state-architecture.md](architecture/global-state-architecture.md)** *(16K)* - Global state architecture design
  - Hypothesis persistence across blocks
  - State management patterns
  - Rewinding mechanism

---

## Implementation

Implementation guides, summaries, and technical notes.

- **[decoder-readme.md](implementation/decoder-readme.md)** *(11K)* - Main decoder implementation guide
  - Component overview
  - Usage examples
  - API documentation

- **[summary.md](implementation/summary.md)** *(9.4K)* - Implementation progress summary
  - Completed phases
  - Testing results
  - Current status

- **[weight-loading.md](implementation/weight-loading.md)** *(2.8K)* - Model weight loading notes
  - ESPnet checkpoint compatibility
  - Weight mapping strategies
  - Vocab size inference

- **[root-cause-analysis.md](implementation/root-cause-analysis.md)** *(5.3K)* - Root cause findings
  - Key issues discovered
  - Resolution strategies

---

## Analysis & Comparisons

Detailed analysis and comparison documents between native and ESPnet implementations.

- **[deep-dive-ctc.md](analysis/deep-dive-ctc.md)** *(33K)* - Deep dive into CTC implementation
  - CTC prefix scoring algorithm
  - Forward variable computation
  - Performance optimization strategies

- **[decoder-comparison.md](analysis/decoder-comparison.md)** *(27K)* - Detailed decoder comparison
  - ESPnet vs native implementation
  - Architectural differences
  - Behavior analysis

- **[ctc-comparison.md](analysis/ctc-comparison.md)** *(24K)* - CTC implementation comparison
  - Scoring differences
  - State management
  - Recommended fixes

- **[ctc-detailed-analysis.md](analysis/ctc-detailed-analysis.md)** *(18K)* - CTC detailed analysis
  - Algorithm walkthrough
  - Implementation notes

- **[streaming-analysis.md](analysis/streaming-analysis.md)** *(4.8K)* - Streaming behavior analysis

- **[initial-comparison.md](analysis/initial-comparison.md)** *(3.6K)* - Initial comparison findings

---

## Debugging & Fixes

Debugging investigations and fix documentation.

- **[eos-handling-fix.md](debugging/eos-handling-fix.md)** *(9.4K)* - EOS token handling fix
  - Problem description
  - Solution implementation
  - Test results

- **[segment2-analysis.md](debugging/segment2-analysis.md)** *(7.9K)* - Segment 2 debugging analysis
  - Specific test case investigation
  - Output comparison

- **[investigation.md](debugging/investigation.md)** *(5.0K)* - General investigation notes
  - Debugging techniques
  - Issue tracking

---

## Planning & Roadmap

Development roadmap and planning documents.

- **[roadmap.md](planning/roadmap.md)** *(18K)* - Development roadmap
  - Future phases
  - Planned features
  - Optimization goals
  - Performance targets

- **[session-summary.md](planning/session-summary.md)** *(6.0K)* - Development session summary
  - Major milestones
  - Key decisions

---

## Development Sessions

Chronological development session notes from LLM-assisted implementation.

### Phase 0: Foundation (Sessions 1-7)

- **[llm1.md](sessions/llm1.md)** *(14K)* - Oct 7, 2025: Initial investigation
  - Current state analysis
  - Debugging attempts
  - Known issues

- **[llm2.md](sessions/llm2.md)** *(16K)* - State structure implementation

- **[llm3.md](sessions/llm3.md)** *(21K)* - Hypothesis updates

- **[llm4.md](sessions/llm4.md)** *(17K)* - Beam search updates

- **[llm5.md](sessions/llm5.md)** *(13K)* - CTC scorer integration

- **[llm6.md](sessions/llm6.md)** *(19K)* - Streaming flow fixes

- **[llm7.md](sessions/llm7.md)** *(6.6K)* - Phase 0 completion

- **[phase0-summary.md](sessions/phase0-summary.md)** *(6.9K)* - Phase 0 summary
  - Foundation implementation
  - Test results

### Phase 1: CTC Integration (Sessions 8-9)

- **[llm8.md](sessions/llm8.md)** *(13K)* - CTC prefix scorer implementation

- **[llm9.md](sessions/llm9.md)** *(13K)* - CTC optimization

### Phase 2: Global State & Parity (Sessions 10-12)

- **[llm10.md](sessions/llm10.md)** *(9.5K)* - Global state architecture
  - Hypothesis persistence
  - Rewinding mechanism

- **[llm11.md](sessions/llm11.md)** *(7.9K)* - ESPnet parity achievement
  - 92% word-level parity
  - Final debugging

- **[llm12.md](sessions/llm12.md)** *(10K)* - Code quality improvements
  - Default decoder switch to ESPnet
  - Dynamic token ID calculation
  - Individual scorer score tracking

---

## Current Status

**As of LLM Session 12 (Oct 12, 2025):**

### Native Decoder
- ✅ **92% ESPnet parity** (768/835 words correct)
- ✅ Blockwise Synchronous Beam Search (BSBS) implemented
- ✅ CTC prefix scoring with full forward algorithm
- ✅ Global state management with rewinding
- ✅ Block Boundary Detection (BBD)
- ✅ Dynamic vocab size support
- ✅ Individual scorer score tracking

### ESPnet Decoder (Reference)
- ✅ **100% parity** (835/835 words correct)
- ✅ Default decoder implementation
- ✅ Production-ready

### Key Achievements
1. **Streaming ASR Pipeline**: Complete streaming architecture matching ESPnet's design
2. **CTC Integration**: Full CTC prefix scoring with state management
3. **Code Quality**: Well-documented, no hardcoded values, flexible configuration
4. **Testing**: Comprehensive validation against ESPnet reference

### Known Limitations
- Native decoder at 92% parity due to model-level limitations (see [llm11.md](sessions/llm11.md))
- ESPnet decoder recommended for production use

---

## Quick Links

### For New Developers
1. Start with **[decoder-readme.md](implementation/decoder-readme.md)** for implementation overview
2. Read **[decoder.md](architecture/decoder.md)** for architecture details
3. Check **[summary.md](implementation/summary.md)** for current status

### For Debugging
1. See **[eos-handling-fix.md](debugging/eos-handling-fix.md)** for EOS token issues
2. Check **[deep-dive-ctc.md](analysis/deep-dive-ctc.md)** for CTC problems
3. Review **[decoder-comparison.md](analysis/decoder-comparison.md)** for behavioral differences

### For Understanding Design Decisions
1. Read session docs in chronological order: [llm1.md](sessions/llm1.md) → [llm12.md](sessions/llm12.md)
2. See **[roadmap.md](planning/roadmap.md)** for future direction
3. Check **[global-state-architecture.md](architecture/global-state-architecture.md)** for key architectural decisions

---

## File Organization

```
docs/
├── README.md                    # This file
├── architecture/                # Core architecture docs
├── implementation/              # Implementation guides
├── analysis/                    # Comparative analysis
├── debugging/                   # Debugging & fixes
├── planning/                    # Roadmap & planning
└── sessions/                    # Development session notes
```

All documentation follows lowercase naming (except README.md) for consistency.
