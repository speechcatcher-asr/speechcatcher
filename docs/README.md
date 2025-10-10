# Documentation Index

This directory contains all technical documentation for the speechcatcher internal decoder implementation.

## Main Documentation

### Architecture & Implementation
- **[compare.md](compare.md)** - Detailed comparison between custom decoder and ESPnet's implementation
  - Critical differences identified
  - Root cause analysis
  - Recommended fixes with priority order

- **[DECODER_README.md](DECODER_README.md)** - Comprehensive decoder documentation
  - Architecture overview
  - Component descriptions
  - Usage examples

- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation progress summary
  - Completed phases
  - Testing results
  - Known issues

### Technical Deep Dives
- **[decoder.md](decoder.md)** - Decoder implementation details

- **[WEIGHT_LOADING_NOTES.md](WEIGHT_LOADING_NOTES.md)** - Notes on checkpoint loading
  - ESPnet model compatibility
  - Weight mapping strategies

- **[ESPNET_DECODER_ROADMAP.md](ESPNET_DECODER_ROADMAP.md)** - Development roadmap
  - Future phases
  - Planned features
  - Optimization goals

## Session Notes

The [sessions/](sessions/) directory contains detailed notes from LLM-assisted development sessions:

- **[sessions/LLM.md](sessions/LLM.md)** - Oct 7, 2025 session notes
  - Current state analysis
  - Debugging attempts
  - Known issues

- **[sessions/PHASE_0_SUMMARY.md](sessions/PHASE_0_SUMMARY.md)** - Phase 0 completion summary
  - Foundation implementation
  - Test results

## Quick Links

### Current Status (as of Oct 10, 2025)
- ✅ State structure fixed to match ESPnet (dict-based)
- ✅ Hypothesis structure updated (torch.Tensor yseq, xpos tracking)
- ✅ Beam search updated for dict-based states
- ⚠️ Decoder runs but produces repetitive output without CTC
- ⚠️ CTC scorer causes timeout (needs optimization)

### Next Steps
See [compare.md](compare.md) section "Recommended Fixes (Priority Order)" for actionable next steps.
