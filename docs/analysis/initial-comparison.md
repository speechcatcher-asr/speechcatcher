# Segment-by-Segment Comparison: Native vs ESPnet Decoder

## Summary Statistics

| Segment | Duration | Native Words | ESPnet Words | Difference | Native % | Status |
|---------|----------|--------------|--------------|------------|----------|---------|
| 0       | 60.0s    | 70           | 105          | -35        | 66.7%    | ⚠️ Moderate loss |
| 1       | 60.0s    | 37           | 92           | -55        | 40.2%    | 🔴 Severe loss + repetition |
| 2       | 60.0s    | 4            | 96           | -92        | 4.2%     | 🔴 CRITICAL failure |
| 3       | 60.0s    | 6            | 111          | -105       | 5.4%     | 🔴 CRITICAL failure |
| 4       | 60.0s    | 5            | 5            | 0          | 100.0%   | ✅ Identical |
| 5       | 60.0s    | 53           | 53           | 0          | 100.0%   | ✅ Identical |
| 6       | 60.0s    | 78           | 78           | 0          | 100.0%   | ✅ Identical |
| 7       | 65.0s    | 4            | 4            | 0          | 100.0%   | ✅ Identical |
| **TOTAL** | **485.0s** | **257**  | **544**      | **-287**   | **47.2%** | - |

## Critical Findings

### 1. **Bimodal Performance Pattern**
- **Segments 0-3** (first 4 minutes): Native achieves only 29.0% of ESPnet output (117 vs 404 words)
- **Segments 4-7** (last 4 minutes): Native matches ESPnet 100% (140 vs 140 words)

### 2. **Worst Offenders**

#### Segment 2 (4.2% accuracy) - WORST PERFORMANCE
**ESPnet (96 words):** Full coherent transcription about university, administration, libraries, past vs future, flexibility, innovation, etc.

**Native (4 words):** `und mussten sich,<sos/eos><sos/eos><sos/eos> äh,`

**Issue:** Decoder produced almost nothing, with special tokens appearing in output

#### Segment 3 (5.4% accuracy) - SECOND WORST
**ESPnet (111 words):** Full transcription about fear of the future, virus, vigilance vs anxiety, etc.

**Native (6 words):** `andere. Ja, das ist so.`

**Issue:** Decoder produced trivial output, completely missing the content

#### Segment 1 (40.2% accuracy) - REPETITION FAILURE
**ESPnet (92 words):** Full transcription about current situation, temporal phases, university past, etc.

**Native (37 words):** Contains massive repetition: `Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Under. Und`

**Issue:** BBD failed completely, allowing massive token repetition

### 3. **Perfect Segments (4-7)**
All four segments in the second half produced **identical output** between native and ESPnet decoders, suggesting:
- No BBD issues in second half
- No state management issues
- Identical decoder behavior

## Hypothesis

The dramatic difference between first half (0-3) and second half (4-7) suggests:

1. **NOT a temporal embedding issue** - If it were temporal embeddings, we'd expect degradation after 180s (3 minutes), but problems START immediately and then IMPROVE after 4 minutes

2. **Possible audio content dependency** - The decoder may handle certain acoustic conditions or speech patterns differently

3. **BBD sensitivity to speech content** - The first 4 minutes may contain speech patterns that trigger BBD inappropriately, while the second half does not

4. **Encoder buffer issues** - First segments may have problematic encoder state handling that resolves by segment 4

## Next Steps

**Priority 1:** Investigate segments 2 and 3 (worst offenders with <6% accuracy)
**Priority 2:** Investigate segment 1 (BBD repetition failure)
**Priority 3:** Understand why segments 4-7 work perfectly
