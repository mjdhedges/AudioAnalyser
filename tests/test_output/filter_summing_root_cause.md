# Root Cause Analysis: Why Octave Band Filters Don't Sum to Unity

## Problem Statement

Octave band filters implemented according to ISO 266:1997 standard do not naturally sum to a flat (0 dB) magnitude response. When summed, they exhibit ripple ranging from -11.46 dB to +3.48 dB.

## Key Findings

### 1. Filters Are Already Correctly Normalized
- Each individual filter has a peak magnitude of 1.0 (0 dB)
- Filters are designed correctly according to ISO standard
- No per-filter normalization is needed

### 2. The Problem: Phase Interference in Overlap Regions

When overlapping bandpass filters are summed:
- **At any frequency in the overlap region**, multiple filters contribute
- Each filter has a different **phase response**
- Phase differences cause **constructive interference** (peaks > 0 dB) or **destructive interference** (valleys < 0 dB)

Example:
- At 1000 Hz, the 500 Hz, 1000 Hz, and 2000 Hz bands all contribute
- Their phases are different, causing interference
- Result: summed magnitude ≠ sum of individual magnitudes

### 3. This Is a Fundamental Limitation

**Butterworth bandpass filters are NOT designed for perfect reconstruction**
- They're designed for **independent frequency analysis**
- They're **not complementary** - they don't naturally sum to unity
- Standard octave band analysis (e.g., MATLAB `musicanalyser.m`) **does NOT reconstruct** the signal

## Why Normalization Is a "Bodge"

The current frequency-dependent normalization:
- ✅ Achieves flat magnitude response (0 dB)
- ✅ Preserves phase relationships
- ❌ But it's a **post-processing fix**, not a fundamental solution
- ❌ Doesn't address the root cause: **filter design is wrong for reconstruction**

## Real Solutions

### Option 1: Complementary Filter Design
Design filters specifically to sum to unity:
- Requires custom filter design (not standard Butterworth)
- Filters must have complementary magnitude/phase relationships
- Complex to implement correctly

### Option 2: Constant-Q Filter Banks
Use constant-Q filters designed for perfect reconstruction:
- Gammatone filters
- Auditory filter banks
- Designed with perfect reconstruction in mind

### Option 3: Accept Limitation
Recognize that perfect reconstruction isn't the intended use case:
- Octave bands are for **analysis**, not reconstruction
- Accept some ripple as inherent to the method
- Use filters for analysis only, don't reconstruct

### Option 4: Non-Overlapping Filters
Use filters that don't overlap:
- Not standard octave bands (would violate ISO standard)
- Would lose frequency coverage

## Recommended Approach

**For crest factor testing**: The test is trying to verify that splitting and recombining doesn't affect crest factor. However:

1. **This may not be possible** with standard octave bands due to phase interference
2. **Phase errors** (not just magnitude) affect peak amplitudes
3. **The real question**: Is perfect reconstruction even necessary for the use case?

## Next Steps

1. **Determine the actual requirement**: 
   - Is perfect reconstruction needed?
   - Or is approximate reconstruction acceptable?

2. **If perfect reconstruction is required**:
   - Implement complementary filter design
   - Or use a filter bank designed for reconstruction (e.g., constant-Q)

3. **If approximate is acceptable**:
   - Keep current implementation
   - Document expected ripple
   - Measure actual impact on crest factor

4. **Remove normalization bodge**:
   - If moving to proper filter design, remove frequency-dependent normalization
   - Implement correct filter design from the start

