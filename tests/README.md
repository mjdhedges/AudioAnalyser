# Tests

This directory contains test suites for the Music Analyser project.

## Test Structure

### Core Tests

- **`test_octave_band_quality.py`** - Comprehensive quality tests for octave band filtering
  - Tests octave alignment (ISO 266:1997 standard)
  - Tests frequency response ripple (< 1 dB target, with baseline thresholds)
  - Tests crest factor preservation (< 1 dB target)
  - Tests non-overlapping bands for cascade method
  - Tests linear phase (zero-phase via filtfilt)
  - Tests filter stability (no NaN/Inf values)
  - Tests band count correctness

- **`test_octave_filter.py`** - Basic functionality tests for OctaveBandFilter
  - Tests initialization
  - Tests filter design
  - Tests filter application
  - Tests octave bank creation

- **`test_audio_processor.py`** - Tests for audio file loading and processing
- **`test_music_analyzer.py`** - Tests for music analysis functionality
- **`test_main.py`** - Tests for CLI interface
- **`test_peak_envelope.py`** - Tests for peak envelope processor implementation

## Quality Test Criteria

The `test_octave_band_quality.py` test suite enforces strict quality criteria:

### Pass/Fail Thresholds

1. **Octave Alignment**: Bands must follow ISO 266:1997 standard
   - Lower frequency = center / sqrt(2)
   - Upper frequency = center * sqrt(2)
   - Frequency ratio = 2.0 (one octave)

2. **Frequency Response Ripple**: 
   - **Target**: < 1 dB max ripple
   - **Current Baseline**: < 50 dB (test fails if worse than baseline)
   - Prevents accidental degradation

3. **Crest Factor Preservation**:
   - **Target**: < 1 dB deviation
   - **Current**: Passes with current implementation
   - Critical for accurate analysis

4. **Non-Overlapping Bands**:
   - **Target**: < 20% reconstruction error
   - **Current Baseline**: < 150% (test fails if worse)
   - Ensures bands partition spectrum correctly

5. **Linear Phase**:
   - Group delay variation < 100 samples
   - Ensures waveform preservation

6. **Filter Stability**:
   - No NaN or Inf values in output
   - Ensures numerical stability

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_octave_band_quality.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Output

Test output files are stored in `tests/test_output/`. This directory is cleaned automatically to keep only essential results.

## Notes

- Quality tests use baseline thresholds for metrics that haven't achieved target values yet
- Tests will fail if performance degrades beyond current baseline
- This helps catch accidental regressions during development

