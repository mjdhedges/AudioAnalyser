# Tests

This directory contains test suites for the Audio Analyser project.

## Octave Filter Tests

- `test_octave_filter.py`: basic FFT octave-bank construction, residual bands,
  energy closure, block mode, and analysis helper coverage.
- `test_octave_filter_extended.py`: power-complementary weight validation,
  amplitude-sum transparency, block-vs-full-file agreement, and constructor
  validation.
- `test_octave_band_quality.py`: signal-level quality checks proving that
  octave-band RMS values power-sum back to the full-band RMS and that each band
  returns a finite same-length time series.

The octave filter quality criterion is summed power, not simple amplitude sum:

```text
sum(weight_band(f) ** 2) = 1.0
```

This matches the production FFT power-complementary octave filter and the proof
in `proofs/octave_band_energy_closure/`.
