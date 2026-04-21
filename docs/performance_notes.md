# Performance notes

This file captures intentional performance optimizations, their measured impact, and any known numerical deltas.

## 2026-04: Sliding-window max acceleration (`max_abs_over_window`)

- **Change**: Replace Python `deque` hot path with an algorithm-equivalent Numba implementation, keeping the original as a reference fallback.
- **Files**:
  - `src/signal_metrics.py`: `_max_abs_over_window_py`, `_max_abs_over_window_numba`, `max_abs_over_window`
  - `tests/test_signal_metrics.py`: asserts **exact equality** between Python reference and accelerated path.

### Measured impact (single track, Win64)

Profiling command:

```powershell
python -m cProfile -o profile_single.pstats -m src.main --input "tracks\Music\Allegro maestoso.flac" --output-dir "analysis\Music" --skip-post
```

Result summary:

- **Before**: ~1208.7s total profiled wall time
- **After**:  ~716.3s total profiled wall time
- **Net**: ~40% reduction

Hotspot shift:

- Before: `signal_metrics.max_abs_over_window` dominated cumulative time.
- After: primary CPU sinks became `compute_slow_rms_envelope`, `compute_peak_hold_envelope`, and `visualization.create_octave_crest_factor_time_plot`.

## 2026-04: IEC SLOW RMS envelope acceleration (`compute_slow_rms_envelope`)

- **Change**: Add Numba implementation `_compute_slow_rms_envelope_numba` and make it the default when Numba is available.
  - Python reference remains available via `use_numba=False`.
- **Files**:
  - `src/signal_metrics.py`: `_compute_slow_rms_envelope_py`, `_compute_slow_rms_envelope_numba`, `compute_slow_rms_envelope(use_numba=...)`
  - `tests/test_signal_metrics.py`:
    - Python reference path: **bit-for-bit equality** to `_compute_slow_rms_envelope_py`
    - Numba path: asserts “very small” differences only (`rtol=1e-7`, `atol=2e-9`)

### Observed numerical delta

On Win64 with float32 inputs, the Numba path differs from the Python reference by approximately:

- **max abs diff**: ~2e-9
- **max rel diff**: ~1.6e-8

These deltas are considered acceptable for performance work, but any downstream “golden” validations should treat the Python path as the canonical reference if bit-identical outputs are required.

## 2026-04: Skip expensive octave crest-factor time plot

- **Change**: CLI flag `--skip-octave-cf-time` to skip `octave_crest_factor_time.png` generation.
- **Why**: profiling showed `visualization.create_octave_crest_factor_time_plot` as a major wall-time contributor.
- **Note**: per-octave crest-factor-over-time data is not currently persisted in `analysis_results.csv`.

## 2026-04: Peak-hold envelope + sampled window maxima acceleration

- **Change**: Add Numba implementations for:
  - `compute_peak_hold_envelope` (opt-out via `use_numba=False`)
  - `sampled_max_abs` (exact-match test vs reference)
- **Why**: these were still significant contributors after removing the `max_abs_over_window` hotspot.

