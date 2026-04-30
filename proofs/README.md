# Audio Analyser Proofs

This folder contains reproducible evidence for analysis methods that are easy to
misunderstand, difficult to validate by inspection, or important enough to need a
recorded technical justification.

Each proof should be self-contained: include the script, generated or licensed
source material, numeric outputs, plots, and a README explaining the result.

## Completed Proofs

### Octave Band Energy Closure

Status: **complete**

Folder: `octave_band_energy_closure/`

Purpose: validates the FFT power-complementary octave bank used by the main
pipeline.

Decision:

- Octave-band RMS/energy must be summed in **linear power**, not by adding dB.
- The current filter bank closes summed power to numerical precision.
- Amplitude sum is not the validation target; raised-cosine overlaps can exceed
  flat amplitude while summed power remains flat.
- Block FFT mode is acceptable for large-file processing, with small band RMS
  deltas versus full-file FFT.

Key outputs:

- `energy_closure_results.csv`
- `time_series_validation.csv`
- `block_fft_comparison.csv`
- `band_power_sum.png`
- `captured_power_percent.png`
- `energy_closure_error.png`

### Crest Factor Window and Peak-Hold Study

Status: **complete**

Folder: `peak_hold_tau/`

Purpose: validates the crest-factor-over-time method and explains why the older
1-second slow/peak-hold target was too fragile.

Decision:

- Use **fixed-window crest factor** as the main programme/report metric.
- Default window: `2.0 s`.
- Default hop: `1.0 s`.
- Formula: `max(abs(window)) / rms(window)` on the same window.
- Windows below the configured RMS floor are invalid and should be stored/plotted
  as `NaN`, not clamped to `0 dB`.
- Slow RMS + peak-hold can remain a meter-style diagnostic, but not the
  authoritative report trace.

Key outputs:

- `tau_sweep_results.csv`
- `block_window_summary.csv`
- `tau_by_window_summary.csv`
- `best_tau_trace.png`
- `block_window_traces.png`
- `best_tau_by_block_window.png`

### Octave-Band Peak Expansion

Status: **complete**

Folder: `octave_band_peak_expansion/`

Purpose: explains why filtered octave-band sample peaks can exceed the source
full-band sample peak after filtering, even when the channel gain reference is
correct.

Decision:

- The octave bank is **power-complementary**, not peak-limiting.
- Filtered octave-band peaks are derived post-filter peaks.
- Small positive octave-band peak readings are not automatically a gain bug or a
  phase bug.
- Source headroom should be read from the full-spectrum channel peak; octave-band
  peaks should be treated as post-filter diagnostics.

Key outputs:

- `peak_expansion_results.csv`
- `peak_expansion_by_signal.png`
- `rms_power_closure_by_signal.png`

## Incomplete / Planned Proofs

### Octave Filter Band Edges

Status: **planned**

Verify that octave filters place energy into the intended centre bands and that
crossover behaviour matches the configured filter-bank design. Useful material:
sine sweeps, stepped sines, centre-frequency tones, and crossover-frequency
tones.

The existing `octave_band_energy_closure/` and `octave_band_peak_expansion/`
proofs cover power closure and peak behaviour, but not a full band-edge
acceptance matrix.

### LFE Band-Limited Analysis

Status: **planned**

Prove that LFE and low-frequency deep-dive metrics respond correctly to in-band
low-frequency content and reject out-of-band content. This should include
content around LFE crossover regions and confirm the report/deep-dive outputs.

### Mono, Stereo, and Multichannel Grouping

Status: **planned**

Prove channel grouping rules for mono, stereo, screen, LFE, surround, and height
content. This should confirm that group plots and reports do not silently drop,
duplicate, or mislabel channels.

### Normalization and dBFS Reference

Status: **planned**

Prove that peak, RMS, and dBFS values move correctly under gain changes, while
crest factor remains gain-invariant. This should cover per-channel normalization,
track-wide peak metadata, and filtered-band metrics.
