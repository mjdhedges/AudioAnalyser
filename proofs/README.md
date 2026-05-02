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

### Octave Filter Band Edges

Status: **complete; band filtering passes, crest-factor expansion recorded**

Folder: `octave_filter_band_edges/`

Purpose: validates octave filter centre placement, crossover behaviour, and
phase neutrality. It also records a separate EIA-426B-style compressed pink-noise
crest-factor diagnostic for the derived octave-band signals.

Decision:

- Centre-frequency tones land fully in the intended band.
- Geometric crossover tones split power equally between adjacent bands.
- FFT weights are real and non-negative, so retained bins receive no phase
  rotation from the filter bank.
- Full-spectrum compressed pink noise remains at the generated `6 dB` crest
  factor because the full-spectrum path is the unfiltered source.
- Active derived octave bands do show crest-factor expansion on compressed pink
  noise, so the current filter bank should **not** be described as preserving
  compressed-noise crest factor per band.
- Directly summing the octave-band waveforms is not perfect reconstruction; the
  bank reconstructs to numerical precision only when the same FFT weights are
  applied as synthesis weights before summing.

Key outputs:

- `band_edge_acceptance.csv`
- `phase_response.csv`
- `compressed_pink_noise_crest_factor.csv`
- `reconstruction_analysis.csv`
- `band_edge_acceptance.png`
- `compressed_pink_noise_crest_factor.png`
- `reconstruction_error.png`

### LFE Band-Limited Analysis

Status: **complete**

Folder: `lfe_band_limited_analysis/`

Purpose: proves that LFE deep-dive target plots respond to intended
low-frequency content, split adjacent crossover content correctly, and reject
screen-range content.

Decision:

- LFE target bands are `8 Hz`, `16 Hz`, `31.25 Hz`, `62.5 Hz`, `125 Hz`, and
  `250 Hz`.
- Target-centre tones land fully in the intended LFE plot band.
- Adjacent LFE crossover tones split power as expected, allowing small
  finite-tone FFT leakage for irrational geometric crossover frequencies.
- The `250/500 Hz` crossover is correctly treated as the upper transition: about
  half the power remains in the plotted `250 Hz` LFE target.
- `500 Hz` and `1 kHz` screen-range centre tones are rejected by the plotted LFE
  target set.
- The `4 Hz` residual octave is excluded from the plotted LFE target set.

Key outputs:

- `lfe_band_acceptance.csv`
- `lfe_band_metrics.csv`
- `lfe_target_power_by_case.png`
- `lfe_band_power_matrix.png`

### Mono, Stereo, and Multichannel Grouping

Status: **complete**

Folder: `channel_grouping/`

Purpose: proves channel grouping rules for mono, stereo, screen, LFE, surround,
and height content across group plots and markdown report grouping.

Decision:

- Mono report grouping is represented by root-level `analysis_results.csv` and
  maps to `Mono`.
- Stereo folders are not cinema-screen channels; they group under
  `All Channels`.
- `FL`, `FR`, and `FC` group as `Screen`.
- `LFE` and `Low Frequency Effects` group as `LFE`.
- Surround, back, top, and height channels group as `Surround+Height`.
- Channel grouping should classify channel-name tokens, not loose substrings, so
  `TFL` and `TFR` do not get mistaken for `FL` and `FR`.

Key outputs:

- `grouping_results.csv`
- `channel_classification.csv`
- `channel_mapping_results.csv`
- `grouped_channel_counts.png`

## Incomplete / Planned Proofs

### Normalization and dBFS Reference

Status: **planned**

Prove that peak, RMS, and dBFS values move correctly under gain changes, while
crest factor remains gain-invariant. This should cover per-channel normalization,
track-wide peak metadata, and filtered-band metrics.
