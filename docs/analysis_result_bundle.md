# Analysis Result Bundle

Audio Analyser writes a per-track `.aaresults` bundle during analysis. The
bundle is the hand-off between audio processing and graph/report rendering, so
rendering code can recreate plots without reloading the source audio.

## Bundle Layout

```text
<track>.aaresults/
  manifest.json
  channels/
    channel_01/
      metadata.json
      analysis_config.json
      plotting_config.json
      envelope_config.json
      octave_band_analysis.csv
      time_domain_analysis.csv
      extreme_chunks_octave_analysis.csv
      histogram_linear.csv
      histogram_log_db.csv
      octave_time_metrics.csv
      envelope_plot_data.json
```

## Plot Replay Data

- `octave_spectrum.png` and `crest_factor.png` should use
  `octave_band_analysis.csv` plus `extreme_chunks_octave_analysis.csv`.
- `crest_factor_time.png` should use `time_domain_analysis.csv`.
- `histograms.png` should use `histogram_linear.csv`.
- `histograms_log_db.png` should use `histogram_log_db.csv`.
- `octave_crest_factor_time.png`, LFE octave-time plots, and
  Screen/Surround+Height deep-dive plots should use
  `octave_time_metrics.csv`.
- Pattern and independent envelope plots should use `envelope_plot_data.json`.
- Group plots and report prose should use the same per-channel CSV/JSON tables
  through the bundle `manifest.json`.

The bundle stores derived time-series and histogram data rather than full
filtered waveforms by default. Full waveform storage can be added later as an
optional schema extension if a future UI needs sample-level replay.
