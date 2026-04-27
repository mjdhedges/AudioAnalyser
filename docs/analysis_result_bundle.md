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
      time_domain_summary.csv
      advanced_statistics.csv
      extreme_chunks_octave_analysis.csv
      histogram_linear.csv
      histogram_log_db.csv
      octave_time_metrics.csv
      envelope_statistics.csv
      envelope_pattern_analysis.csv
      sustained_peaks_summary.csv
      sustained_peaks_events.csv
      envelope_plot_data.json
```

Analysis writes `.aaresults` bundles by default. The old per-channel
`analysis_results.csv` export is controlled by `export.generate_legacy_csv` and
is disabled by default.

## Plot Replay Data

- `octave_spectrum.png` and `crest_factor.png` use
  `octave_band_analysis.csv` plus `extreme_chunks_octave_analysis.csv`.
- `crest_factor_time.png` should use `time_domain_analysis.csv`.
- `histograms.png` should use `histogram_linear.csv`.
- `histograms_log_db.png` should use `histogram_log_db.csv`.
- `octave_crest_factor_time.png`, LFE octave-time plots, and
  Screen/Surround+Height deep-dive plots should use
  `octave_time_metrics.csv`.
- Pattern and independent envelope plots should use `envelope_plot_data.json`.
- Report sections that previously came from legacy CSV blocks use
  `advanced_statistics.csv`, `time_domain_summary.csv`,
  `envelope_statistics.csv`, `envelope_pattern_analysis.csv`,
  `sustained_peaks_summary.csv`, and `sustained_peaks_events.csv`.
- Group plots and report prose should use the same per-channel CSV/JSON tables
  through the bundle `manifest.json`.

The bundle stores derived time-series and histogram data rather than full
filtered waveforms by default. Full waveform storage can be added later as an
optional schema extension if a future UI needs sample-level replay.

## Rendering From A Bundle

Plot rendering now runs as a second pass over `.aaresults` data:

```powershell
python -m src.render --results analysis/Track.aaresults --output-dir rendered
```

`--results` may point at one bundle or a directory containing multiple bundles.
Render DPI defaults to `plotting.render_dpi` in `config.toml`; use `--dpi` to
override it for one run.

Use `--no-spectrum-plots`, `--no-histograms`, `--no-time-plots`,
`--no-envelope-plots`, or `--no-group-plots` to skip a plot family.

Add `--reports` to write `analysis.md` next to the rendered plots. The report
uses bundle tables directly and does not require legacy `analysis_results.csv`
files.

Legacy `src.main --post-only` and `--run-post` post-processing reads the old
sectioned CSV files and is skipped for bundle-only output. Use `src.render` for
new render/report workflows.
