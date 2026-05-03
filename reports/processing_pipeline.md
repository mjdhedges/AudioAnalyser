# Audio Analyser Processing Pipeline

This document explains how Audio Analyser processes source material and turns it into portable `.aaresults` bundles, channel-level measurements, group-level plots, Markdown reports, and PDF reports. It is written for audio professionals who need to understand what the numbers mean, how they are produced, and how much confidence to place in them.

## Executive Summary

Audio Analyser is a measurement pipeline for audio programme material. It is designed to answer practical questions about level, crest factor, frequency-dependent behaviour, peak density, low-frequency energy, and peak recovery characteristics.

The pipeline processes each channel independently, then combines selected results into channel-group summaries such as Screen, LFE, and Surround+Height. This matters because a single full-mix statistic can hide the channel that actually drives amplifier, loudspeaker, or system-headroom demand.

The main outputs are:

- Portable `.aaresults` result bundles containing per-channel metadata, octave-band statistics, time-domain crest factor data, histogram tables, envelope/recovery statistics, and compact replay data for rendering the configured envelope plots.
- Per-channel plots for octave spectrum, crest factor, time-domain behaviour, histograms, and envelope events rendered from those bundles.
- Group-level plots, deep-dive plots, Markdown reports, and PDF reports that identify demanding channels and present the results in a reviewable form.
- Local report image folders, so each report folder contains compact PDF-friendly copies of the graphs it needs under `images/`.

## End-to-End Flow

The processing flow is:

1. The CLI finds one input file or recursively discovers supported audio files in a batch folder.
2. Existing analysis results can be reused when result-cache metadata indicates that the source file and analysis configuration have not changed.
3. The audio file is decoded through `ffmpeg` at the source stream's native sample rate and preserved as mono or multi-channel audio.
4. Multi-channel material is split into individual channels.
5. Each channel is normalized for analysis while retaining the original peak reference so exported levels remain tied to the source file's dBFS scale.
6. An octave-band filter bank is created for the channel.
7. The analyzer calculates whole-interval octave-band statistics, configured time-domain crest factor, fixed-window extreme chunks, peak/envelope statistics, and histograms.
8. A `.aaresults` bundle is written or updated for the track, including the derived tables needed to render plots later without reloading the source audio.
9. In the normal GUI flow, each worker continues immediately from analysis into bundle rendering and report generation for that track before it is marked finished.
10. Rendered outputs include per-channel plots, group plots, deep-dive plots, worst-channel manifests, Markdown reports, and PDF reports.

## Input Handling

Audio Analyser supports common audio formats such as WAV, FLAC, MP3, AIFF, M4A, AAC, MKV, and MTS where `ffmpeg`/`ffprobe` support is available.

All source decoding now goes through `ffmpeg`. This keeps stereo music files and multi-channel film containers on the same decoding path and avoids format-specific fallbacks. `ffprobe` is used to inspect audio stream metadata, channel counts, and channel layout where available. Decoded audio is written to a temporary PCM WAV representation for loading into the Python analysis pipeline.

The analysis pipeline intentionally does not resample source material. Source sample-peak traces must remain tied to the original decoded samples; resampling can create intersample/reconstruction peaks that exceed the original source sample peak. Bundle metadata and reports record the native sample rate used for each track.

## Channel Handling

Multi-channel files are not downmixed for analysis. Each channel is processed separately and written to its own bundle channel entry. Channel names may be long legacy labels or short layout labels, for example:

- `Channel 1 FL`
- `Channel 2 FR`
- `Channel 3 FC`
- `Channel 4 LFE`
- `Channel 7 SL`

Channel names are inferred from the available layout information where possible, including RP22-style naming for cinema layouts. The renderer and report generator classify both long names such as `Channel 1 FL` and short names such as `FL`, `LFE`, or `SL`.

Group-level reports then combine or select channel results into practical review groups:

- Screen
- LFE
- Surround+Height
- Mono or All Channels where a cinema grouping does not apply

## Level Reference and Normalization

Per-channel audio is normalized for internal processing. The original per-channel source peak is retained and used when converting analysis values back to dBFS. This allows the pipeline to gain the numerical stability benefits of normalized processing while still reporting levels relative to the original source scale.

In bundle metadata, each channel records the channel original peak and original peak in dBFS. It also records the track-wide original peak for context. Plot and export calculations use the channel reference when reporting peak and RMS levels in dBFS.

## Octave-Band Analysis

The frequency-domain analysis uses 1/1-octave centre frequencies configured in `config.toml`. The current configuration includes a 4 Hz-and-below residual band, 8 Hz plus the IEC-style octave series from 16 Hz to 16 kHz, and a high residual band above the 16 kHz octave region:

- 4 Hz and below
- 8 Hz
- 16 Hz
- 31.25 Hz
- 62.5 Hz
- 125 Hz
- 250 Hz
- 500 Hz
- 1 kHz
- 2 kHz
- 4 kHz
- 8 kHz
- 16 kHz
- Above the 16 kHz octave region

The default processing path uses the proven FFT power-complementary octave-bank design. It applies raised-cosine weights in the frequency domain and inverse FFTs each band back to a time-domain waveform. Adjacent bands overlap in amplitude, but their squared weights sum to 1.0 at every FFT bin, so octave-band RMS values can be summed as linear power without losing or double-counting energy.

The octave FFT mode is configurable. In `auto` mode, the analyser estimates the full-file FFT memory requirement and uses full-file FFT processing when it fits under the configured RAM budget. If the estimate exceeds `analysis.octave_max_memory_gb`, it switches to large-block FFT processing. If the octave-bank time-series output itself exceeds the budget, the bank is held in temporary disk-backed storage while downstream metrics run. Bundle metadata records the requested mode, effective mode, block length, storage method, residual-band settings, and RAM budget.

For each octave band and for the full-spectrum signal, the analyzer exports:

- Peak amplitude and peak dBFS
- RMS and RMS dBFS
- Dynamic range ratio and dB value
- Crest factor ratio and dB value
- Crest factor validity and method metadata
- Mean, standard deviation, and amplitude percentiles

These whole-track octave statistics are different from the time-domain samples. They describe the channel over the whole analyzed duration, not just one moment in time.

Crest factor follows a two-path rule:

- **No time axis**: full-channel and octave-band summary rows use whole-interval crest factor, `max(abs(signal)) / rms(signal)`, with peak and RMS measured over the same complete interval.
- **Time axis**: crest-factor graphs and tables use the configured time-domain method (`fixed_window`, `slow`, or `fixed_chunk`).

The previous hybrid definition (whole-band RMS combined with a separate 1-second sliding peak) is no longer used for octave-band spectrum rows because it created a third interpretation of crest factor.

## Time-Domain Crest Factor

Time-domain crest factor is configuration-driven. Bundle tables and reports record the method and timing metadata so plots can be interpreted without guessing the calculation.

Three time-series modes are supported:

### Fixed-Window Mode

Fixed-window mode is the default report method. It calculates peak and RMS over the same overlapping window:

```text
crest_factor = max(abs(window)) / rms(window)
```

The current default settings are:

- `analysis.time_domain_crest_factor_mode = "fixed_window"`
- `analysis.crest_factor_window_seconds = 2.0`
- `analysis.crest_factor_step_seconds = 1.0`
- `analysis.crest_factor_rms_floor_dbfs = -80.0`

Windows below the configured RMS floor, or true-silence windows, are invalid crest-factor measurements. They are stored as `NaN` and plotted as gaps, not clamped to `0 dB`. Reports include a note under the first crest-factor-over-time plot explaining those gaps.

### Slow Mode

Slow mode uses a peak-hold envelope and a slow RMS envelope, sampled at a configured time step. The slow RMS envelope is intended to follow the IEC sound level meter "Slow" time weighting concept, where the exponential time constant is 1 second. In practical terms, this makes the RMS level trend behave like the level movement an engineer would expect to see from an SPL meter set to Slow, rather than a simple fixed-block RMS calculation.

This does not by itself turn dBFS into calibrated SPL. Absolute agreement with an SPL meter still requires a calibrated playback chain, microphone, acoustic weighting, and measurement position. However, when the analysis is calibrated to the same reference and uses equivalent weighting, the slow RMS behaviour should be comparable to what a professional would see on a Slow SPL meter.

The peak-hold envelope is a diagnostic crest-factor aid. It tracks programme peaks with its own configured decay constant so peak-to-RMS behaviour can be reviewed alongside the slow RMS trend. It is no longer treated as the authoritative report crest-factor trace unless `time_domain_crest_factor_mode = "slow"` is explicitly selected.

The bundle tables record:

- `time_domain_mode`
- `time_domain_time_step_seconds`
- `time_domain_rms_method`
- `time_domain_peak_method`
- `crest_factor_method`
- `chunk_duration_seconds`
- `crest_factor_window_seconds`
- `crest_factor_step_seconds`
- `is_valid_crest_factor`

For example, a run may state that time-domain crest factor uses fixed-window mode, a 2-second window, a 1-second step, and `crest_factor_method = "fixed_window_peak_rms"`.

### Fixed-Chunk Mode

Fixed-chunk mode calculates peak and RMS over non-overlapping windows based on the configured `analysis.chunk_duration_seconds`. This is a straightforward block-based method and is useful when repeatable fixed-duration windows are required or for legacy comparison.

## Extreme Chunk Analysis

Extreme chunk analysis identifies sections with minimum and maximum crest factor. This currently uses fixed windows based on `analysis.chunk_duration_seconds`, even when the exported time-domain plot mode is `fixed_window` or `slow`.

This distinction is important:

- Time-domain plots may use `fixed_window`, `slow`, or `fixed_chunk` metadata.
- Extreme chunk octave overlays are selected from fixed analysis windows.

The report generator should describe these methods separately when needed.

## Peak, Envelope, and Recovery Analysis

The envelope analysis looks beyond single peak values and asks how peaks behave over time. This is useful for understanding practical system stress, especially in LFE and effects-heavy material.

The pipeline detects peak events and can report:

- Counts of discrete peaks above -3 dBFS, -1 dBFS, and -0.1 dBFS.
- Peak rates per second.
- Loud duty cycle, showing how much time the signal spends at high levels.
- Sustained-peak hold time.
- Recovery time to configured thresholds such as -3 dB, -6 dB, -9 dB, and -12 dB relative to each peak.
- Pattern and independent envelope plots for repeated and non-repeating events.

These statistics are intended to help distinguish a brief transient from a sustained high-energy event. Two signals can share a similar maximum peak but place very different demands on a playback system if one recovers quickly and the other remains near peak.

## Bundle Output Structure

The primary analysis output is a `.aaresults` bundle. A bundle contains a manifest, per-channel folders, metadata JSON, and derived tables. These are the stable boundary between source analysis and later rendering/reporting.

Important per-channel bundle artifacts include:

- `metadata.json`: source path, channel identity, sample rate, duration, original peak, analysis time, and processing metadata.
- `octave_band_analysis.csv`: whole-interval full-spectrum and octave-band statistics, including `crest_factor_method = "whole_interval_peak_rms"` and `is_valid_crest_factor`.
- `time_domain_analysis.csv`: exported time-domain samples for crest factor, peak, RMS, validity, window/step settings, and method metadata.
- `time_domain_summary.csv`: summary statistics and method metadata for the time-domain method actually used.
- `octave_time_metrics.csv`: fixed-window octave-band crest-factor time data used for deep-dive plots, including validity, window/step settings, and `crest_factor_method = "fixed_window_peak_rms"`.
- `histogram_linear.csv` and `histogram_log_db.csv`: amplitude distribution data.
- Envelope and sustained-peak tables where the relevant analysis is available.
- `envelope_plot_data.json`: compact replay data for envelope plots. It stores only the configured top-N pattern and independent envelope windows used by the plots, plus compact time-axis metadata. If more envelope examples are needed later, the source analysis can be rerun with larger envelope plot counts.

Legacy `analysis_results.csv` export can still exist when explicitly enabled, but it is no longer the preferred reporting boundary. Reports and the GUI render path should prefer `.aaresults` bundles.

## Report Generation

Reports are generated from `.aaresults` bundles and the rendered plot folder. The renderer:

1. Finds one bundle or recursively discovers bundles below a selected results folder.
2. Renders per-channel plots from bundle tables.
3. Renders group plots and worst-channel manifests from bundle channel data.
4. Renders LFE, Screen, and Surround+Height deep-dive plots when the required groups and octave-time data are available.
5. Builds Markdown reports from bundle data and rendered plots.
6. Copies referenced graph images into a local `images/` folder beside each report. These report-local copies are downsampled and flattened to JPEG for compact PDF embedding; the original rendered PNG plots remain available in their channel and group folders.
7. Converts Markdown reports to PDF using local image URLs and page-aware plot blocks.

Keeping image assets local to each report folder makes the report folders portable and easier to archive or convert to PDF. The render CLI and GUI use `plotting.render_dpi` by default for standalone plot generation, while PDF size is controlled primarily by the compact report-local image copies.

In the GUI's normal analysis workflow, rendering and report generation are now part of the same per-track worker lifecycle. A worker analyzes a track, writes/updates its `.aaresults` bundle, renders that bundle, generates reports when enabled, and only then marks the item finished. The GUI progress bar therefore represents total work: analysis plus render/report steps. The table shows the current stage (`Analysis` or `Render`) and the item status.

The File menu's "Render existing results" command remains available for re-rendering existing `.aaresults` bundles without reprocessing source audio.

## How to Read the Reports

Use the report as a structured review:

- Start with the crest factor table to compare group dynamics.
- Review the crest factor over time plots to identify whether low crest factor occurs during high-level passages.
- Treat gaps in crest-factor-over-time plots as intentional invalid-window gaps. They indicate silence or windows below the configured RMS floor, not missing plot rendering.
- Use octave spectrum plots to see which bands dominate peak and RMS demand.
- Use LFE and group deep dives for frequency-specific behaviour over time.
- Use sustained-peak recovery tables to judge whether events are transient or sustained.
- Use peak occurrence and duty cycle to understand how often the signal approaches full scale.
- Treat envelope plot folders as detailed supporting artefacts. The main report lists pattern and independent envelope plot counts by channel rather than embedding every envelope plot in the PDF.

Low crest factor is not automatically a problem. It matters most when peak level and RMS level are both high, because that combination implies sustained demand and reduced headroom.

## Reproducibility Notes

The analysis is controlled by `config.toml` and command-line overrides. Important settings include:

- Sample rate
- Octave centre frequencies
- Filter-bank mode
- Time-domain crest factor mode
- Fixed-window crest factor window, step, and RMS floor
- Slow RMS and peak-hold constants for diagnostic slow mode
- Fixed-chunk duration
- Envelope detection thresholds
- Export and plotting options
- Result caching options

Result cache metadata records the source path, source modification time, configuration hash, and cache date. If the source or configuration changes, analysis should be regenerated.

To render or regenerate plots/reports after analysis bundles already exist:

```powershell
.\venv\Scripts\python.exe -m src.render --results "AudioAnalyserProject\analysis" --output-dir "AudioAnalyserProject\rendered" --reports
```

To regenerate this processing overview in the future, read `docs/processing_pipeline_report_instructions.md`, review the source files listed there, then update this document to reflect the current pipeline.

## Practical Limitations

This pipeline is designed for analytical consistency, not for replacing calibrated acoustic measurement. It works from digital source files and reports dBFS-referenced programme characteristics. It does not know the playback chain, room response, loudspeaker sensitivity, limiter behaviour, or amplifier voltage/current limits unless those are added as separate calibration layers.

The strongest use case is comparative programme analysis: identifying which content, channel group, frequency band, and time region is most demanding before moving to playback-system-specific interpretation.
