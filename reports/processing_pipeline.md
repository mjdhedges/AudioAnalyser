# Audio Analyser Processing Pipeline

This document explains how Audio Analyser processes source material and turns it into channel-level measurements, group-level plots, CSV exports, and Markdown reports. It is written for audio professionals who need to understand what the numbers mean, how they are produced, and how much confidence to place in them.

## Executive Summary

Audio Analyser is a measurement pipeline for audio programme material. It is designed to answer practical questions about level, crest factor, frequency-dependent behaviour, peak density, low-frequency energy, and peak recovery characteristics.

The pipeline processes each channel independently, then combines selected results into channel-group summaries such as Screen, LFE, and Surround+Height. This matters because a single full-mix statistic can hide the channel that actually drives amplifier, loudspeaker, or system-headroom demand.

The main outputs are:

- Per-channel CSV files containing metadata, octave-band statistics, time-domain crest factor data, peak statistics, histograms, and envelope/recovery statistics.
- Per-channel plots for octave spectrum, crest factor, time-domain behaviour, histograms, and envelope events.
- Group-level plots and Markdown reports that identify the most demanding channels and present the results in a reviewable form.
- Local report image folders, so each report folder contains the graphs it needs under `images/`.

## End-to-End Flow

The processing flow is:

1. The CLI finds one input file or recursively discovers supported audio files in a batch folder.
2. Existing analysis results can be reused when result-cache metadata indicates that the source file and analysis configuration have not changed.
3. The audio file is loaded, resampled to the configured sample rate if required, and preserved as mono or multi-channel audio.
4. Multi-channel material is split into individual channels.
5. Each channel is normalized for analysis while retaining the original peak reference so exported levels remain tied to the source file's dBFS scale.
6. An octave-band filter bank is created for the channel.
7. The analyzer calculates whole-track octave-band statistics, time-domain crest factor, fixed-window extreme chunks, peak/envelope statistics, and histograms.
8. CSV files and per-channel plots are written.
9. Post-processing generates worst-channel manifests and group-level plots.
10. Markdown reports are generated from the CSV exports and copied graph assets.

## Input Handling

Audio Analyser supports common audio formats such as WAV, FLAC, MP3, AIFF, M4A, and MKV/MTS-style container workflows where ffmpeg/ffprobe support is available.

For file loading, the pipeline first tries `soundfile` so multi-channel layouts are preserved. If that fails, it falls back to `librosa`. When the source sample rate differs from the configured analysis rate, audio is resampled to the configured rate.

For MKV and TrueHD-style workflows, ffprobe is used to inspect audio streams. TrueHD streams can be extracted and decoded to temporary PCM WAV for analysis. MTS files are probed for channel layout information and can be loaded directly when supported.

## Channel Handling

Multi-channel files are not downmixed for analysis. Each channel is processed separately and written to its own channel folder, for example:

- `Channel 1 FL`
- `Channel 2 FR`
- `Channel 3 FC`
- `Channel 4 LFE`
- `Channel 7 SL`

Channel names are inferred from the available layout information where possible, including RP22-style naming for cinema layouts. Mono material is processed directly in the track output folder.

Group-level reports then combine or select channel results into practical review groups:

- Screen
- LFE
- Surround+Height
- Mono or All Channels where a cinema grouping does not apply

## Level Reference and Normalization

Per-channel audio is normalized for internal processing. The original source peak is retained and used when converting analysis values back to dBFS. This allows the pipeline to gain the numerical stability benefits of normalized processing while still reporting levels relative to the original source scale.

In the CSV export, track metadata includes the original peak and original peak in dBFS. Plot and export calculations use this reference when reporting peak and RMS levels in dBFS.

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

The octave FFT mode is configurable. In `auto` mode, the analyser estimates the full-file FFT memory requirement and uses full-file FFT processing when it fits under the configured RAM budget. If the estimate exceeds `analysis.octave_max_memory_gb`, it switches to large-block FFT processing. If the octave-bank time-series output itself exceeds the budget, the bank is held in temporary disk-backed storage while downstream metrics run. CSV metadata records the requested mode, effective mode, block length, storage method, residual-band settings, and RAM budget.

For each octave band and for the full-spectrum signal, the analyzer exports:

- Peak amplitude and peak dBFS
- RMS and RMS dBFS
- Dynamic range ratio and dB value
- Crest factor ratio and dB value
- Mean, standard deviation, and amplitude percentiles

These whole-track octave statistics are different from the time-domain samples. They describe the channel over the whole analyzed duration, not just one moment in time.

## Time-Domain Crest Factor

Time-domain crest factor is configuration-driven. Reports should not assume a fixed period; they should read `[TIME_DOMAIN_SUMMARY]` from the relevant CSV export.

Two modes are supported:

### Slow Mode

Slow mode uses a peak-hold envelope and a slow RMS envelope, sampled at a configured time step. The slow RMS envelope is intended to follow the IEC sound level meter "Slow" time weighting concept, where the exponential time constant is 1 second. In practical terms, this makes the RMS level trend behave like the level movement an engineer would expect to see from an SPL meter set to Slow, rather than a simple fixed-block RMS calculation.

This does not by itself turn dBFS into calibrated SPL. Absolute agreement with an SPL meter still requires a calibrated playback chain, microphone, acoustic weighting, and measurement position. However, when the analysis is calibrated to the same reference and uses equivalent weighting, the slow RMS behaviour should be comparable to what a professional would see on a Slow SPL meter.

The peak-hold envelope is a separate crest-factor aid. It tracks programme peaks with its own configured decay constant so peak-to-RMS behaviour can be reviewed alongside the slow RMS trend.

The CSV metadata records:

- `time_domain_mode`
- `time_domain_time_step_seconds`
- `time_domain_rms_method`
- `time_domain_peak_method`
- `chunk_duration_seconds`

For example, a run may state that time-domain crest factor uses slow mode, RMS uses `slow_rms_tau=1.0`, peaks use `peak_hold_tau=2.0`, and values are exported every 1-second interval.

### Fixed-Chunk Mode

Fixed-chunk mode calculates peak and RMS over non-overlapping windows based on the configured `analysis.chunk_duration_seconds`. This is a straightforward block-based method and is useful when repeatable fixed-duration windows are required.

## Extreme Chunk Analysis

Extreme chunk analysis identifies sections with minimum and maximum crest factor. This currently uses fixed windows based on `analysis.chunk_duration_seconds`, even when the exported time-domain plot mode is slow.

This distinction is important:

- Time-domain plots may use slow metering metadata.
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

## CSV Export Structure

Each `analysis_results.csv` is organized into named sections:

- `[TRACK_METADATA]`: source path, channel identity, sample rate, duration, original peak, analysis time.
- `[ADVANCED_STATISTICS]`: peak rates, duty cycle, spectral balance, temporal consistency, and related metrics.
- `[OCTAVE_BAND_ANALYSIS]`: whole-track full-spectrum and octave-band statistics.
- `[TIME_DOMAIN_ANALYSIS]`: exported time-domain samples for crest factor, peak, and RMS.
- `[TIME_DOMAIN_SUMMARY]`: metadata and summary statistics for the time-domain method actually used.
- `[HISTOGRAM_DATA]`: amplitude distribution data where exported.
- Envelope and sustained-peak sections where the relevant analysis is available.

Reports should prefer these exported sections over assumptions. In particular, the explanatory text for time-domain analysis should be driven by `[TIME_DOMAIN_SUMMARY]`.

## Report Generation

Track reports are generated from the analysis folder into the reports folder. The report generator:

1. Finds track analysis folders.
2. Determines channel groups from the worst-channel manifest or inferred channel folders.
3. Reads CSV exports for the selected channels.
4. Builds the Markdown report text.
5. Copies referenced graph images into a local `images/` folder beside the report.
6. Links graphs using simple local paths such as `images/Screen_crest_factor_time.png`.

Keeping image assets local to each report folder makes the report folders portable and easier to archive or convert to PDF.

## How to Read the Reports

Use the report as a structured review:

- Start with the crest factor table to compare group dynamics.
- Review the crest factor over time plots to identify whether low crest factor occurs during high-level passages.
- Use octave spectrum plots to see which bands dominate peak and RMS demand.
- Use LFE and group deep dives for frequency-specific behaviour over time.
- Use sustained-peak recovery tables to judge whether events are transient or sustained.
- Use peak occurrence and duty cycle to understand how often the signal approaches full scale.

Low crest factor is not automatically a problem. It matters most when peak level and RMS level are both high, because that combination implies sustained demand and reduced headroom.

## Reproducibility Notes

The analysis is controlled by `config.toml` and command-line overrides. Important settings include:

- Sample rate
- Octave centre frequencies
- Filter-bank mode
- Time-domain crest factor mode
- Slow RMS and peak-hold constants
- Fixed-window chunk duration
- Envelope detection thresholds
- Export and plotting options
- Result caching options

Result cache metadata records the source path, source modification time, configuration hash, and cache date. If the source or configuration changes, analysis should be regenerated.

To regenerate track reports after analysis outputs already exist:

```powershell
.\venv\Scripts\python.exe -m src.generate_reports
```

To regenerate this processing overview in the future, read `docs/processing_pipeline_report_instructions.md`, review the source files listed there, then update this document to reflect the current pipeline.

## Practical Limitations

This pipeline is designed for analytical consistency, not for replacing calibrated acoustic measurement. It works from digital source files and reports dBFS-referenced programme characteristics. It does not know the playback chain, room response, loudspeaker sensitivity, limiter behaviour, or amplifier voltage/current limits unless those are added as separate calibration layers.

The strongest use case is comparative programme analysis: identifying which content, channel group, frequency band, and time region is most demanding before moving to playback-system-specific interpretation.
