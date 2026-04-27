# Processing Pipeline Report Instructions

Use this file as the source brief when regenerating `reports/processing_pipeline.md`.
The report is intended for audio professionals, not software developers, so explain
the engineering in plain language while staying technically accurate.

## Audience

- Cinema, recording, loudspeaker, amplifier, and system-integration professionals.
- Readers understand dBFS, RMS, crest factor, octave bands, LFE, and channel layouts.
- Readers should not need to understand the Python codebase.

## Tone and Style

- Professional, concise, and explanatory.
- Avoid marketing language and avoid overstating standards compliance.
- Prefer "what is measured", "how it is measured", and "why it matters".
- Call out configuration-driven behavior where appropriate. Do not hard-code timing
  assumptions; cite metadata such as `[TIME_DOMAIN_SUMMARY]` where report wording
  depends on a completed analysis run.

## Source Files to Review

- `src/main.py`: CLI flow, batch/single-file processing, bundle output layout, and result caching.
- `src/audio_processor.py`: input loading, ffprobe/ffmpeg handling, sample-rate conversion,
  multi-channel preservation, normalization helpers.
- `src/channel_mapping.py`: RP22-style channel naming and bundle metadata labels.
- `src/track_processor.py`: per-channel pipeline orchestration.
- `src/octave_filter.py`: octave-band filter bank design and processing modes.
- `src/music_analyzer.py`: octave-band statistics, time-domain analysis selection, extreme
  chunk analysis.
- `src/time_domain_metrics.py`: slow and fixed-window crest-factor methods.
- `src/envelope_analyzer.py`: peak envelope, pattern, independent event, and sustained-peak
  analysis.
- `src/data_export.py`: legacy CSV sections and metric definitions still reused by bundle exports.
- `src/results/bundle.py` and `src/results/reader.py`: `.aaresults` writer and reader.
- `src/results/render.py` and `src/render.py`: bundle graph rendering, group outputs, and render CLI.
- `src/report_generator.py`: bundle-native Markdown report generation and local image asset handling.
- `config.toml`: current default analysis, filtering, envelope, plotting, export, and
  performance settings.

## Required Report Structure

1. Title and short purpose statement.
2. Executive summary of what the pipeline measures.
3. End-to-end processing flow from input file to `.aaresults` bundle to rendered reports.
4. Input handling and channel mapping.
5. Normalization and dBFS reference explanation.
6. Octave-band filtering approach.
7. Time-domain crest factor approach, including slow vs fixed-window behavior.
8. Envelope, peak, and recovery analysis.
9. Bundle export and report generation.
10. Interpretation guidance and limitations.
11. Reproducibility notes.

## Accuracy Rules

- Describe multi-channel processing as per-channel analysis with group-level summaries.
- State that rendered report image assets are referenced from the render output folder.
- Explain that generated report prose should read run metadata and tables from `.aaresults` bundle artifacts.
- Distinguish whole-track octave statistics from time-domain sampled statistics.
- Explain that octave-band filtering uses the FFT power-complementary design
  proven in `proofs/octave_band_energy_closure/`, including low/high residual
  bands when enabled in config.
- State the octave processing mode from exported metadata whenever available:
  requested mode, effective mode, FFT block length, output storage, and
  configured octave RAM limit.
- Explain that octave-band RMS values must be summed as linear power, not by
  adding dB values.
- In the Slow Mode section, reference IEC sound level meter Slow time weighting:
  the RMS envelope time constant is intended to follow the 1-second Slow
  response, so the RMS trend should be comparable to what an engineer would
  see from a Slow SPL meter when the analysis is calibrated to the same
  reference and uses equivalent weighting.
- Make the calibration caveat explicit: dBFS analysis only matches absolute SPL
  after playback-chain, microphone, acoustic weighting, and measurement-position
  calibration.
- Keep the peak-hold envelope distinct from the IEC Slow RMS behaviour; peak hold
  is a crest-factor aid with its own configured decay constant.
- Do not say RMS is calculated over 2-second periods unless the CSV metadata for that run
  actually says fixed-window mode with a 2-second window.
- Mention that extreme chunk octave analysis currently uses fixed windows based on
  `analysis.chunk_duration_seconds`, even when the time-domain plot/export mode is slow.

## Regeneration Checklist

- Re-read the source files above before editing the report.
- Update the report if defaults in `config.toml` change.
- Regenerate track reports from bundles with:

```powershell
.\venv\Scripts\python.exe -m src.render --results analysis --output-dir rendered --reports
```

- Verify report image links after regeneration.
- Keep `reports/processing_pipeline.md` free of track-specific numeric results unless the
  report is intentionally being made run-specific.
