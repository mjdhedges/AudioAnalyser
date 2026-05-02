# Normalization and dBFS Reference Proof

Status: **PASS**

This proof validates the channel-normalization contract used by Audio Analyser:
analysis runs on peak-normalized channel data, while the channel's original peak
is carried as the dBFS reference.

## Method

- Deterministic source with low-frequency and mid-band tones
- Gain cases: `0.25x, 0.5x, 1x`
- Production whole-interval metrics: `compute_whole_interval_crest_factor`
- Production octave-band metrics: `MusicAnalyzer.analyze_octave_bands`
- Production fixed-window metrics: `FixedWindowTimeDomainCalculator`
- Production octave time export path: `results.bundle._write_octave_time_metrics`
- Checked bands: `Full Spectrum, 62.500, 1000.000`

Each raw source is scaled by gain, then normalized to peak `1.0` before analysis.
The raw channel peak is passed as `original_peak`.

## Results

- Gain-shift checks: **PASS**
- Maximum peak/RMS shift error: `0.000000017498 dB`
- Maximum crest-factor gain-invariance error: `0.000000000000 dB`
- Acceptance tolerance: `1.0e-07 dB`
- Normalized RMS span across gain cases: `0.000000000000e+00`
- Reference `1.0x` full-band peak: `-0.915150 dBFS`
- Reference `1.0x` full-band RMS: `-8.925351 dBFS`
- Octave whole-interval reference rows checked: `3`
- Octave time export reference rows checked: `2`

## Interpretation

Peak and RMS dBFS values move by the exact source gain change because dBFS is
computed from the normalized analysis level multiplied by `original_peak`:

```text
level_dbfs = 20 * log10(normalized_level * original_peak)
```

Crest factor remains gain-invariant because it is a ratio of peak to RMS from
the same normalized signal path:

```text
crest_factor = peak / rms
```

The same behavior holds for filtered octave-band whole-interval metrics and for
the exported octave time metrics. This confirms that octave-band dBFS values use
the channel peak reference consistently, while crest factor remains independent
of source gain.

## Outputs

- `normalization_results.csv`
- `octave_band_reference.csv`
- `octave_time_reference.csv`
- `gain_shift_acceptance.csv`
- `gain_shift_dbfs.png`
- `crest_factor_invariance.png`
- `source_material/*.wav`
