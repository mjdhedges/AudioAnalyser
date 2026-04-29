# Crest Factor Window and Peak-Hold Study

## Decision Summary

Use **fixed-window crest factor** as the main programme/report metric:

- Window: **2.0 seconds**
- Hop/step: **1.0 second**
- Formula: `max(abs(window)) / rms(window)` on the same signal window
- Silence/no-signal handling: windows below the configured RMS floor are
  **invalid** and should be stored/plotted as missing (`NaN`), not clamped to
  `0 dB`
- Slow RMS + peak-hold remains useful as a **meter-style diagnostic**, but it
  should not be the authoritative crest-factor-over-time value in reports

This replaces the earlier interpretation that a 1-second slow RMS + peak-hold
trace was a suitable primary crest-factor trace.

This study explores two linked crest-factor questions:

1. How fixed-block window length changes measured crest factor.
2. Which peak-hold time constant best aligns the slow-mode crest-factor
   algorithm with a fixed-block reference while holding the RMS tau at the IEC
   Slow value of 1.0 second.

## Method

- Source material: `source_material/peak_hold_tau_test.wav`
- Sample rate: 48000 Hz
- Slow RMS tau: 1.0 second
- Primary tau reference: non-overlapping 2s block peak / RMS crest factor
- Candidate peak-hold tau values: 0.05, 0.08, 0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.56, 0.63, 0.71, 0.8, 0.9, 1.0, 1.12, 1.25, 1.4, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0
- Fixed block windows studied: 5ms, 10ms, 20ms, 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s, 72s
- Warm-up excluded from scoring: 3.0 seconds
- Programme noise bed: -80 dBFS pink noise
- Final true-silence edge case: 10 seconds

The generated signal contains level-stepped click trains, steady-state sine
waves at multiple levels, low-frequency sine waves, and pink-noise bursts of
different durations and levels. A low-level pink-noise bed keeps programme gaps
from becoming exact digital zero, while the final true-silence section remains
as a separate edge case. This stresses fast attack behaviour, missed peak
response, stable crest factor, RMS settling, and invalid/absent signal handling
without relying on copyrighted programme material.

## Key Findings

### 1. The original 1-second target was too fragile

The earlier proof used a 1-second fixed-block reference. That reference produced
0 dB crest-factor points when blocks landed on exact digital zero. Those were not
real crest-factor measurements; they were invalid/no-signal windows represented
as `0 dB` by the fallback/clamp:

```python
crest = peak / rms
crest_db = 20 * log10(max(crest, 1.0))
```

For a true silent block, `rms == 0`, so the proof encoded "no measurement" as a
linear ratio of `1.0`, which becomes `0 dB`.

### 2. A realistic programme floor removes false active-programme 0 dB values

After adding a `-80 dBFS` pink-noise bed to the programme section, fixed-block
crest factor no longer collapses to `0 dB` during programme gaps:

| Window | Programme min | Programme 0 dB count | Programme median |
| --- | ---: | ---: | ---: |
| 1s | 3.0106 dB | 0 | 11.7758 dB |
| 2s | 3.0107 dB | 0 | 14.4928 dB |
| 5s | 4.2967 dB | 0 | 10.4388 dB |
| 10s | 5.0671 dB | 0 | 12.7203 dB |

The remaining `0 dB` values in the full-window summaries come from the explicit
10-second true-silence tail. Those are useful for testing invalid-window
handling, but they must not be reported as real crest factor.

### 3. Narrow-band real signals should not repeatedly hit 0 dB

For a sine wave, crest factor is approximately `3.01 dB`. A square wave can reach
`0 dB`, but narrow octave-band signals should not behave like ideal square waves
because the octave filter limits bandwidth. Repeated 0 dB points in octave-band
crest-factor graphs are therefore a sign of measurement/windowing/silence
handling, not a meaningful programme result.

### 4. 2-second fixed windows are the better reporting target

The block-window study still shows that very short windows move toward local
instantaneous behaviour and can understate crest factor. The 2-second window is
the smallest tested window that gives a stable programme-region floor while
retaining useful time resolution.

Recommended reporting window:

```text
crest_factor_window_seconds = 2.0
crest_factor_step_seconds = 1.0
```

## Result

### 2s reference tau sweep

Best peak-hold tau by mean absolute error against the 2s fixed-block
reference:

- Peak-hold tau: **1.4 seconds**
- Mean error: -2.903 dB
- Mean absolute error: 3.415 dB
- RMSE: 5.848 dB
- P95 absolute error: 11.495 dB
- Max absolute error: 20.030 dB
- Windows within +/-0.5 dB: 40.0%
- Windows within +/-1.0 dB: 45.7%

For comparison:

- 1.0 second peak tau: MAE 4.057 dB, RMSE 7.109 dB
- 2.0 second peak tau: MAE 4.083 dB, RMSE 6.353 dB

### Detector design rationale

The RMS detector and peak detector intentionally do different jobs. The IEC Slow
RMS detector squares the signal, exponentially averages mean-square energy with
`slow_rms_tau=1.0`, and then converts back to RMS. This is the part of the
analysis anchored to sound-level-meter behaviour.

The peak detector is not another RMS detector with a different tau. It takes the
absolute sample value, attacks immediately when a new sample exceeds the held
value, then releases with exponential decay. This preserves short peak events
that would otherwise be diluted by an energy average, which is essential for
crest factor because crest factor compares peak demand against RMS energy.

That split is the right shape for this function: RMS should represent sustained
energy, while peak should represent recent maximum excursion. The tau sweep in
this proof is therefore not trying to make the peak detector behave like RMS; it
is finding the peak-release time that gives a useful short-term crest-factor
trace when RMS remains anchored to IEC Slow.

### Block window effect

The fixed-block window length changes both the value and usefulness of the crest
factor measurement:

- Shortest block tested (5ms):
  mean crest factor 5.079 dB, P95 9.905 dB,
  max 15.240 dB.
- 1-second block: mean crest factor 11.178 dB,
  P95 33.313 dB,
  max 33.313 dB.
- Whole-track block (72s):
  crest factor 10.484 dB. This is a broad single-number
  description and contains no time variation.

Very short windows move toward local instantaneous behaviour: the 5-100ms blocks
measure much lower average crest factor because the RMS and peak are being taken
over nearly the same tiny event. In the limit, a one-sample block has peak equal
to RMS and crest factor becomes 0 dB. For this proof material, the smallest block
that gives a stable and meaningful short-term crest-factor result is about 2
seconds. Larger blocks give broadly similar crest-factor conclusions, but average
them over more time and therefore reduce time resolution. Very long windows
describe the track broadly but hide short-term changes.

### Best tau by block reference

| Window | Best peak tau | Best MAE | 1s tau MAE | 2s tau MAE |
| --- | ---: | ---: | ---: | ---: |
| 5ms | 0.05s | 3.656 | 5.097 | 7.425 |
| 10ms | 0.2s | 3.642 | 4.848 | 7.107 |
| 20ms | 0.2s | 3.638 | 4.668 | 6.851 |
| 50ms | 0.2s | 3.568 | 4.401 | 6.497 |
| 100ms | 0.25s | 3.487 | 4.115 | 6.122 |
| 250ms | 0.315s | 3.281 | 3.412 | 5.352 |
| 500ms | 1.25s | 2.532 | 2.587 | 4.084 |
| 1s | 1.4s | 2.304 | 2.652 | 3.369 |
| 2s | 1.4s | 3.415 | 4.057 | 4.083 |
| 5s | 3.15s | 4.995 | 5.750 | 5.095 |
| 10s | 3.15s | 5.010 | 8.413 | 6.541 |
| 72s | 2s | 0.486 | 10.484 | 0.486 |

### Practical conclusion

For short-term crest-factor analysis, keep RMS anchored to IEC Slow with
`slow_rms_tau=1.0`. Treat whole-track crest factor as a useful headline number
only, because it intentionally removes time variation. Avoid very small block
windows for programme-dynamics reporting: below roughly 100ms, the measurement
is dominated by local instantaneous behaviour and trends toward 0 dB as the
window approaches one sample.

For this proof material, the minimum stable fixed-block crest-factor window is
about 2 seconds. Smaller windows increasingly collapse toward local
instantaneous behaviour and can understate crest factor; larger windows remain
stable but smear the result over more programme time. The useful engineering
choice is therefore a block/window length long enough to avoid collapse, but not
so long that it hides dynamic changes.

The primary reference is now the 2s block
because the 1-second reference was too vulnerable to exact-zero gaps and local
collapse. The result should be read as a stability check for the slow metering
trace, not as a replacement for fixed-window crest factor when reporting
programme dynamics.

## Outputs

- `tau_sweep_results.csv`: numeric results for every candidate tau
- `tau_sweep_error.png`: error metrics vs peak-hold tau
- `best_tau_trace.png`: crest-factor trace for the best tau against reference
- `block_window_summary.csv`: crest-factor distribution vs fixed block length
- `block_window_effect.png`: visual summary of block length effect
- `block_window_traces.png`: representative crest-factor traces by block length
- `tau_by_window_summary.csv`: best peak-hold tau for each block length
- `best_tau_by_block_window.png`: best tau and error vs block length

## Interpretation

The fixed-block method is treated as the reference because it directly computes
maximum absolute sample divided by RMS inside each window. Slow-mode crest factor
is a metering-style method: RMS follows the IEC Slow 1-second exponential
behaviour, while peak hold has instantaneous attack and configurable exponential
decay. The best tau is therefore the tau that makes that metering-style trace
closest to the fixed-window reference for this proof material.

This result should not be treated as a universal standard. It is evidence for
this stress-test material and scoring method. Clicks, steady tones, and short
noise bursts stress the peak-hold and RMS envelopes differently, so the optimum
can move depending on source material, chosen block reference, and whether mean
error, mean absolute error, RMSE, or percentile error is considered the primary
criterion.

The IEC Slow RMS tau remains important because it anchors the RMS side of the
analysis to a real metering behaviour. The block-length study is about selecting
a time scale that gives useful short-term insight, not replacing that RMS
anchor.

## Required Main Programme Updates

### Processing Model

Keep crest-factor processing to two explicit paths:

- **Whole-interval crest factor** for any summary value without a time axis:
  full-track, full-channel, octave-band spectrum, and report table summary rows.
  Peak and RMS must be measured over the same complete interval:
  `max(abs(signal)) / rms(signal)`.
- **Configured time-series crest factor** for any graph/table with time on the
  x-axis. This path must use `time_domain_crest_factor_mode`, selecting
  `fixed_window`, `slow`, or `fixed_chunk`.

Do not use hybrid definitions such as whole-band RMS combined with a separate
1-second sliding peak for spectrum rows. That produces a third interpretation
that is hard to explain and does not match either the whole-interval summary or
the configured time-series behaviour.

### Configuration

Add explicit fixed-window crest-factor settings:

```toml
[analysis]
crest_factor_window_seconds = 2.0
crest_factor_step_seconds = 1.0
crest_factor_rms_floor_dbfs = -80.0
crest_factor_primary_mode = "fixed_window"
```

Keep the existing slow metering settings, but treat them as diagnostic:

```toml
[analysis]
peak_hold_tau_seconds = 1.4  # only for slow metering diagnostic traces
time_domain_slow_rms_tau_seconds = 1.0
```

### Primary Time-Domain Crest Factor

For each channel, compute primary crest factor from fixed windows:

```python
window_samples = int(crest_factor_window_seconds * sample_rate)
step_samples = int(crest_factor_step_seconds * sample_rate)

for start in range(0, len(signal) - window_samples + 1, step_samples):
    window = signal[start:start + window_samples]
    peak = max(abs(window))
    rms = sqrt(mean(window ** 2))

    if dbfs(rms, original_peak) < crest_factor_rms_floor_dbfs:
        crest_factor_db = NaN
    else:
        crest_factor_db = 20 * log10(max(peak / rms, 1.0))
```

Use this for:

- per-channel `crest_factor_time.png`
- group `crest_factor_time.png`
- report percentile tables and narrative
- selecting stable low/high crest-factor regions when "real crest factor" is the
  intended metric

### Octave-Band Crest Factor Time

For each octave/residual band, apply the same fixed-window calculation to the
band-limited signal:

```text
octave_crest_factor_db = max(abs(band_window)) / rms(band_window)
```

Do not derive octave-band report crest factor from a slow RMS + decaying
peak-hold trace. That trace can decay below the slow RMS tail, forcing the clamp
to `0 dB`, which is visually misleading.

### Silence and Low-Level Windows

Never encode "no valid crest-factor measurement" as `0 dB`.

Use:

- `NaN` in internal arrays/tables
- blank/null in CSV/JSON exports if needed
- gaps in plots
- explicit "below floor / silence" counts in summaries if useful

This allows the final true-silence tail to test the pipeline without corrupting
crest-factor distributions.

### Bundle Output Changes

Update `.aaresults` output to make the metric explicit. Suggested columns for
`time_domain_analysis.csv` and `octave_time_metrics.csv`:

```text
time_seconds
crest_factor_db
peak_level_dbfs
rms_level_dbfs
is_valid_crest_factor
crest_factor_window_seconds
crest_factor_step_seconds
crest_factor_method
```

Suggested values:

```text
crest_factor_method = "fixed_window_peak_rms"
crest_factor_window_seconds = 2.0
crest_factor_step_seconds = 1.0
```

If slow metering traces are retained, store them separately, for example:

```text
slow_peak_hold_crest_factor_db
slow_peak_hold_tau_seconds
slow_rms_tau_seconds
```

### Plotting Changes

Plot fixed-window crest factor as the main line. For invalid windows:

- leave gaps in the line (`NaN`)
- do not clamp to `0 dB`
- optionally annotate silent/below-floor regions lightly

If the slow meter trace is shown, label it clearly:

```text
Slow meter diagnostic: peak-hold / IEC Slow RMS
```

It should not be named simply "crest factor" in reports.
