# Peak-Hold Tau and Block Window Study

This study explores two linked crest-factor questions:

1. How fixed-block window length changes measured crest factor.
2. Which peak-hold time constant best aligns the slow-mode crest-factor
   algorithm with a fixed-block reference while holding the RMS tau at the IEC
   Slow value of 1.0 second.

## Method

- Source material: `source_material/peak_hold_tau_test.wav`
- Sample rate: 48000 Hz
- Slow RMS tau: 1.0 second
- Primary tau reference: non-overlapping 1-second block peak / RMS crest factor
- Candidate peak-hold tau values: 0.05, 0.08, 0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.56, 0.63, 0.71, 0.8, 0.9, 1.0, 1.12, 1.25, 1.4, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0
- Fixed block windows studied: 5ms, 10ms, 20ms, 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s, 62s
- Warm-up excluded from scoring: 3.0 seconds

The generated signal contains level-stepped click trains, steady-state sine
waves at multiple levels, low-frequency sine waves, and pink-noise bursts of
different durations and levels. This stresses fast attack behaviour, missed peak
response, stable crest factor, and RMS settling without relying on copyrighted
programme material.

## Result

### 1-second reference tau sweep

Best peak-hold tau by mean absolute error against the 1-second fixed-block
reference:

- Peak-hold tau: **0.8 seconds**
- Mean error: -1.190 dB
- Mean absolute error: 2.959 dB
- RMSE: 5.313 dB
- P95 absolute error: 11.923 dB
- Max absolute error: 19.687 dB
- Windows within +/-0.5 dB: 39.0%
- Windows within +/-1.0 dB: 52.5%

For comparison:

- 1.0 second peak tau: MAE 3.008 dB, RMSE 5.409 dB
- 2.0 second peak tau: MAE 3.231 dB, RMSE 6.615 dB

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
  mean crest factor 2.418 dB, P95 8.872 dB,
  max 15.240 dB.
- 1-second block: mean crest factor 11.042 dB,
  P95 33.313 dB,
  max 33.313 dB.
- Whole-track block (62s):
  crest factor 9.834 dB. This is a broad single-number
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
| 5ms | 0.05s | 1.600 | 8.154 | 9.612 |
| 10ms | 0.05s | 1.459 | 7.960 | 9.412 |
| 20ms | 0.05s | 1.365 | 7.811 | 9.259 |
| 50ms | 0.05s | 1.085 | 7.502 | 8.970 |
| 100ms | 0.05s | 1.098 | 7.058 | 8.517 |
| 250ms | 0.125s | 1.729 | 5.768 | 7.335 |
| 500ms | 0.4s | 2.849 | 3.935 | 5.062 |
| 1s | 0.8s | 2.959 | 3.008 | 3.231 |
| 2s | 5s | 2.027 | 4.734 | 2.936 |
| 5s | 31.5s | 2.584 | 5.119 | 4.355 |
| 10s | 63s | 3.587 | 6.134 | 5.779 |
| 62s | 1.6s | 0.058 | 3.490 | 1.136 |

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

The 1-second reference remains useful for judging the peak-hold algorithm because
it sits on the IEC Slow RMS time scale. Against that reference, the best
peak-hold tau is 0.8 seconds, with 1.0 second close behind. That supports using a
peak-hold tau around the IEC Slow scale, while recognizing that changing the
reference block length changes the best-matching peak tau.

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

The fixed-block method is treated as the reference for 1-second window crest
factor because it directly computes maximum absolute sample divided by RMS inside
each window. Slow-mode crest factor is a metering-style method: RMS follows the
IEC Slow 1-second exponential behaviour, while peak hold has instantaneous attack
and configurable exponential decay. The best tau is therefore the tau that makes
that metering-style trace closest to the fixed-window reference for this proof
material.

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
