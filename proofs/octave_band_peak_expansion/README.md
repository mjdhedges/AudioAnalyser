# Octave-Band Peak Expansion Proof

## Decision Summary

The Audio Analyser octave bank is **power-complementary**, not
**peak-limiting**.

This means:

- Band RMS/power values are valid to sum in linear power.
- A filtered octave-band sample peak can legitimately be higher than the
  source full-band sample peak.
- Small positive filtered-band peak readings, for example `+1 dBFS` at 62.5 Hz,
  are not automatically a gain-reference bug or a phase bug.
- Reports should treat these values as **derived filtered-band peaks**, not as
  source-file headroom.

This proof was added after `analysis/v0.3.2` still showed a few octave-band
peaks above `0 dBFS` after the channel gain-reference bug was fixed. The
remaining positives were isolated to filtered octave-band metrics.

## Why This Can Happen

The source waveform is the sum of many frequency components. Its full-band
sample peak depends on their instantaneous phase and cancellation. Filtering can
remove harmonics or neighbouring bands that were limiting or cancelling the
component of interest, changing the waveform shape of the extracted band.

That extracted component can therefore have a sample peak above the source
full-band sample peak, even though the filter has not applied broadband gain.

This is the same practical issue that can occur after DSP, for example a
subwoofer path high-passed around 80 Hz: the post-DSP waveform can have slightly
different peaks from the pre-DSP full-band waveform.

## Method

The proof uses the production `OctaveBandFilter` from `src/octave_filter.py`:

- Sample rate: `48000 Hz`
- Duration: `20 s`
- Processing mode: `full_file`
- Filter bank: FFT power-complementary octave bank
- Source signals are all peak-normalized to a full-band sample peak of `1.0`
  (`0 dBFS`)

Signals tested:

- `62.5 Hz sine`
- `62.5 Hz + 125 Hz burst`
- `Unit impulse`
- `Clipped 62.5 Hz square`
- `Pink noise`

Outputs:

- `peak_expansion_results.csv`
- `peak_expansion_by_signal.png`
- `rms_power_closure_by_signal.png`
- `real_event_metrics.csv`
- `real_event_waveform_trace.csv`
- `real_event_source_vs_filtered_62_5hz.png`

## Results

Maximum filtered-band peak expansion per signal:

| Signal | Max band | Band peak | Expansion vs source peak | Summed band power |
| --- | ---: | ---: | ---: | ---: |
| 62.5 Hz sine | 62.5 Hz | 1.000000 | 0.0000 dB | 1.000000 |
| 62.5 Hz + 125 Hz burst | 125 Hz | 0.568316 | -4.9082 dB | 1.000000 |
| Unit impulse | 16000 Hz | 0.515605 | -5.7537 dB | 1.000000 |
| Clipped 62.5 Hz square | 62.5 Hz | 1.275064 | +2.1106 dB | 1.000000 |
| Pink noise | 4 Hz | 0.369875 | -8.6389 dB | 1.000000 |

Only the clipped/square-like low-frequency case exceeds the source full-band
peak. The same run still closes RMS power exactly to numerical precision for
all cases.

## Real-World Time-Domain Event

The proof also includes a production-data case from:

- Track: `Ready Player One (2018) - Race.mts`
- Channel: `FL`
- Band: `62.5 Hz`
- Exported event row: `132 s`

The exported `132 s` row is a fixed-window row whose timestamp marks the end of
the 2-second window, so the event window is `130-132 s`. The proof decodes the
source through the same `AudioProcessor.load_audio()` path as the main program,
filters the full decoded FL channel with the same full-file FFT octave weight,
then crops a 1-second waveform view around the strongest filtered-band sample.

Result from `real_event_metrics.csv`:

| Source peak in 1 s view | Filtered 62.5 Hz peak | Expansion vs source peak |
| ---: | ---: | ---: |
| `-0.7758 dBFS` | `+1.0360 dBFS` | `+1.8119 dB` |

The waveform plot shows the full-band source repeatedly approaching the channel
peak limit while the band-limited 62.5 Hz component continues as a smoother
low-frequency waveform. Around the strongest sample, neighbouring frequency
content and clipping/limiting-like waveform shape constrain the source samples,
but the extracted 62.5 Hz component is no longer constrained by those other
components and rises above `0 dBFS`.

## Interpretation

The positive band peak in the clipped square case is expected. The square wave
is bounded to `±1`, but the full-band square contains harmonics. The octave band
extracts the 62.5 Hz fundamental and removes harmonics that help define the
flat-topped waveform. The resulting band-limited sinusoidal component can peak
above the original bounded square waveform.

The real `Ready Player One` event is the same mechanism in programme material:
it is not a mathematically pure square wave, but the source waveform in the
event window is strongly constrained while a large low-frequency component is
present. Filtering removes the neighbouring content that constrained the
full-band sample stream, so the derived band-limited waveform has a higher
sample peak than the original source waveform.

This is not the previous gain-offset issue. A gain-offset issue causes the
**full-spectrum** channel peak to be wrong. In the fixed `v0.3.2` data, full
spectrum peaks differ by channel and match each channel's stored
`original_peak_dbfs`. The remaining positives are filtered-band-only values.

This is also not evidence of an unwanted phase expansion in the octave bank. A
steady 62.5 Hz sine remains at `0 dB` after filtering, and the summed RMS power
remains closed for all test cases.

## Reporting Guidance

Keep small positive octave-band peak values when they come from filtered-band
signals; they are meaningful derived post-filter peaks.

Recommended wording for reports and documentation:

> Filtered octave-band peak levels are derived after octave-band filtering. They
> can exceed the source full-band sample peak because filtering changes waveform
> shape and can remove cancelling components. Treat them as post-filter band
> peaks, not source-file headroom.

For source headroom, use the full-spectrum channel peak. For band comparisons,
prefer RMS level and crest factor, with filtered peak interpreted as an
additional post-filter diagnostic.
