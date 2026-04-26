# Octave Band Energy Closure Proof

This proof validates an FFT-based, power-complementary octave analysis bank.
The goal is for the total linear power of all octave bands to equal the original
full-band signal power.

This is an explicit requirement for this analyser. The octave bands are used for
RMS, energy distribution, and crest-factor analysis, so the filter bank must be
flat in summed power. It does not need to be flat in simple amplitude sum.

## Method

- Source material: generated deterministic WAV files in `source_material/`
- Sample rate: 48000 Hz
- Analysis method: one full-signal FFT, octave-spaced raised-cosine weights, and
  inverse FFT per band
- Power complementarity rule: `sum(weight_band(f) ** 2) = 1.0` at every FFT bin
- Large-block mode: the same method is also tested in non-overlapping
  30-second FFT blocks for lower peak RAM use
- Time-series requirement: every band must return a finite time-domain signal
  with the same sample count as the input so downstream peak, RMS, crest-factor,
  histogram, and envelope analysis can operate on it.
- Octave bands:

- `4 Hz and below`
- `8 Hz`
- `16 Hz`
- `31.25 Hz`
- `62.5 Hz`
- `125 Hz`
- `250 Hz`
- `500 Hz`
- `1000 Hz`
- `2000 Hz`
- `4000 Hz`
- `8000 Hz`
- `16000 Hz`
- `>22627 Hz`

The low residual band captures the virtual 4 Hz octave and all energy below it.
The high residual band captures energy above the 16 kHz octave region up to
Nyquist.

## Results

Status: **PASS**

Mean absolute closure error: **0.000000000 dB**

Worst closure case: **mid_1khz_sine**, -0.000000000 dB
(100.000000% captured power).

Summed response:

- Amplitude sum minimum: 0.000 dB
- Amplitude sum maximum: 3.010 dB
- Power sum minimum: -0.000000000 dB
- Power sum maximum: 0.000000000 dB
- Power sum P95 absolute deviation: 0.000000000 dB

| Signal | Closure error (dB) | Captured power | Waveform sum error (dB) | Loudest band |
| --- | ---: | ---: | ---: | ---: |
| pink_noise | -0.000000000 | 100.000000% | -12.913 | 4 Hz and below |
| white_noise | 0.000000000 | 100.000000% | -10.200 | 16000 Hz |
| octave_multitone | -0.000000000 | 100.000000% | -162.136 | 1000 Hz |
| log_sweep | -0.000000000 | 100.000000% | -10.424 | 8 Hz |
| lfe_31hz_sine | 0.000000000 | 100.000000% | -161.754 | 31.25 Hz |
| mid_1khz_sine | -0.000000000 | 100.000000% | -161.574 | 1000 Hz |

Time-series validation: **PASS**

- Total 1-second band crest-factor windows evaluated: 5040
- Maximum band crest factor observed in validation: 45.034 dB

| Signal | Samples | Bands | Same length | All finite | Crest windows | Max band CF (dB) |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| pink_noise | 2880000 | 14 | True | True | 840 | 14.908 |
| white_noise | 2880000 | 14 | True | True | 840 | 14.809 |
| octave_multitone | 2880000 | 14 | True | True | 840 | 19.065 |
| log_sweep | 2880000 | 14 | True | True | 840 | 45.034 |
| lfe_31hz_sine | 2880000 | 14 | True | True | 840 | 18.975 |
| mid_1khz_sine | 2880000 | 14 | True | True | 840 | 32.952 |

Large-block FFT validation: **PASS**

- Block duration tested: 30 seconds
- Maximum band RMS difference vs full-file FFT: 0.035623 dB
- Band RMS comparison ignores bands below -120 dB relative to full-band RMS,
  avoiding meaningless ratios against numerical residuals in inactive bands

| Signal | Block length (s) | Blocks | Block closure error (dB) | Max band RMS delta (dB) | Mean band RMS delta (dB) | Same shape | All finite |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| pink_noise | 30 | 2 | -0.000000000 | 0.035623 | 0.007097 | True | True |
| white_noise | 30 | 2 | 0.000000000 | 0.035349 | 0.004714 | True | True |
| octave_multitone | 30 | 2 | 0.000000000 | 0.003956 | 0.000647 | True | True |
| log_sweep | 30 | 2 | -0.000000000 | 0.001869 | 0.000207 | True | True |
| lfe_31hz_sine | 30 | 2 | -0.000000000 | 0.003975 | 0.003975 | True | True |
| mid_1khz_sine | 30 | 2 | -0.000000000 | 0.000000 | 0.000000 | True | True |

## Interpretation

Octave-band RMS values must be summed as linear power, not by adding dB values:

```text
combined_rms = sqrt(sum(band_rms ** 2))
combined_dbfs = 20 * log10(combined_rms)
```

The amplitude-sum plot is included for transparency, but it is not the criterion
for energy closure. Adjacent raised-cosine bands overlap in amplitude, so the
simple amplitude sum rises in transition regions. The acoustically relevant
energy check is the power sum, and the power sum is flat by construction.

That distinction matters for our processing:

- RMS is an energy measure, so octave-band RMS totals must be power-summed.
- Peak levels are measured per band or from the original full-band waveform; they
  are not summed across octave bands.
- Crest factor is calculated from each signal path's own peak and RMS values.
  Band crest factors are not summed into a full-spectrum crest factor.

Therefore, the required filter-bank behaviour is:

```text
sum(weight_band(f) ** 2) = 1.0
```

not:

```text
sum(weight_band(f)) = 1.0
```

This prototype is also computationally attractive for offline analysis: one FFT
is calculated for the signal and reused for every band. Time-domain band signals
are still available through inverse FFT, so downstream RMS, peak, crest-factor,
histogram, and envelope analysis can continue to operate on band signals.

The large-block mode gives a lower-memory route for longer files. Blocks are
still tens of seconds long, which gives the 4 Hz and sub-8 Hz content enough
context for offline analysis. The proof checks that block processing remains
energy-closed and returns the same shape of time-series band output.

## Outputs

- `band_frequency_responses.png`: frequency response of every FFT band
- `band_amplitude_sum.png`: simple amplitude sum of the band weights
- `band_power_sum.png`: power sum of the band weights
- `fft_power_complementary_weights.csv`: numeric weight and sum data
- `energy_closure_results.csv`: numeric closure results
- `time_series_validation.csv`: proof that band outputs are valid time-series
  signals for downstream metrics
- `block_fft_comparison.csv`: comparison between full-file FFT and
  30-second block FFT processing
- `energy_closure_error.png`: closure error by source signal
- `captured_power_percent.png`: captured power by source signal
- `source_material/*.wav`: generated deterministic source signals
