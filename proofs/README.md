# Audio Analyser Proofs

This folder contains reproducible evidence for analysis methods that are easy to
misunderstand, difficult to validate by inspection, or important enough to need a
recorded technical justification.

Each proof should be self-contained: include the script, generated or licensed
source material, numeric outputs, plots, and a README explaining the result.

## Proof Backlog

### 1. Octave Band Energy Closure

Prove how octave-band energy should be summed and how closely the combined band
energy agrees with full-band RMS/energy. This should show that band RMS values
must be converted to linear power before summing; dB values must not be added
directly.

Status: in progress in `octave_band_energy_closure/`.

### 2. Octave Filter Band Edges

Verify that octave filters place energy into the intended centre bands and that
crossover behaviour matches the configured filter-bank design. Useful source
material includes sine sweeps, stepped sines, and tones placed at centre
frequencies and crossover frequencies.

Status: planned.

### 3. Crest Factor Short-Term Method

Validate the short-term crest-factor method: IEC Slow RMS tau, peak-hold tau,
sampling interval, and practical block/time scale. The existing
`peak_hold_tau/` proof is the first version of this evidence.

Status: started in `peak_hold_tau/`.

### 4. LFE Band-Limited Analysis

Prove that LFE and low-frequency deep-dive metrics respond correctly to in-band
low-frequency content and reject out-of-band content.

Status: planned.

### 5. Mono, Stereo, and Multichannel Grouping

Prove channel grouping rules for mono, stereo, screen, LFE, surround, and height
content. This should confirm that group plots and reports do not silently drop,
duplicate, or mislabel channels.

Status: planned.

### 6. Normalization and dBFS Reference

Prove that peak, RMS, and dBFS values move correctly under gain changes, while
crest factor remains gain-invariant.

Status: planned.
