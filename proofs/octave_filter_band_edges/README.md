# Octave Filter Band Edges Proof

Filter-bank status: **PASS**

EIA-426B crest-factor check: **expansion observed**

This proof validates the production FFT octave filter bank used by Audio
Analyser. The primary question is whether octave centre frequencies and
crossover frequencies are assigned to the intended bands. That check passes.

The proof also records a separate EIA-426B-style compressed pink-noise
crest-factor check. That later check is intentionally diagnostic: it documents
that derived octave-band signals can expand crest factor even though the filter
bank itself has no phase rotation and can reconstruct the source with matched
power-complementary synthesis.

## Band Filtering Result

Status: **PASS**

- Centre-frequency tones land fully in the intended band.
- Geometric crossover tones split power equally between adjacent bands.
- Minimum centre-frequency primary power: `100.000000000%`
- Maximum crossover split error: `0.000000000%`
- Phase audit: **PASS**
- Maximum retained-bin phase shift: `0.000000000000 degrees`

## Method

- Sample rate: `48000 Hz`
- Filter implementation: production `src.octave_filter.OctaveBandFilter`
- Band centres:

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

The acceptance matrix checks every in-band centre frequency and every geometric
crossover between adjacent bands. Centre tones should land fully in the intended
band. Crossover tones should split power equally between the two adjacent bands.

The phase audit samples the production FFT weights over the audible range. The
weights are real and non-negative, so retained FFT bins keep their original
phase and only their magnitude changes.

## Band Filtering Interpretation

The octave bank places centre-frequency energy in the expected band and splits
geometric crossover energy exactly between adjacent bands. That confirms the
band-edge behaviour implied by the raised-cosine power-complementary design.

The phase check confirms that the production bank is magnitude-only in the FFT
domain. It does not introduce per-bin phase rotation, so any crest-factor change
seen in derived band signals is not caused by all-pass or minimum-phase
rotation.

## EIA-426B Crest-Factor Check

Status: **expansion observed**

The crest-factor stimulus is deterministic EIA-426B-style compressed pink noise:
pink spectrum, 20 Hz to 20 kHz band limit, then hard-limited to a
`6 dB` source crest factor. Active octave
bands were checked against a `0.25 dB` expansion
tolerance; inactive bands below `-45 dB` relative RMS are
reported but excluded from that acceptance gate.

- Full-spectrum compressed pink-noise crest factor: `6.000 dB`
- Active-band compressed pink-noise crest-factor check:
  **FAIL**
- Worst active-band crest expansion: `2000 Hz`,
  `+8.426 dB`

The compressed pink-noise check is intentionally separate from the existing
peak-expansion proof. The full-spectrum path remains at the generated
`6 dB` crest factor because column 0 of the
octave bank is the unfiltered source. The derived octave-band paths do show
crest-factor expansion on this compressed stimulus, so the current bank should
not be described as preserving compressed-noise crest factor per band.

## Reconstruction Check

- Direct time-domain band sum error:
  `-10.455 dB`
  relative to source RMS
- Power-complementary synthesis sum error:
  `-301.982 dB`
  relative to source RMS

The separated octave-band signals are not a direct perfect-reconstruction bank
if they are simply summed in the time domain. Direct summing applies
`sum(weight_band(f))`, which rises through crossover regions and changes the
source waveform. The bank is power-complementary instead: applying the matching
FFT weights as synthesis weights and then summing applies
`sum(weight_band(f) ** 2) = 1`, reconstructing the source to numerical
precision.

## Outputs

- `band_edge_acceptance.csv`
- `phase_response.csv`
- `compressed_pink_noise_crest_factor.csv`
- `reconstruction_analysis.csv`
- `band_edge_acceptance.png`
- `compressed_pink_noise_crest_factor.png`
- `reconstruction_error.png`
- `source_material/eia426b_style_compressed_pink_noise.wav`
