# LFE Band-Limited Analysis Proof

Status: **PASS**

This proof validates the production LFE deep-dive target-band behavior. It checks
that low-frequency content appears in the intended LFE target plots, crossover
content is split as expected, and clearly out-of-band screen-range content is
rejected by the LFE target set.

## Scope

- Production target frequencies: `8 Hz`, `16 Hz`, `31.25 Hz`, `62.5 Hz`, `125 Hz`, `250 Hz`
- Filter implementation: `src.octave_filter.OctaveBandFilter`
- Target mapping helper: `src.post.lfe_octave_time._get_octave_band_indices`
- Time-domain metrics: `FixedWindowTimeDomainCalculator`
- Window: `2 s`
- Step: `1 s`

## Results

- LFE centre-frequency capture: **PASS**
- Minimum target-centre captured power: `100.000000000%`
- Adjacent LFE crossover split: **PASS**
- Maximum adjacent crossover total-power error: `0.096584265%`
- Crossover tolerance allows `0.5%` finite-tone
  FFT leakage around irrational geometric crossover frequencies.
- Upper transition at `sqrt(250 * 500) Hz`: **PASS**
- Upper transition power retained in plotted LFE targets:
  `49.997403638%`
- 500 Hz and 1 kHz screen-range rejection: **PASS**
- Maximum out-of-LFE target power: `0.000000000%`
- 4 Hz residual exclusion from plotted LFE targets:
  **PASS**

## Interpretation

The plotted LFE deep-dive target set responds correctly to the intended LFE
octaves: `8 Hz`, `16 Hz`, `31.25 Hz`, `62.5 Hz`, `125 Hz`, and `250 Hz`.
Centre tones land in the expected target band, and geometric crossovers between
adjacent LFE targets split power equally.

The `250 Hz` plot is the top LFE target and represents the upper transition into
screen-range content. At the `250/500 Hz` crossover, half the power remains in
the plotted `250 Hz` target and half moves to the unplotted `500 Hz` octave. At
the `500 Hz` and `1 kHz` centres, the plotted LFE target set rejects the content.

The `4 Hz` residual octave is also excluded from the plotted LFE target set. That
keeps the LFE deep-dive plots focused on the configured visible target bands
rather than residual-band bookkeeping.

## Outputs

- `lfe_band_acceptance.csv`
- `lfe_band_metrics.csv`
- `lfe_target_power_by_case.png`
- `lfe_band_power_matrix.png`
- `source_material/*.wav`
