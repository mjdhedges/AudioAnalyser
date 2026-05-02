# Channel Grouping Proof

Status: **PASS**

This proof validates mono, stereo, and multichannel grouping rules used by
report and post-processing outputs. It creates synthetic analysis-output folders
with valid minimal CSV sections, then runs the production grouping helpers.

## Scope

- Channel naming: `src.channel_mapping`
- Time-series group plots: `src.post.group_crest_factor_time._group_channels`
- Octave-spectrum group plots: `src.post.group_octave_spectrum._group_channels`
- Markdown report grouping: `src.report_generator._determine_channel_groups`
- Bundle renderer classification: `src.results.render._classify_channel_name`

## Results

- Filesystem grouping helpers: **PASS**
- Channel-name classification: **PASS**
- Classification failures: None

| Case | Helper | Observed channel count | Pass |
| --- | --- | ---: | --- |
| mono_root | group_crest_factor_time | 0 | True |
| mono_root | group_octave_spectrum | 0 | True |
| mono_root | report_generator | 1 | True |
| stereo | group_crest_factor_time | 2 | True |
| stereo | group_octave_spectrum | 2 | True |
| stereo | report_generator | 2 | True |
| ffmpeg_5_1 | group_crest_factor_time | 6 | True |
| ffmpeg_5_1 | group_octave_spectrum | 6 | True |
| ffmpeg_5_1 | report_generator | 6 | True |
| ffmpeg_5_1_2 | group_crest_factor_time | 8 | True |
| ffmpeg_5_1_2 | group_octave_spectrum | 8 | True |
| ffmpeg_5_1_2 | report_generator | 8 | True |
| ffmpeg_7_1_4 | group_crest_factor_time | 12 | True |
| ffmpeg_7_1_4 | group_octave_spectrum | 12 | True |
| ffmpeg_7_1_4 | report_generator | 12 | True |

## Interpretation

Mono report grouping is represented by a root-level `analysis_results.csv` and
maps to the `Mono` report group. Stereo channel folders are intentionally not
cinema-screen channels; they group under `All Channels`.

For cinema layouts, `FL`, `FR`, and `FC` group as `Screen`; `LFE` groups as
`LFE`; surround, back, and top/height folders group as `Surround+Height` in the
post-processing group plots.

The markdown report fallback classifies channel folders by channel-name tokens,
not by loose substring matching. This matters for height channels such as
`TFL/TFR`: they contain `FL/FR` as substrings but must group with
`Surround+Height`, not `Screen`.

## Outputs

- `grouping_results.csv`
- `channel_classification.csv`
- `channel_mapping_results.csv`
- `grouped_channel_counts.png`
- `synthetic_tracks/`
