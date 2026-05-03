"""Tests for rendering plots from analysis result bundles."""

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from src.music_analyzer import MusicAnalyzer
from src.report_pdf import markdown_report_to_pdf
from src.render import _resolve_render_dpi, main as render_main
from src.results import generate_bundle_report, load_result_bundle
from src.results.bundle import write_channel_result_bundle
from src.results.render import (
    render_bundle_envelope_plots,
    render_bundle_group_outputs,
    render_bundle_histograms,
    render_bundle_spectrum_plots,
    render_bundle_time_plots,
)


class _DummyConfig:
    def __init__(self, values):
        self.values = values

    def get(self, key_path, default=None):
        return self.values.get(key_path, default)


def _write_test_bundle(
    tmp_path,
    include_envelopes=False,
    channel_index=0,
    channel_name="FL",
    center_frequency=10.0,
):
    sample_rate = 1000
    time = np.arange(sample_rate * 2) / sample_rate
    channel_data = 0.5 * np.sin(2 * np.pi * 10 * time)
    octave_bank = np.column_stack([channel_data, channel_data * 0.8])
    center_frequencies = [center_frequency]

    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    analysis_results = analyzer.analyze_octave_bands(octave_bank, center_frequencies)
    analysis_results["band_data"] = {
        "Full Spectrum": octave_bank[:, 0],
        f"{center_frequency:.3f}": octave_bank[:, 1],
    }
    time_analysis = {
        "time_points": np.array([1.0, 2.0]),
        "crest_factors": np.array([1.5, 1.6]),
        "crest_factors_db": np.array([3.52, 4.08]),
        "peak_levels": np.array([0.5, 0.5]),
        "rms_levels": np.array([0.25, 0.25]),
        "peak_levels_dbfs": np.array([-6.0, -6.0]),
        "rms_levels_dbfs": np.array([-12.0, -12.0]),
    }
    envelope_statistics = {}
    if include_envelopes:
        envelope_statistics = {
            "10.000": {
                "pattern_analysis": {
                    "patterns_detected": 1,
                    "pattern_1": {
                        "peak_times_seconds": [1.0],
                        "envelope_windows": [np.array([-18.0, -6.0, -12.0])],
                        "time_windows_ms": [np.array([-10.0, 0.0, 10.0])],
                    },
                },
                "worst_case_envelopes": [
                    {
                        "rank": 1,
                        "peak_value_db": -6.0,
                        "peak_time_seconds": 1.0,
                        "envelope_window": np.array([-18.0, -6.0, -12.0]),
                        "time_window_ms": np.array([-10.0, 0.0, 10.0]),
                        "decay_times": {"decay_6db_ms": 10.0},
                    }
                ],
            }
        }
    return write_channel_result_bundle(
        track_output_dir=tmp_path,
        track_metadata={
            "track_name": "render_test.wav",
            "track_path": "Tracks/render_test.wav",
            "content_type": "Test Signal",
            "channel_index": channel_index,
            "channel_name": channel_name,
            "total_channels": 1,
            "duration_seconds": 2.0,
            "sample_rate": sample_rate,
            "samples": len(channel_data),
            "original_peak": 1.0,
            "analysis_date": "2026-04-26T22:30:00",
        },
        analysis_results=analysis_results,
        time_analysis=time_analysis,
        chunk_octave_analysis=None,
        envelope_statistics=envelope_statistics,
        octave_bank=octave_bank,
        center_frequencies=center_frequencies,
        channel_data=channel_data,
        plotting_config={
            "histogram_bins": 11,
            "histogram_range": [-1.0, 1.0],
            "log_histogram_noise_floor_db": -60.0,
            "log_histogram_max_db": 0.0,
            "log_histogram_max_bin_size_db": 6.0,
            "octave_spectrum_xlim": [7.0, 20000.0],
            "octave_spectrum_ylim": [-60.0, 3.0],
            "crest_factor_xlim": [7.0, 20000.0],
            "crest_factor_ylim_min": 0.0,
            "crest_factor_ylim_max": 30.0,
        },
        envelope_config={},
        analysis_config={"peak_hold_tau_seconds": 1.0},
    )


def test_load_result_bundle_reads_channel_artifacts(tmp_path):
    """Reader loads manifest metadata and channel tables."""
    bundle_dir = _write_test_bundle(tmp_path)
    bundle = load_result_bundle(bundle_dir)
    channel = bundle.channels()[0]

    assert bundle.schema_version == 1
    assert channel.channel_id == "channel_01"
    assert channel.read_json("metadata")["channel_name"] == "FL"

    histogram = channel.read_table("histogram_linear")
    assert isinstance(histogram, pd.DataFrame)
    assert {"frequency_label", "bin_center", "bin_density"}.issubset(histogram.columns)


def test_render_bundle_histograms_writes_pngs(tmp_path):
    """Histogram rendering uses only pre-binned bundle data."""
    bundle_dir = _write_test_bundle(tmp_path)
    output_dir = tmp_path / "rendered"

    output_paths = render_bundle_histograms(
        bundle=load_result_bundle(bundle_dir),
        output_dir=output_dir,
        dpi=80,
    )

    assert len(output_paths) == 2
    assert (output_dir / "channel_01" / "histograms.png").exists()
    assert (output_dir / "channel_01" / "histograms_log_db.png").exists()


def test_render_bundle_spectrum_plots_writes_pngs(tmp_path):
    """Spectrum rendering uses only stored bundle tables."""
    bundle_dir = _write_test_bundle(tmp_path)
    output_dir = tmp_path / "rendered_spectrum"

    output_paths = render_bundle_spectrum_plots(
        bundle=load_result_bundle(bundle_dir),
        output_dir=output_dir,
        dpi=80,
    )

    assert len(output_paths) == 2
    assert (output_dir / "channel_01" / "octave_spectrum.png").exists()
    assert (output_dir / "channel_01" / "crest_factor.png").exists()


def test_render_bundle_time_plots_writes_pngs(tmp_path):
    """Time rendering uses only stored bundle tables."""
    bundle_dir = _write_test_bundle(tmp_path)
    output_dir = tmp_path / "rendered_time"

    output_paths = render_bundle_time_plots(
        bundle=load_result_bundle(bundle_dir),
        output_dir=output_dir,
        dpi=80,
    )

    assert len(output_paths) == 2
    assert (output_dir / "channel_01" / "crest_factor_time.png").exists()
    assert (output_dir / "channel_01" / "octave_crest_factor_time.png").exists()


def test_render_bundle_envelope_plots_writes_pngs(tmp_path):
    """Envelope rendering uses stored bundle window data."""
    bundle_dir = _write_test_bundle(tmp_path, include_envelopes=True)
    output_dir = tmp_path / "rendered_envelopes"

    output_paths = render_bundle_envelope_plots(
        bundle=load_result_bundle(bundle_dir),
        output_dir=output_dir,
        dpi=80,
    )

    assert len(output_paths) == 2
    assert (
        output_dir / "channel_01" / "pattern_envelopes" / "pattern_envelopes_10_000.png"
    ).exists()
    assert (
        output_dir
        / "channel_01"
        / "independent_envelopes"
        / "independent_envelopes_10_000.png"
    ).exists()


def test_render_bundle_group_outputs_writes_plots_and_manifest(tmp_path):
    """Group rendering uses bundle channel tables."""
    bundle_dir = _write_test_bundle(tmp_path)
    output_dir = tmp_path / "rendered_groups"

    output_paths = render_bundle_group_outputs(
        bundle=load_result_bundle(bundle_dir),
        output_dir=output_dir,
        dpi=80,
    )

    assert len(output_paths) == 3
    assert (output_dir / "Screen" / "crest_factor_time.png").exists()
    assert (output_dir / "Screen" / "octave_spectrum.png").exists()
    assert (output_dir / "worst_channels_manifest.csv").exists()


def test_bundle_render_and_report_write_cinema_deep_dives(tmp_path):
    """Cinema bundles get legacy-equivalent group deep-dive outputs."""
    _write_test_bundle(
        tmp_path,
        channel_index=0,
        channel_name="FL",
        center_frequency=8.0,
    )
    _write_test_bundle(
        tmp_path,
        channel_index=3,
        channel_name="LFE",
        center_frequency=8.0,
    )
    bundle_dir = _write_test_bundle(
        tmp_path,
        channel_index=6,
        channel_name="SL",
        center_frequency=8.0,
    )
    bundle = load_result_bundle(bundle_dir)
    output_dir = tmp_path / "rendered_cinema"

    render_bundle_group_outputs(bundle=bundle, output_dir=output_dir, dpi=80)
    report_path = generate_bundle_report(bundle=bundle, rendered_output_dir=output_dir)

    assert (output_dir / "Screen" / "screen_8_0Hz.png").exists()
    assert (output_dir / "LFE" / "lfe_full_channel.png").exists()
    assert (output_dir / "LFE" / "lfe_octave_time_8_0Hz.png").exists()
    assert (output_dir / "Surround+Height" / "surround_height_8_0Hz.png").exists()
    assert (output_dir / "lfe_deep_dive.md").exists()
    assert (output_dir / "screen_deep_dive.md").exists()
    assert (output_dir / "surround_height_deep_dive.md").exists()
    assert (output_dir / "lfe_deep_dive.pdf").exists()
    assert (output_dir / "screen_deep_dive.pdf").exists()
    assert (output_dir / "surround_height_deep_dive.pdf").exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "LFE Deep Dive" in report_text
    assert "Screen Channel Deep Dive" in report_text
    assert "Surround+Height Channel Deep Dive" in report_text
    lfe_text = (output_dir / "lfe_deep_dive.md").read_text(encoding="utf-8")
    screen_text = (output_dir / "screen_deep_dive.md").read_text(encoding="utf-8")
    surround_text = (output_dir / "surround_height_deep_dive.md").read_text(
        encoding="utf-8"
    )
    assert "## Contents" in lfe_text
    assert "- [Full Channel](#full-channel)" in lfe_text
    assert "## Full Channel" in lfe_text
    assert "- [8 Hz](#8-hz)" in lfe_text
    assert "## 8 Hz" in lfe_text
    assert "## Contents" in screen_text
    assert "- [8 Hz](#8-hz)" in screen_text
    assert "## 8 Hz" in screen_text
    assert "## Contents" in surround_text
    assert "- [8 Hz](#8-hz)" in surround_text
    assert "## 8 Hz" in surround_text


def test_generate_bundle_report_writes_markdown(tmp_path):
    """Reports can be generated from bundle tables without legacy CSVs."""
    bundle_dir = _write_test_bundle(tmp_path)
    bundle = load_result_bundle(bundle_dir)
    output_dir = tmp_path / "rendered_report"
    render_bundle_group_outputs(bundle=bundle, output_dir=output_dir, dpi=80)
    render_bundle_spectrum_plots(bundle=bundle, output_dir=output_dir, dpi=80)
    render_bundle_time_plots(bundle=bundle, output_dir=output_dir, dpi=80)
    render_bundle_histograms(bundle=bundle, output_dir=output_dir, dpi=80)

    report_path = generate_bundle_report(
        bundle=bundle,
        rendered_output_dir=output_dir,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert report_path == output_dir / "analysis.md"
    assert "# render_test - Audio Signal Analysis" in report_text
    assert "Source bundle: `render_test.aaresults`" in report_text
    assert "## Contents" in report_text
    assert "- [Group Overview Plots](#group-overview-plots)" in report_text
    assert "Crest Factor Analysis" in report_text
    assert "## Group Overview Plots" in report_text
    assert "### Screen" in report_text
    assert "### Screen Channels" in report_text
    assert "#### FL" in report_text
    assert "##### Octave Spectrum" in report_text
    assert 'alt="Crest Factor Over Time - Screen"' in report_text
    assert "T3, T6, T9, and T12 are recovery-time measurements" in report_text
    pdf_path = output_dir / "analysis.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_markdown_report_to_pdf_writes_pdf(tmp_path):
    """Markdown reports can be exported as portable PDFs."""
    markdown_path = tmp_path / "analysis.md"
    markdown_path.write_text(
        "# Test Report\n\n"
        "This is a **PDF smoke test**.\n\n"
        "| Metric | Value |\n"
        "| --- | --- |\n"
        "| Crest factor | 12 dB |\n",
        encoding="utf-8",
    )

    pdf_path = markdown_report_to_pdf(markdown_path)

    assert pdf_path == tmp_path / "analysis.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_render_dpi_uses_override_then_config():
    """Render DPI is configurable but still overridable from the CLI."""
    config = _DummyConfig({"plotting.render_dpi": 150, "plotting.dpi": 300})

    assert _resolve_render_dpi(config, None) == 150
    assert _resolve_render_dpi(config, 80) == 80


def test_render_cli_renders_histograms_from_bundle(tmp_path):
    """Render CLI scans bundles and writes plots."""
    bundle_dir = _write_test_bundle(tmp_path)
    output_dir = tmp_path / "cli_rendered"

    result = CliRunner().invoke(
        render_main,
        [
            "--results",
            str(bundle_dir),
            "--output-dir",
            str(output_dir),
            "--dpi",
            "80",
            "--reports",
        ],
    )

    if result.exception:
        raise result.exception

    assert result.exit_code == 0
    assert (output_dir / "render_test" / "channel_01" / "octave_spectrum.png").exists()
    assert (output_dir / "render_test" / "channel_01" / "crest_factor.png").exists()
    assert (output_dir / "render_test" / "channel_01" / "histograms.png").exists()
    assert (
        output_dir / "render_test" / "channel_01" / "histograms_log_db.png"
    ).exists()
    assert (
        output_dir / "render_test" / "channel_01" / "crest_factor_time.png"
    ).exists()
    assert (
        output_dir / "render_test" / "channel_01" / "octave_crest_factor_time.png"
    ).exists()
    assert (output_dir / "render_test" / "analysis.md").exists()
    assert (output_dir / "render_test" / "analysis.pdf").exists()

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["track"]["track_name"] == "render_test.wav"
