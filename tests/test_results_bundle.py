"""Tests for portable analysis result bundles."""

import json

import numpy as np
import pandas as pd

from src.music_analyzer import MusicAnalyzer
from src.results import write_channel_result_bundle


def test_write_channel_result_bundle_contains_plot_replay_data(tmp_path):
    """Bundle writer stores the derived data needed to recreate current graphs."""
    sample_rate = 1000
    duration_seconds = 2.0
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    channel_data = 0.5 * np.sin(2 * np.pi * 10 * time)
    octave_bank = np.column_stack(
        [
            channel_data,
            channel_data * 0.8,
            channel_data * 0.2,
        ]
    )
    center_frequencies = [10.0, 20.0]

    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    analysis_results = analyzer.analyze_octave_bands(
        octave_bank,
        center_frequencies,
    )
    analysis_results["band_data"] = {
        "Full Spectrum": octave_bank[:, 0],
        "10.000": octave_bank[:, 1],
        "20.000": octave_bank[:, 2],
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
    envelope_statistics = {
        "10.000": {
            "pattern_analysis": {
                "patterns_detected": 1,
                "pattern_1": {
                    "num_repetitions": 2,
                    "peak_times_seconds": [0.5, 1.0, 1.5],
                    "envelope_windows": [
                        np.array([-12.0, -8.0, -9.0]),
                        np.array([-12.0, -6.0, -9.0]),
                        np.array([-12.0, -7.0, -9.0]),
                    ],
                    "time_windows_ms": [
                        np.array([-1.0, 0.0, 1.0]),
                        np.array([-1.0, 0.0, 1.0]),
                        np.array([-1.0, 0.0, 1.0]),
                    ],
                },
            },
            "worst_case_envelopes": [
                {
                    "rank": 1,
                    "peak_value_db": -6.0,
                    "peak_time_seconds": 1.0,
                    "envelope_window": np.array([-12.0, -6.0, -9.0]),
                    "time_window_ms": np.array([-1.0, 0.0, 1.0]),
                    "decay_times": {"decay_6db_ms": 20.0},
                },
                {
                    "rank": 2,
                    "peak_value_db": -7.0,
                    "peak_time_seconds": 1.5,
                    "envelope_window": np.array([-12.0, -7.0, -9.0]),
                    "time_window_ms": np.array([-1.0, 0.0, 1.0]),
                    "decay_times": {"decay_6db_ms": 10.0},
                },
            ],
        }
    }
    track_metadata = {
        "track_name": "test_track.wav",
        "track_path": "Tracks/test_track.wav",
        "content_type": "Test Signal",
        "channel_index": 0,
        "channel_name": "FL",
        "total_channels": 1,
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "samples": len(channel_data),
        "original_peak": 1.0,
        "analysis_date": "2026-04-26T22:00:00",
    }

    bundle_dir = write_channel_result_bundle(
        track_output_dir=tmp_path,
        track_metadata=track_metadata,
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
        },
        envelope_config={
            "envelope_plots_num_pattern_envelopes": 1,
            "envelope_plots_num_independent_envelopes": 1,
        },
        analysis_config={"peak_hold_tau_seconds": 1.0},
        advanced_statistics={"true_peak_to_rms_ratio_db": 12.0},
    )

    channel_dir = bundle_dir / "channels" / "channel_01"
    manifest = json.loads((bundle_dir / "manifest.json").read_text())

    assert manifest["schema_version"] == 1
    assert manifest["channels"][0]["channel_name"] == "FL"
    assert (channel_dir / "octave_band_analysis.csv").exists()
    assert (channel_dir / "time_domain_analysis.csv").exists()
    assert (channel_dir / "histogram_linear.csv").exists()
    assert (channel_dir / "histogram_log_db.csv").exists()
    assert (channel_dir / "octave_time_metrics.csv").exists()
    assert (channel_dir / "advanced_statistics.csv").exists()
    assert (channel_dir / "time_domain_summary.csv").exists()
    assert (channel_dir / "envelope_statistics.csv").exists()
    assert (channel_dir / "envelope_pattern_analysis.csv").exists()
    assert (channel_dir / "sustained_peaks_summary.csv").exists()
    assert (channel_dir / "sustained_peaks_events.csv").exists()
    assert (channel_dir / "envelope_plot_data.json").exists()

    octave_band = pd.read_csv(channel_dir / "octave_band_analysis.csv")
    assert "is_valid_crest_factor" in octave_band.columns
    assert "crest_factor_method" in octave_band.columns
    assert set(octave_band["crest_factor_method"]) == {"whole_interval_peak_rms"}

    octave_time = pd.read_csv(channel_dir / "octave_time_metrics.csv")
    assert {
        "frequency_hz",
        "time_seconds",
        "crest_factor_db",
        "peak_dbfs",
        "rms_dbfs",
        "is_valid_crest_factor",
        "crest_factor_window_seconds",
        "crest_factor_step_seconds",
        "crest_factor_method",
    }.issubset(octave_time.columns)
    assert set(octave_time["frequency_hz"]) == {10.0, 20.0}
    assert set(octave_time["crest_factor_method"]) == {"fixed_window_peak_rms"}

    advanced_statistics = pd.read_csv(channel_dir / "advanced_statistics.csv")
    assert "true_peak_to_rms_ratio_db" in set(advanced_statistics["parameter"])

    envelope_summary = pd.read_csv(channel_dir / "envelope_statistics.csv")
    assert envelope_summary.loc[0, "peak_value_db"] == -6.0

    envelope_data = json.loads((channel_dir / "envelope_plot_data.json").read_text())
    assert envelope_data["10.000"]["pattern_analysis"]["pattern_1"][
        "envelope_windows"
    ] == [[-12.0, -6.0, -9.0]]
    assert envelope_data["10.000"]["pattern_analysis"]["pattern_1"][
        "peak_times_seconds"
    ] == [1.0]
    assert envelope_data["10.000"]["pattern_analysis"]["pattern_1"][
        "time_window_axes"
    ] == [{"start_ms": -1.0, "step_ms": 1.0}]
    assert (
        "time_windows_ms"
        not in envelope_data["10.000"]["pattern_analysis"]["pattern_1"]
    )
    assert len(envelope_data["10.000"]["worst_case_envelopes"]) == 1
    assert envelope_data["10.000"]["worst_case_envelopes"][0]["envelope_window"] == [
        -12.0,
        -6.0,
        -9.0,
    ]
    assert envelope_data["10.000"]["worst_case_envelopes"][0]["time_window_axis"] == {
        "start_ms": -1.0,
        "step_ms": 1.0,
    }
    assert "time_window_ms" not in envelope_data["10.000"]["worst_case_envelopes"][0]
