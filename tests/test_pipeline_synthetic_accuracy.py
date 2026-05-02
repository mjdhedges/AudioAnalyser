"""Synthetic end-to-end data accuracy checks for the audio pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audio_processor import AudioProcessor
from src.channel_mapping import get_channel_folder_name, get_channel_name
from src.music_analyzer import MusicAnalyzer
from src.octave_filter import OctaveBandFilter
from src.results import load_result_bundle, write_channel_result_bundle


SAMPLE_RATE = 8000
DURATION_SECONDS = 4.0
CENTER_FREQUENCIES = [
    8.0,
    16.0,
    31.25,
    62.5,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
]
ANALYSIS_CONFIG = {
    "crest_factor_window_seconds": 1.0,
    "crest_factor_step_seconds": 1.0,
    "crest_factor_rms_floor_dbfs": -120.0,
}
PLOTTING_CONFIG = {
    "histogram_bins": 33,
    "histogram_range": [-1.0, 1.0],
    "log_histogram_noise_floor_db": -80.0,
    "log_histogram_max_db": 0.0,
    "log_histogram_max_bin_size_db": 6.0,
}


def _sine(frequency_hz: float, peak: float) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * DURATION_SECONDS)) / SAMPLE_RATE
    return (peak * np.sin(2.0 * np.pi * frequency_hz * time)).astype(np.float32)


def _square(frequency_hz: float, peak: float) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * DURATION_SECONDS)) / SAMPLE_RATE
    wave = np.sign(np.sin(2.0 * np.pi * frequency_hz * time))
    wave[wave == 0.0] = 1.0
    return (peak * wave).astype(np.float32)


def _windowed_impulses(peak: float) -> np.ndarray:
    signal = np.zeros(int(SAMPLE_RATE * DURATION_SECONDS), dtype=np.float32)
    for window_idx in range(int(DURATION_SECONDS)):
        signal[window_idx * SAMPLE_RATE + 100] = peak
    return signal


def _build_5_1_fixture() -> tuple[np.ndarray, list[dict[str, object]]]:
    """Build a 5.1 fixture with unique channel tones and gains."""
    specs = [
        {"index": 0, "name": "FL", "frequency": 250.0, "peak": 0.50},
        {"index": 1, "name": "FR", "frequency": 500.0, "peak": 0.40},
        {"index": 2, "name": "FC", "frequency": 1000.0, "peak": 0.30},
        {"index": 3, "name": "LFE", "frequency": 62.5, "peak": 0.80},
        {"index": 4, "name": "SBL", "frequency": 125.0, "peak": 0.20},
        {"index": 5, "name": "SBR", "frequency": 2000.0, "peak": 0.10},
    ]
    channels = [_sine(float(spec["frequency"]), float(spec["peak"])) for spec in specs]
    return np.column_stack(channels).astype(np.float32), specs


def _read_channel_tables(
    bundle_dir: Path,
    channel_index: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    channel_dir = bundle_dir / "channels" / f"channel_{channel_index + 1:02d}"
    metadata = json.loads((channel_dir / "metadata.json").read_text(encoding="utf-8"))
    return (
        metadata,
        pd.read_csv(channel_dir / "octave_band_analysis.csv"),
        pd.read_csv(channel_dir / "time_domain_analysis.csv"),
        pd.read_csv(channel_dir / "octave_time_metrics.csv"),
    )


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-20)))


def _median_window_rms_dbfs(signal: np.ndarray) -> float:
    window_samples = SAMPLE_RATE
    windows = signal.reshape(-1, window_samples)
    rms_values = np.sqrt(np.mean(np.square(windows), axis=1))
    return _db(float(np.median(rms_values)))


def _analyze_channel(
    *,
    track_output_dir: Path,
    track_name: str,
    channel_data: np.ndarray,
    channel_index: int,
    total_channels: int,
    channel_layout: str,
) -> Path:
    """Run the production analysis/bundle path for one decoded channel."""
    original_peak = float(np.max(np.abs(channel_data)))
    normalized = AudioProcessor().normalize_audio(channel_data)

    octave_filter = OctaveBandFilter(
        sample_rate=SAMPLE_RATE,
        processing_mode="full_file",
        include_low_residual_band=True,
        include_high_residual_band=True,
    )
    center_frequencies = octave_filter.get_band_center_frequencies(CENTER_FREQUENCIES)
    octave_bank = octave_filter.create_octave_bank(normalized, CENTER_FREQUENCIES)

    analyzer = MusicAnalyzer(
        sample_rate=SAMPLE_RATE,
        original_peak=original_peak,
        time_domain_crest_factor_mode="fixed_window",
        analysis_config=ANALYSIS_CONFIG,
    )
    comprehensive = analyzer.analyze_comprehensive(
        normalized,
        octave_bank,
        center_frequencies,
        chunk_duration=1.0,
    )
    analysis_results = comprehensive["main_analysis"]
    analysis_results["band_data"] = {
        "Full Spectrum": octave_bank[:, 0],
        **{
            f"{freq:.3f}": octave_bank[:, idx + 1]
            for idx, freq in enumerate(center_frequencies)
        },
    }

    channel_name = get_channel_name(channel_index, total_channels, channel_layout)
    track_metadata = {
        "track_name": track_name,
        "track_path": f"synthetic/{track_name}",
        "content_type": "Test Signal",
        "channel_index": channel_index,
        "channel_name": channel_name,
        "channel_folder_name": get_channel_folder_name(
            channel_index,
            total_channels,
            channel_layout,
        ),
        "total_channels": total_channels,
        "duration_seconds": DURATION_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "samples": int(channel_data.size),
        "original_peak": original_peak,
        "analysis_date": "2026-05-02T00:00:00",
    }
    return write_channel_result_bundle(
        track_output_dir=track_output_dir,
        track_metadata=track_metadata,
        analysis_results=analysis_results,
        time_analysis=comprehensive["time_analysis"],
        chunk_octave_analysis=comprehensive["chunk_octave_analysis"],
        envelope_statistics=None,
        octave_bank=octave_bank,
        center_frequencies=center_frequencies,
        channel_data=normalized,
        plotting_config=PLOTTING_CONFIG,
        envelope_config={},
        analysis_config=ANALYSIS_CONFIG,
        advanced_statistics={},
    )


def test_synthetic_5_1_pipeline_exports_correct_channel_metrics(tmp_path: Path) -> None:
    """Known 5.1 tones survive channel extraction, normalization, and export."""
    audio_data, specs = _build_5_1_fixture()
    processor = AudioProcessor()
    channels = processor.extract_channels(audio_data)

    assert len(channels) == 6
    assert [idx for _channel, idx in channels] == list(range(6))
    assert get_channel_folder_name(3, 6, "5.1") == "Channel 4 LFE"

    bundle_dir: Path | None = None
    for channel_data, channel_index in channels:
        bundle_dir = _analyze_channel(
            track_output_dir=tmp_path,
            track_name="synthetic_5_1.wav",
            channel_data=channel_data,
            channel_index=channel_index,
            total_channels=6,
            channel_layout="5.1",
        )

    assert bundle_dir is not None
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [channel["channel_name"] for channel in manifest["channels"]] == [
        "FL",
        "FR",
        "FC",
        "LFE",
        "SBL",
        "SBR",
    ]

    bundle = load_result_bundle(bundle_dir)
    assert [channel.channel_name for channel in bundle.channels()] == [
        "FL",
        "FR",
        "FC",
        "LFE",
        "SBL",
        "SBR",
    ]
    lfe_channel = bundle.get_channel("channel_04")
    assert lfe_channel.read_json("metadata")["channel_name"] == "LFE"
    assert not lfe_channel.read_table("octave_band_analysis").empty

    for spec in specs:
        metadata, octave_rows, time_rows, octave_time_rows = _read_channel_tables(
            bundle_dir,
            int(spec["index"]),
        )

        assert metadata["channel_name"] == spec["name"]
        assert metadata["original_peak"] == np.max(
            np.abs(audio_data[:, int(spec["index"])])
        )

        full = octave_rows[octave_rows["frequency_hz"] == "Full Spectrum"].iloc[0]
        expected_peak_dbfs = _db(float(spec["peak"]))
        expected_rms_dbfs = _db(float(spec["peak"]) / np.sqrt(2.0))
        assert float(full["max_amplitude"]) == pytest_approx(1.0)
        assert float(full["rms"]) == pytest_approx(1.0 / np.sqrt(2.0), abs=2e-6)
        assert float(full["max_amplitude_db"]) == pytest_approx(
            expected_peak_dbfs,
            abs=2e-5,
        )
        assert float(full["rms_db"]) == pytest_approx(expected_rms_dbfs, abs=2e-5)
        assert float(full["crest_factor_db"]) == pytest_approx(
            _db(np.sqrt(2.0)),
            abs=2e-5,
        )

        valid_time = time_rows[time_rows["is_valid_crest_factor"]]
        assert set(valid_time["peak_level_dbfs"].round(4)) == {
            round(expected_peak_dbfs, 4)
        }
        assert set(valid_time["rms_level_dbfs"].round(4)) == {
            round(expected_rms_dbfs, 4)
        }

        band_rows = octave_rows[octave_rows["frequency_hz"] != "Full Spectrum"].copy()
        band_rows["frequency_numeric"] = band_rows["frequency_hz"].astype(float)
        dominant = band_rows.loc[band_rows["rms"].idxmax()]
        assert float(dominant["frequency_numeric"]) == pytest_approx(
            float(spec["frequency"])
        )
        assert float(dominant["rms_db"]) == pytest_approx(expected_rms_dbfs, abs=2e-5)

        target_time = octave_time_rows[
            np.isclose(
                octave_time_rows["frequency_hz"].astype(float),
                float(spec["frequency"]),
            )
        ]
        assert not target_time.empty
        assert int(target_time["is_valid_crest_factor"].sum()) == 4
        assert float(target_time["rms_dbfs"].median()) == pytest_approx(
            expected_rms_dbfs,
            abs=2e-5,
        )


def test_non_bin_centered_tone_keeps_correct_reference_and_band_dominance(
    tmp_path: Path,
) -> None:
    """A slightly off-bin tone keeps full-band metrics and sensible band dominance."""
    frequency = 510.3
    peak = 0.45
    signal = _sine(frequency, peak)
    bundle_dir = _analyze_channel(
        track_output_dir=tmp_path,
        track_name="synthetic_off_bin.wav",
        channel_data=signal,
        channel_index=0,
        total_channels=1,
        channel_layout="mono",
    )

    _metadata, octave_rows, time_rows, _octave_time_rows = _read_channel_tables(
        bundle_dir,
        0,
    )
    full = octave_rows[octave_rows["frequency_hz"] == "Full Spectrum"].iloc[0]
    expected_peak_dbfs = _db(float(np.max(np.abs(signal))))
    expected_rms_dbfs = _db(float(np.sqrt(np.mean(np.square(signal)))))
    expected_window_rms_dbfs = _median_window_rms_dbfs(signal)

    assert float(full["max_amplitude_db"]) == pytest_approx(
        expected_peak_dbfs,
        abs=2e-5,
    )
    assert float(full["rms_db"]) == pytest_approx(expected_rms_dbfs, abs=2e-5)

    band_rows = octave_rows[octave_rows["frequency_hz"] != "Full Spectrum"].copy()
    band_rows["frequency_numeric"] = band_rows["frequency_hz"].astype(float)
    dominant = band_rows.loc[band_rows["rms"].idxmax()]
    assert float(dominant["frequency_numeric"]) == 500.0
    assert float(dominant["rms"]) > 0.99 / np.sqrt(2.0)

    valid_time = time_rows[time_rows["is_valid_crest_factor"]]
    assert len(valid_time) == int(DURATION_SECONDS)
    assert float(valid_time["rms_level_dbfs"].median()) == pytest_approx(
        expected_window_rms_dbfs,
        abs=2e-5,
    )


def test_synthetic_stereo_pipeline_preserves_shape_and_gain(tmp_path: Path) -> None:
    """Stereo sine/square channels keep separate names, gains, and crest factors."""
    left = _sine(500.0, 0.7)
    right = _square(1000.0, 0.35)
    audio_data = np.column_stack([left, right]).astype(np.float32)

    bundle_dir: Path | None = None
    for channel_data, channel_index in AudioProcessor().extract_channels(audio_data):
        bundle_dir = _analyze_channel(
            track_output_dir=tmp_path,
            track_name="synthetic_stereo.wav",
            channel_data=channel_data,
            channel_index=channel_index,
            total_channels=2,
            channel_layout="stereo",
        )

    assert bundle_dir is not None
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [channel["channel_name"] for channel in manifest["channels"]] == [
        "Channel 1 Left",
        "Channel 2 Right",
    ]

    for channel_index, peak, expected_crest_db in (
        (0, 0.7, _db(np.sqrt(2.0))),
        (1, 0.35, 0.0),
    ):
        metadata, octave_rows, time_rows, _octave_time_rows = _read_channel_tables(
            bundle_dir,
            channel_index,
        )
        full = octave_rows[octave_rows["frequency_hz"] == "Full Spectrum"].iloc[0]
        assert metadata["original_peak"] == pytest_approx(peak)
        assert float(full["max_amplitude_db"]) == pytest_approx(_db(peak), abs=2e-5)
        assert float(full["crest_factor_db"]) == pytest_approx(
            expected_crest_db,
            abs=2e-5,
        )
        valid_time = time_rows[time_rows["is_valid_crest_factor"]]
        assert float(valid_time["crest_factor_db"].median()) == pytest_approx(
            expected_crest_db,
            abs=2e-5,
        )


def test_near_silence_exports_invalid_crest_factor_windows(tmp_path: Path) -> None:
    """Very quiet channels should not turn invalid crest windows into 0 dB data."""
    quiet = _sine(500.0, 1e-8)
    bundle_dir = _analyze_channel(
        track_output_dir=tmp_path,
        track_name="synthetic_near_silence.wav",
        channel_data=quiet,
        channel_index=0,
        total_channels=1,
        channel_layout="mono",
    )

    metadata, octave_rows, time_rows, octave_time_rows = _read_channel_tables(
        bundle_dir,
        0,
    )
    full = octave_rows[octave_rows["frequency_hz"] == "Full Spectrum"].iloc[0]

    assert metadata["original_peak"] == pytest_approx(1e-8)
    assert float(full["max_amplitude_db"]) == pytest_approx(_db(1e-8), abs=2e-5)
    assert not time_rows["is_valid_crest_factor"].any()
    assert time_rows["crest_factor_db"].isna().all()
    assert set(time_rows["peak_level_dbfs"].round(1)) == {-160.0}
    assert set(time_rows["rms_level_dbfs"].round(1)) == {-163.0}

    target_rows = octave_time_rows[
        np.isclose(octave_time_rows["frequency_hz"].astype(float), 500.0)
    ]
    assert not target_rows["is_valid_crest_factor"].any()
    assert target_rows["crest_factor_db"].isna().all()


def test_sparse_impulse_pipeline_exports_high_crest_factor(tmp_path: Path) -> None:
    """Sparse transient content keeps high crest factor through exported metrics."""
    impulse = _windowed_impulses(0.6)
    bundle_dir = _analyze_channel(
        track_output_dir=tmp_path,
        track_name="synthetic_impulse.wav",
        channel_data=impulse,
        channel_index=0,
        total_channels=1,
        channel_layout="mono",
    )

    metadata, octave_rows, time_rows, _octave_time_rows = _read_channel_tables(
        bundle_dir,
        0,
    )
    full = octave_rows[octave_rows["frequency_hz"] == "Full Spectrum"].iloc[0]
    expected_normalized_rms = 1.0 / np.sqrt(SAMPLE_RATE)
    expected_crest_db = _db(np.sqrt(SAMPLE_RATE))
    expected_rms_dbfs = _db(0.6 / np.sqrt(SAMPLE_RATE))

    assert metadata["channel_name"] == "FC"
    assert metadata["original_peak"] == pytest_approx(0.6)
    assert float(full["max_amplitude"]) == pytest_approx(1.0)
    assert float(full["rms"]) == pytest_approx(expected_normalized_rms)
    assert float(full["crest_factor_db"]) == pytest_approx(expected_crest_db)
    assert float(full["rms_db"]) == pytest_approx(expected_rms_dbfs)

    valid_time = time_rows[time_rows["is_valid_crest_factor"]]
    assert len(valid_time) == int(DURATION_SECONDS)
    assert set(valid_time["peak_level_dbfs"].round(4)) == {round(_db(0.6), 4)}
    assert set(valid_time["rms_level_dbfs"].round(4)) == {round(expected_rms_dbfs, 4)}
    assert float(valid_time["crest_factor_db"].median()) == pytest_approx(
        expected_crest_db,
        abs=2e-5,
    )


def test_7_1_4_height_channel_smoke_uses_surround_height_identity(
    tmp_path: Path,
) -> None:
    """A height channel in 7.1.4 keeps its identity and metric reference."""
    total_channels = 12
    layout = "7.1.4"
    folders = [
        get_channel_folder_name(index, total_channels, layout)
        for index in range(total_channels)
    ]
    assert folders[8] == "Channel 9 TFL"
    assert folders[9] == "Channel 10 TFR"
    assert folders[10] == "Channel 11 TBL"
    assert folders[11] == "Channel 12 TBR"

    tfl = _sine(1000.0, 0.25)
    bundle_dir = _analyze_channel(
        track_output_dir=tmp_path,
        track_name="synthetic_7_1_4_height.wav",
        channel_data=tfl,
        channel_index=8,
        total_channels=total_channels,
        channel_layout=layout,
    )
    metadata, octave_rows, _time_rows, _octave_time_rows = _read_channel_tables(
        bundle_dir,
        8,
    )
    full = octave_rows[octave_rows["frequency_hz"] == "Full Spectrum"].iloc[0]
    band_rows = octave_rows[octave_rows["frequency_hz"] != "Full Spectrum"].copy()
    band_rows["frequency_numeric"] = band_rows["frequency_hz"].astype(float)
    dominant = band_rows.loc[band_rows["rms"].idxmax()]

    assert metadata["channel_name"] == "TFL"
    assert metadata["channel_folder_name"] == "Channel 9 TFL"
    assert float(full["max_amplitude_db"]) == pytest_approx(_db(0.25), abs=2e-5)
    assert float(dominant["frequency_numeric"]) == 1000.0


def pytest_approx(value: float, **kwargs) -> object:
    """Small wrapper to keep imports local to assertion helpers."""
    return pytest.approx(value, **kwargs)
