import numpy as np

from src.music_analyzer import MusicAnalyzer


def test_octave_band_statistics_use_whole_interval_crest_factor():
    sr = 1000
    t = np.arange(sr * 2) / sr
    sine = 0.5 * np.sin(2 * np.pi * 50 * t)
    analyzer = MusicAnalyzer(sample_rate=sr, original_peak=1.0)

    results = analyzer.analyze_octave_bands(
        np.column_stack([sine, sine]),
        center_frequencies=[50.0],
    )

    stats = results["statistics"]["50.000"]
    assert stats["crest_factor_method"] == "whole_interval_peak_rms"
    assert stats["is_valid_crest_factor"]
    assert np.isclose(stats["crest_factor_db"], 3.0103, atol=0.01)


def test_octave_band_statistics_mark_silence_invalid():
    analyzer = MusicAnalyzer(sample_rate=1000, original_peak=1.0)

    results = analyzer.analyze_octave_bands(
        np.zeros((2000, 1), dtype=np.float64),
        center_frequencies=[],
    )

    stats = results["statistics"]["Full Spectrum"]
    assert stats["crest_factor_method"] == "whole_interval_peak_rms"
    assert not stats["is_valid_crest_factor"]
    assert np.isnan(stats["crest_factor_db"])


def test_time_domain_mode_fixed_window_marks_silence_invalid():
    sr = 1000
    t = np.arange(sr * 4) / sr
    sine = 0.5 * np.sin(2 * np.pi * 50 * t)
    x = np.concatenate([sine, np.zeros(sr * 3)]).astype(np.float32)
    analyzer = MusicAnalyzer(
        sample_rate=sr,
        original_peak=1.0,
        time_domain_crest_factor_mode="fixed_window",
        analysis_config={
            "crest_factor_window_seconds": 2.0,
            "crest_factor_step_seconds": 1.0,
            "crest_factor_rms_floor_dbfs": -80.0,
        },
    )

    out = analyzer.analyze_crest_factor_over_time(x, chunk_duration=2.0)

    assert out["time_domain_mode"] == "fixed_window"
    assert out["chunk_duration"] == 2.0
    assert out["time_step_seconds"] == 1.0
    assert out["crest_factor_method"] == "fixed_window_peak_rms"
    assert np.isclose(out["crest_factors_db"][0], 3.0103, atol=0.01)
    assert out["is_valid_crest_factor"][0]
    assert not out["is_valid_crest_factor"][-1]
    assert np.isnan(out["crest_factors_db"][-1])


def test_time_domain_mode_slow_includes_metadata_keys():
    sr = 1000
    x = np.zeros(sr * 3, dtype=np.float32)
    x[100] = 1.0
    analyzer = MusicAnalyzer(
        sample_rate=sr,
        original_peak=1.0,
        time_domain_crest_factor_mode="slow",
        analysis_config={
            "time_domain_slow_window_seconds": 1.0,
            "time_domain_slow_step_seconds": 1.0,
            "time_domain_slow_rms_tau_seconds": 1.0,
        },
    )

    out = analyzer.analyze_crest_factor_over_time(x, chunk_duration=2.0)
    assert out["time_domain_mode"] == "slow"
    assert "time_domain_rms_method" in out
    assert "time_domain_peak_method" in out
    assert out["chunk_duration"] == 1.0
    assert out["time_step_seconds"] == 1.0


def test_time_domain_mode_fixed_chunk_uses_chunk_duration():
    sr = 1000
    x = np.zeros(sr * 4, dtype=np.float32)
    x[1500] = 1.0
    analyzer = MusicAnalyzer(
        sample_rate=sr,
        original_peak=1.0,
        time_domain_crest_factor_mode="fixed_chunk",
        analysis_config={},
    )

    out = analyzer.analyze_crest_factor_over_time(x, chunk_duration=2.0)
    assert out["time_domain_mode"] == "fixed_chunk"
    assert out["chunk_duration"] == 2.0
    assert out["time_step_seconds"] == 2.0
    assert out["num_chunks"] == 2
