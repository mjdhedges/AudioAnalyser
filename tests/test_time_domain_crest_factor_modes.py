import numpy as np

from src.music_analyzer import MusicAnalyzer


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

