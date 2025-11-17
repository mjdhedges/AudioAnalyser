import numpy as np
from src.envelope_analyzer import EnvelopeAnalyzer


def test_sustained_peak_recovery_times_basic():
    sr = 1000  # 1 kHz sample rate for easy ms math
    ea = EnvelopeAnalyzer(sample_rate=sr, original_peak=1.0)

    # Create a synthetic signal with a clear peak then exponential decay
    # Peak at t=100ms, amplitude 1.0, then decay with tau ~ 50ms
    duration_s = 1.0
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=float)
    peak_idx = int(0.1 * sr)  # 100ms
    signal[peak_idx] = 1.0
    decay_len = n - peak_idx - 1
    if decay_len > 0:
        decay = np.exp(-np.arange(decay_len) / (0.05 * sr))  # tau=50ms
        signal[peak_idx + 1 :] = decay

    # One band (Full Spectrum): octave_bank shape (samples, bands)
    octave_bank = signal.reshape(-1, 1)
    center_freqs = []  # no bands, only Full Spectrum (0 Hz is added internally)

    config = {
        "envelope_method": "peak_envelope",
        "rms_envelope_window_ms": 10.0,
        "peak_detection_min_height_db": -40.0,
        "peak_detection_min_distance_ms": 20.0,
        "worst_case_num_envelopes": 1,
        "worst_case_sort_by": "peak_value",
        "pattern_min_repetitions": 3,
        "pattern_max_patterns_per_band": 0,  # disable pattern analysis
        "pattern_similarity_threshold": 0.85,
        "attack_threshold_db": -20.0,
        "peak_hold_threshold_db": -1.0,
        "decay_thresholds_db": [-3.0, -6.0, -9.0, -12.0],
        # Sustained peaks config
        "sustained_peaks_enable": True,
        "sustained_peaks_min_peak_dbfs": -3.0,
        "sustained_peaks_thresholds_db": [-3.0, -6.0, -9.0, -12.0],
        "sustained_peaks_relative": True,
        "sustained_peaks_export_events": False,
    }

    stats = ea.analyze_envelope_statistics(octave_bank, center_freqs, config=config)
    full = stats["Full Spectrum"]
    sustained = full.get("sustained_peaks_summary", {})

    # There should be at least one detected peak
    assert sustained.get("n_peaks", 0) >= 1

    # Mean recovery times should be monotonically increasing with deeper thresholds
    t3 = sustained["t3_ms"]["mean"]
    t6 = sustained["t6_ms"]["mean"]
    t9 = sustained["t9_ms"]["mean"]
    t12 = sustained["t12_ms"]["mean"]
    assert t3 <= t6 <= t9 <= t12


