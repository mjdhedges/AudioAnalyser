from __future__ import annotations

import numpy as np

from src.signal_metrics import (
    compute_peak_hold_envelope,
    compute_slow_rms_envelope,
    _compute_slow_rms_envelope_py,
    _compute_peak_hold_envelope_py,
    max_abs_over_window,
    _max_abs_over_window_py,
    sampled_max_abs,
    _sampled_max_abs_py,
)


def test_compute_slow_rms_envelope_on_constant_signal() -> None:
    sample_rate = 10
    signal = np.ones(sample_rate * 2)
    envelope = compute_slow_rms_envelope(signal, sample_rate)

    assert envelope.size == signal.size
    # Constant signal should converge close to 1.0
    assert np.isclose(envelope[-1], 1.0, atol=1e-3)


def test_compute_slow_rms_envelope_numba_matches_python_reference() -> None:
    rng = np.random.default_rng(0)
    # Use float32 input to match common audio dtype; reference computes in float64.
    signal = rng.standard_normal(50_000).astype(np.float32)
    for sample_rate in (8000, 44100):
        for tau in (0.1, 1.0, 2.0):
            # Reference path must match reference exactly.
            got = compute_slow_rms_envelope(signal, sample_rate, tau=tau, use_numba=False)
            ref = _compute_slow_rms_envelope_py(signal, sample_rate, tau=tau)
            assert got.dtype == ref.dtype == np.float64
            assert got.shape == ref.shape
            np.testing.assert_array_equal(got, ref)

            # Default (accelerated) path: allow tiny FP differences (must be very small).
            fast = compute_slow_rms_envelope(signal, sample_rate, tau=tau, use_numba=True)
            np.testing.assert_allclose(fast, ref, rtol=1e-7, atol=2e-9)


def test_max_abs_over_window_detects_local_peak() -> None:
    signal = np.array([0.0, 0.5, 1.0, 0.2, 0.1])
    result = max_abs_over_window(signal, window_samples=2)
    assert np.isclose(result, 1.0)


def test_max_abs_over_window_numba_matches_python_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(10_000).astype(np.float32)
    for window in (1, 2, 7, 128, 1024, signal.size, signal.size + 1):
        got = max_abs_over_window(signal, window_samples=window)
        ref = _max_abs_over_window_py(signal, window_samples=window)
        assert got == ref


def test_sampled_max_abs_returns_sequence() -> None:
    signal = np.array([0.0, 0.5, 1.0, 0.2, 0.1, 0.8, 0.3, 0.6])
    peaks = sampled_max_abs(signal, window_samples=2, step_samples=2)
    assert peaks.size == 4
    assert np.allclose(peaks, [0.5, 1.0, 0.8, 0.6])


def test_sampled_max_abs_numba_matches_python_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(50_000).astype(np.float32)
    peaks_fast = sampled_max_abs(signal, window_samples=256, step_samples=128, use_numba=True)
    peaks_ref = _sampled_max_abs_py(signal, window_samples=256, step_samples=128)
    assert peaks_fast.shape == peaks_ref.shape
    np.testing.assert_array_equal(peaks_fast, peaks_ref)


def test_compute_peak_hold_envelope_has_attack_and_decay() -> None:
    sample_rate = 10
    signal = np.zeros(sample_rate * 2)
    signal[5] = 1.0
    envelope = compute_peak_hold_envelope(signal, sample_rate)

    assert np.isclose(envelope[5], 1.0)
    decay = np.exp(-1.0 / sample_rate)
    assert np.isclose(envelope[6], decay, atol=1e-6)


def test_compute_peak_hold_envelope_numba_close_to_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(50_000).astype(np.float32)
    ref = _compute_peak_hold_envelope_py(signal, sample_rate=44100, tau=1.0)
    fast = compute_peak_hold_envelope(signal, sample_rate=44100, tau=1.0, use_numba=True)
    np.testing.assert_allclose(fast, ref, rtol=1e-7, atol=2e-9)

