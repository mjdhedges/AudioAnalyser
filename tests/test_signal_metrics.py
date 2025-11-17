from __future__ import annotations

import numpy as np

from src.signal_metrics import compute_slow_rms_envelope, max_abs_over_window


def test_compute_slow_rms_envelope_on_constant_signal() -> None:
    sample_rate = 10
    signal = np.ones(sample_rate * 2)
    envelope = compute_slow_rms_envelope(signal, sample_rate)

    assert envelope.size == signal.size
    # Constant signal should converge close to 1.0
    assert np.isclose(envelope[-1], 1.0, atol=1e-3)


def test_max_abs_over_window_detects_local_peak() -> None:
    signal = np.array([0.0, 0.5, 1.0, 0.2, 0.1])
    result = max_abs_over_window(signal, window_samples=2)
    assert np.isclose(result, 1.0)

