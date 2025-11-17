"""Signal processing utilities for level weighting and peak analysis."""

from __future__ import annotations

from collections import deque

import numpy as np


def compute_slow_rms_envelope(
    signal: np.ndarray,
    sample_rate: int,
    tau: float = 1.0,
) -> np.ndarray:
    """Compute the IEC SLOW (τ = 1s) RMS envelope of a signal.

    Args:
        signal: Input audio samples.
        sample_rate: Sampling rate in Hz.
        tau: Time constant for the exponential average (defaults to 1 second).

    Returns:
        Array of the same length as ``signal`` containing the slow-weighted RMS
        envelope (linear amplitude units).
    """
    if signal.size == 0:
        return np.array([], dtype=np.float64)

    dt = 1.0 / float(sample_rate)
    alpha = dt / (tau + dt)

    slow_squares = np.empty(signal.size, dtype=np.float64)
    prev = float(signal[0]) ** 2
    slow_squares[0] = prev

    for idx in range(1, signal.size):
        squared = float(signal[idx]) ** 2
        prev += alpha * (squared - prev)
        slow_squares[idx] = prev

    slow_squares = np.clip(slow_squares, 0.0, None)
    return np.sqrt(slow_squares)


def max_abs_over_window(
    signal: np.ndarray,
    window_samples: int,
) -> float:
    """Return the maximum absolute level observed in any sliding window.

    Args:
        signal: Input audio samples.
        window_samples: Window length in samples.

    Returns:
        Maximum absolute amplitude contained within any contiguous
        ``window_samples`` segment of ``signal``.
    """
    if signal.size == 0:
        return 0.0

    abs_signal = np.abs(signal)
    if window_samples <= 1 or window_samples >= abs_signal.size:
        return float(np.max(abs_signal))

    max_val = 0.0
    candidates: deque[int] = deque()

    for idx, value in enumerate(abs_signal):
        while candidates and abs_signal[candidates[-1]] <= value:
            candidates.pop()
        candidates.append(idx)

        if candidates[0] <= idx - window_samples:
            candidates.popleft()

        if idx >= window_samples - 1:
            current = abs_signal[candidates[0]]
            if current > max_val:
                max_val = float(current)

    return max_val

