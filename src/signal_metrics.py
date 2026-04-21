"""Signal processing utilities for level weighting and peak analysis."""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional accel
    njit = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False


def _compute_slow_rms_envelope_py(
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


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _compute_slow_rms_envelope_numba(
        signal: np.ndarray,
        sample_rate: int,
        tau: float,
    ) -> np.ndarray:
        """Numba-accelerated IEC SLOW RMS envelope.

        Intended to be numerically identical to `_compute_slow_rms_envelope_py` by
        preserving float64 intermediates and the exact recurrence.
        """
        n = signal.size
        if n == 0:
            return np.empty(0, dtype=np.float64)

        dt = 1.0 / float(sample_rate)
        alpha = dt / (tau + dt)

        slow_squares = np.empty(n, dtype=np.float64)
        prev = float(signal[0]) ** 2
        slow_squares[0] = prev

        for idx in range(1, n):
            v = float(signal[idx])
            squared = v ** 2
            prev = prev + alpha * (squared - prev)
            slow_squares[idx] = prev

        # Clip to [0, +inf) (kept for parity with Python implementation).
        for idx in range(n):
            if slow_squares[idx] < 0.0:
                slow_squares[idx] = 0.0

        return np.sqrt(slow_squares)


def compute_slow_rms_envelope(
    signal: np.ndarray,
    sample_rate: int,
    tau: float = 1.0,
    use_numba: bool = True,
) -> np.ndarray:
    """Compute the IEC SLOW (τ = 1s) RMS envelope of a signal.

    By default, uses a Numba-accelerated implementation when available.
    This provides large speedups but may introduce extremely small floating-point
    differences (observed on Win64: ~2e-9 absolute, ~1.6e-8 relative). You can
    force the exact reference path with ``use_numba=False``.
    """
    if use_numba and _NUMBA_AVAILABLE:
        sig = np.ascontiguousarray(signal)
        return _compute_slow_rms_envelope_numba(sig, int(sample_rate), float(tau))
    return _compute_slow_rms_envelope_py(signal, sample_rate, tau=tau)


def _compute_peak_hold_envelope_py(
    signal: np.ndarray,
    sample_rate: int,
    tau: float = 1.0,
) -> np.ndarray:
    """Compute a peak-hold envelope with exponential decay.

    Mimics the IEC SLOW behaviour for peak indication: instantaneous attack
    followed by exponential decay with the specified time constant.

    Args:
        signal: Input audio samples.
        sample_rate: Sampling rate in Hz.
        tau: Time constant controlling the decay speed (seconds).

    Returns:
        Array of peak-hold envelope samples in linear amplitude units.
    """
    if signal.size == 0:
        return np.array([], dtype=np.float64)

    dt = 1.0 / float(sample_rate)
    decay = np.exp(-dt / max(tau, dt))
    abs_signal = np.abs(signal).astype(np.float64, copy=False)
    envelope = np.empty(abs_signal.size, dtype=np.float64)
    prev = 0.0

    for idx, value in enumerate(abs_signal):
        decayed = prev * decay
        prev = value if value > decayed else decayed
        envelope[idx] = prev

    return envelope


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _compute_peak_hold_envelope_numba(signal: np.ndarray, sample_rate: int, tau: float) -> np.ndarray:
        n = signal.size
        if n == 0:
            return np.empty(0, dtype=np.float64)

        dt = 1.0 / float(sample_rate)
        decay = np.exp(-dt / max(tau, dt))

        envelope = np.empty(n, dtype=np.float64)
        prev = 0.0
        for idx in range(n):
            value = abs(float(signal[idx]))
            decayed = prev * decay
            prev = value if value > decayed else decayed
            envelope[idx] = prev
        return envelope


def compute_peak_hold_envelope(
    signal: np.ndarray,
    sample_rate: int,
    tau: float = 1.0,
    use_numba: bool = True,
) -> np.ndarray:
    """Compute a peak-hold envelope with exponential decay.

    Defaults to Numba acceleration when available; set ``use_numba=False`` for
    the reference implementation.
    """
    if use_numba and _NUMBA_AVAILABLE:
        sig = np.ascontiguousarray(signal)
        return _compute_peak_hold_envelope_numba(sig, int(sample_rate), float(tau))
    return _compute_peak_hold_envelope_py(signal, sample_rate, tau=tau)


def _max_abs_over_window_py(
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


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _max_abs_over_window_numba(signal: np.ndarray, window_samples: int) -> float:
        """Numba-accelerated monotonic-queue sliding max of abs(signal).

        Algorithm is equivalent to `_max_abs_over_window_py`, implemented without
        Python deque overhead. This is intended to be numerically identical.
        """
        n = signal.size
        if n == 0:
            return 0.0

        if window_samples <= 1 or window_samples >= n:
            max_val = 0.0
            for i in range(n):
                v = abs(float(signal[i]))
                if v > max_val:
                    max_val = v
            return max_val

        # Monotonic deque of candidate indices implemented via a ring buffer.
        # We store indices in `dq`, and maintain [head, tail) as the active span.
        dq = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0

        max_val = 0.0
        for idx in range(n):
            value = abs(float(signal[idx]))

            # Pop from tail while the new value dominates.
            while tail > head:
                last_idx = dq[tail - 1]
                last_val = abs(float(signal[last_idx]))
                if last_val <= value:
                    tail -= 1
                else:
                    break

            dq[tail] = idx
            tail += 1

            # Expire head if it falls out of window.
            if dq[head] <= idx - window_samples:
                head += 1

            if idx >= window_samples - 1:
                current = abs(float(signal[dq[head]]))
                if current > max_val:
                    max_val = current

        return max_val


def max_abs_over_window(
    signal: np.ndarray,
    window_samples: int,
) -> float:
    """Return the maximum absolute level observed in any sliding window.

    Uses a Numba-accelerated implementation when available; otherwise falls back
    to the original Python deque algorithm.
    """
    if _NUMBA_AVAILABLE:
        # Ensure contiguous input for best performance/predictability.
        sig = np.ascontiguousarray(signal)
        return float(_max_abs_over_window_numba(sig, int(window_samples)))
    return _max_abs_over_window_py(signal, window_samples)


def _sampled_max_abs_py(
    signal: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> np.ndarray:
    """Compute maximum absolute value for evenly spaced sliding windows.

    Args:
        signal: Input audio samples.
        window_samples: Length of each analysis window in samples.
        step_samples: Hop size between consecutive window starts in samples.

    Returns:
        Array of window maxima (linear amplitude units) sampled every
        ``step_samples`` with window length ``window_samples``.
    """
    if signal.size == 0 or window_samples <= 0 or step_samples <= 0:
        return np.array([], dtype=np.float64)

    abs_signal = np.abs(signal)
    n = abs_signal.size
    if n < window_samples:
        return np.array([], dtype=np.float64)

    num_windows = (n - window_samples) // step_samples + 1
    peaks = np.empty(num_windows, dtype=np.float64)
    window_end_indices = window_samples - 1 + np.arange(num_windows) * step_samples

    dq: deque[int] = deque()
    window_idx = 0

    for idx, value in enumerate(abs_signal):
        while dq and abs_signal[dq[-1]] <= value:
            dq.pop()
        dq.append(idx)

        while dq and dq[0] <= idx - window_samples:
            dq.popleft()

        while (
            window_idx < num_windows
            and idx == window_end_indices[window_idx]
            and dq
        ):
            peaks[window_idx] = abs_signal[dq[0]]
            window_idx += 1

        if window_idx >= num_windows:
            break

    return peaks


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _sampled_max_abs_numba(signal: np.ndarray, window_samples: int, step_samples: int) -> np.ndarray:
        n = signal.size
        if n == 0 or window_samples <= 0 or step_samples <= 0:
            return np.empty(0, dtype=np.float64)
        if n < window_samples:
            return np.empty(0, dtype=np.float64)

        num_windows = (n - window_samples) // step_samples + 1
        peaks = np.empty(num_windows, dtype=np.float64)
        window_end_indices = window_samples - 1 + np.arange(num_windows) * step_samples

        dq = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        window_idx = 0

        for idx in range(n):
            value = abs(float(signal[idx]))

            while tail > head:
                last_idx = dq[tail - 1]
                last_val = abs(float(signal[last_idx]))
                if last_val <= value:
                    tail -= 1
                else:
                    break
            dq[tail] = idx
            tail += 1

            while tail > head and dq[head] <= idx - window_samples:
                head += 1

            while window_idx < num_windows and idx == window_end_indices[window_idx] and tail > head:
                peaks[window_idx] = abs(float(signal[dq[head]]))
                window_idx += 1
                if window_idx >= num_windows:
                    break

            if window_idx >= num_windows:
                break

        return peaks


def sampled_max_abs(
    signal: np.ndarray,
    window_samples: int,
    step_samples: int,
    use_numba: bool = True,
) -> np.ndarray:
    """Compute maximum absolute value for evenly spaced sliding windows.

    Defaults to Numba acceleration when available; set ``use_numba=False`` for
    the reference implementation.
    """
    if use_numba and _NUMBA_AVAILABLE:
        sig = np.ascontiguousarray(signal)
        return _sampled_max_abs_numba(sig, int(window_samples), int(step_samples))
    return _sampled_max_abs_py(signal, window_samples, step_samples)

