"""Study script for crest-factor block length and peak-hold tau behaviour.

This script is intentionally self-contained. It generates deterministic audio,
computes fixed-block crest factor at several window lengths, then sweeps
peak-hold time constants while keeping the IEC Slow RMS tau fixed at 1 second.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


PROOF_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 48_000
RMS_TAU_SECONDS = 1.0
WINDOW_SECONDS = 2.0
STEP_SECONDS = 1.0
WARMUP_SECONDS = 3.0
PROGRAM_NOISE_FLOOR_DBFS = -80.0
FINAL_TRUE_SILENCE_SECONDS = 10.0
BLOCK_WINDOWS_SECONDS = [
    0.005,
    0.010,
    0.020,
    0.050,
    0.100,
    0.250,
    0.500,
    1.000,
    2.000,
    5.000,
    10.000,
]

PEAK_TAU_CANDIDATES_SECONDS = [
    0.05,
    0.08,
    0.10,
    0.125,
    0.16,
    0.20,
    0.25,
    0.315,
    0.40,
    0.50,
    0.56,
    0.63,
    0.71,
    0.80,
    0.90,
    1.00,
    1.12,
    1.25,
    1.40,
    1.60,
    2.00,
    2.50,
    3.15,
    4.00,
    5.00,
    6.30,
    8.00,
    10.00,
    12.50,
    16.00,
    20.00,
    25.00,
    31.50,
    40.00,
    50.00,
    63.00,
]


@dataclass(frozen=True)
class TauResult:
    """Summary error metrics for one peak-hold tau candidate."""

    peak_tau_seconds: float
    mean_error_db: float
    mean_abs_error_db: float
    rmse_db: float
    p95_abs_error_db: float
    max_abs_error_db: float
    within_0_5_db_percent: float
    within_1_0_db_percent: float


@dataclass(frozen=True)
class BlockWindowSummary:
    """Crest-factor distribution for one fixed block length."""

    window_seconds: float
    num_windows: int
    min_cf_db: float
    mean_cf_db: float
    median_cf_db: float
    p95_cf_db: float
    max_cf_db: float
    std_cf_db: float


@dataclass(frozen=True)
class TauWindowSummary:
    """Best peak tau when comparing slow metering to one block length."""

    window_seconds: float
    best_peak_tau_seconds: float
    best_mean_abs_error_db: float
    best_rmse_db: float
    best_p95_abs_error_db: float
    tau_1s_mean_abs_error_db: float
    tau_2s_mean_abs_error_db: float


def generate_test_signal(sample_rate: int) -> np.ndarray:
    """Create deterministic programme material that stresses peak/RMS tracking."""
    rng = np.random.default_rng(20260425)

    sections: list[np.ndarray] = []

    def silence(duration: float) -> np.ndarray:
        return np.zeros(int(duration * sample_rate), dtype=np.float64)

    def sine(duration: float, freq: float, amp: float) -> np.ndarray:
        t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
        return amp * np.sin(2.0 * np.pi * freq * t)

    def click_train(duration: float, levels: list[float], spacing: float) -> np.ndarray:
        n = int(duration * sample_rate)
        signal = np.zeros(n, dtype=np.float64)
        click_len = int(0.0025 * sample_rate)
        click_t = np.arange(click_len, dtype=np.float64) / sample_rate
        # Windowed impulse with a short HF tone gives a realistic peak without a DC step.
        click_shape = np.sin(2.0 * np.pi * 6000.0 * click_t) * np.hanning(click_len)
        click_shape /= np.max(np.abs(click_shape))
        for idx, start_sec in enumerate(np.arange(0.5, duration - 0.1, spacing)):
            start = int(start_sec * sample_rate)
            end = min(start + click_len, n)
            level = levels[idx % len(levels)]
            signal[start:end] += level * click_shape[: end - start]
        return signal

    def pink_noise(duration: float, amp: float) -> np.ndarray:
        n = int(duration * sample_rate)
        white = rng.normal(0.0, 1.0, n)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        scale = np.ones_like(freqs)
        scale[1:] = 1.0 / np.sqrt(freqs[1:])
        spectrum *= scale
        pink = np.fft.irfft(spectrum, n=n)
        pink -= np.mean(pink)
        pink /= max(np.max(np.abs(pink)), 1e-12)
        return amp * pink

    def pink_burst(duration: float, amp: float, burst_duration: float) -> np.ndarray:
        n = int(duration * sample_rate)
        burst_n = int(burst_duration * sample_rate)
        signal = np.zeros(n, dtype=np.float64)
        start = max((n - burst_n) // 2, 0)
        end = min(start + burst_n, n)
        burst = pink_noise((end - start) / sample_rate, amp)
        fade_n = min(int(0.01 * sample_rate), burst.size // 4)
        if fade_n > 0:
            fade = np.sin(np.linspace(0.0, np.pi / 2.0, fade_n)) ** 2
            burst[:fade_n] *= fade
            burst[-fade_n:] *= fade[::-1]
        signal[start:end] = burst
        return signal

    # 1) Clicks at different levels to test fast attack and missed-peak behaviour.
    sections.append(silence(1.0))
    sections.append(click_train(12.0, [0.20, 0.35, 0.50, 0.70, 0.90], spacing=0.75))

    # 2) Steady-state sine waves at different levels to test stable crest factor.
    sections.append(silence(1.0))
    for amp in [0.18, 0.35, 0.55, 0.78, 0.92]:
        sections.append(sine(3.0, 1000.0, amp))

    # 3) Low-frequency steady-state sine levels to test slow-cycle behaviour.
    sections.append(silence(1.0))
    for amp in [0.30, 0.60, 0.88]:
        sections.append(sine(4.0, 63.0, amp))

    # 4) Pink-noise bursts with different durations and levels to stress RMS settling.
    sections.append(silence(1.0))
    for burst_duration, amp in [
        (0.10, 0.90),
        (0.25, 0.75),
        (0.50, 0.80),
        (1.00, 0.65),
        (2.00, 0.55),
        (4.00, 0.45),
    ]:
        sections.append(pink_burst(max(2.0, burst_duration + 1.0), amp, burst_duration))
        sections.append(silence(0.5))

    programme = np.concatenate(sections)
    # Real programme material rarely sits at exact digital zero between events.
    # Add a very low pink-noise bed to the programme region so gap behaviour is
    # exercised with a realistic floor, then append true silence as a distinct
    # edge case for invalid/absent signal handling.
    noise_floor_amp = 10.0 ** (PROGRAM_NOISE_FLOOR_DBFS / 20.0)
    programme = programme + pink_noise(programme.size / sample_rate, noise_floor_amp)
    signal = np.concatenate([programme, silence(FINAL_TRUE_SILENCE_SECONDS)])
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.98 * signal / peak
    return signal.astype(np.float32)


def slow_rms_envelope(signal: np.ndarray, sample_rate: int, tau: float) -> np.ndarray:
    """Independent copy of the IEC-style exponential RMS recurrence."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=np.float64)
    dt = 1.0 / float(sample_rate)
    alpha = 1.0 - np.exp(-dt / tau)
    y2 = np.empty_like(x, dtype=np.float64)
    prev = float(x[0]) ** 2
    y2[0] = prev
    for idx in range(1, x.size):
        prev += alpha * (float(x[idx]) ** 2 - prev)
        y2[idx] = prev
    return np.sqrt(np.clip(y2, 0.0, None))


def peak_hold_envelope(signal: np.ndarray, sample_rate: int, tau: float) -> np.ndarray:
    """Independent peak-hold envelope with instantaneous attack and exponential decay."""
    x_abs = np.abs(np.asarray(signal, dtype=np.float64))
    if x_abs.size == 0:
        return np.array([], dtype=np.float64)
    dt = 1.0 / float(sample_rate)
    decay = np.exp(-dt / max(tau, dt))
    env = np.empty_like(x_abs, dtype=np.float64)
    prev = 0.0
    for idx, value in enumerate(x_abs):
        decayed = prev * decay
        prev = value if value > decayed else decayed
        env[idx] = prev
    return env


def block_reference(
    signal: np.ndarray,
    sample_rate: int,
    window_seconds: float = WINDOW_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-block time points and crest factor in dB."""
    window_samples = max(int(window_seconds * sample_rate), 1)
    num_blocks = len(signal) // window_samples
    if num_blocks == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    trimmed = np.asarray(signal[: num_blocks * window_samples], dtype=np.float64)
    blocks = trimmed.reshape(num_blocks, window_samples)
    peak = np.max(np.abs(blocks), axis=1)
    rms = np.sqrt(np.mean(np.square(blocks), axis=1))
    crest = np.divide(peak, rms, out=np.ones_like(peak), where=rms > 0)
    crest_db = 20.0 * np.log10(np.maximum(crest, 1.0))
    times = (np.arange(num_blocks, dtype=np.float64) + 1.0) * window_seconds
    return times, crest_db


def slow_crest_db_for_tau(
    signal: np.ndarray,
    sample_rate: int,
    peak_tau: float,
    window_seconds: float = WINDOW_SECONDS,
    rms_env: np.ndarray | None = None,
    peak_env: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slow-mode crest factor sampled at fixed block endpoints."""
    if rms_env is None:
        rms_env = slow_rms_envelope(signal, sample_rate, tau=RMS_TAU_SECONDS)
    if peak_env is None:
        peak_env = peak_hold_envelope(signal, sample_rate, tau=peak_tau)

    window_samples = max(int(window_seconds * sample_rate), 1)
    step_samples = window_samples
    end_indices = np.arange(window_samples - 1, len(signal), step_samples)
    times = (end_indices + 1) / float(sample_rate)
    rms = rms_env[end_indices]
    peak = peak_env[end_indices]
    crest = np.divide(peak, rms, out=np.ones_like(peak), where=rms > 0)
    crest_db = 20.0 * np.log10(np.maximum(crest, 1.0))
    return times, crest_db


def compare_tau(
    reference_db: np.ndarray,
    candidate_db: np.ndarray,
    peak_tau: float,
    window_seconds: float = WINDOW_SECONDS,
) -> TauResult:
    """Compare a candidate tau against the fixed-block crest-factor reference."""
    count = min(reference_db.size, candidate_db.size)
    reference = reference_db[:count]
    candidate = candidate_db[:count]

    warmup_blocks = int(WARMUP_SECONDS / window_seconds)
    if count > warmup_blocks:
        reference = reference[warmup_blocks:]
        candidate = candidate[warmup_blocks:]

    error = candidate - reference
    abs_error = np.abs(error)
    return TauResult(
        peak_tau_seconds=peak_tau,
        mean_error_db=float(np.mean(error)),
        mean_abs_error_db=float(np.mean(abs_error)),
        rmse_db=float(np.sqrt(np.mean(np.square(error)))),
        p95_abs_error_db=float(np.percentile(abs_error, 95)),
        max_abs_error_db=float(np.max(abs_error)),
        within_0_5_db_percent=float(np.mean(abs_error <= 0.5) * 100.0),
        within_1_0_db_percent=float(np.mean(abs_error <= 1.0) * 100.0),
    )


def write_results_csv(results: list[TauResult], output_path: Path) -> None:
    """Write tau sweep metrics."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "peak_tau_seconds",
                "mean_error_db",
                "mean_abs_error_db",
                "rmse_db",
                "p95_abs_error_db",
                "max_abs_error_db",
                "within_0_5_db_percent",
                "within_1_0_db_percent",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.peak_tau_seconds,
                    row.mean_error_db,
                    row.mean_abs_error_db,
                    row.rmse_db,
                    row.p95_abs_error_db,
                    row.max_abs_error_db,
                    row.within_0_5_db_percent,
                    row.within_1_0_db_percent,
                ]
            )


def summarize_block_window(window_seconds: float, crest_db: np.ndarray) -> BlockWindowSummary:
    """Summarize crest-factor distribution for one fixed block length."""
    valid = crest_db[np.isfinite(crest_db)]
    if valid.size == 0:
        valid = np.array([0.0], dtype=np.float64)
    return BlockWindowSummary(
        window_seconds=window_seconds,
        num_windows=int(crest_db.size),
        min_cf_db=float(np.min(valid)),
        mean_cf_db=float(np.mean(valid)),
        median_cf_db=float(np.median(valid)),
        p95_cf_db=float(np.percentile(valid, 95)),
        max_cf_db=float(np.max(valid)),
        std_cf_db=float(np.std(valid)),
    )


def write_block_window_summary_csv(
    summaries: list[BlockWindowSummary],
    output_path: Path,
) -> None:
    """Write fixed-block crest-factor summaries."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_seconds",
                "num_windows",
                "min_cf_db",
                "mean_cf_db",
                "median_cf_db",
                "p95_cf_db",
                "max_cf_db",
                "std_cf_db",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.window_seconds,
                    row.num_windows,
                    row.min_cf_db,
                    row.mean_cf_db,
                    row.median_cf_db,
                    row.p95_cf_db,
                    row.max_cf_db,
                    row.std_cf_db,
                ]
            )


def write_tau_by_window_csv(
    summaries: list[TauWindowSummary],
    output_path: Path,
) -> None:
    """Write best tau and comparison values for each block length."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_seconds",
                "best_peak_tau_seconds",
                "best_mean_abs_error_db",
                "best_rmse_db",
                "best_p95_abs_error_db",
                "tau_1s_mean_abs_error_db",
                "tau_2s_mean_abs_error_db",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.window_seconds,
                    row.best_peak_tau_seconds,
                    row.best_mean_abs_error_db,
                    row.best_rmse_db,
                    row.best_p95_abs_error_db,
                    row.tau_1s_mean_abs_error_db,
                    row.tau_2s_mean_abs_error_db,
                ]
            )


def run_block_window_study(
    signal: np.ndarray,
    sample_rate: int,
) -> tuple[list[BlockWindowSummary], list[TauWindowSummary]]:
    """Study fixed-block crest factor and best tau vs block length."""
    duration_seconds = len(signal) / float(sample_rate)
    windows = [w for w in BLOCK_WINDOWS_SECONDS if w < duration_seconds]
    windows.append(duration_seconds)

    rms_env = slow_rms_envelope(signal, sample_rate, tau=RMS_TAU_SECONDS)
    peak_env_by_tau = {
        tau: peak_hold_envelope(signal, sample_rate, tau=tau)
        for tau in PEAK_TAU_CANDIDATES_SECONDS
    }

    block_summaries: list[BlockWindowSummary] = []
    tau_summaries: list[TauWindowSummary] = []

    for window_seconds in windows:
        _times, reference_db = block_reference(signal, sample_rate, window_seconds)
        block_summaries.append(summarize_block_window(window_seconds, reference_db))

        tau_results: list[TauResult] = []
        for peak_tau in PEAK_TAU_CANDIDATES_SECONDS:
            _candidate_times, candidate_db = slow_crest_db_for_tau(
                signal,
                sample_rate,
                peak_tau,
                window_seconds=window_seconds,
                rms_env=rms_env,
                peak_env=peak_env_by_tau[peak_tau],
            )
            tau_results.append(
                compare_tau(reference_db, candidate_db, peak_tau, window_seconds)
            )

        tau_results.sort(key=lambda row: (row.mean_abs_error_db, row.rmse_db))
        best = tau_results[0]
        tau_1s = _result_for_tau(tau_results, 1.0)
        tau_2s = _result_for_tau(tau_results, 2.0)
        tau_summaries.append(
            TauWindowSummary(
                window_seconds=window_seconds,
                best_peak_tau_seconds=best.peak_tau_seconds,
                best_mean_abs_error_db=best.mean_abs_error_db,
                best_rmse_db=best.rmse_db,
                best_p95_abs_error_db=best.p95_abs_error_db,
                tau_1s_mean_abs_error_db=(
                    tau_1s.mean_abs_error_db if tau_1s is not None else np.nan
                ),
                tau_2s_mean_abs_error_db=(
                    tau_2s.mean_abs_error_db if tau_2s is not None else np.nan
                ),
            )
        )

    write_block_window_summary_csv(
        block_summaries,
        PROOF_DIR / "block_window_summary.csv",
    )
    write_tau_by_window_csv(tau_summaries, PROOF_DIR / "tau_by_window_summary.csv")
    return block_summaries, tau_summaries


def plot_results(
    results: list[TauResult],
    times: np.ndarray,
    reference_db: np.ndarray,
    best_tau: float,
    best_candidate_db: np.ndarray,
) -> None:
    """Create proof plots."""
    tau = np.array([r.peak_tau_seconds for r in results], dtype=np.float64)
    mae = np.array([r.mean_abs_error_db for r in results], dtype=np.float64)
    rmse = np.array([r.rmse_db for r in results], dtype=np.float64)
    p95 = np.array([r.p95_abs_error_db for r in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(tau, mae, "o-", label="Mean absolute error")
    ax.semilogx(tau, rmse, "s-", label="RMSE")
    ax.semilogx(tau, p95, "^-", label="P95 absolute error")
    ax.axvline(best_tau, color="black", linestyle="--", label=f"Best tau = {best_tau:g}s")
    ax.set_xlabel("Peak-hold tau (seconds)")
    ax.set_ylabel(f"Error vs {_format_window_label(WINDOW_SECONDS)} block reference (dB)")
    ax.set_title("Peak-hold tau sweep")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "tau_sweep_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    count = min(times.size, reference_db.size, best_candidate_db.size)
    ax.plot(
        times[:count],
        reference_db[:count],
        label=f"{_format_window_label(WINDOW_SECONDS)} block reference",
        linewidth=2,
    )
    ax.plot(
        times[:count],
        best_candidate_db[:count],
        label=f"Slow RMS + peak hold tau {best_tau:g}s",
        linewidth=2,
        alpha=0.85,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Crest factor (dB)")
    ax.set_title("Best tau crest-factor trace")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "best_tau_trace.png", dpi=160)
    plt.close(fig)


def plot_block_window_study(
    block_summaries: list[BlockWindowSummary],
    tau_summaries: list[TauWindowSummary],
) -> None:
    """Create plots for block length and best tau behaviour."""
    windows = np.array([row.window_seconds for row in block_summaries], dtype=np.float64)
    mean_cf = np.array([row.mean_cf_db for row in block_summaries], dtype=np.float64)
    p95_cf = np.array([row.p95_cf_db for row in block_summaries], dtype=np.float64)
    max_cf = np.array([row.max_cf_db for row in block_summaries], dtype=np.float64)
    std_cf = np.array([row.std_cf_db for row in block_summaries], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(windows, mean_cf, "o-", label="Mean")
    ax.semilogx(windows, p95_cf, "s-", label="P95")
    ax.semilogx(windows, max_cf, "^-", label="Max")
    ax.semilogx(windows, std_cf, "d-", label="Std dev")
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.6, label="1s IEC Slow anchor")
    ax.set_xlabel("Fixed block length (seconds)")
    ax.set_ylabel("Crest factor distribution (dB)")
    ax.set_title("Effect of block length on fixed-block crest factor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "block_window_effect.png", dpi=160)
    plt.close(fig)

    tau_windows = np.array([row.window_seconds for row in tau_summaries], dtype=np.float64)
    best_tau = np.array([row.best_peak_tau_seconds for row in tau_summaries], dtype=np.float64)
    best_mae = np.array([row.best_mean_abs_error_db for row in tau_summaries], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.semilogx(tau_windows, best_tau, "o-", color="tab:blue", label="Best peak-hold tau")
    ax1.axhline(1.0, color="tab:blue", linestyle="--", alpha=0.5, label="1s RMS tau")
    ax1.set_xlabel("Fixed block reference length (seconds)")
    ax1.set_ylabel("Best peak-hold tau (seconds)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.semilogx(tau_windows, best_mae, "s-", color="tab:red", label="Best MAE")
    ax2.set_ylabel("Best mean absolute error (dB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    ax1.set_title("Best peak-hold tau vs fixed-block reference length")
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "best_tau_by_block_window.png", dpi=160)
    plt.close(fig)


def plot_block_window_traces(signal: np.ndarray, sample_rate: int) -> None:
    """Plot representative fixed-block crest-factor traces."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for window_seconds in [0.05, 0.5, 1.0, 2.0, 5.0]:
        times, crest_db = block_reference(signal, sample_rate, window_seconds)
        ax.plot(
            times,
            crest_db,
            marker="o" if window_seconds >= 1.0 else None,
            linewidth=1.5 if window_seconds >= 1.0 else 1.0,
            alpha=0.85,
            label=f"{_format_window_label(window_seconds)} block",
        )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Fixed-block crest factor (dB)")
    ax.set_title("Crest-factor time variation at different block lengths")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "block_window_traces.png", dpi=160)
    plt.close(fig)


def _result_for_tau(results: list[TauResult], peak_tau: float) -> TauResult | None:
    """Return the result for a candidate tau if present."""
    for result in results:
        if abs(result.peak_tau_seconds - peak_tau) < 1e-9:
            return result
    return None


def _format_window_label(seconds: float) -> str:
    """Format a block length for Markdown."""
    if seconds >= 1.0:
        return f"{seconds:g}s"
    return f"{seconds * 1000:g}ms"


def write_summary(
    best: TauResult,
    results: list[TauResult],
    block_summaries: list[BlockWindowSummary],
    tau_summaries: list[TauWindowSummary],
) -> None:
    """Write a human-readable proof summary."""
    one_second = _result_for_tau(results, 1.0)
    two_seconds = _result_for_tau(results, 2.0)
    comparison_lines: list[str] = []
    if one_second is not None:
        comparison_lines.append(
            f"- 1.0 second peak tau: MAE {one_second.mean_abs_error_db:.3f} dB, "
            f"RMSE {one_second.rmse_db:.3f} dB"
        )
    if two_seconds is not None:
        comparison_lines.append(
            f"- 2.0 second peak tau: MAE {two_seconds.mean_abs_error_db:.3f} dB, "
            f"RMSE {two_seconds.rmse_db:.3f} dB"
        )
    comparison_text = "\n".join(comparison_lines)
    shortest = block_summaries[0]
    one_second_summary = min(
        block_summaries,
        key=lambda row: abs(row.window_seconds - 1.0),
    )
    whole_track = block_summaries[-1]
    best_tau_rows = tau_summaries
    best_tau_table = "\n".join(
        (
            "| Window | Best peak tau | Best MAE | 1s tau MAE | 2s tau MAE |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                "| "
                f"{_format_window_label(row.window_seconds)} | "
                f"{row.best_peak_tau_seconds:g}s | "
                f"{row.best_mean_abs_error_db:.3f} | "
                f"{row.tau_1s_mean_abs_error_db:.3f} | "
                f"{row.tau_2s_mean_abs_error_db:.3f} |"
                for row in best_tau_rows
            ],
        )
    )

    summary = f"""# Peak-Hold Tau and Block Window Study

This study explores two linked crest-factor questions:

1. How fixed-block window length changes measured crest factor.
2. Which peak-hold time constant best aligns the slow-mode crest-factor
   algorithm with a fixed-block reference while holding the RMS tau at the IEC
   Slow value of 1.0 second.

## Method

- Source material: `source_material/peak_hold_tau_test.wav`
- Sample rate: {SAMPLE_RATE} Hz
- Slow RMS tau: {RMS_TAU_SECONDS:.1f} second
- Primary tau reference: non-overlapping {_format_window_label(WINDOW_SECONDS)} block peak / RMS crest factor
- Candidate peak-hold tau values: {", ".join(str(x) for x in PEAK_TAU_CANDIDATES_SECONDS)}
- Fixed block windows studied: {", ".join(_format_window_label(x.window_seconds) for x in block_summaries)}
- Warm-up excluded from scoring: {WARMUP_SECONDS:.1f} seconds
- Programme noise bed: {PROGRAM_NOISE_FLOOR_DBFS:.0f} dBFS pink noise
- Final true-silence edge case: {FINAL_TRUE_SILENCE_SECONDS:.0f} seconds

The generated signal contains level-stepped click trains, steady-state sine
waves at multiple levels, low-frequency sine waves, and pink-noise bursts of
different durations and levels. A low-level pink-noise bed keeps programme gaps
from becoming exact digital zero, while the final true-silence section remains
as a separate edge case. This stresses fast attack behaviour, missed peak
response, stable crest factor, RMS settling, and invalid/absent signal handling
without relying on copyrighted programme material.

## Result

### {_format_window_label(WINDOW_SECONDS)} reference tau sweep

Best peak-hold tau by mean absolute error against the {_format_window_label(WINDOW_SECONDS)} fixed-block
reference:

- Peak-hold tau: **{best.peak_tau_seconds:g} seconds**
- Mean error: {best.mean_error_db:.3f} dB
- Mean absolute error: {best.mean_abs_error_db:.3f} dB
- RMSE: {best.rmse_db:.3f} dB
- P95 absolute error: {best.p95_abs_error_db:.3f} dB
- Max absolute error: {best.max_abs_error_db:.3f} dB
- Windows within +/-0.5 dB: {best.within_0_5_db_percent:.1f}%
- Windows within +/-1.0 dB: {best.within_1_0_db_percent:.1f}%

For comparison:

{comparison_text}

### Detector design rationale

The RMS detector and peak detector intentionally do different jobs. The IEC Slow
RMS detector squares the signal, exponentially averages mean-square energy with
`slow_rms_tau=1.0`, and then converts back to RMS. This is the part of the
analysis anchored to sound-level-meter behaviour.

The peak detector is not another RMS detector with a different tau. It takes the
absolute sample value, attacks immediately when a new sample exceeds the held
value, then releases with exponential decay. This preserves short peak events
that would otherwise be diluted by an energy average, which is essential for
crest factor because crest factor compares peak demand against RMS energy.

That split is the right shape for this function: RMS should represent sustained
energy, while peak should represent recent maximum excursion. The tau sweep in
this proof is therefore not trying to make the peak detector behave like RMS; it
is finding the peak-release time that gives a useful short-term crest-factor
trace when RMS remains anchored to IEC Slow.

### Block window effect

The fixed-block window length changes both the value and usefulness of the crest
factor measurement:

- Shortest block tested ({_format_window_label(shortest.window_seconds)}):
  mean crest factor {shortest.mean_cf_db:.3f} dB, P95 {shortest.p95_cf_db:.3f} dB,
  max {shortest.max_cf_db:.3f} dB.
- 1-second block: mean crest factor {one_second_summary.mean_cf_db:.3f} dB,
  P95 {one_second_summary.p95_cf_db:.3f} dB,
  max {one_second_summary.max_cf_db:.3f} dB.
- Whole-track block ({_format_window_label(whole_track.window_seconds)}):
  crest factor {whole_track.mean_cf_db:.3f} dB. This is a broad single-number
  description and contains no time variation.

Very short windows move toward local instantaneous behaviour: the 5-100ms blocks
measure much lower average crest factor because the RMS and peak are being taken
over nearly the same tiny event. In the limit, a one-sample block has peak equal
to RMS and crest factor becomes 0 dB. For this proof material, the smallest block
that gives a stable and meaningful short-term crest-factor result is about 2
seconds. Larger blocks give broadly similar crest-factor conclusions, but average
them over more time and therefore reduce time resolution. Very long windows
describe the track broadly but hide short-term changes.

### Best tau by block reference

{best_tau_table}

### Practical conclusion

For short-term crest-factor analysis, keep RMS anchored to IEC Slow with
`slow_rms_tau=1.0`. Treat whole-track crest factor as a useful headline number
only, because it intentionally removes time variation. Avoid very small block
windows for programme-dynamics reporting: below roughly 100ms, the measurement
is dominated by local instantaneous behaviour and trends toward 0 dB as the
window approaches one sample.

For this proof material, the minimum stable fixed-block crest-factor window is
about 2 seconds. Smaller windows increasingly collapse toward local
instantaneous behaviour and can understate crest factor; larger windows remain
stable but smear the result over more programme time. The useful engineering
choice is therefore a block/window length long enough to avoid collapse, but not
so long that it hides dynamic changes.

The primary reference is now the {_format_window_label(WINDOW_SECONDS)} block
because the 1-second reference was too vulnerable to exact-zero gaps and local
collapse. The result should be read as a stability check for the slow metering
trace, not as a replacement for fixed-window crest factor when reporting
programme dynamics.

## Outputs

- `tau_sweep_results.csv`: numeric results for every candidate tau
- `tau_sweep_error.png`: error metrics vs peak-hold tau
- `best_tau_trace.png`: crest-factor trace for the best tau against reference
- `block_window_summary.csv`: crest-factor distribution vs fixed block length
- `block_window_effect.png`: visual summary of block length effect
- `block_window_traces.png`: representative crest-factor traces by block length
- `tau_by_window_summary.csv`: best peak-hold tau for each block length
- `best_tau_by_block_window.png`: best tau and error vs block length

## Interpretation

The fixed-block method is treated as the reference because it directly computes
maximum absolute sample divided by RMS inside each window. Slow-mode crest factor
is a metering-style method: RMS follows the IEC Slow 1-second exponential
behaviour, while peak hold has instantaneous attack and configurable exponential
decay. The best tau is therefore the tau that makes that metering-style trace
closest to the fixed-window reference for this proof material.

This result should not be treated as a universal standard. It is evidence for
this stress-test material and scoring method. Clicks, steady tones, and short
noise bursts stress the peak-hold and RMS envelopes differently, so the optimum
can move depending on source material, chosen block reference, and whether mean
error, mean absolute error, RMSE, or percentile error is considered the primary
criterion.

The IEC Slow RMS tau remains important because it anchors the RMS side of the
analysis to a real metering behaviour. The block-length study is about selecting
a time scale that gives useful short-term insight, not replacing that RMS
anchor.
"""
    (PROOF_DIR / "README.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the proof."""
    source_dir = PROOF_DIR / "source_material"
    source_dir.mkdir(parents=True, exist_ok=True)

    signal = generate_test_signal(SAMPLE_RATE)
    wav_path = source_dir / "peak_hold_tau_test.wav"
    sf.write(wav_path, signal, SAMPLE_RATE, subtype="PCM_24")

    times, reference_db = block_reference(signal, SAMPLE_RATE)
    block_summaries, tau_summaries = run_block_window_study(signal, SAMPLE_RATE)

    results: list[TauResult] = []
    candidates_by_tau: dict[float, np.ndarray] = {}
    rms_env = slow_rms_envelope(signal, SAMPLE_RATE, tau=RMS_TAU_SECONDS)
    for peak_tau in PEAK_TAU_CANDIDATES_SECONDS:
        peak_env = peak_hold_envelope(signal, SAMPLE_RATE, tau=peak_tau)
        _, candidate_db = slow_crest_db_for_tau(
            signal,
            SAMPLE_RATE,
            peak_tau,
            rms_env=rms_env,
            peak_env=peak_env,
        )
        candidates_by_tau[peak_tau] = candidate_db
        results.append(compare_tau(reference_db, candidate_db, peak_tau))

    results.sort(key=lambda row: (row.mean_abs_error_db, row.rmse_db))
    best = results[0]
    write_results_csv(results, PROOF_DIR / "tau_sweep_results.csv")
    plot_results(results, times, reference_db, best.peak_tau_seconds, candidates_by_tau[best.peak_tau_seconds])
    plot_block_window_study(block_summaries, tau_summaries)
    plot_block_window_traces(signal, SAMPLE_RATE)
    write_summary(best, results, block_summaries, tau_summaries)

    print(f"Wrote proof outputs to: {PROOF_DIR}")
    print(f"Best peak-hold tau: {best.peak_tau_seconds:g}s")
    print(f"Mean absolute error: {best.mean_abs_error_db:.3f} dB")
    print(f"RMSE: {best.rmse_db:.3f} dB")


if __name__ == "__main__":
    main()
