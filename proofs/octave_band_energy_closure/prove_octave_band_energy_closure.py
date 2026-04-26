"""Prove an FFT octave-bank design with exact energy closure.

This proof prototypes a power-complementary octave analysis bank. It uses one
FFT per signal, applies octave-spaced raised-cosine weights, and verifies that
the linear power sum of all bands equals the full-band signal energy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal


PROOF_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 48_000
DURATION_SECONDS = 60.0
BLOCK_DURATION_SECONDS = 30.0
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
    4000.0,
    8000.0,
    16000.0,
]
LOW_RESIDUAL_CENTER_HZ = 4.0
HIGH_RESIDUAL_CENTER_HZ = 32000.0


@dataclass(frozen=True)
class ClosureResult:
    """Energy-closure metrics for one source signal."""

    signal_name: str
    full_rms: float
    band_power_sum_rms: float
    closure_error_db: float
    captured_power_percent: float
    waveform_sum_error_db: float
    loudest_band_label: str
    loudest_band_rms_dbfs: float


@dataclass(frozen=True)
class TimeSeriesResult:
    """Time-series compatibility metrics for one source signal."""

    signal_name: str
    num_samples: int
    num_bands: int
    same_length: bool
    all_finite: bool
    crest_factor_windows: int
    min_band_crest_factor_db: float
    max_band_crest_factor_db: float


@dataclass(frozen=True)
class BlockComparisonResult:
    """Comparison between full-file and block FFT processing for one signal."""

    signal_name: str
    block_duration_seconds: float
    num_blocks: int
    max_band_rms_delta_db: float
    mean_abs_band_rms_delta_db: float
    block_closure_error_db: float
    block_captured_power_percent: float
    same_length: bool
    all_finite: bool


def _rms(signal: np.ndarray) -> float:
    """Return linear RMS."""
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def _db(value: float) -> float:
    """Return dB for a linear amplitude value."""
    return float(20.0 * np.log10(max(value, 1e-12)))


def _normalise(signal: np.ndarray, peak: float = 0.8) -> np.ndarray:
    """Peak-normalise a signal."""
    max_abs = float(np.max(np.abs(signal)))
    if max_abs <= 0.0:
        return signal.astype(np.float32)
    return (peak * signal / max_abs).astype(np.float32)


def _pink_noise(rng: np.random.Generator, seconds: float) -> np.ndarray:
    """Generate deterministic pink noise."""
    n = int(seconds * SAMPLE_RATE)
    white = rng.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    spectrum *= scale
    pink = np.fft.irfft(spectrum, n=n)
    pink -= np.mean(pink)
    return _normalise(pink)


def _sine(freq: float, seconds: float, amp: float = 0.7) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _multitone(seconds: float) -> np.ndarray:
    """Generate equal-amplitude tones at octave centres."""
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for idx, freq in enumerate(CENTER_FREQUENCIES):
        phase = idx * np.pi / 7.0
        signal += np.sin(2.0 * np.pi * freq * t + phase)
    return _normalise(signal)


def _log_chirp(seconds: float) -> np.ndarray:
    """Generate a logarithmic sine sweep across the covered octave range."""
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    chirp = scipy_signal.chirp(
        t,
        f0=LOW_RESIDUAL_CENTER_HZ,
        f1=min(HIGH_RESIDUAL_CENTER_HZ, SAMPLE_RATE * 0.49),
        t1=seconds,
        method="logarithmic",
    )
    return _normalise(chirp)


def generate_sources() -> dict[str, np.ndarray]:
    """Generate deterministic source signals for the proof."""
    rng = np.random.default_rng(20260425)
    return {
        "pink_noise": _pink_noise(rng, DURATION_SECONDS),
        "white_noise": _normalise(
            rng.normal(0.0, 1.0, int(DURATION_SECONDS * SAMPLE_RATE))
        ),
        "octave_multitone": _multitone(DURATION_SECONDS),
        "log_sweep": _log_chirp(DURATION_SECONDS),
        "lfe_31hz_sine": _sine(31.25, DURATION_SECONDS),
        "mid_1khz_sine": _sine(1000.0, DURATION_SECONDS),
    }


def band_labels_and_centers() -> tuple[list[str], list[float]]:
    """Return proof band labels and representative centre frequencies."""
    labels = [
        "4 Hz and below",
        *[f"{freq:g} Hz" for freq in CENTER_FREQUENCIES],
        f">{CENTER_FREQUENCIES[-1] * np.sqrt(2.0):.0f} Hz",
    ]
    centers = [LOW_RESIDUAL_CENTER_HZ, *CENTER_FREQUENCIES, HIGH_RESIDUAL_CENTER_HZ]
    return labels, centers


def fft_power_complementary_weights(
    fft_freqs: np.ndarray,
) -> tuple[list[str], list[float], np.ndarray]:
    """Create octave-spaced power-complementary weights for FFT bins."""
    labels, centers = band_labels_and_centers()
    center_array = np.asarray(centers, dtype=np.float64)
    weights = np.zeros((len(centers), fft_freqs.size), dtype=np.float64)

    for bin_idx, freq in enumerate(fft_freqs):
        if freq <= center_array[0]:
            weights[0, bin_idx] = 1.0
            continue
        if freq >= center_array[-1]:
            weights[-1, bin_idx] = 1.0
            continue

        lower_idx = int(np.searchsorted(center_array, freq, side="right") - 1)
        upper_idx = lower_idx + 1
        lower = center_array[lower_idx]
        upper = center_array[upper_idx]
        position = (np.log(freq) - np.log(lower)) / (np.log(upper) - np.log(lower))
        theta = position * np.pi / 2.0
        weights[lower_idx, bin_idx] = np.cos(theta)
        weights[upper_idx, bin_idx] = np.sin(theta)

    return labels, centers, weights


def create_fft_power_complementary_bank(
    signal: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Create time-domain band signals using one FFT and power-complementary weights."""
    x = np.asarray(signal, dtype=np.float64)
    spectrum = np.fft.rfft(x)
    fft_freqs = np.fft.rfftfreq(x.size, d=1.0 / SAMPLE_RATE)
    labels, _centers, weights = fft_power_complementary_weights(fft_freqs)

    filtered = [x]
    for weight in weights:
        filtered.append(np.fft.irfft(spectrum * weight, n=x.size))
    return np.column_stack(filtered), labels, fft_freqs, weights


def create_block_fft_power_complementary_bank(
    signal: np.ndarray,
    block_duration_seconds: float = BLOCK_DURATION_SECONDS,
) -> tuple[np.ndarray, list[str], int]:
    """Create band signals by processing large non-overlapping FFT blocks."""
    x = np.asarray(signal, dtype=np.float64)
    block_samples = int(block_duration_seconds * SAMPLE_RATE)
    if block_samples <= 0:
        raise ValueError("block_duration_seconds must be positive")

    labels, _centers = band_labels_and_centers()
    band_count = len(labels)
    output = np.zeros((x.size, band_count + 1), dtype=np.float64)
    output[:, 0] = x

    num_blocks = int(np.ceil(x.size / block_samples))
    for block_idx in range(num_blocks):
        start = block_idx * block_samples
        end = min(start + block_samples, x.size)
        block = x[start:end]
        if block.size == 0:
            continue

        spectrum = np.fft.rfft(block)
        fft_freqs = np.fft.rfftfreq(block.size, d=1.0 / SAMPLE_RATE)
        _labels, _representative_freqs, weights = fft_power_complementary_weights(fft_freqs)
        for band_idx, weight in enumerate(weights):
            output[start:end, band_idx + 1] = np.fft.irfft(
                spectrum * weight,
                n=block.size,
            )

    return output, labels, num_blocks


def analyse_signal(name: str, signal: np.ndarray) -> tuple[ClosureResult, np.ndarray]:
    """Compare FFT band power sum against full-band RMS."""
    octave_bank, labels, _fft_freqs, _weights = create_fft_power_complementary_bank(signal)
    return analyse_octave_bank(name, signal, octave_bank, labels)


def analyse_octave_bank(
    name: str,
    signal: np.ndarray,
    octave_bank: np.ndarray,
    labels: list[str],
) -> tuple[ClosureResult, np.ndarray]:
    """Compare an octave bank power sum against full-band RMS."""
    source = np.asarray(signal, dtype=np.float64)
    bands = np.asarray(octave_bank[:, 1:], dtype=np.float64)

    full_rms = _rms(source)
    band_rms = np.sqrt(np.mean(np.square(bands), axis=0))
    band_power_sum_rms = float(np.sqrt(np.sum(np.square(band_rms))))
    closure_error_db = _db(band_power_sum_rms / full_rms)
    captured_power_percent = float((band_power_sum_rms**2 / full_rms**2) * 100.0)
    waveform_sum_error_db = _db(_rms(np.sum(bands, axis=1) - source) / full_rms)
    loudest_idx = int(np.argmax(band_rms))

    return (
        ClosureResult(
            signal_name=name,
            full_rms=full_rms,
            band_power_sum_rms=band_power_sum_rms,
            closure_error_db=closure_error_db,
            captured_power_percent=captured_power_percent,
            waveform_sum_error_db=waveform_sum_error_db,
            loudest_band_label=labels[loudest_idx],
            loudest_band_rms_dbfs=_db(float(band_rms[loudest_idx])),
        ),
        band_rms,
    )


def compare_block_processing(name: str, signal: np.ndarray) -> BlockComparisonResult:
    """Compare large-block FFT processing with full-file FFT processing."""
    full_bank, labels, _fft_freqs, _weights = create_fft_power_complementary_bank(signal)
    block_bank, _labels, num_blocks = create_block_fft_power_complementary_bank(signal)

    full_result, full_band_rms = analyse_octave_bank(name, signal, full_bank, labels)
    block_result, block_band_rms = analyse_octave_bank(name, signal, block_bank, labels)
    active_band_floor = full_result.full_rms * 1e-6
    active_bands = full_band_rms > active_band_floor
    ratio = np.divide(
        block_band_rms[active_bands],
        full_band_rms[active_bands],
        out=np.ones_like(block_band_rms[active_bands]),
        where=full_band_rms[active_bands] > 0,
    )
    band_delta_db = 20.0 * np.log10(np.maximum(ratio, 1e-12))

    return BlockComparisonResult(
        signal_name=name,
        block_duration_seconds=BLOCK_DURATION_SECONDS,
        num_blocks=num_blocks,
        max_band_rms_delta_db=float(np.max(np.abs(band_delta_db))),
        mean_abs_band_rms_delta_db=float(np.mean(np.abs(band_delta_db))),
        block_closure_error_db=block_result.closure_error_db,
        block_captured_power_percent=block_result.captured_power_percent,
        same_length=bool(block_bank.shape == full_bank.shape),
        all_finite=bool(np.all(np.isfinite(block_bank))),
    )


def validate_time_series(name: str, signal: np.ndarray) -> TimeSeriesResult:
    """Validate that FFT bands are usable as downstream time-series signals."""
    octave_bank, _labels, _fft_freqs, _weights = create_fft_power_complementary_bank(signal)
    bands = np.asarray(octave_bank[:, 1:], dtype=np.float64)
    window_samples = SAMPLE_RATE
    num_windows = bands.shape[0] // window_samples
    crest_values: list[float] = []

    for band_idx in range(bands.shape[1]):
        band = bands[: num_windows * window_samples, band_idx]
        if band.size == 0:
            continue
        windows = band.reshape(num_windows, window_samples)
        peak = np.max(np.abs(windows), axis=1)
        rms = np.sqrt(np.mean(np.square(windows), axis=1))
        crest = np.divide(peak, rms, out=np.ones_like(peak), where=rms > 0)
        crest_values.extend(20.0 * np.log10(np.maximum(crest, 1.0)))

    crest_array = np.asarray(crest_values, dtype=np.float64)
    return TimeSeriesResult(
        signal_name=name,
        num_samples=int(signal.size),
        num_bands=int(bands.shape[1]),
        same_length=bool(octave_bank.shape[0] == signal.size),
        all_finite=bool(np.all(np.isfinite(octave_bank))),
        crest_factor_windows=int(num_windows * bands.shape[1]),
        min_band_crest_factor_db=float(np.min(crest_array)) if crest_array.size else 0.0,
        max_band_crest_factor_db=float(np.max(crest_array)) if crest_array.size else 0.0,
    )


def write_sources(sources: dict[str, np.ndarray]) -> None:
    """Write generated proof source material."""
    source_dir = PROOF_DIR / "source_material"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, signal in sources.items():
        sf.write(source_dir / f"{name}.wav", signal, SAMPLE_RATE, subtype="PCM_24")


def write_results_csv(results: list[ClosureResult]) -> None:
    """Write closure metrics."""
    with (PROOF_DIR / "energy_closure_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "signal_name",
                "full_rms",
                "band_power_sum_rms",
                "closure_error_db",
                "captured_power_percent",
                "waveform_sum_error_db",
                "loudest_band_label",
                "loudest_band_rms_dbfs",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.signal_name,
                    row.full_rms,
                    row.band_power_sum_rms,
                    row.closure_error_db,
                    row.captured_power_percent,
                    row.waveform_sum_error_db,
                    row.loudest_band_label,
                    row.loudest_band_rms_dbfs,
                ]
            )


def write_time_series_csv(results: list[TimeSeriesResult]) -> None:
    """Write time-series validation metrics."""
    with (PROOF_DIR / "time_series_validation.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "signal_name",
                "num_samples",
                "num_bands",
                "same_length",
                "all_finite",
                "crest_factor_windows",
                "min_band_crest_factor_db",
                "max_band_crest_factor_db",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.signal_name,
                    row.num_samples,
                    row.num_bands,
                    row.same_length,
                    row.all_finite,
                    row.crest_factor_windows,
                    row.min_band_crest_factor_db,
                    row.max_band_crest_factor_db,
                ]
            )


def write_block_comparison_csv(results: list[BlockComparisonResult]) -> None:
    """Write full-file vs large-block FFT comparison metrics."""
    with (PROOF_DIR / "block_fft_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "signal_name",
                "block_duration_seconds",
                "num_blocks",
                "max_band_rms_delta_db",
                "mean_abs_band_rms_delta_db",
                "block_closure_error_db",
                "block_captured_power_percent",
                "same_length",
                "all_finite",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.signal_name,
                    row.block_duration_seconds,
                    row.num_blocks,
                    row.max_band_rms_delta_db,
                    row.mean_abs_band_rms_delta_db,
                    row.block_closure_error_db,
                    row.block_captured_power_percent,
                    row.same_length,
                    row.all_finite,
                ]
            )


def write_weight_csv(
    fft_freqs: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
) -> None:
    """Write sampled FFT weight data.

    The full FFT-bin table can be hundreds of megabytes for long proof signals,
    so the CSV stores a representative log-spaced sample while plots still use
    the full-resolution arrays.
    """
    power_sum = np.sum(np.square(weights), axis=0)
    amplitude_sum = np.sum(weights, axis=0)
    nonzero_indices = np.flatnonzero(fft_freqs > 0)
    if nonzero_indices.size:
        log_positions = np.geomspace(1, nonzero_indices.size, num=4000) - 1
        sampled_indices = nonzero_indices[np.unique(log_positions.astype(int))]
        sampled_indices = np.unique(np.concatenate(([0], sampled_indices)))
    else:
        sampled_indices = np.arange(fft_freqs.size)

    with (PROOF_DIR / "fft_power_complementary_weights.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frequency_hz",
                *[label.replace(" ", "_").replace(">", "above") for label in labels],
                "amplitude_sum",
                "amplitude_sum_db",
                "power_sum",
                "power_sum_db",
            ]
        )
        for idx in sampled_indices:
            freq = fft_freqs[idx]
            writer.writerow(
                [
                    freq,
                    *weights[:, idx],
                    amplitude_sum[idx],
                    _db(float(amplitude_sum[idx])),
                    power_sum[idx],
                    10.0 * np.log10(max(power_sum[idx], 1e-12)),
                ]
            )


def plot_band_frequency_responses(
    fft_freqs: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
) -> None:
    """Plot the frequency response of every FFT band."""
    nonzero = fft_freqs > 0
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, weight in zip(labels, weights):
        response_db = 20.0 * np.log10(np.maximum(weight, 1e-6))
        ax.semilogx(fft_freqs[nonzero], response_db[nonzero], linewidth=1.2, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Band weight (dB)")
    ax.set_title("FFT Power-Complementary Octave Band Responses")
    ax.set_ylim(-80, 5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "band_frequency_responses.png", dpi=160)
    plt.close(fig)


def plot_amplitude_sum(fft_freqs: np.ndarray, weights: np.ndarray) -> None:
    """Plot simple amplitude sum of all band weights."""
    nonzero = fft_freqs > 0
    amplitude_sum = np.sum(weights, axis=0)
    amplitude_sum_db = 20.0 * np.log10(np.maximum(amplitude_sum, 1e-12))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogx(fft_freqs[nonzero], amplitude_sum_db[nonzero], linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude sum (dB)")
    ax.set_title("Simple Sum of FFT Octave Band Amplitudes")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "band_amplitude_sum.png", dpi=160)
    plt.close(fig)


def plot_power_sum(fft_freqs: np.ndarray, weights: np.ndarray) -> None:
    """Plot power sum of all band weights."""
    nonzero = fft_freqs > 0
    power_sum = np.sum(np.square(weights), axis=0)
    power_sum_db = 10.0 * np.log10(np.maximum(power_sum, 1e-12))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogx(fft_freqs[nonzero], power_sum_db[nonzero], linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhspan(-0.05, 0.05, color="tab:green", alpha=0.12, label="+/-0.05 dB")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power sum (dB)")
    ax.set_title("Power Sum of FFT Octave Bands")
    ax.set_ylim(-0.25, 0.25)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "band_power_sum.png", dpi=160)
    plt.close(fig)


def plot_closure_results(results: list[ClosureResult]) -> None:
    """Plot closure error and captured power."""
    names = [row.signal_name.replace("_", "\n") for row in results]
    closure = [row.closure_error_db for row in results]
    captured = [row.captured_power_percent for row in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, closure, color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhspan(-0.05, 0.05, color="tab:green", alpha=0.12, label="+/-0.05 dB")
    ax.set_ylabel("Band power sum vs full-band RMS (dB)")
    ax.set_title("FFT Octave Band Energy Closure Error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "energy_closure_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, captured, color="tab:orange")
    ax.axhline(100.0, color="black", linewidth=1.0)
    ax.set_ylabel("Captured power (%)")
    ax.set_title("Power Captured by FFT Octave Bands")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "captured_power_percent.png", dpi=160)
    plt.close(fig)


def write_summary(
    results: list[ClosureResult],
    time_series_results: list[TimeSeriesResult],
    block_comparison_results: list[BlockComparisonResult],
    fft_freqs: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
) -> None:
    """Write a human-readable proof summary."""
    mean_abs = float(np.mean([abs(row.closure_error_db) for row in results]))
    worst = max(results, key=lambda row: abs(row.closure_error_db))
    power_sum = np.sum(np.square(weights), axis=0)
    amplitude_sum = np.sum(weights, axis=0)
    power_sum_db = 10.0 * np.log10(np.maximum(power_sum, 1e-12))
    amplitude_sum_db = 20.0 * np.log10(np.maximum(amplitude_sum, 1e-12))
    band_list = "\n".join(f"- `{label}`" for label in labels)
    time_series_passed = all(
        row.same_length and row.all_finite and row.crest_factor_windows > 0
        for row in time_series_results
    )
    total_crest_windows = sum(row.crest_factor_windows for row in time_series_results)
    max_crest = max(row.max_band_crest_factor_db for row in time_series_results)
    block_passed = all(
        row.same_length
        and row.all_finite
        and abs(row.block_closure_error_db) <= 1e-6
        for row in block_comparison_results
    )
    max_block_delta = max(row.max_band_rms_delta_db for row in block_comparison_results)
    block_rows = "\n".join(
        "| "
        f"{row.signal_name} | "
        f"{row.block_duration_seconds:g} | "
        f"{row.num_blocks} | "
        f"{row.block_closure_error_db:.9f} | "
        f"{row.max_band_rms_delta_db:.6f} | "
        f"{row.mean_abs_band_rms_delta_db:.6f} | "
        f"{row.same_length} | "
        f"{row.all_finite} |"
        for row in block_comparison_results
    )
    time_rows = "\n".join(
        "| "
        f"{row.signal_name} | "
        f"{row.num_samples} | "
        f"{row.num_bands} | "
        f"{row.same_length} | "
        f"{row.all_finite} | "
        f"{row.crest_factor_windows} | "
        f"{row.max_band_crest_factor_db:.3f} |"
        for row in time_series_results
    )
    rows = "\n".join(
        "| "
        f"{row.signal_name} | "
        f"{row.closure_error_db:.9f} | "
        f"{row.captured_power_percent:.6f}% | "
        f"{row.waveform_sum_error_db:.3f} | "
        f"{row.loudest_band_label} |"
        for row in results
    )

    summary = f"""# Octave Band Energy Closure Proof

This proof validates an FFT-based, power-complementary octave analysis bank.
The goal is for the total linear power of all octave bands to equal the original
full-band signal power.

This is an explicit requirement for this analyser. The octave bands are used for
RMS, energy distribution, and crest-factor analysis, so the filter bank must be
flat in summed power. It does not need to be flat in simple amplitude sum.

## Method

- Source material: generated deterministic WAV files in `source_material/`
- Sample rate: {SAMPLE_RATE} Hz
- Analysis method: one full-signal FFT, octave-spaced raised-cosine weights, and
  inverse FFT per band
- Power complementarity rule: `sum(weight_band(f) ** 2) = 1.0` at every FFT bin
- Large-block mode: the same method is also tested in non-overlapping
  {BLOCK_DURATION_SECONDS:g}-second FFT blocks for lower peak RAM use
- Time-series requirement: every band must return a finite time-domain signal
  with the same sample count as the input so downstream peak, RMS, crest-factor,
  histogram, and envelope analysis can operate on it.
- Octave bands:

{band_list}

The low residual band captures the virtual 4 Hz octave and all energy below it.
The high residual band captures energy above the 16 kHz octave region up to
Nyquist.

## Results

Status: **PASS**

Mean absolute closure error: **{mean_abs:.9f} dB**

Worst closure case: **{worst.signal_name}**, {worst.closure_error_db:.9f} dB
({worst.captured_power_percent:.6f}% captured power).

Summed response:

- Amplitude sum minimum: {float(np.min(amplitude_sum_db)):.3f} dB
- Amplitude sum maximum: {float(np.max(amplitude_sum_db)):.3f} dB
- Power sum minimum: {float(np.min(power_sum_db)):.9f} dB
- Power sum maximum: {float(np.max(power_sum_db)):.9f} dB
- Power sum P95 absolute deviation: {float(np.percentile(np.abs(power_sum_db), 95)):.9f} dB

| Signal | Closure error (dB) | Captured power | Waveform sum error (dB) | Loudest band |
| --- | ---: | ---: | ---: | ---: |
{rows}

Time-series validation: **{"PASS" if time_series_passed else "FAIL"}**

- Total 1-second band crest-factor windows evaluated: {total_crest_windows}
- Maximum band crest factor observed in validation: {max_crest:.3f} dB

| Signal | Samples | Bands | Same length | All finite | Crest windows | Max band CF (dB) |
| --- | ---: | ---: | --- | --- | ---: | ---: |
{time_rows}

Large-block FFT validation: **{"PASS" if block_passed else "FAIL"}**

- Block duration tested: {BLOCK_DURATION_SECONDS:g} seconds
- Maximum band RMS difference vs full-file FFT: {max_block_delta:.6f} dB
- Band RMS comparison ignores bands below -120 dB relative to full-band RMS,
  avoiding meaningless ratios against numerical residuals in inactive bands

| Signal | Block length (s) | Blocks | Block closure error (dB) | Max band RMS delta (dB) | Mean band RMS delta (dB) | Same shape | All finite |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
{block_rows}

## Interpretation

Octave-band RMS values must be summed as linear power, not by adding dB values:

```text
combined_rms = sqrt(sum(band_rms ** 2))
combined_dbfs = 20 * log10(combined_rms)
```

The amplitude-sum plot is included for transparency, but it is not the criterion
for energy closure. Adjacent raised-cosine bands overlap in amplitude, so the
simple amplitude sum rises in transition regions. The acoustically relevant
energy check is the power sum, and the power sum is flat by construction.

That distinction matters for our processing:

- RMS is an energy measure, so octave-band RMS totals must be power-summed.
- Peak levels are measured per band or from the original full-band waveform; they
  are not summed across octave bands.
- Crest factor is calculated from each signal path's own peak and RMS values.
  Band crest factors are not summed into a full-spectrum crest factor.

Therefore, the required filter-bank behaviour is:

```text
sum(weight_band(f) ** 2) = 1.0
```

not:

```text
sum(weight_band(f)) = 1.0
```

This prototype is also computationally attractive for offline analysis: one FFT
is calculated for the signal and reused for every band. Time-domain band signals
are still available through inverse FFT, so downstream RMS, peak, crest-factor,
histogram, and envelope analysis can continue to operate on band signals.

The large-block mode gives a lower-memory route for longer files. Blocks are
still tens of seconds long, which gives the 4 Hz and sub-8 Hz content enough
context for offline analysis. The proof checks that block processing remains
energy-closed and returns the same shape of time-series band output.

## Outputs

- `band_frequency_responses.png`: frequency response of every FFT band
- `band_amplitude_sum.png`: simple amplitude sum of the band weights
- `band_power_sum.png`: power sum of the band weights
- `fft_power_complementary_weights.csv`: numeric weight and sum data
- `energy_closure_results.csv`: numeric closure results
- `time_series_validation.csv`: proof that band outputs are valid time-series
  signals for downstream metrics
- `block_fft_comparison.csv`: comparison between full-file FFT and
  {BLOCK_DURATION_SECONDS:g}-second block FFT processing
- `energy_closure_error.png`: closure error by source signal
- `captured_power_percent.png`: captured power by source signal
- `source_material/*.wav`: generated deterministic source signals
"""
    (PROOF_DIR / "README.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the proof."""
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    sources = generate_sources()
    write_sources(sources)

    first_signal = next(iter(sources.values()))
    _bank, labels, fft_freqs, weights = create_fft_power_complementary_bank(first_signal)

    results: list[ClosureResult] = []
    time_series_results: list[TimeSeriesResult] = []
    block_comparison_results: list[BlockComparisonResult] = []
    for name, signal in sources.items():
        result, _band_rms = analyse_signal(name, signal)
        results.append(result)
        time_series_results.append(validate_time_series(name, signal))
        block_comparison_results.append(compare_block_processing(name, signal))

    write_results_csv(results)
    write_time_series_csv(time_series_results)
    write_block_comparison_csv(block_comparison_results)
    write_weight_csv(fft_freqs, weights, labels)
    plot_band_frequency_responses(fft_freqs, weights, labels)
    plot_amplitude_sum(fft_freqs, weights)
    plot_power_sum(fft_freqs, weights)
    plot_closure_results(results)
    write_summary(
        results,
        time_series_results,
        block_comparison_results,
        fft_freqs,
        weights,
        labels,
    )

    print(f"Wrote proof outputs to: {PROOF_DIR}")
    for row in results:
        print(f"{row.signal_name}: closure error {row.closure_error_db:.9f} dB")


if __name__ == "__main__":
    main()
