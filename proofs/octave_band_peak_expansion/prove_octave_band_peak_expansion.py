"""Prove filtered octave-band peaks can exceed full-band sample peaks.

The Audio Analyser octave bank is designed for power/RMS closure, not for
sample-peak limiting. This proof uses deterministic bounded signals to show
when a filtered band can legitimately exceed the source full-band peak.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter


PROOF_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 48_000
DURATION_SECONDS = 20.0
REAL_CASE_METADATA = (
    REPO_ROOT
    / "analysis"
    / "v0.3.5"
    / "analysis"
    / "Ready Player One (2018) - Race.aaresults"
    / "channels"
    / "channel_01"
    / "metadata.json"
)
REAL_CASE_BAND_HZ = 62.5
REAL_CASE_EVENT_SECONDS = 132.0
REAL_CASE_EVENT_CHUNK_SECONDS = 2.0
REAL_CASE_PLOT_SECONDS = 1.0
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


@dataclass(frozen=True)
class SignalCase:
    """One deterministic source signal for the proof."""

    name: str
    signal: np.ndarray
    note: str


def db(value: float) -> float:
    """Convert a positive linear amplitude to dB."""
    return float(20.0 * np.log10(max(float(value), 1e-20)))


def normalize_peak(signal: np.ndarray) -> np.ndarray:
    """Normalize a signal so the full-band sample peak is exactly 1.0."""
    signal = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return signal / peak if peak > 0 else signal


def pink_noise(sample_count: int, seed: int = 1234) -> np.ndarray:
    """Generate deterministic pink-ish noise for a music-like broadband case."""
    rng = np.random.default_rng(seed)
    white = rng.normal(size=sample_count)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(sample_count, 1.0 / SAMPLE_RATE)
    weight = np.zeros_like(freqs)
    weight[1:] = 1.0 / np.sqrt(freqs[1:])
    return np.fft.irfft(spectrum * weight, n=sample_count)


def generate_cases() -> list[SignalCase]:
    """Create bounded signals that exercise steady-state and transient cases."""
    sample_count = int(SAMPLE_RATE * DURATION_SECONDS)
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    burst_window = np.hanning(sample_count)

    return [
        SignalCase(
            name="62.5 Hz sine",
            signal=normalize_peak(np.sin(2.0 * np.pi * 62.5 * time)),
            note="Steady single tone exactly at the octave centre.",
        ),
        SignalCase(
            name="62.5 Hz + 125 Hz burst",
            signal=normalize_peak(
                (
                    np.sin(2.0 * np.pi * 62.5 * time)
                    + np.sin(2.0 * np.pi * 125.0 * time)
                )
                * burst_window
            ),
            note="Two low tones with a smooth amplitude envelope.",
        ),
        SignalCase(
            name="Unit impulse",
            signal=normalize_peak(np.eye(1, sample_count, sample_count // 2)[0]),
            note="Worst-case broadband transient for impulse-response behaviour.",
        ),
        SignalCase(
            name="Clipped 62.5 Hz square",
            signal=normalize_peak(np.sign(np.sin(2.0 * np.pi * 62.5 * time))),
            note=(
                "Bounded square-like source; removing harmonics changes waveform "
                "shape and can raise the extracted fundamental peak."
            ),
        ),
        SignalCase(
            name="Pink noise",
            signal=normalize_peak(pink_noise(sample_count)),
            note="Broadband noise with a music-like spectral slope.",
        ),
    ]


def analyze_case(
    octave_filter: OctaveBandFilter,
    band_centers: list[float],
    signal_case: SignalCase,
) -> list[dict[str, object]]:
    """Run one signal through the octave bank and return per-band metrics."""
    source = signal_case.signal
    octave_bank = octave_filter.create_octave_bank(source, CENTER_FREQUENCIES)
    source_peak = float(np.max(np.abs(source)))
    source_rms = float(np.sqrt(np.mean(np.square(source))))
    source_power = float(np.mean(np.square(source)))
    band_labels: list[object] = ["Full Spectrum", *band_centers]

    rows = []
    for idx, label in enumerate(band_labels):
        band = np.asarray(octave_bank[:, idx], dtype=np.float64)
        peak = float(np.max(np.abs(band)))
        rms = float(np.sqrt(np.mean(np.square(band))))
        rows.append(
            {
                "case": signal_case.name,
                "band_hz": label,
                "source_peak": source_peak,
                "source_peak_db": db(source_peak),
                "source_rms": source_rms,
                "source_rms_db": db(source_rms),
                "band_peak": peak,
                "band_peak_db": db(peak),
                "band_peak_minus_source_peak_db": db(peak / source_peak)
                if source_peak > 0
                else np.nan,
                "band_rms": rms,
                "band_rms_db": db(rms),
                "band_power_fraction": float((rms * rms) / source_power)
                if source_power > 0
                else np.nan,
                "exceeds_source_peak": bool(peak > source_peak + 1e-9),
                "note": signal_case.note,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write proof rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_real_time_domain_event() -> dict[str, object] | None:
    """Analyze the real Ready Player One 62.5 Hz peak-expansion event."""
    if not REAL_CASE_METADATA.exists():
        print(f"Skipping real case; metadata not found: {REAL_CASE_METADATA}")
        return None

    metadata = json.loads(REAL_CASE_METADATA.read_text(encoding="utf-8"))
    source_path = Path(str(metadata["track_path"]))
    if not source_path.exists():
        print(f"Skipping real case; source media not found: {source_path}")
        return None

    original_peak = float(metadata["original_peak"])
    channel_index = int(metadata["channel_index"])
    channel_name = str(metadata["channel_name"])
    decoded_audio, sample_rate = AudioProcessor().load_audio(source_path)
    if decoded_audio.ndim == 1:
        source_segment = decoded_audio
    else:
        source_segment = decoded_audio[:, channel_index]
    source_segment = np.asarray(source_segment, dtype=np.float64)
    if source_segment.size == 0:
        raise RuntimeError("Decoded real-case channel is empty")

    normalized_source = source_segment / original_peak if original_peak > 0 else source_segment
    octave_filter = OctaveBandFilter(
        sample_rate=sample_rate,
        processing_mode="full_file",
        max_memory_gb=8.0,
        include_low_residual_band=True,
        include_high_residual_band=True,
    )
    band_centers = octave_filter.get_band_center_frequencies(CENTER_FREQUENCIES)
    band_index = band_centers.index(REAL_CASE_BAND_HZ)
    spectrum = np.fft.rfft(normalized_source)
    fft_freqs = np.fft.rfftfreq(normalized_source.size, d=1.0 / sample_rate)
    weight = octave_filter._fft_power_complementary_weight(
        fft_freqs,
        band_centers,
        band_index,
    )
    filtered_band = np.fft.irfft(spectrum * weight, n=normalized_source.size)

    relative_time = np.arange(source_segment.size, dtype=np.float64) / sample_rate
    absolute_time = relative_time
    # Fixed-window exports label each row by the window end time. The 132 s
    # event therefore covers the production chunk from 130 s to 132 s.
    search_mask = (
        (absolute_time >= REAL_CASE_EVENT_SECONDS - REAL_CASE_EVENT_CHUNK_SECONDS)
        & (absolute_time <= REAL_CASE_EVENT_SECONDS)
    )
    search_indices = np.flatnonzero(search_mask)
    if search_indices.size == 0:
        raise RuntimeError("Real-case search window did not overlap decoded segment")
    peak_index = int(
        search_indices[
            np.argmax(np.abs(filtered_band[search_indices] * original_peak))
        ]
    )
    peak_time = float(absolute_time[peak_index])

    plot_start = peak_time - REAL_CASE_PLOT_SECONDS / 2.0
    plot_end = peak_time + REAL_CASE_PLOT_SECONDS / 2.0
    plot_mask = (absolute_time >= plot_start) & (absolute_time < plot_end)
    zoom_mask = (absolute_time >= peak_time - 0.08) & (absolute_time <= peak_time + 0.08)

    source_dbfs_signal = normalized_source * original_peak
    band_dbfs_signal = filtered_band * original_peak
    source_peak = float(np.max(np.abs(source_dbfs_signal[plot_mask])))
    band_peak = float(np.max(np.abs(band_dbfs_signal[plot_mask])))
    expansion_db = db(band_peak / source_peak) if source_peak > 0 else np.nan

    metrics = {
        "track": source_path.name,
        "channel": channel_name,
        "channel_index": channel_index,
        "sample_rate": sample_rate,
        "band_hz": REAL_CASE_BAND_HZ,
        "analysis_window_start_seconds": plot_start,
        "analysis_window_end_seconds": plot_end,
        "filtered_peak_time_seconds": peak_time,
        "source_peak_linear_dbfs": source_peak,
        "source_peak_dbfs": db(source_peak),
        "filtered_band_peak_linear_dbfs": band_peak,
        "filtered_band_peak_dbfs": db(band_peak),
        "filtered_minus_source_peak_db": expansion_db,
        "channel_original_peak_dbfs": db(original_peak),
    }

    write_csv(PROOF_DIR / "real_event_metrics.csv", [metrics])
    write_real_event_trace_csv(
        absolute_time,
        source_dbfs_signal,
        band_dbfs_signal,
        plot_mask,
    )
    plot_real_event_waveforms(
        absolute_time,
        source_dbfs_signal,
        band_dbfs_signal,
        plot_mask,
        zoom_mask,
        metrics,
    )
    return metrics


def write_real_event_trace_csv(
    absolute_time: np.ndarray,
    source_signal: np.ndarray,
    filtered_signal: np.ndarray,
    plot_mask: np.ndarray,
) -> None:
    """Write the 1-second real-event waveform trace."""
    rows = []
    for time_seconds, source, filtered in zip(
        absolute_time[plot_mask],
        source_signal[plot_mask],
        filtered_signal[plot_mask],
    ):
        rows.append(
            {
                "time_seconds": float(time_seconds),
                "source_sample_dbfs_linear": float(source),
                "filtered_62_5hz_sample_dbfs_linear": float(filtered),
                "source_abs_dbfs": db(abs(source)),
                "filtered_62_5hz_abs_dbfs": db(abs(filtered)),
            }
        )
    write_csv(PROOF_DIR / "real_event_waveform_trace.csv", rows)


def plot_real_event_waveforms(
    absolute_time: np.ndarray,
    source_signal: np.ndarray,
    filtered_signal: np.ndarray,
    plot_mask: np.ndarray,
    zoom_mask: np.ndarray,
    metrics: dict[str, object],
) -> None:
    """Plot full-band and filtered-band waveforms around the real event."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    for ax, mask, title in (
        (
            axes[0],
            plot_mask,
            "1-second event window",
        ),
        (
            axes[1],
            zoom_mask,
            "160 ms zoom around filtered-band peak",
        ),
    ):
        ax.plot(
            absolute_time[mask],
            source_signal[mask],
            color="#1f77b4",
            linewidth=1.0,
            label="Source full-band sample",
        )
        ax.plot(
            absolute_time[mask],
            filtered_signal[mask],
            color="#d62728",
            linewidth=1.2,
            label="Filtered 62.5 Hz octave band",
        )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.axhline(-1.0, color="black", linestyle=":", linewidth=1.0)
        ax.axvline(
            float(metrics["filtered_peak_time_seconds"]),
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Sample amplitude (linear dBFS)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle(
        (
            "Ready Player One FL real event: source waveform vs 62.5 Hz "
            f"filtered band ({metrics['filtered_band_peak_dbfs']:.2f} dBFS)"
        )
    )
    plt.tight_layout()
    fig.savefig(PROOF_DIR / "real_event_source_vs_filtered_62_5hz.png", dpi=150)
    plt.close(fig)


def plot_peak_expansion(rows: list[dict[str, object]]) -> None:
    """Plot maximum filtered-band peak expansion per case."""
    case_names = []
    peak_expansion = []
    for case_name in dict.fromkeys(str(row["case"]) for row in rows):
        case_rows = [
            row
            for row in rows
            if row["case"] == case_name and row["band_hz"] != "Full Spectrum"
        ]
        max_row = max(
            case_rows,
            key=lambda row: float(row["band_peak_minus_source_peak_db"]),
        )
        case_names.append(f"{case_name}\n({max_row['band_hz']} Hz)")
        peak_expansion.append(float(max_row["band_peak_minus_source_peak_db"]))

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2ca02c" if value <= 0 else "#d62728" for value in peak_expansion]
    ax.bar(case_names, peak_expansion, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Max Filtered-Band Peak Minus Source Peak (dB)")
    ax.set_title("Octave Filtering Can Raise Band-Limited Sample Peaks")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(PROOF_DIR / "peak_expansion_by_signal.png", dpi=150)
    plt.close(fig)


def plot_rms_closure(rows: list[dict[str, object]]) -> None:
    """Plot summed band RMS power error for each case."""
    case_names = []
    closure_error_db = []
    for case_name in dict.fromkeys(str(row["case"]) for row in rows):
        case_rows = [
            row
            for row in rows
            if row["case"] == case_name and row["band_hz"] != "Full Spectrum"
        ]
        power_sum = float(sum(float(row["band_rms"]) ** 2 for row in case_rows))
        source_power = float(case_rows[0]["source_rms"]) ** 2
        case_names.append(case_name)
        closure_error_db.append(db(power_sum / source_power) if source_power > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(case_names, closure_error_db, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Summed Band Power / Source Power (dB)")
    ax.set_title("Power-Complementary Bank Preserves RMS Power, Not Peaks")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(PROOF_DIR / "rms_power_closure_by_signal.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the octave-band peak-expansion proof."""
    octave_filter = OctaveBandFilter(
        sample_rate=SAMPLE_RATE,
        processing_mode="full_file",
        max_memory_gb=8.0,
        include_low_residual_band=True,
        include_high_residual_band=True,
    )
    band_centers = octave_filter.get_band_center_frequencies(CENTER_FREQUENCIES)
    rows: list[dict[str, object]] = []
    for signal_case in generate_cases():
        rows.extend(analyze_case(octave_filter, band_centers, signal_case))

    write_csv(PROOF_DIR / "peak_expansion_results.csv", rows)
    plot_peak_expansion(rows)
    plot_rms_closure(rows)
    real_metrics = analyze_real_time_domain_event()

    print("Wrote:")
    print(f"- {PROOF_DIR / 'peak_expansion_results.csv'}")
    print(f"- {PROOF_DIR / 'peak_expansion_by_signal.png'}")
    print(f"- {PROOF_DIR / 'rms_power_closure_by_signal.png'}")
    if real_metrics is not None:
        print(f"- {PROOF_DIR / 'real_event_metrics.csv'}")
        print(f"- {PROOF_DIR / 'real_event_waveform_trace.csv'}")
        print(f"- {PROOF_DIR / 'real_event_source_vs_filtered_62_5hz.png'}")


if __name__ == "__main__":
    main()
