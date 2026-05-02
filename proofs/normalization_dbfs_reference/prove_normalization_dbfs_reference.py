"""Prove normalization and dBFS reference behavior.

The production pipeline normalizes each channel before analysis, then carries the
channel's original peak as the dBFS reference. This proof verifies that peak and
RMS dBFS values move with source gain while crest factor remains gain-invariant,
including filtered octave-band metrics and exported octave time metrics.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.music_analyzer import MusicAnalyzer
from src.octave_filter import OctaveBandFilter
from src.results.bundle import _write_octave_time_metrics
from src.time_domain_metrics import (
    FixedWindowTimeDomainCalculator,
    compute_whole_interval_crest_factor,
)


PROOF_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 48_000
DURATION_SECONDS = 12.0
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
GAINS = [0.25, 0.5, 1.0]
REFERENCE_GAIN = 1.0
CHECK_BANDS = ["Full Spectrum", "62.500", "1000.000"]
ACCEPTANCE_TOLERANCE_DB = 1e-7
ANALYSIS_CONFIG = {
    "crest_factor_window_seconds": 2.0,
    "crest_factor_step_seconds": 1.0,
    "crest_factor_rms_floor_dbfs": -120.0,
}


@dataclass(frozen=True)
class GainCase:
    """One source gain case."""

    gain: float
    raw_signal: np.ndarray
    normalized_signal: np.ndarray
    original_peak: float


def db_amplitude(value: float) -> float:
    """Convert a positive linear amplitude ratio to dB."""
    return float(20.0 * np.log10(max(float(value), 1e-20)))


def rms(signal: np.ndarray) -> float:
    """Return linear RMS."""
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def crest_factor_db(signal: np.ndarray) -> float:
    """Return crest factor in dB."""
    peak = float(np.max(np.abs(signal)))
    signal_rms = rms(signal)
    return db_amplitude(peak / signal_rms) if signal_rms > 0.0 else float("nan")


def normalize_to_peak(signal: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Peak-normalize a deterministic signal."""
    max_abs = float(np.max(np.abs(signal)))
    if max_abs <= 0.0:
        return signal.astype(np.float32)
    return (peak * signal / max_abs).astype(np.float32)


def source_shape() -> np.ndarray:
    """Create deterministic broadband-ish source material."""
    time = np.arange(int(SAMPLE_RATE * DURATION_SECONDS), dtype=np.float64) / SAMPLE_RATE
    signal = (
        0.75 * np.sin(2.0 * np.pi * 62.5 * time)
        + 0.35 * np.sin(2.0 * np.pi * 1000.0 * time + 0.4)
        + 0.15 * np.sin(2.0 * np.pi * 353.5533905932738 * time + 1.1)
    )
    envelope = 0.65 + 0.35 * np.sin(2.0 * np.pi * 0.25 * time) ** 2
    return normalize_to_peak(signal * envelope)


def make_gain_case(base_signal: np.ndarray, gain: float) -> GainCase:
    """Create normalized channel data and original peak reference for one gain."""
    raw = np.asarray(base_signal, dtype=np.float64) * float(gain)
    original_peak = float(np.max(np.abs(raw)))
    normalized = raw / original_peak if original_peak > 0.0 else raw
    return GainCase(
        gain=float(gain),
        raw_signal=raw.astype(np.float32),
        normalized_signal=normalized.astype(np.float32),
        original_peak=original_peak,
    )


def write_wav(path: Path, signal: np.ndarray) -> None:
    """Write mono 16-bit PCM source material."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def create_octave_bank(signal: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Create a production octave bank for a normalized channel signal."""
    octave_filter = OctaveBandFilter(
        sample_rate=SAMPLE_RATE,
        processing_mode="full_file",
        max_memory_gb=8.0,
        include_low_residual_band=True,
        include_high_residual_band=True,
    )
    center_frequencies = octave_filter.get_band_center_frequencies(CENTER_FREQUENCIES)
    return octave_filter.create_octave_bank(signal, CENTER_FREQUENCIES), center_frequencies


def analyze_full_band(case: GainCase) -> dict[str, object]:
    """Analyze full-band whole-interval and fixed-window metrics."""
    whole = compute_whole_interval_crest_factor(
        case.normalized_signal,
        original_peak=case.original_peak,
    )
    time_result = FixedWindowTimeDomainCalculator().compute(
        case.normalized_signal,
        sample_rate=SAMPLE_RATE,
        original_peak=case.original_peak,
        config=ANALYSIS_CONFIG,
    )
    return {
        "gain": case.gain,
        "original_peak": case.original_peak,
        "raw_peak": float(np.max(np.abs(case.raw_signal))),
        "raw_rms": rms(case.raw_signal),
        "normalized_peak": float(np.max(np.abs(case.normalized_signal))),
        "normalized_rms": rms(case.normalized_signal),
        "whole_peak_dbfs": whole.peak_level_dbfs,
        "whole_rms_dbfs": whole.rms_level_dbfs,
        "whole_crest_factor_db": whole.crest_factor_db,
        "median_window_peak_dbfs": float(np.nanmedian(time_result.peak_levels_dbfs)),
        "median_window_rms_dbfs": float(np.nanmedian(time_result.rms_levels_dbfs)),
        "median_window_crest_factor_db": float(
            np.nanmedian(time_result.crest_factors_db)
        ),
    }


def analyze_octave_bands(
    case: GainCase,
    octave_bank: np.ndarray,
    center_frequencies: list[float],
) -> list[dict[str, object]]:
    """Analyze octave-band statistics using production MusicAnalyzer."""
    analyzer = MusicAnalyzer(
        sample_rate=SAMPLE_RATE,
        original_peak=case.original_peak,
        analysis_config=ANALYSIS_CONFIG,
    )
    results = analyzer.analyze_octave_bands(octave_bank, center_frequencies)
    rows = []
    for band_label, stats in results["statistics"].items():
        if band_label not in CHECK_BANDS:
            continue
        rows.append(
            {
                "gain": case.gain,
                "band": band_label,
                "linear_peak": stats["max_amplitude"],
                "linear_rms": stats["rms"],
                "peak_dbfs": stats["max_amplitude_db"],
                "rms_dbfs": stats["rms_db"],
                "crest_factor_db": stats["crest_factor_db"],
            }
        )
    return rows


def analyze_exported_octave_time(
    case: GainCase,
    octave_bank: np.ndarray,
    center_frequencies: list[float],
) -> list[dict[str, object]]:
    """Analyze exported octave time metrics from the result bundle writer."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "octave_time_metrics.csv"
        _write_octave_time_metrics(
            output_path,
            octave_bank,
            center_frequencies,
            case.normalized_signal,
            SAMPLE_RATE,
            case.original_peak,
            ANALYSIS_CONFIG,
        )
        data = pd.read_csv(output_path)

    rows = []
    for band in [62.5, 1000.0]:
        band_rows = data[np.isclose(data["frequency_hz"].astype(float), band)]
        rows.append(
            {
                "gain": case.gain,
                "frequency_hz": band,
                "median_peak_dbfs": float(np.nanmedian(band_rows["peak_dbfs"])),
                "median_rms_dbfs": float(np.nanmedian(band_rows["rms_dbfs"])),
                "median_crest_factor_db": float(
                    np.nanmedian(band_rows["crest_factor_db"])
                ),
                "valid_windows": int(band_rows["is_valid_crest_factor"].sum()),
            }
        )
    return rows


def compare_gain_shifts(
    full_rows: list[dict[str, object]],
    octave_rows: list[dict[str, object]],
    time_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compare metric shifts against expected gain shifts."""
    rows: list[dict[str, object]] = []
    reference_full = next(row for row in full_rows if row["gain"] == REFERENCE_GAIN)
    for row in full_rows:
        expected_shift = db_amplitude(float(row["gain"]) / REFERENCE_GAIN)
        rows.extend(
            [
                metric_shift_row(
                    "full_band_whole_peak",
                    str(row["gain"]),
                    expected_shift,
                    float(row["whole_peak_dbfs"]) - float(reference_full["whole_peak_dbfs"]),
                    0.0,
                    float(row["whole_crest_factor_db"])
                    - float(reference_full["whole_crest_factor_db"]),
                ),
                metric_shift_row(
                    "full_band_whole_rms",
                    str(row["gain"]),
                    expected_shift,
                    float(row["whole_rms_dbfs"]) - float(reference_full["whole_rms_dbfs"]),
                    0.0,
                    float(row["whole_crest_factor_db"])
                    - float(reference_full["whole_crest_factor_db"]),
                ),
                metric_shift_row(
                    "full_band_window_peak",
                    str(row["gain"]),
                    expected_shift,
                    float(row["median_window_peak_dbfs"])
                    - float(reference_full["median_window_peak_dbfs"]),
                    0.0,
                    float(row["median_window_crest_factor_db"])
                    - float(reference_full["median_window_crest_factor_db"]),
                ),
                metric_shift_row(
                    "full_band_window_rms",
                    str(row["gain"]),
                    expected_shift,
                    float(row["median_window_rms_dbfs"])
                    - float(reference_full["median_window_rms_dbfs"]),
                    0.0,
                    float(row["median_window_crest_factor_db"])
                    - float(reference_full["median_window_crest_factor_db"]),
                ),
            ]
        )

    for band in CHECK_BANDS:
        reference = next(
            row
            for row in octave_rows
            if row["gain"] == REFERENCE_GAIN and row["band"] == band
        )
        for row in [row for row in octave_rows if row["band"] == band]:
            expected_shift = db_amplitude(float(row["gain"]) / REFERENCE_GAIN)
            rows.extend(
                [
                    metric_shift_row(
                        "octave_band_peak",
                        band,
                        expected_shift,
                        float(row["peak_dbfs"]) - float(reference["peak_dbfs"]),
                        0.0,
                        float(row["crest_factor_db"])
                        - float(reference["crest_factor_db"]),
                    ),
                    metric_shift_row(
                        "octave_band_rms",
                        band,
                        expected_shift,
                        float(row["rms_dbfs"]) - float(reference["rms_dbfs"]),
                        0.0,
                        float(row["crest_factor_db"])
                        - float(reference["crest_factor_db"]),
                    ),
                ]
            )

    for frequency in [62.5, 1000.0]:
        reference = next(
            row
            for row in time_rows
            if row["gain"] == REFERENCE_GAIN and row["frequency_hz"] == frequency
        )
        for row in [row for row in time_rows if row["frequency_hz"] == frequency]:
            expected_shift = db_amplitude(float(row["gain"]) / REFERENCE_GAIN)
            rows.extend(
                [
                    metric_shift_row(
                        "octave_time_peak",
                        f"{frequency:g} Hz",
                        expected_shift,
                        float(row["median_peak_dbfs"])
                        - float(reference["median_peak_dbfs"]),
                        0.0,
                        float(row["median_crest_factor_db"])
                        - float(reference["median_crest_factor_db"]),
                    ),
                    metric_shift_row(
                        "octave_time_rms",
                        f"{frequency:g} Hz",
                        expected_shift,
                        float(row["median_rms_dbfs"]) - float(reference["median_rms_dbfs"]),
                        0.0,
                        float(row["median_crest_factor_db"])
                        - float(reference["median_crest_factor_db"]),
                    ),
                ]
            )
    return rows


def metric_shift_row(
    metric: str,
    target: str,
    expected_shift_db: float,
    observed_shift_db: float,
    expected_crest_delta_db: float,
    observed_crest_delta_db: float,
) -> dict[str, object]:
    """Return one gain-shift acceptance row."""
    shift_error = observed_shift_db - expected_shift_db
    crest_error = observed_crest_delta_db - expected_crest_delta_db
    return {
        "metric": metric,
        "target": target,
        "expected_shift_db": expected_shift_db,
        "observed_shift_db": observed_shift_db,
        "shift_error_db": shift_error,
        "expected_crest_delta_db": expected_crest_delta_db,
        "observed_crest_delta_db": observed_crest_delta_db,
        "crest_error_db": crest_error,
        "pass": bool(
            abs(shift_error) <= ACCEPTANCE_TOLERANCE_DB
            and abs(crest_error) <= ACCEPTANCE_TOLERANCE_DB
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_gain_shifts(rows: list[dict[str, object]]) -> None:
    """Plot observed versus expected gain shifts."""
    peak_rms_rows = [
        row
        for row in rows
        if row["target"] in {"0.25", "0.5", "1.0"}
        and str(row["metric"]).endswith(("peak", "rms"))
    ]
    labels = [f"{row['metric']}\n{row['target']}x" for row in peak_rms_rows]
    expected = [float(row["expected_shift_db"]) for row in peak_rms_rows]
    observed = [float(row["observed_shift_db"]) for row in peak_rms_rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(peak_rms_rows))
    width = 0.4
    ax.bar(x - width / 2, expected, width, label="Expected")
    ax.bar(x + width / 2, observed, width, label="Observed")
    ax.set_ylabel("Shift versus 1.0x gain (dB)")
    ax.set_title("Peak and RMS dBFS Track Source Gain")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "gain_shift_dbfs.png", dpi=160)
    plt.close(fig)


def plot_crest_invariance(rows: list[dict[str, object]]) -> None:
    """Plot crest-factor deltas across checks."""
    labels = [f"{row['metric']}\n{row['target']}" for row in rows]
    deltas = [float(row["observed_crest_delta_db"]) for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(rows)), deltas)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Crest-factor delta versus 1.0x gain (dB)")
    ax.set_title("Crest Factor Is Gain-Invariant")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "crest_factor_invariance.png", dpi=160)
    plt.close(fig)


def write_summary(
    full_rows: list[dict[str, object]],
    octave_rows: list[dict[str, object]],
    time_rows: list[dict[str, object]],
    shift_rows: list[dict[str, object]],
) -> None:
    """Write a human-readable proof summary."""
    passed = all(bool(row["pass"]) for row in shift_rows)
    max_shift_error = max(abs(float(row["shift_error_db"])) for row in shift_rows)
    max_crest_error = max(abs(float(row["crest_error_db"])) for row in shift_rows)
    reference = next(row for row in full_rows if row["gain"] == REFERENCE_GAIN)
    normalized_rms_values = [float(row["normalized_rms"]) for row in full_rows]
    normalized_rms_span = max(normalized_rms_values) - min(normalized_rms_values)
    octave_reference_count = len(
        [row for row in octave_rows if row["gain"] == REFERENCE_GAIN]
    )
    time_reference_count = len(
        [row for row in time_rows if row["gain"] == REFERENCE_GAIN]
    )

    summary = f"""# Normalization and dBFS Reference Proof

Status: **{"PASS" if passed else "FAIL"}**

This proof validates the channel-normalization contract used by Audio Analyser:
analysis runs on peak-normalized channel data, while the channel's original peak
is carried as the dBFS reference.

## Method

- Deterministic source with low-frequency and mid-band tones
- Gain cases: `{", ".join(f"{gain:g}x" for gain in GAINS)}`
- Production whole-interval metrics: `compute_whole_interval_crest_factor`
- Production octave-band metrics: `MusicAnalyzer.analyze_octave_bands`
- Production fixed-window metrics: `FixedWindowTimeDomainCalculator`
- Production octave time export path: `results.bundle._write_octave_time_metrics`
- Checked bands: `{", ".join(CHECK_BANDS)}`

Each raw source is scaled by gain, then normalized to peak `1.0` before analysis.
The raw channel peak is passed as `original_peak`.

## Results

- Gain-shift checks: **{"PASS" if passed else "FAIL"}**
- Maximum peak/RMS shift error: `{max_shift_error:.12f} dB`
- Maximum crest-factor gain-invariance error: `{max_crest_error:.12f} dB`
- Acceptance tolerance: `{ACCEPTANCE_TOLERANCE_DB:.1e} dB`
- Normalized RMS span across gain cases: `{normalized_rms_span:.12e}`
- Reference `1.0x` full-band peak: `{float(reference["whole_peak_dbfs"]):.6f} dBFS`
- Reference `1.0x` full-band RMS: `{float(reference["whole_rms_dbfs"]):.6f} dBFS`
- Octave whole-interval reference rows checked: `{octave_reference_count}`
- Octave time export reference rows checked: `{time_reference_count}`

## Interpretation

Peak and RMS dBFS values move by the exact source gain change because dBFS is
computed from the normalized analysis level multiplied by `original_peak`:

```text
level_dbfs = 20 * log10(normalized_level * original_peak)
```

Crest factor remains gain-invariant because it is a ratio of peak to RMS from
the same normalized signal path:

```text
crest_factor = peak / rms
```

The same behavior holds for filtered octave-band whole-interval metrics and for
the exported octave time metrics. This confirms that octave-band dBFS values use
the channel peak reference consistently, while crest factor remains independent
of source gain.

## Outputs

- `normalization_results.csv`
- `octave_band_reference.csv`
- `octave_time_reference.csv`
- `gain_shift_acceptance.csv`
- `gain_shift_dbfs.png`
- `crest_factor_invariance.png`
- `source_material/*.wav`
"""
    (PROOF_DIR / "README.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the proof."""
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    base_signal = source_shape()
    source_dir = PROOF_DIR / "source_material"

    full_rows: list[dict[str, object]] = []
    octave_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []

    for gain in GAINS:
        case = make_gain_case(base_signal, gain)
        write_wav(source_dir / f"source_gain_{gain:g}x.wav", case.raw_signal)
        octave_bank, center_frequencies = create_octave_bank(case.normalized_signal)
        full_rows.append(analyze_full_band(case))
        octave_rows.extend(analyze_octave_bands(case, octave_bank, center_frequencies))
        time_rows.extend(
            analyze_exported_octave_time(case, octave_bank, center_frequencies)
        )

    shift_rows = compare_gain_shifts(full_rows, octave_rows, time_rows)

    write_csv(PROOF_DIR / "normalization_results.csv", full_rows)
    write_csv(PROOF_DIR / "octave_band_reference.csv", octave_rows)
    write_csv(PROOF_DIR / "octave_time_reference.csv", time_rows)
    write_csv(PROOF_DIR / "gain_shift_acceptance.csv", shift_rows)
    plot_gain_shifts(shift_rows)
    plot_crest_invariance(shift_rows)
    write_summary(full_rows, octave_rows, time_rows, shift_rows)

    print(f"Status: {'PASS' if all(bool(row['pass']) for row in shift_rows) else 'FAIL'}")
    print(f"Wrote proof outputs to: {PROOF_DIR}")


if __name__ == "__main__":
    main()
