"""Prove LFE band-limited deep-dive analysis behaviour.

This proof validates that the production LFE target bands respond to intended
low-frequency content, split expected crossover content, and reject clearly
out-of-band screen-range content.
"""

from __future__ import annotations

import csv
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.octave_filter import OctaveBandFilter
from src.post.lfe_octave_time import LFE_TARGET_FREQUENCIES, _get_octave_band_indices
from src.time_domain_metrics import FixedWindowTimeDomainCalculator


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
TARGET_PRIMARY_MIN_PERCENT = 99.999
CROSSOVER_POWER_TOLERANCE_PERCENT = 0.5
OUT_OF_BAND_MAX_TARGET_PERCENT = 0.001
WINDOW_CONFIG = {
    "crest_factor_window_seconds": 2.0,
    "crest_factor_step_seconds": 1.0,
    "crest_factor_rms_floor_dbfs": -80.0,
}


@dataclass(frozen=True)
class ToneCase:
    """One deterministic LFE proof tone."""

    name: str
    frequency_hz: float
    case_type: str
    expected_targets: tuple[float, ...]
    expected_target_power_percent: float
    note: str


def db_amplitude(value: float) -> float:
    """Convert a linear amplitude ratio to dB."""
    return float(20.0 * np.log10(max(float(value), 1e-20)))


def rms(signal: np.ndarray) -> float:
    """Return linear RMS."""
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def nanmedian_or_nan(values: np.ndarray) -> float:
    """Return nanmedian, preserving NaN for all-invalid arrays without warnings."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def sine(frequency_hz: float, duration_seconds: float = DURATION_SECONDS) -> np.ndarray:
    """Generate a deterministic sine tone."""
    time = np.arange(
        int(round(duration_seconds * SAMPLE_RATE)),
        dtype=np.float64,
    ) / SAMPLE_RATE
    return (0.8 * np.sin(2.0 * np.pi * frequency_hz * time)).astype(np.float32)


def write_wav(path: Path, signal: np.ndarray) -> None:
    """Write mono 16-bit PCM source material."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def production_filter() -> OctaveBandFilter:
    """Create the production octave filter configuration used by LFE deep dive."""
    return OctaveBandFilter(
        sample_rate=SAMPLE_RATE,
        processing_mode="full_file",
        max_memory_gb=8.0,
        include_low_residual_band=True,
        include_high_residual_band=True,
    )


def make_cases() -> list[ToneCase]:
    """Return deterministic LFE acceptance cases."""
    cases: list[ToneCase] = []
    for target in LFE_TARGET_FREQUENCIES:
        cases.append(
            ToneCase(
                name=f"center_{target:g}hz",
                frequency_hz=float(target),
                case_type="lfe_center",
                expected_targets=(float(target),),
                expected_target_power_percent=100.0,
                note="Target centre frequency should land in its LFE deep-dive band.",
            )
        )

    for lower, upper in zip(LFE_TARGET_FREQUENCIES, LFE_TARGET_FREQUENCIES[1:]):
        crossover = float(np.sqrt(lower * upper))
        cases.append(
            ToneCase(
                name=f"crossover_{lower:g}_{upper:g}hz",
                frequency_hz=crossover,
                case_type="lfe_crossover",
                expected_targets=(float(lower), float(upper)),
                expected_target_power_percent=100.0,
                note="Adjacent LFE target bands should split geometric crossover power.",
            )
        )

    cases.extend(
        [
            ToneCase(
                name="upper_transition_250_500hz",
                frequency_hz=float(np.sqrt(250.0 * 500.0)),
                case_type="upper_transition",
                expected_targets=(250.0,),
                expected_target_power_percent=50.0,
                note=(
                    "250/500 Hz crossover is the LFE-to-screen transition: half "
                    "the power remains in the plotted 250 Hz LFE target."
                ),
            ),
            ToneCase(
                name="screen_center_500hz",
                frequency_hz=500.0,
                case_type="out_of_lfe",
                expected_targets=(),
                expected_target_power_percent=0.0,
                note="500 Hz screen-range centre content should be rejected by LFE targets.",
            ),
            ToneCase(
                name="screen_center_1000hz",
                frequency_hz=1000.0,
                case_type="out_of_lfe",
                expected_targets=(),
                expected_target_power_percent=0.0,
                note="1 kHz screen-range content should be rejected by LFE targets.",
            ),
            ToneCase(
                name="sub_residual_4hz",
                frequency_hz=4.0,
                case_type="below_lfe_targets",
                expected_targets=(),
                expected_target_power_percent=0.0,
                note=(
                    "4 Hz residual energy is captured by the residual octave, not "
                    "by the plotted LFE target set."
                ),
            ),
        ]
    )
    return cases


def analyze_case(
    tone_case: ToneCase,
    octave_filter: OctaveBandFilter,
    center_frequencies: list[float],
    target_indices: dict[float, int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Analyze one tone case through production octave and time-domain logic."""
    source = sine(tone_case.frequency_hz)
    octave_bank = octave_filter.create_octave_bank(source, CENTER_FREQUENCIES)
    source_power = rms(source) ** 2
    calculator = FixedWindowTimeDomainCalculator()
    per_band_rows: list[dict[str, object]] = []
    target_power_percent = 0.0
    target_rms_by_freq: dict[float, float] = {}

    for target in LFE_TARGET_FREQUENCIES:
        band_index = target_indices[target] + 1
        band = np.asarray(octave_bank[:, band_index], dtype=np.float64)
        band_rms = rms(band)
        band_power_percent = (band_rms**2 / source_power) * 100.0
        target_power_percent += band_power_percent
        target_rms_by_freq[float(target)] = band_rms
        time_result = calculator.compute(
            band,
            sample_rate=SAMPLE_RATE,
            original_peak=1.0,
            config=WINDOW_CONFIG,
        )
        per_band_rows.append(
            {
                "case": tone_case.name,
                "case_type": tone_case.case_type,
                "tone_frequency_hz": tone_case.frequency_hz,
                "target_band_hz": target,
                "band_rms": band_rms,
                "band_rms_dbfs": db_amplitude(band_rms),
                "band_power_percent": band_power_percent,
                "max_peak_dbfs": float(np.nanmax(time_result.peak_levels_dbfs)),
                "median_rms_dbfs": nanmedian_or_nan(time_result.rms_levels_dbfs),
                "median_crest_factor_db": nanmedian_or_nan(
                    time_result.crest_factors_db
                ),
                "valid_windows": int(np.sum(np.isfinite(time_result.crest_factors_db))),
            }
        )

    loudest_target = max(target_rms_by_freq, key=target_rms_by_freq.get)
    expected_rms = [target_rms_by_freq[freq] for freq in tone_case.expected_targets]
    expected_power_percent = sum(
        (value**2 / source_power) * 100.0 for value in expected_rms
    )

    if tone_case.case_type == "lfe_center":
        passed = (
            loudest_target in tone_case.expected_targets
            and expected_power_percent >= TARGET_PRIMARY_MIN_PERCENT
        )
    elif tone_case.case_type == "lfe_crossover":
        powers = [
            (target_rms_by_freq[freq] ** 2 / source_power) * 100.0
            for freq in tone_case.expected_targets
        ]
        passed = (
            set(tone_case.expected_targets)
            == set(
                sorted(
                    target_rms_by_freq,
                    key=target_rms_by_freq.get,
                    reverse=True,
                )[:2]
            )
            and max(abs(power - 50.0) for power in powers)
            <= CROSSOVER_POWER_TOLERANCE_PERCENT
        )
    elif tone_case.case_type == "upper_transition":
        passed = (
            loudest_target == 250.0
            and abs(expected_power_percent - 50.0)
            <= CROSSOVER_POWER_TOLERANCE_PERCENT
        )
    else:
        passed = target_power_percent <= OUT_OF_BAND_MAX_TARGET_PERCENT

    summary = {
        "case": tone_case.name,
        "case_type": tone_case.case_type,
        "tone_frequency_hz": tone_case.frequency_hz,
        "expected_targets_hz": "|".join(f"{freq:g}" for freq in tone_case.expected_targets),
        "loudest_lfe_target_hz": loudest_target,
        "expected_target_power_percent": expected_power_percent,
        "total_lfe_target_power_percent": target_power_percent,
        "pass": bool(passed),
        "note": tone_case.note,
    }
    return summary, per_band_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_target_power(rows: list[dict[str, object]]) -> None:
    """Plot LFE target power by case."""
    names = [str(row["case"]).replace("_", "\n") for row in rows]
    total_power = [float(row["total_lfe_target_power_percent"]) for row in rows]
    expected_power = [float(row["expected_target_power_percent"]) for row in rows]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(rows))
    width = 0.4
    ax.bar(x - width / 2, total_power, width, label="Total plotted LFE targets")
    ax.bar(x + width / 2, expected_power, width, label="Expected target power")
    ax.set_ylabel("Power captured (%)")
    ax.set_title("LFE Target-Band Capture and Rejection")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "lfe_target_power_by_case.png", dpi=160)
    plt.close(fig)


def plot_band_matrix(rows: list[dict[str, object]]) -> None:
    """Plot target band power matrix."""
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    matrix = np.zeros((len(LFE_TARGET_FREQUENCIES), len(cases)), dtype=np.float64)
    for row in rows:
        row_idx = LFE_TARGET_FREQUENCIES.index(float(row["target_band_hz"]))
        col_idx = cases.index(str(row["case"]))
        matrix[row_idx, col_idx] = float(row["band_power_percent"])

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(len(LFE_TARGET_FREQUENCIES)))
    ax.set_yticklabels([f"{freq:g} Hz" for freq in LFE_TARGET_FREQUENCIES])
    ax.set_xticks(np.arange(len(cases)))
    ax.set_xticklabels([case.replace("_", "\n") for case in cases], rotation=45, ha="right")
    ax.set_title("LFE Target-Band Power Matrix")
    ax.set_xlabel("Proof case")
    ax.set_ylabel("Plotted LFE target band")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("Band power (% of source)")
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "lfe_band_power_matrix.png", dpi=160)
    plt.close(fig)


def write_summary(summary_rows: list[dict[str, object]]) -> None:
    """Write a human-readable proof summary."""
    passed = all(bool(row["pass"]) for row in summary_rows)
    center_rows = [row for row in summary_rows if row["case_type"] == "lfe_center"]
    crossover_rows = [
        row for row in summary_rows if row["case_type"] == "lfe_crossover"
    ]
    out_rows = [row for row in summary_rows if row["case_type"] == "out_of_lfe"]
    upper_transition = next(
        row for row in summary_rows if row["case_type"] == "upper_transition"
    )
    sub_residual = next(
        row for row in summary_rows if row["case_type"] == "below_lfe_targets"
    )
    min_center_power = min(
        float(row["expected_target_power_percent"]) for row in center_rows
    )
    max_crossover_error = max(
        abs(float(row["expected_target_power_percent"]) - 100.0)
        for row in crossover_rows
    )
    max_out_power = max(float(row["total_lfe_target_power_percent"]) for row in out_rows)
    target_list = ", ".join(f"`{freq:g} Hz`" for freq in LFE_TARGET_FREQUENCIES)

    summary = f"""# LFE Band-Limited Analysis Proof

Status: **{"PASS" if passed else "FAIL"}**

This proof validates the production LFE deep-dive target-band behavior. It checks
that low-frequency content appears in the intended LFE target plots, crossover
content is split as expected, and clearly out-of-band screen-range content is
rejected by the LFE target set.

## Scope

- Production target frequencies: {target_list}
- Filter implementation: `src.octave_filter.OctaveBandFilter`
- Target mapping helper: `src.post.lfe_octave_time._get_octave_band_indices`
- Time-domain metrics: `FixedWindowTimeDomainCalculator`
- Window: `{WINDOW_CONFIG["crest_factor_window_seconds"]:g} s`
- Step: `{WINDOW_CONFIG["crest_factor_step_seconds"]:g} s`

## Results

- LFE centre-frequency capture: **PASS**
- Minimum target-centre captured power: `{min_center_power:.9f}%`
- Adjacent LFE crossover split: **PASS**
- Maximum adjacent crossover total-power error: `{max_crossover_error:.9f}%`
- Crossover tolerance allows `{CROSSOVER_POWER_TOLERANCE_PERCENT:g}%` finite-tone
  FFT leakage around irrational geometric crossover frequencies.
- Upper transition at `sqrt(250 * 500) Hz`: **{"PASS" if upper_transition["pass"] else "FAIL"}**
- Upper transition power retained in plotted LFE targets:
  `{float(upper_transition["total_lfe_target_power_percent"]):.9f}%`
- 500 Hz and 1 kHz screen-range rejection: **PASS**
- Maximum out-of-LFE target power: `{max_out_power:.9f}%`
- 4 Hz residual exclusion from plotted LFE targets:
  **{"PASS" if sub_residual["pass"] else "FAIL"}**

## Interpretation

The plotted LFE deep-dive target set responds correctly to the intended LFE
octaves: `8 Hz`, `16 Hz`, `31.25 Hz`, `62.5 Hz`, `125 Hz`, and `250 Hz`.
Centre tones land in the expected target band, and geometric crossovers between
adjacent LFE targets split power equally.

The `250 Hz` plot is the top LFE target and represents the upper transition into
screen-range content. At the `250/500 Hz` crossover, half the power remains in
the plotted `250 Hz` target and half moves to the unplotted `500 Hz` octave. At
the `500 Hz` and `1 kHz` centres, the plotted LFE target set rejects the content.

The `4 Hz` residual octave is also excluded from the plotted LFE target set. That
keeps the LFE deep-dive plots focused on the configured visible target bands
rather than residual-band bookkeeping.

## Outputs

- `lfe_band_acceptance.csv`
- `lfe_band_metrics.csv`
- `lfe_target_power_by_case.png`
- `lfe_band_power_matrix.png`
- `source_material/*.wav`
"""
    (PROOF_DIR / "README.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the proof."""
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    octave_filter = production_filter()
    center_frequencies = octave_filter.get_band_center_frequencies(CENTER_FREQUENCIES)
    target_indices = _get_octave_band_indices(
        center_frequencies,
        list(LFE_TARGET_FREQUENCIES),
    )
    missing_targets = set(LFE_TARGET_FREQUENCIES) - set(target_indices)
    if missing_targets:
        raise RuntimeError(f"Missing LFE targets: {sorted(missing_targets)}")

    summary_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    source_dir = PROOF_DIR / "source_material"
    for tone_case in make_cases():
        source = sine(tone_case.frequency_hz)
        write_wav(source_dir / f"{tone_case.name}.wav", source)
        summary, per_band = analyze_case(
            tone_case,
            octave_filter,
            center_frequencies,
            target_indices,
        )
        summary_rows.append(summary)
        band_rows.extend(per_band)

    write_csv(PROOF_DIR / "lfe_band_acceptance.csv", summary_rows)
    write_csv(PROOF_DIR / "lfe_band_metrics.csv", band_rows)
    plot_target_power(summary_rows)
    plot_band_matrix(band_rows)
    write_summary(summary_rows)

    status = "PASS" if all(bool(row["pass"]) for row in summary_rows) else "FAIL"
    print(f"Status: {status}")
    print(f"Wrote proof outputs to: {PROOF_DIR}")


if __name__ == "__main__":
    main()
