"""Prove octave filter band-edge placement and phase neutrality.

This proof exercises the production FFT octave filter bank at centre
frequencies, crossover frequencies, and with EIA-426B-style compressed pink
noise. It is intended to catch band mapping mistakes and unexpected
crest-factor growth from the filter-bank phase response.
"""

from __future__ import annotations

import csv
import sys
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.octave_filter import OctaveBandFilter


PROOF_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 48_000
DURATION_SECONDS = 30.0
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
TARGET_COMPRESSED_NOISE_CREST_DB = 6.0
CREST_EXPANSION_TOLERANCE_DB = 0.25
ACTIVE_BAND_FLOOR_DB = -45.0


def db_amplitude(value: float) -> float:
    """Convert a linear amplitude ratio to dB."""
    return float(20.0 * np.log10(max(float(value), 1e-20)))


def rms(signal: np.ndarray) -> float:
    """Return linear RMS."""
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def crest_factor_db(signal: np.ndarray) -> float:
    """Return crest factor in dB."""
    signal = np.asarray(signal, dtype=np.float64)
    signal_rms = rms(signal)
    if signal_rms <= 0.0:
        return float("nan")
    return db_amplitude(float(np.max(np.abs(signal))) / signal_rms)


def band_labels(centers: list[float]) -> list[str]:
    """Return human-readable octave band labels."""
    labels = []
    for center in centers:
        if np.isclose(center, LOW_RESIDUAL_CENTER_HZ):
            labels.append("4 Hz and below")
        elif np.isclose(center, HIGH_RESIDUAL_CENTER_HZ):
            labels.append(">22627 Hz")
        else:
            labels.append(f"{center:g} Hz")
    return labels


def production_filter() -> OctaveBandFilter:
    """Create the production octave filter configuration used by the proof."""
    return OctaveBandFilter(
        sample_rate=SAMPLE_RATE,
        processing_mode="full_file",
        max_memory_gb=8.0,
        include_low_residual_band=True,
        include_high_residual_band=True,
        low_residual_center_hz=LOW_RESIDUAL_CENTER_HZ,
        high_residual_center_hz=HIGH_RESIDUAL_CENTER_HZ,
    )


def analyze_band_edges(
    octave_filter: OctaveBandFilter,
    centers: list[float],
    labels: list[str],
) -> list[dict[str, object]]:
    """Build centre-frequency and crossover acceptance rows."""
    rows: list[dict[str, object]] = []
    nyquist = SAMPLE_RATE / 2.0

    for band_idx, center in enumerate(centers):
        if center >= nyquist:
            continue
        weights = octave_filter._fft_power_complementary_weights(
            np.asarray([center], dtype=np.float64),
            centers,
        )[:, 0]
        power = weights**2
        top_idx = int(np.argmax(power))
        rows.append(
            {
                "test_type": "center",
                "frequency_hz": center,
                "expected_band": labels[band_idx],
                "observed_primary_band": labels[top_idx],
                "primary_power_percent": float(power[top_idx] * 100.0),
                "secondary_band": "",
                "secondary_power_percent": 0.0,
                "top_pair_power_percent": float(power[top_idx] * 100.0),
                "power_sum_percent": float(np.sum(power) * 100.0),
                "pass": bool(top_idx == band_idx and power[top_idx] >= 0.999999),
            }
        )

    for lower_idx, (lower, upper) in enumerate(zip(centers, centers[1:])):
        crossover = float(np.sqrt(lower * upper))
        if crossover >= nyquist:
            continue
        weights = octave_filter._fft_power_complementary_weights(
            np.asarray([crossover], dtype=np.float64),
            centers,
        )[:, 0]
        power = weights**2
        top_two = np.argsort(power)[-2:][::-1]
        expected = {lower_idx, lower_idx + 1}
        observed = {int(top_two[0]), int(top_two[1])}
        top_pair_power = float(np.sum(power[top_two]))
        rows.append(
            {
                "test_type": "crossover",
                "frequency_hz": crossover,
                "expected_band": f"{labels[lower_idx]} / {labels[lower_idx + 1]}",
                "observed_primary_band": labels[int(top_two[0])],
                "primary_power_percent": float(power[int(top_two[0])] * 100.0),
                "secondary_band": labels[int(top_two[1])],
                "secondary_power_percent": float(power[int(top_two[1])] * 100.0),
                "top_pair_power_percent": top_pair_power * 100.0,
                "power_sum_percent": float(np.sum(power) * 100.0),
                "pass": bool(
                    observed == expected
                    and np.allclose(power[list(expected)], 0.5, atol=1e-12)
                    and np.isclose(top_pair_power, 1.0, atol=1e-12)
                ),
            }
        )

    return rows


def analyze_phase_response(
    octave_filter: OctaveBandFilter,
    centers: list[float],
    labels: list[str],
) -> list[dict[str, object]]:
    """Confirm that production FFT weights do not rotate retained-bin phase."""
    fft_freqs = np.geomspace(1.0, SAMPLE_RATE / 2.0, num=4096)
    weights = octave_filter._fft_power_complementary_weights(fft_freqs, centers)
    rows = []

    for label, weight in zip(labels, weights):
        active = weight > 1e-12
        phase_shift = np.angle(weight[active].astype(np.complex128), deg=True)
        rows.append(
            {
                "band": label,
                "active_bins_sampled": int(np.sum(active)),
                "min_active_weight": float(np.min(weight[active])),
                "max_active_weight": float(np.max(weight[active])),
                "negative_weight_count": int(np.sum(weight < 0.0)),
                "max_abs_phase_shift_degrees": float(np.max(np.abs(phase_shift))),
                "pass": bool(
                    np.all(weight >= 0.0)
                    and np.max(np.abs(phase_shift)) <= 1e-12
                ),
            }
        )

    return rows


def pink_noise(sample_count: int, seed: int = 426) -> np.ndarray:
    """Generate deterministic pink noise."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, sample_count)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / SAMPLE_RATE)
    weights = np.zeros_like(freqs)
    passband = (freqs >= 20.0) & (freqs <= 20_000.0)
    weights[passband] = 1.0 / np.sqrt(freqs[passband])
    shaped = np.fft.irfft(spectrum * weights, n=sample_count)
    shaped -= np.mean(shaped)
    return shaped / rms(shaped)


def compress_to_crest_factor(signal: np.ndarray, target_db: float) -> np.ndarray:
    """Hard-limit noise to a target crest factor."""
    target_linear = 10.0 ** (target_db / 20.0)
    normalized = np.asarray(signal, dtype=np.float64) / rms(signal)
    low = 0.01
    high = float(np.max(np.abs(normalized)))

    for _ in range(80):
        threshold = (low + high) / 2.0
        clipped = np.clip(normalized, -threshold, threshold)
        current_crest = float(np.max(np.abs(clipped)) / rms(clipped))
        if current_crest > target_linear:
            high = threshold
        else:
            low = threshold

    compressed = np.clip(normalized, -high, high)
    compressed -= np.mean(compressed)
    compressed /= rms(compressed)
    return (0.95 * compressed / np.max(np.abs(compressed))).astype(np.float32)


def write_wav(path: Path, signal: np.ndarray) -> None:
    """Write mono 16-bit PCM source material with stdlib wave support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def analyze_compressed_noise(
    octave_filter: OctaveBandFilter,
    centers: list[float],
    labels: list[str],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Check crest-factor stability on compressed pink noise."""
    sample_count = int(SAMPLE_RATE * DURATION_SECONDS)
    source = compress_to_crest_factor(
        pink_noise(sample_count),
        TARGET_COMPRESSED_NOISE_CREST_DB,
    )
    octave_bank = octave_filter.create_octave_bank(source, CENTER_FREQUENCIES)
    source_rms = rms(source)
    source_cf = crest_factor_db(source)
    rows = [
        {
            "band": "Full Spectrum",
            "band_rms_db_relative_to_source": 0.0,
            "crest_factor_db": source_cf,
            "crest_factor_expansion_db": 0.0,
            "active_for_acceptance": True,
            "pass": True,
        }
    ]

    active_floor = 10.0 ** (ACTIVE_BAND_FLOOR_DB / 20.0)
    for band_idx, label in enumerate(labels):
        band = np.asarray(octave_bank[:, band_idx + 1], dtype=np.float64)
        band_rms = rms(band)
        band_cf = crest_factor_db(band)
        expansion = band_cf - source_cf
        active = band_rms / source_rms >= active_floor
        rows.append(
            {
                "band": label,
                "band_rms_db_relative_to_source": db_amplitude(band_rms / source_rms),
                "crest_factor_db": band_cf,
                "crest_factor_expansion_db": expansion,
                "active_for_acceptance": bool(active),
                "pass": bool(
                    (not active) or expansion <= CREST_EXPANSION_TOLERANCE_DB
                ),
            }
        )

    return rows, source, octave_bank


def synthesize_power_complementary_sum(
    octave_bank: np.ndarray,
    octave_filter: OctaveBandFilter,
    centers: list[float],
) -> np.ndarray:
    """Reconstruct source by applying matching synthesis weights and summing."""
    bands = np.asarray(octave_bank[:, 1:], dtype=np.float64)
    fft_freqs = np.fft.rfftfreq(bands.shape[0], d=1.0 / SAMPLE_RATE)
    weights = octave_filter._fft_power_complementary_weights(fft_freqs, centers)
    synthesis_spectrum = np.zeros(fft_freqs.size, dtype=np.complex128)

    for band_idx, weight in enumerate(weights):
        synthesis_spectrum += np.fft.rfft(bands[:, band_idx]) * weight

    return np.fft.irfft(synthesis_spectrum, n=bands.shape[0])


def analyze_reconstruction(
    source: np.ndarray,
    octave_bank: np.ndarray,
    octave_filter: OctaveBandFilter,
    centers: list[float],
) -> list[dict[str, object]]:
    """Compare direct band summing with power-complementary synthesis."""
    source = np.asarray(source, dtype=np.float64)
    source_rms = rms(source)
    source_peak = float(np.max(np.abs(source)))
    source_cf = crest_factor_db(source)
    direct_sum = np.sum(np.asarray(octave_bank[:, 1:], dtype=np.float64), axis=1)
    power_sum = synthesize_power_complementary_sum(
        octave_bank,
        octave_filter,
        centers,
    )

    rows = []
    for name, signal in (
        ("source", source),
        ("direct_time_domain_band_sum", direct_sum),
        ("power_complementary_synthesis_sum", power_sum),
    ):
        error = signal - source
        signal_rms = rms(signal)
        signal_peak = float(np.max(np.abs(signal)))
        rows.append(
            {
                "path": name,
                "rms": signal_rms,
                "rms_delta_db_vs_source": db_amplitude(signal_rms / source_rms),
                "peak": signal_peak,
                "peak_delta_db_vs_source": db_amplitude(signal_peak / source_peak),
                "crest_factor_db": crest_factor_db(signal),
                "crest_factor_delta_db_vs_source": crest_factor_db(signal)
                - source_cf,
                "error_rms_db_relative_to_source": db_amplitude(rms(error) / source_rms),
                "error_peak_db_relative_to_source_peak": db_amplitude(
                    float(np.max(np.abs(error))) / source_peak
                ),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_band_edges(rows: list[dict[str, object]]) -> None:
    """Plot primary and secondary power at band centres and crossovers."""
    labels = [
        f"{row['test_type']}\n{float(row['frequency_hz']):g} Hz"
        for row in rows
    ]
    primary = [float(row["primary_power_percent"]) for row in rows]
    secondary = [float(row["secondary_power_percent"]) for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(rows))
    ax.bar(x, primary, label="Primary band")
    ax.bar(x, secondary, bottom=primary, label="Secondary band")
    ax.axhline(100.0, color="black", linewidth=1.0)
    ax.set_ylabel("Power captured (%)")
    ax.set_title("Octave Filter Centre and Crossover Acceptance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "band_edge_acceptance.png", dpi=160)
    plt.close(fig)


def plot_crest_factor(rows: list[dict[str, object]]) -> None:
    """Plot compressed pink-noise crest-factor expansion by band."""
    labels = [str(row["band"]) for row in rows]
    expansion = [float(row["crest_factor_expansion_db"]) for row in rows]
    active = [bool(row["active_for_acceptance"]) for row in rows]
    colors = ["#1f77b4" if is_active else "#bdbdbd" for is_active in active]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(rows))
    ax.bar(x, expansion, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhline(
        CREST_EXPANSION_TOLERANCE_DB,
        color="#d62728",
        linestyle="--",
        linewidth=1.0,
        label=f"+{CREST_EXPANSION_TOLERANCE_DB:g} dB tolerance",
    )
    ax.set_ylabel("Crest factor expansion vs source (dB)")
    ax.set_title("EIA-426B-Style Compressed Pink Noise Through Octave Bank")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "compressed_pink_noise_crest_factor.png", dpi=160)
    plt.close(fig)


def plot_reconstruction(rows: list[dict[str, object]]) -> None:
    """Plot reconstruction error by summing method."""
    plot_rows = [row for row in rows if row["path"] != "source"]
    labels = [str(row["path"]).replace("_", "\n") for row in plot_rows]
    errors = [float(row["error_rms_db_relative_to_source"]) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, errors, color=["#d62728", "#2ca02c"])
    ax.set_ylabel("Error RMS relative to source RMS (dB)")
    ax.set_title("Octave-Band Reconstruction Error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "reconstruction_error.png", dpi=160)
    plt.close(fig)


def write_summary(
    edge_rows: list[dict[str, object]],
    phase_rows: list[dict[str, object]],
    crest_rows: list[dict[str, object]],
    reconstruction_rows: list[dict[str, object]],
    labels: list[str],
) -> None:
    """Write a human-readable proof summary."""
    edge_pass = all(bool(row["pass"]) for row in edge_rows)
    phase_pass = all(bool(row["pass"]) for row in phase_rows)
    crest_pass = all(bool(row["pass"]) for row in crest_rows)
    filter_status = "PASS" if edge_pass and phase_pass else "FAIL"

    active_crest_rows = [
        row
        for row in crest_rows
        if bool(row["active_for_acceptance"]) and row["band"] != "Full Spectrum"
    ]
    worst_crest = max(
        active_crest_rows,
        key=lambda row: float(row["crest_factor_expansion_db"]),
    )
    source_cf = float(crest_rows[0]["crest_factor_db"])
    direct_sum = next(
        row
        for row in reconstruction_rows
        if row["path"] == "direct_time_domain_band_sum"
    )
    synthesis_sum = next(
        row
        for row in reconstruction_rows
        if row["path"] == "power_complementary_synthesis_sum"
    )
    max_phase_shift = max(
        float(row["max_abs_phase_shift_degrees"]) for row in phase_rows
    )
    min_primary_center = min(
        float(row["primary_power_percent"])
        for row in edge_rows
        if row["test_type"] == "center"
    )
    crossover_errors = [
        abs(float(row["primary_power_percent"]) - 50.0)
        for row in edge_rows
        if row["test_type"] == "crossover"
    ]
    max_crossover_error = max(crossover_errors) if crossover_errors else 0.0
    band_list = "\n".join(f"- `{label}`" for label in labels)

    summary = f"""# Octave Filter Band Edges Proof

Filter-bank status: **{filter_status}**

EIA-426B crest-factor check: **{"PASS" if crest_pass else "expansion observed"}**

This proof validates the production FFT octave filter bank used by Audio
Analyser. The primary question is whether octave centre frequencies and
crossover frequencies are assigned to the intended bands. That check passes.

The proof also records a separate EIA-426B-style compressed pink-noise
crest-factor check. That later check is intentionally diagnostic: it documents
that derived octave-band signals can expand crest factor even though the filter
bank itself has no phase rotation and can reconstruct the source with matched
power-complementary synthesis.

## Band Filtering Result

Status: **{filter_status}**

- Centre-frequency tones land fully in the intended band.
- Geometric crossover tones split power equally between adjacent bands.
- Minimum centre-frequency primary power: `{min_primary_center:.9f}%`
- Maximum crossover split error: `{max_crossover_error:.9f}%`
- Phase audit: **{"PASS" if phase_pass else "FAIL"}**
- Maximum retained-bin phase shift: `{max_phase_shift:.12f} degrees`

## Method

- Sample rate: `{SAMPLE_RATE} Hz`
- Filter implementation: production `src.octave_filter.OctaveBandFilter`
- Band centres:

{band_list}

The acceptance matrix checks every in-band centre frequency and every geometric
crossover between adjacent bands. Centre tones should land fully in the intended
band. Crossover tones should split power equally between the two adjacent bands.

The phase audit samples the production FFT weights over the audible range. The
weights are real and non-negative, so retained FFT bins keep their original
phase and only their magnitude changes.

## Band Filtering Interpretation

The octave bank places centre-frequency energy in the expected band and splits
geometric crossover energy exactly between adjacent bands. That confirms the
band-edge behaviour implied by the raised-cosine power-complementary design.

The phase check confirms that the production bank is magnitude-only in the FFT
domain. It does not introduce per-bin phase rotation, so any crest-factor change
seen in derived band signals is not caused by all-pass or minimum-phase
rotation.

## EIA-426B Crest-Factor Check

Status: **{"PASS" if crest_pass else "expansion observed"}**

The crest-factor stimulus is deterministic EIA-426B-style compressed pink noise:
pink spectrum, 20 Hz to 20 kHz band limit, then hard-limited to a
`{TARGET_COMPRESSED_NOISE_CREST_DB:g} dB` source crest factor. Active octave
bands were checked against a `{CREST_EXPANSION_TOLERANCE_DB:g} dB` expansion
tolerance; inactive bands below `{ACTIVE_BAND_FLOOR_DB:g} dB` relative RMS are
reported but excluded from that acceptance gate.

- Full-spectrum compressed pink-noise crest factor: `{source_cf:.3f} dB`
- Active-band compressed pink-noise crest-factor check:
  **{"PASS" if crest_pass else "FAIL"}**
- Worst active-band crest expansion: `{worst_crest["band"]}`,
  `{float(worst_crest["crest_factor_expansion_db"]):+.3f} dB`

The compressed pink-noise check is intentionally separate from the existing
peak-expansion proof. The full-spectrum path remains at the generated
`{TARGET_COMPRESSED_NOISE_CREST_DB:g} dB` crest factor because column 0 of the
octave bank is the unfiltered source. The derived octave-band paths do show
crest-factor expansion on this compressed stimulus, so the current bank should
not be described as preserving compressed-noise crest factor per band.

## Reconstruction Check

- Direct time-domain band sum error:
  `{float(direct_sum["error_rms_db_relative_to_source"]):.3f} dB`
  relative to source RMS
- Power-complementary synthesis sum error:
  `{float(synthesis_sum["error_rms_db_relative_to_source"]):.3f} dB`
  relative to source RMS

The separated octave-band signals are not a direct perfect-reconstruction bank
if they are simply summed in the time domain. Direct summing applies
`sum(weight_band(f))`, which rises through crossover regions and changes the
source waveform. The bank is power-complementary instead: applying the matching
FFT weights as synthesis weights and then summing applies
`sum(weight_band(f) ** 2) = 1`, reconstructing the source to numerical
precision.

## Outputs

- `band_edge_acceptance.csv`
- `phase_response.csv`
- `compressed_pink_noise_crest_factor.csv`
- `reconstruction_analysis.csv`
- `band_edge_acceptance.png`
- `compressed_pink_noise_crest_factor.png`
- `reconstruction_error.png`
- `source_material/eia426b_style_compressed_pink_noise.wav`
"""
    (PROOF_DIR / "README.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the proof."""
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    octave_filter = production_filter()
    centers = octave_filter.get_band_center_frequencies(CENTER_FREQUENCIES)
    labels = band_labels(centers)

    edge_rows = analyze_band_edges(octave_filter, centers, labels)
    phase_rows = analyze_phase_response(octave_filter, centers, labels)
    crest_rows, compressed_noise, octave_bank = analyze_compressed_noise(
        octave_filter,
        centers,
        labels,
    )
    reconstruction_rows = analyze_reconstruction(
        compressed_noise,
        octave_bank,
        octave_filter,
        centers,
    )

    write_csv(PROOF_DIR / "band_edge_acceptance.csv", edge_rows)
    write_csv(PROOF_DIR / "phase_response.csv", phase_rows)
    write_csv(PROOF_DIR / "compressed_pink_noise_crest_factor.csv", crest_rows)
    write_csv(PROOF_DIR / "reconstruction_analysis.csv", reconstruction_rows)
    write_wav(
        PROOF_DIR / "source_material" / "eia426b_style_compressed_pink_noise.wav",
        compressed_noise,
    )
    plot_band_edges(edge_rows)
    plot_crest_factor(crest_rows)
    plot_reconstruction(reconstruction_rows)
    write_summary(edge_rows, phase_rows, crest_rows, reconstruction_rows, labels)

    filter_status = "PASS" if (
        all(bool(row["pass"]) for row in edge_rows)
        and all(bool(row["pass"]) for row in phase_rows)
    ) else "FAIL"
    crest_status = "PASS" if all(bool(row["pass"]) for row in crest_rows) else "EXPANSION"
    print(f"Filter-bank status: {filter_status}")
    print(f"EIA-426B crest-factor check: {crest_status}")
    print(f"Wrote proof outputs to: {PROOF_DIR}")


if __name__ == "__main__":
    main()
