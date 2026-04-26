"""Portable per-track analysis result bundle writer.

The bundle is the stable boundary between analysis and future rendering/UI code.
It stores derived tables and arrays needed to recreate current plots without
reloading the original audio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.signal_metrics import compute_peak_hold_envelope, compute_slow_rms_envelope

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
BUNDLE_SUFFIX = ".aaresults"


def write_channel_result_bundle(
    *,
    track_output_dir: Path,
    track_metadata: Dict[str, Any],
    analysis_results: Dict[str, Any],
    time_analysis: Dict[str, Any],
    chunk_octave_analysis: Optional[Dict[str, Any]],
    envelope_statistics: Optional[Dict[str, Any]],
    octave_bank: np.ndarray,
    center_frequencies: Iterable[float],
    channel_data: np.ndarray,
    plotting_config: Dict[str, Any],
    envelope_config: Dict[str, Any],
    analysis_config: Dict[str, Any],
) -> Path:
    """Write or update one channel inside a per-track result bundle.

    Args:
        track_output_dir: Existing per-track analysis output directory.
        track_metadata: Channel and track metadata exported by the analysis pass.
        analysis_results: Static octave-band analysis results.
        time_analysis: Full-spectrum time-domain analysis results.
        chunk_octave_analysis: Optional min/max chunk octave analysis.
        envelope_statistics: Optional envelope analysis output.
        octave_bank: Full-spectrum plus octave-band time series from analysis.
        center_frequencies: Octave/residual band center frequencies.
        channel_data: Normalized channel signal used by analysis.
        plotting_config: Plotting config used to derive histogram tables.
        envelope_config: Envelope-analysis config used to derive envelope plots.
        analysis_config: Analysis config used to derive octave time tables.

    Returns:
        Path to the per-track result bundle directory.
    """
    track_name = str(track_metadata.get("track_name", "track"))
    bundle_dir = track_output_dir / f"{Path(track_name).stem}{BUNDLE_SUFFIX}"
    channel_index = int(track_metadata.get("channel_index", 0))
    channel_id = f"channel_{channel_index + 1:02d}"
    channel_dir = bundle_dir / "channels" / channel_id
    channel_dir.mkdir(parents=True, exist_ok=True)

    metadata = _json_safe(track_metadata)
    _write_json(channel_dir / "metadata.json", metadata)
    _write_json(channel_dir / "plotting_config.json", _json_safe(plotting_config))
    _write_json(channel_dir / "envelope_config.json", _json_safe(envelope_config))
    _write_json(channel_dir / "analysis_config.json", _json_safe(analysis_config))

    _write_octave_band_table(channel_dir / "octave_band_analysis.csv", analysis_results)
    _write_time_domain_table(channel_dir / "time_domain_analysis.csv", time_analysis)
    _write_extreme_chunk_table(
        channel_dir / "extreme_chunks_octave_analysis.csv",
        analysis_results,
        chunk_octave_analysis,
    )
    _write_histogram_tables(
        channel_dir=channel_dir,
        analysis_results=analysis_results,
        plotting_config=plotting_config,
        original_peak=float(track_metadata.get("original_peak", 1.0) or 1.0),
    )
    _write_octave_time_metrics(
        channel_dir / "octave_time_metrics.csv",
        octave_bank=octave_bank,
        center_frequencies=list(center_frequencies),
        channel_data=channel_data,
        sample_rate=int(track_metadata.get("sample_rate", 44100)),
        peak_hold_tau=float(analysis_config.get("peak_hold_tau_seconds", 1.0)),
    )
    if envelope_statistics is not None:
        _write_json(
            channel_dir / "envelope_plot_data.json",
            _json_safe(envelope_statistics),
        )

    _update_manifest(
        bundle_dir=bundle_dir,
        track_metadata=metadata,
        channel_id=channel_id,
        channel_dir=channel_dir,
    )
    logger.info("Analysis result bundle updated: %s", bundle_dir)
    return bundle_dir


def _update_manifest(
    bundle_dir: Path,
    track_metadata: Dict[str, Any],
    channel_id: str,
    channel_dir: Path,
) -> None:
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "bundle_format": "audio-analyser-results-directory",
            "track": {
                "track_name": track_metadata.get("track_name"),
                "track_path": track_metadata.get("track_path"),
                "content_type": track_metadata.get("content_type"),
                "duration_seconds": track_metadata.get("duration_seconds"),
                "sample_rate": track_metadata.get("sample_rate"),
                "total_channels": track_metadata.get("total_channels"),
                "analysis_date": track_metadata.get("analysis_date"),
            },
        }
    )

    channels = {
        channel["channel_id"]: channel
        for channel in manifest.get("channels", [])
        if "channel_id" in channel
    }
    channels[channel_id] = {
        "channel_id": channel_id,
        "channel_index": track_metadata.get("channel_index"),
        "channel_name": track_metadata.get("channel_name"),
        "relative_path": channel_dir.relative_to(bundle_dir).as_posix(),
        "artifacts": {
            "metadata": "metadata.json",
            "analysis_config": "analysis_config.json",
            "plotting_config": "plotting_config.json",
            "envelope_config": "envelope_config.json",
            "octave_band_analysis": "octave_band_analysis.csv",
            "time_domain_analysis": "time_domain_analysis.csv",
            "extreme_chunks_octave_analysis": "extreme_chunks_octave_analysis.csv",
            "histogram_linear": "histogram_linear.csv",
            "histogram_log_db": "histogram_log_db.csv",
            "octave_time_metrics": "octave_time_metrics.csv",
            "envelope_plot_data": "envelope_plot_data.json",
        },
    }
    manifest["channels"] = [
        channels[key]
        for key in sorted(
            channels,
            key=lambda value: channels[value].get("channel_index", 0),
        )
    ]
    _write_json(manifest_path, manifest)


def _write_octave_band_table(path: Path, analysis_results: Dict[str, Any]) -> None:
    rows = []
    for frequency_hz, stats in analysis_results.get("statistics", {}).items():
        row = {
            "frequency_hz": frequency_hz,
            "max_amplitude": stats.get("max_amplitude"),
            "max_amplitude_db": stats.get("max_amplitude_db"),
            "rms": stats.get("rms"),
            "rms_db": stats.get("rms_db"),
            "dynamic_range": stats.get("dynamic_range"),
            "dynamic_range_db": stats.get("dynamic_range_db"),
            "crest_factor": stats.get("crest_factor"),
            "crest_factor_db": stats.get("crest_factor_db"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
        }
        for percentile, value in stats.get("percentiles", {}).items():
            row[percentile] = value
        rows.append(row)
    _write_dataframe(path, pd.DataFrame(rows))


def _write_time_domain_table(path: Path, time_analysis: Dict[str, Any]) -> None:
    keys = [
        "time_points",
        "crest_factors",
        "crest_factors_db",
        "peak_levels",
        "rms_levels",
        "peak_levels_dbfs",
        "rms_levels_dbfs",
    ]
    names = [
        "time_seconds",
        "crest_factor",
        "crest_factor_db",
        "peak_level",
        "rms_level",
        "peak_level_dbfs",
        "rms_level_dbfs",
    ]
    rows = {
        name: np.asarray(time_analysis.get(key, []), dtype=float)
        for key, name in zip(keys, names)
    }
    _write_dataframe(path, pd.DataFrame(rows))


def _write_extreme_chunk_table(
    path: Path,
    analysis_results: Dict[str, Any],
    chunk_octave_analysis: Optional[Dict[str, Any]],
) -> None:
    rows = []
    if chunk_octave_analysis:
        for chunk_type, chunk_key in (
            ("min_crest", "min_chunk"),
            ("max_crest", "max_chunk"),
        ):
            chunk_data = chunk_octave_analysis.get(chunk_key)
            if not chunk_data:
                continue
            chunk_stats = chunk_data["analysis"].get("statistics", {})
            for freq in analysis_results.get("center_frequencies", []):
                freq_key = f"{float(freq):.3f}"
                stats = chunk_stats.get(freq_key)
                if not stats:
                    continue
                rows.append(
                    {
                        "chunk_type": chunk_type,
                        "time_seconds": chunk_data.get("time"),
                        "crest_factor_db": chunk_data.get("crest_factor_db"),
                        "frequency_hz": freq,
                        "max_amplitude_db": stats.get("max_amplitude_db"),
                        "rms_db": stats.get("rms_db"),
                        "chunk_crest_factor_db": stats.get("crest_factor_db"),
                    }
                )
    _write_dataframe(path, pd.DataFrame(rows))


def _write_histogram_tables(
    channel_dir: Path,
    analysis_results: Dict[str, Any],
    plotting_config: Dict[str, Any],
    original_peak: float,
) -> None:
    band_data = analysis_results.get("band_data", {})
    linear_bins = int(plotting_config.get("histogram_bins", 51))
    linear_range = plotting_config.get("histogram_range", [-1.0, 1.0])
    linear_rows = []
    log_rows = []

    noise_floor_db = float(plotting_config.get("log_histogram_noise_floor_db", -60.0))
    max_db = float(plotting_config.get("log_histogram_max_db", 0.0))
    max_bin_size_db = float(plotting_config.get("log_histogram_max_bin_size_db", 3.0))
    min_log_bins = max(int(np.ceil((max_db - noise_floor_db) / max_bin_size_db)), 2)
    log_bin_edges = np.linspace(noise_floor_db, max_db, min_log_bins)

    for label, values in band_data.items():
        frequency_hz = _frequency_from_label(label)
        signal = np.asarray(values, dtype=float)
        clean_signal = signal[np.isfinite(signal)]

        in_range = clean_signal[
            (clean_signal >= float(linear_range[0]))
            & (clean_signal <= float(linear_range[1]))
        ]
        if in_range.size:
            counts, edges = np.histogram(in_range, bins=linear_bins, range=linear_range)
            density, _ = np.histogram(
                in_range, bins=linear_bins, range=linear_range, density=True
            )
            linear_rows.extend(
                _histogram_rows(label, frequency_hz, counts, density, edges)
            )

        if clean_signal.size:
            abs_signal = np.abs(clean_signal) * original_peak
            abs_signal[abs_signal == 0] = 10 ** (noise_floor_db / 20)
            signal_db = 20 * np.log10(abs_signal)
            signal_db = signal_db[(signal_db >= noise_floor_db) & (signal_db <= max_db)]
            if signal_db.size:
                counts, edges = np.histogram(signal_db, bins=log_bin_edges)
                density, _ = np.histogram(signal_db, bins=log_bin_edges, density=True)
                log_rows.extend(
                    _histogram_rows(label, frequency_hz, counts, density, edges)
                )

    _write_dataframe(channel_dir / "histogram_linear.csv", pd.DataFrame(linear_rows))
    _write_dataframe(channel_dir / "histogram_log_db.csv", pd.DataFrame(log_rows))


def _histogram_rows(
    label: str,
    frequency_hz: Optional[float],
    counts: np.ndarray,
    density: np.ndarray,
    edges: np.ndarray,
) -> list[Dict[str, Any]]:
    rows = []
    for count, density_value, left, right in zip(
        counts,
        density,
        edges[:-1],
        edges[1:],
    ):
        rows.append(
            {
                "frequency_label": label,
                "frequency_hz": frequency_hz,
                "bin_left": left,
                "bin_right": right,
                "bin_center": (left + right) / 2.0,
                "bin_count": count,
                "bin_density": density_value,
            }
        )
    return rows


def _write_octave_time_metrics(
    path: Path,
    octave_bank: np.ndarray,
    center_frequencies: list[float],
    channel_data: np.ndarray,
    sample_rate: int,
    peak_hold_tau: float,
) -> None:
    rows = []
    chunk_duration = 1.0
    chunk_samples = max(int(sample_rate * chunk_duration), 1)
    num_samples = len(channel_data)
    num_complete_chunks = (num_samples - chunk_samples) // chunk_samples + 1
    if num_complete_chunks <= 0:
        _write_dataframe(path, pd.DataFrame(rows))
        return

    end_indices = np.arange(num_complete_chunks) * chunk_samples + (chunk_samples - 1)
    time_points = (np.arange(num_complete_chunks) + 1) * chunk_duration
    channel_peak = float(np.max(np.abs(channel_data))) if channel_data.size else 0.0

    for band_index, center_frequency in enumerate(center_frequencies):
        band_data = np.asarray(octave_bank[:, band_index + 1], dtype=float)
        peak_env = compute_peak_hold_envelope(
            band_data,
            sample_rate,
            tau=peak_hold_tau,
        )
        rms_env = compute_slow_rms_envelope(band_data, sample_rate)

        peak_indices = np.clip(end_indices, 0, max(peak_env.size - 1, 0))
        rms_indices = np.clip(end_indices, 0, max(rms_env.size - 1, 0))
        peaks = (
            peak_env[peak_indices] if peak_env.size else np.zeros(num_complete_chunks)
        )
        rms_values = (
            rms_env[rms_indices] if rms_env.size else np.zeros(num_complete_chunks)
        )

        crest_factors = np.divide(
            peaks,
            rms_values,
            out=np.ones_like(peaks),
            where=rms_values > 0,
        )
        crest_factors = np.maximum(crest_factors, 1.0)
        crest_db = 20 * np.log10(crest_factors)
        peak_dbfs = 20 * np.log10(peaks * channel_peak + 1e-10)
        rms_dbfs = 20 * np.log10(rms_values * channel_peak + 1e-10)

        for time_value, crest, peak, rms in zip(
            time_points,
            np.where(np.isfinite(crest_db), crest_db, 0.0),
            np.where(np.isfinite(peak_dbfs), peak_dbfs, -120.0),
            np.where(np.isfinite(rms_dbfs), rms_dbfs, -120.0),
        ):
            rows.append(
                {
                    "frequency_hz": center_frequency,
                    "time_seconds": time_value,
                    "crest_factor_db": crest,
                    "peak_dbfs": peak,
                    "rms_dbfs": rms,
                }
            )

    _write_dataframe(path, pd.DataFrame(rows))


def _frequency_from_label(label: Any) -> Optional[float]:
    if label == "Full Spectrum":
        return 0.0
    try:
        return float(label)
    except (TypeError, ValueError):
        return None


def _write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
