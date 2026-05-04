"""Portable per-track analysis result bundle writer.

The bundle is the stable boundary between analysis and future rendering/UI code.
It stores derived tables and arrays needed to recreate current plots without
reloading the original audio.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.signal_metrics import sampled_max_abs
from src.version_info import get_application_dict

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
    advanced_statistics: Optional[Dict[str, Any]] = None,
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
    processing_warnings: list[str] = []

    metadata = _json_safe(track_metadata)
    _write_json(channel_dir / "metadata.json", metadata)
    _write_json(channel_dir / "plotting_config.json", _json_safe(plotting_config))
    _write_json(channel_dir / "envelope_config.json", _json_safe(envelope_config))
    _write_json(channel_dir / "analysis_config.json", _json_safe(analysis_config))

    _write_octave_band_table(channel_dir / "octave_band_analysis.csv", analysis_results)
    _write_time_domain_table(channel_dir / "time_domain_analysis.csv", time_analysis)
    _write_time_domain_summary(channel_dir / "time_domain_summary.csv", time_analysis)
    _write_key_value_table(
        channel_dir / "advanced_statistics.csv",
        advanced_statistics or {},
    )
    _write_extreme_chunk_table(
        channel_dir / "extreme_chunks_octave_analysis.csv",
        analysis_results,
        chunk_octave_analysis,
    )
    processing_warnings.extend(
        _write_histogram_tables(
            channel_dir=channel_dir,
            analysis_results=analysis_results,
            plotting_config=plotting_config,
            original_peak=float(track_metadata.get("original_peak", 1.0) or 1.0),
        )
    )
    _write_octave_time_metrics(
        channel_dir / "octave_time_metrics.csv",
        octave_bank=octave_bank,
        center_frequencies=list(center_frequencies),
        channel_data=channel_data,
        sample_rate=int(track_metadata.get("sample_rate", 44100)),
        original_peak=float(track_metadata.get("original_peak", 1.0)),
        analysis_config=analysis_config,
    )
    _write_envelope_summary_tables(channel_dir, envelope_statistics or {})
    if envelope_statistics is not None:
        try:
            _write_json(
                channel_dir / "envelope_plot_data.json",
                _extract_envelope_plot_data(envelope_statistics, envelope_config),
                indent=None,
            )
        except Exception as exc:
            message = f"Could not write envelope plot data: {exc}"
            processing_warnings.append(message)
            logger.warning(message)

    _update_manifest(
        bundle_dir=bundle_dir,
        track_metadata=metadata,
        channel_id=channel_id,
        channel_dir=channel_dir,
        processing_warnings=processing_warnings,
    )

    logger.info("Analysis result bundle updated: %s", bundle_dir)
    return bundle_dir


def _update_manifest(
    bundle_dir: Path,
    track_metadata: Dict[str, Any],
    channel_id: str,
    channel_dir: Path,
    processing_warnings: Optional[list[str]] = None,
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
    manifest["application"] = get_application_dict()

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
            "time_domain_summary": "time_domain_summary.csv",
            "advanced_statistics": "advanced_statistics.csv",
            "extreme_chunks_octave_analysis": "extreme_chunks_octave_analysis.csv",
            "histogram_linear": "histogram_linear.csv",
            "histogram_log_db": "histogram_log_db.csv",
            "octave_time_metrics": "octave_time_metrics.csv",
            "envelope_statistics": "envelope_statistics.csv",
            "envelope_pattern_analysis": "envelope_pattern_analysis.csv",
            "sustained_peaks_summary": "sustained_peaks_summary.csv",
            "sustained_peaks_events": "sustained_peaks_events.csv",
            "envelope_plot_data": "envelope_plot_data.json",
        },
    }
    artifacts = channels[channel_id]["artifacts"]
    present: Dict[str, bool] = {}
    missing: list[str] = []
    for artifact_name, rel_path in artifacts.items():
        is_present = (channel_dir / rel_path).exists()
        present[str(artifact_name)] = bool(is_present)
        if not is_present:
            missing.append(str(artifact_name))
    channels[channel_id]["processing"] = {
        "warnings": list(processing_warnings or []),
        "artifacts_present": present,
        "missing_artifacts": missing,
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
            "is_valid_crest_factor": stats.get("is_valid_crest_factor"),
            "crest_factor_method": stats.get(
                "crest_factor_method", "whole_interval_peak_rms"
            ),
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
    n = len(rows["time_seconds"])
    rows["is_valid_crest_factor"] = np.asarray(
        time_analysis.get(
            "is_valid_crest_factor",
            np.isfinite(rows["crest_factor_db"]),
        ),
        dtype=bool,
    )
    rows["crest_factor_window_seconds"] = np.full(
        n,
        float(time_analysis.get("chunk_duration", np.nan)),
    )
    rows["crest_factor_step_seconds"] = np.full(
        n,
        float(time_analysis.get("time_step_seconds", np.nan)),
    )
    rows["crest_factor_method"] = np.full(
        n,
        str(time_analysis.get("crest_factor_method", "unknown")),
        dtype=object,
    )
    _write_dataframe(path, pd.DataFrame(rows))


def _write_time_domain_summary(path: Path, time_analysis: Dict[str, Any]) -> None:
    valid_crest_db = np.asarray(time_analysis.get("crest_factors_db", []), dtype=float)
    valid_peak_dbfs = np.asarray(time_analysis.get("peak_levels_dbfs", []), dtype=float)
    valid_rms_dbfs = np.asarray(time_analysis.get("rms_levels_dbfs", []), dtype=float)
    valid_crest_db = valid_crest_db[np.isfinite(valid_crest_db)]
    valid_peak_dbfs = valid_peak_dbfs[np.isfinite(valid_peak_dbfs)]
    valid_rms_dbfs = valid_rms_dbfs[np.isfinite(valid_rms_dbfs)]

    summary = {
        "chunk_duration_seconds": time_analysis.get("chunk_duration"),
        "total_chunks": time_analysis.get("num_chunks"),
        "time_domain_mode": time_analysis.get("time_domain_mode", "unknown"),
        "time_domain_time_step_seconds": time_analysis.get("time_step_seconds"),
        "time_domain_rms_method": time_analysis.get(
            "time_domain_rms_method", "unknown"
        ),
        "time_domain_peak_method": time_analysis.get(
            "time_domain_peak_method", "unknown"
        ),
        "valid_crest_factor_windows": int(valid_crest_db.size),
        "invalid_crest_factor_windows": int(
            len(np.asarray(time_analysis.get("crest_factors_db", []), dtype=float))
            - valid_crest_db.size
        ),
        "crest_factor_mean_db": _array_stat(valid_crest_db, np.mean),
        "crest_factor_std_db": _array_stat(valid_crest_db, np.std),
        "crest_factor_min_db": _array_stat(valid_crest_db, np.min),
        "crest_factor_max_db": _array_stat(valid_crest_db, np.max),
        "peak_level_mean_dbfs": _array_stat(valid_peak_dbfs, np.mean),
        "peak_level_std_dbfs": _array_stat(valid_peak_dbfs, np.std),
        "rms_level_mean_dbfs": _array_stat(valid_rms_dbfs, np.mean),
        "rms_level_std_dbfs": _array_stat(valid_rms_dbfs, np.std),
    }
    _write_key_value_table(path, summary)


def _write_key_value_table(path: Path, values: Dict[str, Any]) -> None:
    rows = [
        {"parameter": key, "value": _json_safe(value)} for key, value in values.items()
    ]
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


def _write_envelope_summary_tables(
    channel_dir: Path,
    envelope_statistics: Dict[str, Any],
) -> None:
    envelope_rows = []
    pattern_rows = []
    sustained_rows = []
    event_rows = []

    for frequency_hz, band_data in envelope_statistics.items():
        for envelope in band_data.get("worst_case_envelopes", []):
            decay = envelope.get("decay_times", {})
            envelope_rows.append(
                {
                    "frequency_hz": frequency_hz,
                    "analysis_type": "worst_case",
                    "rank": envelope.get("rank"),
                    "peak_value_db": envelope.get("peak_value_db"),
                    "peak_time_seconds": envelope.get("peak_time_seconds"),
                    "attack_time_ms": envelope.get("attack_time_ms"),
                    "peak_hold_time_ms": envelope.get("peak_hold_time_ms"),
                    "decay_3db_ms": decay.get("decay_3db_ms"),
                    "decay_6db_ms": decay.get("decay_6db_ms"),
                    "decay_9db_ms": decay.get("decay_9db_ms"),
                    "decay_12db_ms": decay.get("decay_12db_ms"),
                    "decay_12db_reached": decay.get("decay_12db_reached", False),
                }
            )

        pattern_analysis = band_data.get("pattern_analysis", {})
        patterns_detected = int(pattern_analysis.get("patterns_detected", 0) or 0)
        for pattern_num in range(1, patterns_detected + 1):
            pattern = pattern_analysis.get(f"pattern_{pattern_num}")
            if not pattern:
                continue
            pattern_rows.append(
                {
                    "frequency_hz": frequency_hz,
                    "pattern_num": pattern_num,
                    "num_repetitions": pattern.get("num_repetitions"),
                    "mean_interval_seconds": pattern.get("mean_interval_seconds"),
                    "std_interval_seconds": pattern.get("std_interval_seconds"),
                    "median_interval_seconds": pattern.get("median_interval_seconds"),
                    "min_interval_seconds": pattern.get("min_interval_seconds"),
                    "max_interval_seconds": pattern.get("max_interval_seconds"),
                    "pattern_regularity_score": pattern.get("pattern_regularity_score"),
                    "pattern_confidence": pattern.get("pattern_confidence"),
                    "beats_per_minute": pattern.get("beats_per_minute"),
                }
            )

        sustained = band_data.get("sustained_peaks_summary", {})
        if sustained:
            row = {"frequency_hz": frequency_hz, "n_peaks": sustained.get("n_peaks", 0)}
            for metric in ("hold_ms", "t3_ms", "t6_ms", "t9_ms", "t12_ms"):
                stats = sustained.get(metric, {})
                for stat_name in ("mean", "median", "p90", "p95", "max"):
                    row[f"{metric}_{stat_name}"] = stats.get(stat_name, 0.0)
            sustained_rows.append(row)

        for event in band_data.get("sustained_peaks_events", []):
            event_rows.append(
                {
                    "frequency_hz": frequency_hz,
                    "peak_time_seconds": event.get("peak_time_seconds", 0.0),
                    "peak_value_db": event.get("peak_value_db", 0.0),
                    "hold_ms": event.get("hold_ms", 0.0),
                    "t3_ms": event.get("t3_ms", 0.0),
                    "t6_ms": event.get("t6_ms", 0.0),
                    "t9_ms": event.get("t9_ms", 0.0),
                    "t12_ms": event.get("t12_ms", 0.0),
                }
            )

    _write_dataframe(
        channel_dir / "envelope_statistics.csv", pd.DataFrame(envelope_rows)
    )
    _write_dataframe(
        channel_dir / "envelope_pattern_analysis.csv",
        pd.DataFrame(pattern_rows),
    )
    _write_dataframe(
        channel_dir / "sustained_peaks_summary.csv",
        pd.DataFrame(sustained_rows),
    )
    _write_dataframe(
        channel_dir / "sustained_peaks_events.csv",
        pd.DataFrame(event_rows),
    )


def _write_histogram_tables(
    channel_dir: Path,
    analysis_results: Dict[str, Any],
    plotting_config: Dict[str, Any],
    original_peak: float,
) -> list[str]:
    warnings: list[str] = []
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
            try:
                counts = np.zeros(len(log_bin_edges) - 1, dtype=np.int64)
                total = 0
                chunk_size = int(plotting_config.get("histogram_chunk_size", 2_000_000))
                floor_linear = 10 ** (noise_floor_db / 20)
                for start in range(0, clean_signal.size, chunk_size):
                    chunk = np.asarray(
                        clean_signal[start : start + chunk_size], dtype=np.float32
                    )
                    if chunk.size == 0:
                        continue
                    chunk = np.abs(chunk) * float(original_peak)
                    chunk[chunk == 0] = floor_linear
                    chunk_db = 20.0 * np.log10(chunk, dtype=np.float32)
                    chunk_db = chunk_db[
                        (chunk_db >= noise_floor_db) & (chunk_db <= max_db)
                    ]
                    if chunk_db.size == 0:
                        continue
                    c, _ = np.histogram(chunk_db, bins=log_bin_edges)
                    counts += c.astype(np.int64, copy=False)
                    total += int(chunk_db.size)

                if total:
                    edges = log_bin_edges
                    bin_widths = np.diff(edges)
                    density = counts / (max(total, 1) * np.maximum(bin_widths, 1e-12))
                    log_rows.extend(
                        _histogram_rows(label, frequency_hz, counts, density, edges)
                    )
            except MemoryError as exc:
                message = (
                    f"Skipping log histogram for {label} due to memory pressure: {exc}"
                )
                warnings.append(message)
                logger.warning(message)

    _write_dataframe(channel_dir / "histogram_linear.csv", pd.DataFrame(linear_rows))
    _write_dataframe(channel_dir / "histogram_log_db.csv", pd.DataFrame(log_rows))
    return warnings


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
    original_peak: float,
    analysis_config: Dict[str, Any],
) -> None:
    rows = []
    window_seconds = float(
        analysis_config.get("crest_factor_window_seconds", 2.0) or 2.0
    )
    step_seconds = float(analysis_config.get("crest_factor_step_seconds", 1.0) or 1.0)
    rms_floor_dbfs = float(
        analysis_config.get("crest_factor_rms_floor_dbfs", -80.0) or -80.0
    )
    window_samples = max(int(sample_rate * window_seconds), 1)
    step_samples = max(int(sample_rate * step_seconds), 1)
    num_samples = len(channel_data)
    num_windows = (num_samples - window_samples) // step_samples + 1
    if num_windows <= 0:
        _write_dataframe(path, pd.DataFrame(rows))
        return

    starts = np.arange(num_windows, dtype=np.int64) * step_samples
    ends = starts + window_samples
    time_points = ends / float(sample_rate)
    channel_peak = float(original_peak)

    for band_index, center_frequency in enumerate(center_frequencies):
        band_data = np.asarray(octave_bank[:, band_index + 1], dtype=float)
        peaks = sampled_max_abs(band_data, window_samples, step_samples)[:num_windows]
        band_squared = np.square(band_data.astype(np.float64, copy=False))
        csum = np.concatenate(([0.0], np.cumsum(band_squared)))
        sum_sq = csum[ends] - csum[starts]
        rms_values = np.sqrt(np.clip(sum_sq / float(window_samples), 0.0, None))
        peak_dbfs = 20 * np.log10(peaks * channel_peak + 1e-10)
        rms_dbfs = 20 * np.log10(rms_values * channel_peak + 1e-10)
        valid = (rms_values > 0) & (rms_dbfs >= rms_floor_dbfs)

        crest_factors = np.full_like(peaks, np.nan, dtype=np.float64)
        crest_factors[valid] = np.divide(
            peaks[valid],
            rms_values[valid],
            out=np.ones_like(peaks[valid]),
            where=rms_values[valid] > 0,
        )
        crest_factors[valid] = np.maximum(crest_factors[valid], 1.0)
        crest_db = np.full_like(peaks, np.nan, dtype=np.float64)
        crest_db[valid] = 20 * np.log10(crest_factors[valid])

        for time_value, crest, peak, rms, is_valid in zip(
            time_points,
            crest_db,
            np.where(np.isfinite(peak_dbfs), peak_dbfs, -120.0),
            np.where(np.isfinite(rms_dbfs), rms_dbfs, -120.0),
            valid,
        ):
            rows.append(
                {
                    "frequency_hz": center_frequency,
                    "time_seconds": time_value,
                    "crest_factor_db": crest,
                    "peak_dbfs": peak,
                    "rms_dbfs": rms,
                    "is_valid_crest_factor": bool(is_valid),
                    "crest_factor_window_seconds": window_seconds,
                    "crest_factor_step_seconds": step_seconds,
                    "crest_factor_method": "fixed_window_peak_rms",
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


def _extract_envelope_plot_data(
    envelope_statistics: Dict[str, Any],
    envelope_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract only the envelope windows needed for current plot replay."""
    plot_data: Dict[str, Any] = {}
    pattern_plot_count = max(
        0,
        int(envelope_config.get("envelope_plots_num_pattern_envelopes", 10) or 0),
    )
    independent_plot_count = max(
        0,
        int(envelope_config.get("envelope_plots_num_independent_envelopes", 3) or 0),
    )
    for frequency_label, band_data in envelope_statistics.items():
        band_plot_data: Dict[str, Any] = {}
        pattern_analysis = band_data.get("pattern_analysis", {})
        patterns_detected = int(pattern_analysis.get("patterns_detected", 0) or 0)
        selected_pattern_windows = _select_top_pattern_window_indices(
            pattern_analysis,
            patterns_detected,
            pattern_plot_count,
        )
        compact_patterns: Dict[str, Any] = {"patterns_detected": patterns_detected}
        for pattern_num in range(1, patterns_detected + 1):
            pattern_key = f"pattern_{pattern_num}"
            pattern = pattern_analysis.get(pattern_key)
            if not pattern:
                continue
            selected_indices = selected_pattern_windows.get(pattern_num, [])
            compact_patterns[pattern_key] = {
                "num_repetitions": pattern.get("num_repetitions"),
                "mean_interval_seconds": pattern.get("mean_interval_seconds"),
                "std_interval_seconds": pattern.get("std_interval_seconds"),
                "median_interval_seconds": pattern.get("median_interval_seconds"),
                "min_interval_seconds": pattern.get("min_interval_seconds"),
                "max_interval_seconds": pattern.get("max_interval_seconds"),
                "pattern_regularity_score": pattern.get("pattern_regularity_score"),
                "pattern_confidence": pattern.get("pattern_confidence"),
                "beats_per_minute": pattern.get("beats_per_minute"),
                "peak_times_seconds": _select_items(
                    pattern.get("peak_times_seconds", []),
                    selected_indices,
                ),
                "envelope_windows": _select_items(
                    pattern.get("envelope_windows", []),
                    selected_indices,
                ),
                "time_window_axes": _select_time_axes(
                    pattern.get("time_windows_ms", []),
                    pattern.get("envelope_windows", []),
                    selected_indices,
                ),
            }
        band_plot_data["pattern_analysis"] = compact_patterns

        compact_worst_cases = []
        for envelope in band_data.get("worst_case_envelopes", [])[
            :independent_plot_count
        ]:
            compact_worst_cases.append(
                {
                    "rank": envelope.get("rank"),
                    "peak_value_db": envelope.get("peak_value_db"),
                    "peak_time_seconds": envelope.get("peak_time_seconds"),
                    "attack_time_ms": envelope.get("attack_time_ms"),
                    "peak_hold_time_ms": envelope.get("peak_hold_time_ms"),
                    "decay_times": envelope.get("decay_times", {}),
                    "envelope_window": envelope.get("envelope_window"),
                    "time_window_axis": _compact_time_axis(
                        envelope.get("time_window_ms"),
                        envelope.get("envelope_window"),
                    ),
                }
            )
        band_plot_data["worst_case_envelopes"] = compact_worst_cases
        plot_data[str(frequency_label)] = band_plot_data
    return _json_safe(plot_data)


def _select_top_pattern_window_indices(
    pattern_analysis: Dict[str, Any],
    patterns_detected: int,
    limit: int,
) -> Dict[int, list[int]]:
    """Return original window indices for the top pattern envelopes by peak level."""
    if limit <= 0:
        return {}
    candidates: list[tuple[float, int, int]] = []
    for pattern_num in range(1, patterns_detected + 1):
        pattern = pattern_analysis.get(f"pattern_{pattern_num}", {})
        windows = pattern.get("envelope_windows", []) or []
        for window_index, envelope_window in enumerate(windows):
            if envelope_window is None:
                continue
            values = np.asarray(envelope_window, dtype=float)
            if values.size == 0:
                continue
            candidates.append((float(np.nanmax(values)), pattern_num, window_index))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: Dict[int, list[int]] = {}
    for _, pattern_num, window_index in candidates[:limit]:
        selected.setdefault(pattern_num, []).append(window_index)
    for indices in selected.values():
        indices.sort()
    return selected


def _select_items(values: Any, indices: list[int]) -> list[Any]:
    """Select list-like values by index while tolerating missing entries."""
    if values is None:
        return []
    try:
        value_count = len(values)
    except TypeError:
        return []
    return [values[index] for index in indices if index < value_count]


def _select_time_axes(
    time_windows_ms: Any,
    envelope_windows: Any,
    indices: list[int],
) -> list[Dict[str, Any]]:
    """Select compact time-axis metadata for the retained envelope windows."""
    axes: list[Dict[str, Any]] = []
    for index in indices:
        time_window = (
            time_windows_ms[index]
            if time_windows_ms is not None and index < len(time_windows_ms)
            else None
        )
        envelope_window = (
            envelope_windows[index]
            if envelope_windows is not None and index < len(envelope_windows)
            else None
        )
        axes.append(_compact_time_axis(time_window, envelope_window))
    return axes


def _compact_time_axis(time_window_ms: Any, envelope_window: Any) -> Dict[str, Any]:
    """Represent a plot time axis without storing every sample timestamp."""
    if time_window_ms is None:
        return {}
    times = np.asarray(time_window_ms, dtype=float)
    if times.size == 0:
        return {}

    if times.size == 1:
        return {"start_ms": float(times[0]), "step_ms": 0.0}

    step_ms = float(times[1] - times[0])
    reconstructed = times[0] + (np.arange(times.size) * step_ms)
    if np.allclose(times, reconstructed, rtol=1e-9, atol=1e-9):
        return {"start_ms": float(times[0]), "step_ms": step_ms}

    return {"values_ms": time_window_ms}


def _write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _write_json(path: Path, data: Dict[str, Any], indent: Optional[int] = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_kwargs = {"indent": indent}
    if indent is None:
        json_kwargs["separators"] = (",", ":")
    payload = json.dumps(data, **json_kwargs)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _array_stat(values: np.ndarray, func) -> Optional[float]:
    return float(func(values)) if values.size else None


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
