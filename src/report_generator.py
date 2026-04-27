"""Report generator for analysis results."""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.results.reader import ChannelResult, ResultBundle

logger = logging.getLogger(__name__)


def _parse_csv_section(
    csv_path: Path, section_name: str
) -> Optional[Dict[str, Dict[str, str]]]:
    """Parse a section from analysis_results.csv."""
    if not csv_path.exists():
        return None
    text = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = text.index(f"[{section_name}]") + 1
    except ValueError:
        return None
    if start >= len(text):
        return None

    header = text[start].split(",")
    key_to_idx = {k: i for i, k in enumerate(header)}
    result: Dict[str, Dict[str, str]] = {}

    for line in text[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        row_key = parts[0]
        row_data: Dict[str, str] = {}
        for k, idx in key_to_idx.items():
            if idx < len(parts):
                row_data[k] = parts[idx]
        result[row_key] = row_data

    return result


def _parse_advanced_stats(csv_path: Path) -> Dict[str, float]:
    """Parse ADVANCED_STATISTICS section."""
    section = _parse_csv_section(csv_path, "ADVANCED_STATISTICS")
    if not section:
        return {}
    # ADVANCED_STATISTICS has parameter,value,description format
    # We need to find the row with the parameter
    result: Dict[str, float] = {}
    text = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = text.index("[ADVANCED_STATISTICS]") + 1
    except ValueError:
        return {}
    if start >= len(text):
        return {}
    # Skip header
    for line in text[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",")
        if len(parts) >= 2:
            key = parts[0]
            try:
                result[key] = float(parts[1])
            except (ValueError, IndexError):
                pass
    return result


def _parse_sustained_peaks_summary(csv_path: Path) -> Optional[Dict[str, float]]:
    """Parse SUSTAINED_PEAKS_SUMMARY section and return Full Spectrum data."""
    section = _parse_csv_section(csv_path, "SUSTAINED_PEAKS_SUMMARY")
    if not section:
        return None
    full_spectrum = section.get("Full Spectrum")
    if not full_spectrum:
        return None
    # Convert string values to float
    result: Dict[str, float] = {}
    for k, v in full_spectrum.items():
        if k == "frequency_hz":
            continue
        try:
            result[k] = float(v)
        except (ValueError, TypeError):
            pass
    return result


def _parse_octave_band_analysis(csv_path: Path) -> Optional[Dict[str, float]]:
    """Parse OCTAVE_BAND_ANALYSIS section and return Full Spectrum data."""
    section = _parse_csv_section(csv_path, "OCTAVE_BAND_ANALYSIS")
    if not section:
        return None
    full_spectrum = section.get("Full Spectrum")
    if not full_spectrum:
        return None
    result: Dict[str, float] = {}
    for k, v in full_spectrum.items():
        if k == "frequency_hz":
            continue
        try:
            result[k] = float(v)
        except (ValueError, TypeError):
            pass
    return result


def _parse_time_domain_summary(csv_path: Path) -> Dict[str, str]:
    """Parse TIME_DOMAIN_SUMMARY section as string values."""
    if not csv_path.exists():
        return {}

    text = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = text.index("[TIME_DOMAIN_SUMMARY]") + 1
    except ValueError:
        return {}
    if start >= len(text):
        return {}

    result: Dict[str, str] = {}
    for line in text[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",", 1)
        if len(parts) == 2:
            result[parts[0].strip()] = parts[1].strip()
    return result


def _parse_track_metadata(csv_path: Path) -> Dict[str, str]:
    """Parse TRACK_METADATA section as string values."""
    if not csv_path.exists():
        return {}

    text = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = text.index("[TRACK_METADATA]") + 1
    except ValueError:
        return {}
    if start >= len(text):
        return {}

    result: Dict[str, str] = {}
    for line in text[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0] != "parameter":
            result[parts[0].strip()] = parts[1].strip()
    return result


def _as_float(value: object) -> Optional[float]:
    """Convert a CSV/config value to float, returning None for invalid values."""
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _format_seconds(value: Optional[float]) -> str:
    """Format seconds for report prose."""
    if value is None:
        return "unknown"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}-second"
    return f"{value:g}-second"


def _select_time_domain_summary(group_data: Dict[str, Dict]) -> Dict[str, str]:
    """Pick the first available time-domain summary for report-level prose."""
    for data in group_data.values():
        summary = data.get("time_summary", {})
        if summary:
            return summary
    return {}


def _select_track_metadata(group_data: Dict[str, Dict]) -> Dict[str, str]:
    """Pick the first available track metadata for report-level prose."""
    for data in group_data.values():
        metadata = data.get("track_metadata", {})
        if metadata:
            return metadata
    return {}


def _octave_processing_sentence(metadata: Dict[str, str]) -> str:
    """Build report prose from exported octave processing metadata."""
    design = metadata.get("octave_filter_design", "").strip()
    effective_mode = metadata.get("octave_effective_processing_mode", "").strip()
    requested_mode = metadata.get("octave_requested_processing_mode", "").strip()
    storage = metadata.get("octave_output_storage", "").strip()
    max_memory = metadata.get("octave_max_memory_gb", "").strip()
    block_seconds = _as_float(metadata.get("octave_fft_block_duration_seconds"))

    if design == "fft_power_complementary":
        mode_text = effective_mode or requested_mode or "unknown"
        sentence = (
            "For this run, octave bands used the FFT power-complementary "
            f"filter bank in `{mode_text}` mode"
        )
        if requested_mode and effective_mode and requested_mode != effective_mode:
            sentence += f" (requested `{requested_mode}`)"
        if effective_mode == "block":
            sentence += f" with {_format_seconds(block_seconds)} FFT blocks"
        if storage:
            sentence += f" and `{storage}` octave-bank storage"
        if max_memory:
            sentence += f" under a configured {max_memory} GB RAM budget"
        sentence += "."
        return sentence

    return (
        "Octave processing metadata was not found in the CSV, so this report "
        "does not know which filter-bank mode produced the octave results."
    )


def _time_domain_sample_label(summary: Dict[str, str]) -> str:
    """Describe the exported time-domain sample unit."""
    mode = summary.get("time_domain_mode", "").strip()
    chunk_duration = _as_float(summary.get("chunk_duration_seconds"))
    step_seconds = _as_float(summary.get("time_domain_time_step_seconds"))

    if mode == "fixed_chunk":
        return f"{_format_seconds(chunk_duration)} fixed windows"
    if mode == "slow":
        return f"{_format_seconds(step_seconds)} slow-mode samples"
    if chunk_duration is not None:
        return f"{_format_seconds(chunk_duration)} time-domain samples"
    return "exported time-domain samples"


def _time_domain_calculation_sentence(summary: Dict[str, str]) -> str:
    """Build report prose from the exported time-domain metadata."""
    mode = summary.get("time_domain_mode", "").strip()
    rms_method = summary.get("time_domain_rms_method", "").strip()
    peak_method = summary.get("time_domain_peak_method", "").strip()
    chunk_duration = _as_float(summary.get("chunk_duration_seconds"))
    step_seconds = _as_float(summary.get("time_domain_time_step_seconds"))

    if mode == "slow":
        return (
            "For this run, time-domain crest factor uses slow mode: "
            f"RMS is calculated with {rms_method or 'the configured slow RMS method'}, "
            f"peaks use {peak_method or 'the configured peak method'}, and values "
            f"are exported every {_format_seconds(step_seconds)} interval."
        )
    if mode == "fixed_chunk":
        return (
            "For this run, time-domain crest factor uses fixed-window mode: "
            f"RMS and peak levels are calculated over "
            f"{_format_seconds(chunk_duration)} windows."
        )
    if summary:
        return (
            "For this run, time-domain calculation metadata is present in "
            "`[TIME_DOMAIN_SUMMARY]`, but the mode is not recognized by this "
            "report generator."
        )
    return (
        "Time-domain calculation metadata was not found in the CSV, so this "
        "report only describes the exported time-domain samples."
    )


def _read_worst_channels_manifest(track_dir: Path) -> Dict[str, Dict[str, str]]:
    """Read worst_channels_manifest.csv and return group -> channel mapping."""
    manifest_path = track_dir / "worst_channels_manifest.csv"
    if not manifest_path.exists():
        return {}
    result: Dict[str, Dict[str, str]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row.get("group", "").lower()
            if group:
                result[group] = {
                    "folder": row.get("folder", ""),
                    "score": row.get("score", ""),
                    "metric_used": row.get("metric_used", ""),
                }
    return result


def _determine_channel_groups(track_dir: Path) -> Dict[str, List[str]]:
    """Determine channel groups based on available channels."""
    manifest = _read_worst_channels_manifest(track_dir)
    groups: Dict[str, List[str]] = {}

    # Check for standard cinema groups
    if "screen" in manifest:
        groups["Screen"] = [manifest["screen"]["folder"]]
    if "lfe" in manifest:
        groups["LFE"] = [manifest["lfe"]["folder"]]
    if "surround" in manifest:
        groups["Surround+Height"] = [manifest["surround"]["folder"]]

    # If no manifest, try to infer from channel folders
    if not groups:
        channel_folders = [
            d.name
            for d in track_dir.iterdir()
            if d.is_dir() and d.name.startswith("Channel ")
        ]
        # Simple heuristic: look for LFE, then stereo
        for folder in channel_folders:
            if "LFE" in folder:
                groups["LFE"] = [folder]
            elif "FC" in folder or "FL" in folder or "FR" in folder:
                if "Screen" not in groups:
                    groups["Screen"] = []
                groups["Screen"].append(folder)
            elif "SL" in folder or "SR" in folder or "SBL" in folder or "SBR" in folder:
                if "Surround+Height" not in groups:
                    groups["Surround+Height"] = []
                groups["Surround+Height"].append(folder)

        # If still no groups, check for mono track (analysis_results.csv in root)
        if not groups:
            mono_csv = track_dir / "analysis_results.csv"
            if mono_csv.exists():
                groups["Mono"] = [""]  # Empty string means root directory
            elif channel_folders:
                groups["All Channels"] = channel_folders

    return groups


def _format_table_row(values: List[str]) -> str:
    """Format a table row."""
    return "| " + " | ".join(values) + " |"


def _format_number(value: float, decimals: int = 1) -> str:
    """Format a number with specified decimals."""
    if value == 0.0:
        return "0.0"
    return f"{value:.{decimals}f}"


def _markdown_image(alt_text: str, rel_path: str) -> str:
    """Return a Markdown image link with a URI-safe relative path."""
    return f"![{alt_text}]({quote(rel_path, safe='/.')})"


def _safe_image_filename(filename: str) -> str:
    """Return a filesystem-safe image filename for report-local assets."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"{safe_stem or 'image'}{suffix.lower()}"


def _copy_report_image(source_path: Path, report_folder: Path, filename: str) -> str:
    """Copy an analysis image beside the report and return a Markdown path."""
    images_dir = report_folder / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dest_path = images_dir / _safe_image_filename(filename)
    shutil.copy2(source_path, dest_path)
    return f"images/{dest_path.name}"


def _markdown_report_image(
    alt_text: str, source_path: Path, report_folder: Path, filename: str
) -> str:
    """Return a Markdown image link to a report-local copied image."""
    rel_path = _copy_report_image(source_path, report_folder, filename)
    return _markdown_image(alt_text, rel_path)


def _group_plot_reference(
    track_dir: Path,
    report_folder: Path,
    base_rel: str,
    group_name: str,
    group_data: Dict,
    filename: str,
) -> Tuple[Path, str, Path]:
    """Return actual, Markdown-relative, and verification paths for a group plot."""
    channel_folder = group_data.get(group_name, {}).get("channel_folder")
    if channel_folder == "":
        rel_path = f"{base_rel}/{filename}"
        plot_path = track_dir / filename
        abs_path_from_reports = (report_folder / base_rel / filename).resolve()
    else:
        rel_path = f"{base_rel}/{group_name}/{filename}"
        plot_path = track_dir / group_name / filename
        abs_path_from_reports = (
            report_folder / base_rel / group_name / filename
        ).resolve()
    return plot_path, rel_path, abs_path_from_reports


def generate_bundle_report(
    bundle: ResultBundle,
    rendered_output_dir: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a rich Markdown report from a `.aaresults` bundle.

    Args:
        bundle: Loaded analysis result bundle.
        rendered_output_dir: Directory containing plots rendered from the bundle.
        output_path: Optional output Markdown path. Defaults to `analysis.md`
            in `rendered_output_dir`.

    Returns:
        Path to the written report.
    """
    report_path = output_path or rendered_output_dir / "analysis.md"
    report_folder = report_path.parent
    report_folder.mkdir(parents=True, exist_ok=True)

    track_name = Path(str(bundle.track.get("track_name") or bundle.path.stem)).stem
    groups = _bundle_channel_groups(bundle)
    group_data = {
        group_name: _bundle_group_summary(channels)
        for group_name, channels in groups.items()
        if channels
    }
    time_summary = _select_time_domain_summary(group_data)
    metadata = _bundle_report_metadata(bundle)
    time_sample_label = _time_domain_sample_label(time_summary)
    time_calculation_sentence = _time_domain_calculation_sentence(time_summary)
    octave_processing_sentence = _octave_processing_sentence(metadata)

    lines: List[str] = []
    lines.append(f"# {track_name} - Audio Signal Analysis")
    lines.append("")
    lines.append("## What This Report Tells You")
    lines.append("")
    lines.append(
        "- **Group comparison:** Statistical analysis across channel groups "
        f"({', '.join(groups.keys())})"
    )
    lines.append(
        "- **Peak occurrence and duty cycle:** Frequency of peaks near full scale "
        "and time spent at peak levels"
    )
    lines.append(
        "- **Peak hold and recovery times:** How long peaks are sustained and "
        "recovery times to -3/-6/-9/-12 dB"
    )
    lines.append(
        "- **Frequency-dependent crest factor behavior:** Wideband and octave-band "
        "dynamic range characteristics"
    )
    lines.append("")
    lines.append("## Processing Summary")
    lines.append("")
    lines.append(f"- Source bundle: `{bundle.path.name}`")
    lines.append(f"- Rendered output folder: `{rendered_output_dir.name}`")
    lines.append(f"- Channels: {len(bundle.channels())}")
    lines.append(
        f"- Duration: {_format_number(float(bundle.track.get('duration_seconds', 0.0)), 1)} s"
    )
    lines.append(
        f"- Sample rate: {_format_number(float(bundle.track.get('sample_rate', 0.0)), 0)} Hz"
    )
    lines.append(f"- {octave_processing_sentence}")
    lines.append("")

    lines.extend(
        _bundle_crest_factor_section(
            group_data, time_sample_label, time_calculation_sentence
        )
    )
    lines.extend(_bundle_group_plot_section(rendered_output_dir, report_folder, groups))
    lines.extend(
        _bundle_frequency_section(group_data, rendered_output_dir, report_folder)
    )
    lines.extend(_bundle_sustained_peak_section(group_data))
    lines.extend(_bundle_peak_decay_section(rendered_output_dir, report_folder))
    lines.extend(_bundle_peak_occurrence_section(group_data))
    lines.extend(_bundle_envelope_section(bundle, rendered_output_dir, report_folder))
    lines.extend(_bundle_appendix(bundle, group_data))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Bundle report written to: %s", report_path)
    return report_path


def _bundle_crest_factor_section(
    group_data: Dict[str, Dict],
    time_sample_label: str,
    time_calculation_sentence: str,
) -> List[str]:
    lines = [
        "## Crest Factor Analysis",
        "",
        "Crest factor (peak-to-RMS ratio) indicates the dynamic range of the audio signal. "
        f"This section provides statistical analysis using {time_sample_label} "
        "from all channels within each group. "
        f"{time_calculation_sentence}",
        "",
        "**Percentile Definitions:** P90 indicates that 90% of exported time-domain "
        "samples have a crest factor at or below this value. P95 indicates the same "
        "for 95% of exported samples.",
        "",
        _format_table_row(
            [
                "Group",
                "Min (dB)",
                "P95 (dB)",
                "P90 (dB)",
                "Mean (dB)",
                "Median (dB)",
                "Max (dB)",
                "Std Dev (dB)",
            ]
        ),
        _format_table_row(["---"] * 8),
    ]
    for group_name in sorted(group_data.keys()):
        values = group_data[group_name].get("crest_values", np.array([]))
        if values.size == 0:
            continue
        lines.append(
            _format_table_row(
                [
                    group_name,
                    _format_number(float(np.min(values)), 1),
                    _format_number(float(np.percentile(values, 95)), 1),
                    _format_number(float(np.percentile(values, 90)), 1),
                    _format_number(float(np.mean(values)), 1),
                    _format_number(float(np.median(values)), 1),
                    _format_number(float(np.max(values)), 1),
                    _format_number(float(np.std(values)), 1),
                ]
            )
        )
    lines.append("")
    return lines


def _bundle_group_plot_section(
    rendered_output_dir: Path,
    report_folder: Path,
    groups: Dict[str, List[ChannelResult]],
) -> List[str]:
    lines = ["## Group Plots", ""]
    for group_name in groups.keys():
        lines.append(f"### {group_name}")
        lines.append("")
        crest_plot = rendered_output_dir / group_name / "crest_factor_time.png"
        octave_plot = rendered_output_dir / group_name / "octave_spectrum.png"
        if crest_plot.exists():
            lines.append(
                _markdown_report_image(
                    f"Crest Factor Over Time - {group_name}",
                    crest_plot,
                    report_folder,
                    f"{group_name}_crest_factor_time.png",
                )
            )
        else:
            lines.append("*Crest factor time plot not available*")
        lines.append("")
        if octave_plot.exists():
            lines.append(
                _markdown_report_image(
                    f"Octave Spectrum - {group_name}",
                    octave_plot,
                    report_folder,
                    f"{group_name}_octave_spectrum.png",
                )
            )
        else:
            lines.append("*Octave spectrum plot not available*")
        lines.append("")
    return lines


def _bundle_frequency_section(
    group_data: Dict[str, Dict],
    rendered_output_dir: Path,
    report_folder: Path,
) -> List[str]:
    lines = [
        "## Frequency Spectrum Summary",
        "",
        "The table below summarizes the wideband full-spectrum row for each group. "
        "Detailed per-channel octave-band plots are generated beside the report.",
        "",
        _format_table_row(
            [
                "Group",
                "Peak (dBFS)",
                "RMS (dBFS)",
                "Crest Factor (dB)",
                "Dynamic Range (dB)",
            ]
        ),
        _format_table_row(["---"] * 5),
    ]
    for group_name, data in group_data.items():
        octave = data.get("octave", {})
        lines.append(
            _format_table_row(
                [
                    group_name,
                    _format_optional_number(octave.get("max_amplitude_db"), 1),
                    _format_optional_number(octave.get("rms_db"), 1),
                    _format_optional_number(octave.get("crest_factor_db"), 1),
                    _format_optional_number(octave.get("dynamic_range_db"), 1),
                ]
            )
        )
    lines.append("")
    lines.append("### Per-Channel Core Plots")
    lines.append("")
    for group in group_data.values():
        for channel in group.get("channels", []):
            channel_dir = rendered_output_dir / channel.channel_id
            lines.append(f"#### {channel.channel_name}")
            lines.append("")
            for filename, title in (
                ("octave_spectrum.png", "Octave Spectrum"),
                ("crest_factor.png", "Crest Factor Spectrum"),
                ("crest_factor_time.png", "Crest Factor Over Time"),
                ("octave_crest_factor_time.png", "Octave Crest Factor Over Time"),
                ("histograms.png", "Linear Histogram"),
                ("histograms_log_db.png", "Log dB Histogram"),
            ):
                plot_path = channel_dir / filename
                if plot_path.exists():
                    lines.append(
                        _markdown_report_image(
                            f"{title} - {channel.channel_name}",
                            plot_path,
                            report_folder,
                            f"{channel.channel_id}_{filename}",
                        )
                    )
                    lines.append("")
    return lines


def _bundle_sustained_peak_section(group_data: Dict[str, Dict]) -> List[str]:
    lines = [
        "## Sustained-Peak Recovery Summary",
        "",
        "This section summarizes independent peaks and recovery times from the "
        "full-spectrum sustained-peak table. P95 values are used as realistic "
        "worst-case recovery indicators.",
        "",
        _format_table_row(
            [
                "Group",
                "Peaks",
                "Hold Mean (ms)",
                "Hold P95 (ms)",
                "T3 P95 (ms)",
                "T6 P95 (ms)",
                "T9 P95 (ms)",
                "T12 P95 (ms)",
            ]
        ),
        _format_table_row(["---"] * 8),
    ]
    for group_name, data in group_data.items():
        sustained = data.get("sustained", {})
        lines.append(
            _format_table_row(
                [
                    group_name,
                    _format_optional_number(sustained.get("n_peaks"), 0),
                    _format_optional_number(sustained.get("hold_ms_mean"), 1),
                    _format_optional_number(sustained.get("hold_ms_p95"), 1),
                    _format_optional_number(sustained.get("t3_ms_p95"), 1),
                    _format_optional_number(sustained.get("t6_ms_p95"), 1),
                    _format_optional_number(sustained.get("t9_ms_p95"), 1),
                    _format_optional_number(sustained.get("t12_ms_p95"), 1),
                ]
            )
        )
    lines.append("")
    return lines


def _bundle_peak_decay_section(
    rendered_output_dir: Path, report_folder: Path
) -> List[str]:
    lines = ["## Peak Decay Characteristics", ""]
    decay_plot = rendered_output_dir / "peak_decay_groups.png"
    if decay_plot.exists():
        lines.append(
            _markdown_report_image(
                "Peak Decay Curves",
                decay_plot,
                report_folder,
                "peak_decay_groups.png",
            )
        )
    else:
        lines.append("*Peak decay plot not available*")
    lines.append("")
    return lines


def _bundle_peak_occurrence_section(group_data: Dict[str, Dict]) -> List[str]:
    lines = [
        "## Peak Occurrence and Duty Cycle",
        "",
        "This section analyzes the frequency of peak events at different amplitude "
        "thresholds and the percentage of time the signal spends above -3 dBFS.",
        "",
        _format_table_row(
            [
                "Group",
                "Peaks >-3dB/s",
                "Peaks >-1dB/s",
                "Peaks >-0.1dB/s",
                "Loud duty % (>-3 dBFS)",
            ]
        ),
        _format_table_row(["---"] * 5),
    ]
    for group_name, data in group_data.items():
        advanced = data.get("advanced", {})
        lines.append(
            _format_table_row(
                [
                    group_name,
                    _format_optional_number(
                        advanced.get("peaks_above_minus3db_per_sec"), 1
                    ),
                    _format_optional_number(
                        advanced.get("peaks_above_minus1db_per_sec"), 1
                    ),
                    _format_optional_number(
                        advanced.get("peaks_above_minus0_1db_per_sec"), 2
                    ),
                    _format_optional_number(
                        advanced.get("peak_distribution_loud_percent"), 4
                    ),
                ]
            )
        )
    lines.append("")
    for group_name, data in group_data.items():
        advanced = data.get("advanced", {})
        peaks_minus0_1 = float(advanced.get("peaks_above_minus0_1db_per_sec", 0.0))
        peaks_minus1 = float(advanced.get("peaks_above_minus1db_per_sec", 0.0))
        loud_duty = float(advanced.get("peak_distribution_loud_percent", 0.0))
        if peaks_minus0_1 > 0.1:
            highlight = (
                f"frequent near-clipping peaks ({peaks_minus0_1:.2f}/s above -0.1 dBFS)"
            )
        elif peaks_minus1 > 1.0:
            highlight = f"moderate hot peaks ({peaks_minus1:.1f}/s above -1 dBFS)"
        else:
            highlight = "low near-clipping activity"
        lines.append(
            f"**{group_name}:** {highlight} with {loud_duty:.4f}% duty cycle above -3 dBFS."
        )
        lines.append("")
    return lines


def _bundle_envelope_section(
    bundle: ResultBundle,
    rendered_output_dir: Path,
    report_folder: Path,
) -> List[str]:
    lines = [
        "## Envelope Analysis",
        "",
        "Pattern and independent envelope plots are rendered from compact stored "
        "envelope windows in `envelope_plot_data.json`.",
        "",
    ]
    for channel in bundle.channels():
        lines.append(f"### {channel.channel_name}")
        lines.append("")
        for folder_name, label in (
            ("pattern_envelopes", "Pattern envelope plots"),
            ("independent_envelopes", "Independent envelope plots"),
        ):
            folder = rendered_output_dir / channel.channel_id / folder_name
            png_count = len(list(folder.glob("*.png"))) if folder.exists() else 0
            rel_folder = Path(os.path.relpath(folder, report_folder)).as_posix()
            lines.append(f"- {label}: `{rel_folder}` ({png_count} plot files)")
        lines.append("")
    return lines


def _bundle_appendix(bundle: ResultBundle, group_data: Dict[str, Dict]) -> List[str]:
    lines = [
        "## Appendix: Data Sources",
        "",
        "### Bundle Artifacts Used",
        "",
        "- `metadata.json`: Track and channel metadata",
        "- `octave_band_analysis.csv`: Wideband and octave-band peak/RMS/crest data",
        "- `time_domain_analysis.csv`: Time-sampled peak/RMS/crest metrics",
        "- `advanced_statistics.csv`: Peak rates and duty-cycle statistics",
        "- `sustained_peaks_summary.csv`: Peak hold and recovery statistics",
        "- `envelope_plot_data.json`: Pattern and independent envelope replay windows",
        "",
        "### Channel Sources",
        "",
    ]
    for group_name, data in group_data.items():
        channels = data.get("channels", [])
        channel_list = ", ".join(channel.channel_name for channel in channels)
        lines.append(f"- **{group_name}:** {channel_list}")
    lines.append("")
    lines.append("### Bundle Location")
    lines.append("")
    lines.append(f"- `{bundle.path}`")
    lines.append("")
    return lines


def _bundle_channel_groups(bundle: ResultBundle) -> Dict[str, List[ChannelResult]]:
    grouped: Dict[str, List[ChannelResult]] = {
        "Screen": [],
        "Surround+Height": [],
        "LFE": [],
        "All Channels": [],
    }
    for channel in bundle.channels():
        name = channel.channel_name
        if name in {
            "Channel 1 FL",
            "Channel 2 FR",
            "Channel 3 FC",
            "Channel 1 Front Left",
            "Channel 2 Front Right",
            "Channel 3 Front Center",
        }:
            grouped["Screen"].append(channel)
        elif "LFE" in name:
            grouped["LFE"].append(channel)
        elif any(
            name.startswith(prefix)
            for prefix in (
                "Channel 5",
                "Channel 6",
                "Channel 7",
                "Channel 8",
                "Channel 9",
                "Channel 10",
                "Channel 11",
                "Channel 12",
            )
        ):
            grouped["Surround+Height"].append(channel)
        else:
            grouped["All Channels"].append(channel)
    return {key: value for key, value in grouped.items() if value}


def _bundle_group_summary(channels: List[ChannelResult]) -> Dict:
    crest_values: List[float] = []
    sustained_rows: List[Dict] = []
    advanced_rows: List[Dict] = []
    octave_rows: List[Dict] = []
    time_summary_rows: List[Dict[str, str]] = []
    for channel in channels:
        time_data = channel.read_table("time_domain_analysis")
        if not time_data.empty and "crest_factor_db" in time_data:
            values = pd.to_numeric(time_data["crest_factor_db"], errors="coerce")
            crest_values.extend(values[np.isfinite(values)].tolist())

        sustained = channel.read_table("sustained_peaks_summary")
        _append_full_spectrum_row(sustained, sustained_rows)

        octave = channel.read_table("octave_band_analysis")
        _append_full_spectrum_row(octave, octave_rows)

        advanced = channel.read_table("advanced_statistics")
        if not advanced.empty:
            advanced_rows.append(dict(zip(advanced["parameter"], advanced["value"])))

        time_summary = channel.read_table("time_domain_summary")
        if not time_summary.empty:
            time_summary_rows.append(
                {
                    str(row["parameter"]): str(row["value"])
                    for _, row in time_summary.iterrows()
                }
            )

    return {
        "channels": channels,
        "channel_folder": channels[0].channel_name if channels else "",
        "crest_values": np.asarray(crest_values, dtype=float),
        "sustained": _mean_numeric_dicts(sustained_rows),
        "advanced": _mean_numeric_dicts(advanced_rows),
        "octave": _mean_numeric_dicts(octave_rows),
        "time_summary": time_summary_rows[0] if time_summary_rows else {},
        "track_metadata": channels[0].read_json("metadata") if channels else {},
    }


def _bundle_report_metadata(bundle: ResultBundle) -> Dict[str, str]:
    for channel in bundle.channels():
        metadata = channel.read_json("metadata")
        return {str(key): str(value) for key, value in metadata.items()}
    return {str(key): str(value) for key, value in bundle.track.items()}


def _append_full_spectrum_row(dataframe, rows: List[Dict]) -> None:
    if dataframe.empty:
        return
    full_spectrum = dataframe[dataframe["frequency_hz"] == "Full Spectrum"]
    if not full_spectrum.empty:
        rows.append(full_spectrum.iloc[0].to_dict())


def _mean_numeric_dicts(rows: List[Dict]) -> Dict[str, float]:
    values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            parsed = _as_float(value)
            if parsed is not None:
                values[str(key)].append(parsed)
    return {key: float(np.mean(items)) for key, items in values.items() if items}


def _format_optional_number(value: object, decimals: int = 1) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "N/A"
    return _format_number(parsed, decimals)


def _generate_lfe_deep_dive(
    track_dir: Path, track_name: str, group_data: Dict, output_dir: Path
) -> bool:
    """Generate LFE deep dive report. Returns True if file was created."""
    if "LFE" not in group_data:
        return False

    lines: List[str] = []
    lines.append(f"# {track_name} - LFE Deep Dive")
    lines.append("")
    lines.append(
        "This section provides a detailed analysis of the LFE (Low Frequency Effects) channel, "
        "focusing on octave bands at 8 Hz, 16 Hz, 31.25 Hz, 62.5 Hz, 125 Hz, and 250 Hz. These frequencies represent "
        "the core LFE range plus the upper transition into the Screen band. The plots show crest factor and level "
        "behavior over time for each octave band, revealing how low-frequency content varies throughout the track."
    )
    lines.append("")

    lfe_data = group_data.get("LFE", {})
    lfe_channel_folder = lfe_data.get("channel_folder", "")

    if lfe_channel_folder:
        base_rel = Path(os.path.relpath(track_dir, output_dir)).as_posix()
        lfe_dir = track_dir / "LFE"

        # First, add the full LFE channel plot
        lfe_full_channel_path = lfe_dir / "lfe_full_channel.png"
        if lfe_full_channel_path.exists():
            lines.append("## LFE Channel (Full Spectrum)")
            lines.append("")
            lines.append(
                _markdown_report_image(
                    "LFE Full Channel Crest Factor Over Time",
                    lfe_full_channel_path,
                    output_dir,
                    "lfe_full_channel.png",
                )
            )
            lines.append("")
            lines.append(
                "*Note: Low crest factors are only important if the peak level is high. "
                "A low crest factor with a high peak level indicates both peak and RMS levels are high, "
                "which places greater demands on the System.*"
            )
            lines.append("")
        else:
            logger.warning(
                f"LFE full channel plot not found for {track_name} at {lfe_full_channel_path}"
            )

        # List of target frequencies for LFE deep dive
        lfe_frequencies = [8.0, 16.0, 31.25, 62.5, 125.0, 250.0]
        found_plots = []

        for freq in lfe_frequencies:
            freq_str = f"{freq:.1f}".replace(".", "_")
            freq_filename = f"lfe_octave_time_{freq_str}Hz.png"
            freq_plot_path = lfe_dir / freq_filename

            if freq_plot_path.exists():
                freq_label = f"{freq:.1f} Hz" if freq < 100 else f"{freq:.0f} Hz"
                lines.append(f"## {freq_label}")
                lines.append("")
                lines.append(
                    _markdown_report_image(
                        f"LFE {freq_label} Octave Band Time Analysis",
                        freq_plot_path,
                        output_dir,
                        freq_filename,
                    )
                )
                lines.append("")
                found_plots.append(freq)
            else:
                logger.warning(
                    f"LFE octave band time plot not found for {track_name} at {freq_plot_path}"
                )

        if not found_plots:
            lines.append("*LFE octave band time plots not available*")
    else:
        lines.append("*LFE channel data not available*")

    # Write deep dive file
    deep_dive_path = output_dir / "lfe_deep_dive.md"
    deep_dive_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"LFE deep dive written to: {deep_dive_path}")
    return True


def _generate_screen_deep_dive(
    track_dir: Path, track_name: str, group_data: Dict, output_dir: Path
) -> bool:
    """Generate Screen deep dive report. Returns True if file was created."""
    if "Screen" not in group_data:
        return False

    lines: List[str] = []
    lines.append(f"# {track_name} - Screen Channel Deep Dive")
    lines.append("")
    lines.append(
        "This section provides a detailed analysis of the Screen channels (FL, FR, FC), "
        "focusing on octave bands across the full spectrum (8 Hz to 16 kHz). The plots show "
        "crest factor and level behavior over time for each octave band, revealing how frequency "
        "content varies throughout the track for the main screen channels."
    )
    lines.append("")

    base_rel = Path(os.path.relpath(track_dir, output_dir)).as_posix()
    screen_dir = track_dir / "Screen"

    # Get all octave band frequencies
    octave_frequencies = [
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
    found_plots = []

    for freq in octave_frequencies:
        freq_str_filename = f"{freq:.1f}".replace(".", "_")
        freq_filename = f"screen_{freq_str_filename}Hz.png"
        freq_plot_path = screen_dir / freq_filename
        freq_label = f"{freq:.1f} Hz" if freq < 100 else f"{freq:.0f} Hz"

        if freq_plot_path.exists():
            lines.append(f"## {freq_label}")
            lines.append("")
            lines.append(
                _markdown_report_image(
                    f"Screen {freq_label} Octave Band Time Analysis (All Channels)",
                    freq_plot_path,
                    output_dir,
                    freq_filename,
                )
            )
            lines.append("")
            found_plots.append(freq)
        else:
            logger.warning(
                f"Screen {freq_label} octave band time plot not found for {track_name} at {freq_plot_path}"
            )

    if not found_plots:
        lines.append("*Screen channel octave band time plots not available*")

    # Write deep dive file
    deep_dive_path = output_dir / "screen_deep_dive.md"
    deep_dive_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Screen deep dive written to: {deep_dive_path}")
    return True


def _generate_surround_height_deep_dive(
    track_dir: Path, track_name: str, group_data: Dict, output_dir: Path
) -> bool:
    """Generate Surround+Height deep dive report. Returns True if file was created."""
    if "Surround+Height" not in group_data:
        return False

    lines: List[str] = []
    lines.append(f"# {track_name} - Surround+Height Channel Deep Dive")
    lines.append("")
    lines.append(
        "This section provides a detailed analysis of the Surround and Height channels, "
        "focusing on octave bands across the full spectrum (8 Hz to 16 kHz). The plots show "
        "crest factor and level behavior over time for each octave band, revealing how frequency "
        "content varies throughout the track for the surround and height channels."
    )
    lines.append("")

    base_rel = Path(os.path.relpath(track_dir, output_dir)).as_posix()
    surround_dir = track_dir / "Surround+Height"

    # Get all octave band frequencies
    octave_frequencies = [
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
    found_plots = []

    for freq in octave_frequencies:
        freq_str_filename = f"{freq:.1f}".replace(".", "_")
        freq_filename = f"surround_height_{freq_str_filename}Hz.png"
        freq_plot_path = surround_dir / freq_filename
        freq_label = f"{freq:.1f} Hz" if freq < 100 else f"{freq:.0f} Hz"

        if freq_plot_path.exists():
            lines.append(f"## {freq_label}")
            lines.append("")
            lines.append(
                _markdown_report_image(
                    f"Surround+Height {freq_label} Octave Band Time Analysis "
                    "(All Channels)",
                    freq_plot_path,
                    output_dir,
                    freq_filename,
                )
            )
            lines.append("")
            found_plots.append(freq)
        else:
            logger.warning(
                f"Surround+Height {freq_label} octave band time plot not found for {track_name} at {freq_plot_path}"
            )

    if not found_plots:
        lines.append("*Surround+Height channel octave band time plots not available*")

    # Write deep dive file
    deep_dive_path = output_dir / "surround_height_deep_dive.md"
    deep_dive_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Surround+Height deep dive written to: {deep_dive_path}")
    return True


def generate_report(track_dir: Path, output_path: Path) -> None:
    """Generate a markdown report for a track."""
    track_name = track_dir.name
    logger.info(f"Generating report for: {track_name}")
    report_folder = output_path.parent
    base_rel = Path(os.path.relpath(track_dir, report_folder)).as_posix()

    # Determine channel groups
    groups = _determine_channel_groups(track_dir)
    if not groups:
        logger.warning(f"No channel groups found for {track_name}")
        return

    # Collect data for each group
    group_data: Dict[str, Dict] = {}
    for group_name, channel_folders in groups.items():
        if not channel_folders:
            continue
        # Use first channel folder (worst channel if manifest exists)
        # Empty string means root directory (mono tracks)
        channel_folder = channel_folders[0]
        if channel_folder:
            csv_path = track_dir / channel_folder / "analysis_results.csv"
        else:
            csv_path = track_dir / "analysis_results.csv"
        if not csv_path.exists():
            logger.warning(
                f"Missing analysis_results.csv for {channel_folder or 'root'}"
            )
            continue

        sustained = _parse_sustained_peaks_summary(csv_path)
        advanced = _parse_advanced_stats(csv_path)
        octave = _parse_octave_band_analysis(csv_path)
        time_summary = _parse_time_domain_summary(csv_path)
        track_metadata = _parse_track_metadata(csv_path)

        # Debug: Log which file is being read and key values
        if sustained:
            logger.debug(
                f"Reading from {csv_path.name}: "
                f"hold_ms_mean={sustained.get('hold_ms_mean', 0):.1f}ms, "
                f"hold_ms_p95={sustained.get('hold_ms_p95', 0):.1f}ms"
            )

        group_data[group_name] = {
            "channel_folder": channel_folder,
            "sustained": sustained or {},
            "advanced": advanced,
            "octave": octave or {},
            "time_summary": time_summary,
            "track_metadata": track_metadata,
        }

    # Generate markdown
    time_summary = _select_time_domain_summary(group_data)
    track_metadata = _select_track_metadata(group_data)
    time_sample_label = _time_domain_sample_label(time_summary)
    time_calculation_sentence = _time_domain_calculation_sentence(time_summary)
    octave_processing_sentence = _octave_processing_sentence(track_metadata)

    lines: List[str] = []
    lines.append(f"# {track_name} - Audio Signal Analysis")
    lines.append("")
    lines.append("## What This Report Tells You")
    lines.append("")
    lines.append(
        "- **Group comparison:** Statistical analysis across channel groups "
        f"({', '.join(groups.keys())})"
    )
    lines.append(
        "- **Peak occurrence and duty cycle:** Frequency of peaks near full scale "
        "and time spent at peak levels"
    )
    lines.append(
        "- **Peak hold and recovery times:** How long peaks are sustained and "
        "recovery times to -3/-6/-9/-12 dB"
    )
    lines.append(
        "- **Frequency-dependent crest factor behavior:** Wideband and octave-band "
        "dynamic range characteristics"
    )
    lines.append("")

    # Crest Factor Analysis (moved to first main section)
    lines.append("## Crest Factor Analysis")
    lines.append("")
    lines.append(
        "Crest factor (peak-to-RMS ratio) indicates the dynamic range of the audio signal. "
        f"This section provides statistical analysis of crest factor using {time_sample_label} "
        "from all channels within each group, showing the distribution and spread of dynamic range "
        f"characteristics over time. {time_calculation_sentence} The mean value represents "
        "the average crest factor across the exported time-domain samples."
    )
    lines.append("")
    lines.append(
        "**Percentile Definitions:** P90 (90th percentile) indicates that 90% of "
        "exported time-domain samples have a crest factor at or below this value. "
        "P95 (95th percentile) indicates that 95% of exported time-domain samples "
        "have a crest factor at or below this value."
    )
    lines.append("")

    # Collect all exported time-domain crest factors from all channels in each group
    # for statistical analysis.
    from src.post.group_crest_factor_time import _parse_time_domain_analysis

    # Group channels similar to group_octave_spectrum
    screen_names = {
        "Channel 1 FL",
        "Channel 2 FR",
        "Channel 3 FC",
        "Channel 1 Front Left",
        "Channel 2 Front Right",
        "Channel 3 Front Center",
    }
    lfe_names = {
        "Channel 4 LFE",
        "Channel 5 LFE",
        "Channel 6 LFE",
        "Channel 4 Low Frequency Effects",
        "Channel 5 Low Frequency Effects",
    }
    surround_prefixes = {
        "Channel 5",
        "Channel 6",
        "Channel 7",
        "Channel 8",
        "Channel 9",
        "Channel 10",
        "Channel 11",
        "Channel 12",
    }

    group_crest_factors: Dict[str, List[float]] = defaultdict(list)

    for sub in track_dir.iterdir():
        if not sub.is_dir():
            continue
        csv_path = sub / "analysis_results.csv"
        if not csv_path.exists():
            continue

        folder = sub.name
        time_data = _parse_time_domain_analysis(csv_path)
        if not time_data:
            continue

        # Extract all exported time-domain crest factors.
        cf_db_values = time_data.get("crest_factor_db", np.array([]))
        if len(cf_db_values) == 0:
            continue

        # Filter out invalid values
        valid_cf = cf_db_values[np.isfinite(cf_db_values)]
        if len(valid_cf) == 0:
            continue

        if folder in screen_names:
            group_crest_factors["Screen"].extend(valid_cf.tolist())
        elif folder in lfe_names:
            group_crest_factors["LFE"].extend(valid_cf.tolist())
        elif any(folder.startswith(prefix) for prefix in surround_prefixes):
            group_crest_factors["Surround+Height"].extend(valid_cf.tolist())
        else:
            # For non-cinema layouts (stereo, mono, etc.), add to "All Channels"
            group_crest_factors["All Channels"].extend(valid_cf.tolist())

    # Handle mono tracks
    mono_csv = track_dir / "analysis_results.csv"
    if mono_csv.exists() and not group_crest_factors:
        time_data = _parse_time_domain_analysis(mono_csv)
        if time_data:
            cf_db_values = time_data.get("crest_factor_db", np.array([]))
            if len(cf_db_values) > 0:
                valid_cf = cf_db_values[np.isfinite(cf_db_values)]
                if len(valid_cf) > 0:
                    group_crest_factors["Mono"].extend(valid_cf.tolist())

    # Generate statistics table
    if group_crest_factors:
        lines.append(
            _format_table_row(
                [
                    "Group",
                    "Min (dB)",
                    "P95 (dB)",
                    "P90 (dB)",
                    "Mean (dB)",
                    "Median (dB)",
                    "Max (dB)",
                    "Std Dev (dB)",
                ]
            )
        )
        lines.append(_format_table_row(["---"] * 8))

        for group_name in sorted(group_crest_factors.keys()):
            cf_values = np.array(group_crest_factors[group_name])
            if len(cf_values) == 0:
                continue

            min_cf = np.min(cf_values)
            p90_cf = np.percentile(cf_values, 90)
            p95_cf = np.percentile(cf_values, 95)
            mean_cf = np.mean(cf_values)
            median_cf = np.median(cf_values)
            max_cf = np.max(cf_values)
            std_cf = np.std(cf_values)

            lines.append(
                _format_table_row(
                    [
                        group_name,
                        _format_number(min_cf, 1),
                        _format_number(p95_cf, 1),
                        _format_number(p90_cf, 1),
                        _format_number(mean_cf, 1),
                        _format_number(median_cf, 1),
                        _format_number(max_cf, 1),
                        _format_number(std_cf, 1),
                    ]
                )
            )
    else:
        lines.append("*Crest factor analysis not available*")

    lines.append("")
    lines.append(
        "*For detailed octave-band crest factor plots, see Octave Spectrum Analysis section below.*"
    )
    lines.append("")

    # Group crest factor time plots
    lines.append("## Crest Factor Over Time by Channel Group")
    lines.append("")
    lines.append(
        f"*Crest factor and level analysis over time. {time_calculation_sentence} "
        "The top plot shows crest factor (peak-to-RMS ratio) for each channel in the group. "
        "The bottom plot shows peak and RMS levels in dBFS, allowing comparison of dynamic "
        "range and level behavior across channels.*"
    )
    lines.append("")

    # Find group crest factor time plots in group-specific folders
    reports_dir = (
        output_path.parent.parent
    )  # Go up from reports/track_name/ to reports/

    for group_name in groups.keys():
        # Plots are now in group-specific folders: track_dir/group_name/crest_factor_time.png
        plot_path, rel_path, abs_path_from_reports = _group_plot_reference(
            track_dir,
            report_folder,
            base_rel,
            group_name,
            group_data,
            "crest_factor_time.png",
        )

        if plot_path.exists():
            if not abs_path_from_reports.exists():
                logger.error(
                    f"Image path verification failed for {group_name} in {track_name}: "
                    f"Expected {abs_path_from_reports} but file does not exist"
                )
                lines.append(
                    f"*⚠️ *FILE NOT FOUND*: Crest factor time plot for {group_name}*"
                )
            else:
                lines.append(f"### {group_name}")
                lines.append(
                    _markdown_report_image(
                        f"Crest Factor Over Time - {group_name}",
                        plot_path,
                        report_folder,
                        f"{group_name}_crest_factor_time.png",
                    )
                )
                lines.append("")
        else:
            logger.warning(
                f"Crest factor time plot not found for {group_name} in {track_name} at {plot_path}"
            )

    # Check if any plots exist
    has_plots = False
    for group_name in groups.keys():
        plot_path, _, _ = _group_plot_reference(
            track_dir,
            report_folder,
            base_rel,
            group_name,
            group_data,
            "crest_factor_time.png",
        )
        if plot_path.exists():
            has_plots = True
            break

    if not has_plots:
        lines.append("*Crest factor time plots not available*")
        lines.append("")

    # Group octave spectrum plots
    lines.append("## Octave Spectrum Analysis by Channel Group")
    lines.append("")
    lines.append(
        "*Frequency-domain analysis showing crest factor and amplitude levels across octave bands, including residual low/high bands. "
        "The top plot shows crest factor by octave band for each channel, indicating dynamic range at different frequencies. "
        "The bottom plot shows peak and RMS levels in dBFS for each octave band, revealing frequency content and energy distribution across the spectrum.*"
    )
    lines.append("")
    lines.append(
        "**Filter Processing:** Octave bands are calculated with an FFT power-complementary filter bank. "
        "Adjacent raised-cosine bands overlap in amplitude but sum flat in power, so octave-band RMS values "
        "can be combined as linear power without losing or double-counting energy. Residual bands capture "
        "energy below the 8 Hz analysis band and above the 16 kHz octave region up to Nyquist."
    )
    lines.append("")
    lines.append(f"**Run Metadata:** {octave_processing_sentence}")
    lines.append("")

    for group_name in groups.keys():
        # Plots are in group-specific folders: track_dir/group_name/octave_spectrum.png
        plot_path, rel_path, abs_path_from_reports = _group_plot_reference(
            track_dir,
            report_folder,
            base_rel,
            group_name,
            group_data,
            "octave_spectrum.png",
        )

        if plot_path.exists():
            if not abs_path_from_reports.exists():
                logger.error(
                    f"Image path verification failed for {group_name} octave spectrum in {track_name}: "
                    f"Expected {abs_path_from_reports} but file does not exist"
                )
                lines.append(
                    f"*⚠️ *FILE NOT FOUND*: Octave spectrum plot for {group_name}*"
                )
            else:
                lines.append(f"### {group_name}")
                lines.append(
                    _markdown_report_image(
                        f"Octave Spectrum - {group_name}",
                        plot_path,
                        report_folder,
                        f"{group_name}_octave_spectrum.png",
                    )
                )
                lines.append("")
        else:
            logger.warning(
                f"Octave spectrum plot not found for {group_name} in {track_name} at {plot_path}"
            )

    # Check if any octave spectrum plots exist
    has_octave_plots = False
    for group_name in groups.keys():
        plot_path, _, _ = _group_plot_reference(
            track_dir,
            report_folder,
            base_rel,
            group_name,
            group_data,
            "octave_spectrum.png",
        )
        if plot_path.exists():
            has_octave_plots = True
            break

    if not has_octave_plots:
        lines.append("*Octave spectrum plots not available*")
        lines.append("")

    # Sustained-peak recovery summary tables
    lines.append("## Sustained-Peak Recovery Summary")
    lines.append("")
    lines.append(
        "This section analyzes how audio peaks decay over time after reaching their maximum level. "
        "For each peak detected in the signal, the analysis measures how long the signal remains near "
        "the peak (hold time) and how quickly it decays to progressively lower levels (-3dB, -6dB, -9dB, -12dB)."
    )
    lines.append("")
    lines.append(
        "*Recovery times are relative to peak value. Thresholds: -3dB, -6dB, -9dB, -12dB.*"
    )
    lines.append("")
    # Get search window from config to display dynamically
    from src.config import config as global_config

    search_window_seconds = global_config.get(
        "envelope_analysis.sustained_peaks_search_window_seconds", 5.0
    )
    search_window_ms = int(search_window_seconds * 1000)
    # Format seconds: show as integer if whole number, otherwise 1 decimal place
    if search_window_seconds == int(search_window_seconds):
        search_window_str = f"{int(search_window_seconds)}"
    else:
        search_window_str = f"{search_window_seconds:.1f}"
    lines.append(
        f"*Hold time measures how long the signal stays within 1.0 dB of the peak value, "
        f"searching up to {search_window_str} seconds after each peak. Values near {search_window_ms}ms indicate the signal "
        f"remained very close to peak for the full search window.*"
    )
    lines.append("")
    lines.append("**Metric Definitions:**")
    lines.append("- **Mean:** Average value across all peaks")
    lines.append("- **Median:** Middle value (50% of peaks above, 50% below)")
    lines.append("- **P90:** 90th percentile - 90% of peaks are at or below this value")
    lines.append("- **P95:** 95th percentile - 95% of peaks are at or below this value")
    lines.append("- **Max:** Worst-case value observed")
    lines.append("")

    for group_name in groups.keys():
        data = group_data.get(group_name, {})
        sustained = data.get("sustained", {})
        if not sustained:
            continue

        lines.append(f"### {group_name}")
        lines.append("")
        lines.append(
            _format_table_row(["Metric", "Mean", "Median", "P90", "P95", "Max"])
        )
        lines.append(_format_table_row(["---"] * 6))

        # Map metric keys to display names
        metric_display_names = {
            "hold_ms": "Hold time (ms)",
            "t3_ms": "Recovery time to -3dB (ms)",
            "t6_ms": "Recovery time to -6dB (ms)",
            "t9_ms": "Recovery time to -9dB (ms)",
            "t12_ms": "Recovery time to -12dB (ms)",
        }

        metrics = ["hold_ms", "t3_ms", "t6_ms", "t9_ms", "t12_ms"]
        for metric in metrics:
            mean_val = sustained.get(f"{metric}_mean", 0.0)
            median_val = sustained.get(f"{metric}_median", 0.0)
            p90_val = sustained.get(f"{metric}_p90", 0.0)
            p95_val = sustained.get(f"{metric}_p95", 0.0)
            max_val = sustained.get(f"{metric}_max", 0.0)

            display_name = metric_display_names.get(metric, metric)

            lines.append(
                _format_table_row(
                    [
                        display_name,
                        _format_number(mean_val, 1),
                        _format_number(median_val, 1),
                        _format_number(p90_val, 1),
                        _format_number(p95_val, 1),
                        _format_number(max_val, 1),
                    ]
                )
            )
        lines.append("")

    # Peak decay visual (moved after Sustained-Peak Recovery Summary)
    lines.append("## Peak Decay Characteristics")
    lines.append("")
    decay_plot_path = track_dir / "peak_decay_groups.png"
    if decay_plot_path.exists():
        rel_path = f"{base_rel}/peak_decay_groups.png"
        abs_path_from_reports = (
            report_folder / base_rel / "peak_decay_groups.png"
        ).resolve()
        if not abs_path_from_reports.exists():
            logger.error(
                f"Image path verification failed for {track_name}: "
                f"Expected {abs_path_from_reports} but file does not exist"
            )
            lines.append(
                f"*ERROR: Peak decay plot not found at expected path: {rel_path}*"
            )
        else:
            lines.append(
                _markdown_report_image(
                    "Peak Decay Curves",
                    decay_plot_path,
                    report_folder,
                    "peak_decay_groups.png",
                )
            )
    else:
        logger.warning(
            f"Peak decay plot not found for {track_name} at {decay_plot_path}"
        )
        lines.append("*Peak decay plot not available*")
    lines.append("")

    # Peak occurrence and duty cycle
    lines.append("## Peak Occurrence and Duty Cycle")
    lines.append("")
    lines.append(
        "This section analyzes the frequency of peak events at different amplitude thresholds and the percentage "
        "of time the signal spends at loud levels. Peak events are discrete occurrences where the signal reaches "
        "or exceeds a specific level. Peaks greater than -0.1 dBFS are effectively clip events, as they approach "
        "the digital full-scale limit. The duty cycle indicates what percentage of the track duration the signal "
        "spends above -3 dBFS, providing insight into how much time the signal operates at high levels."
    )
    lines.append("")
    lines.append(
        _format_table_row(
            [
                "Group",
                "Peaks >-3dB/s",
                "Peaks >-1dB/s",
                "Peaks >-0.1dB/s",
                "Loud duty % (>-3 dBFS)",
            ]
        )
    )
    lines.append(_format_table_row(["---"] * 5))

    for group_name in groups.keys():
        data = group_data.get(group_name, {})
        advanced = data.get("advanced", {})

        peaks_minus3 = advanced.get("peaks_above_minus3db_per_sec", 0.0)
        peaks_minus1 = advanced.get("peaks_above_minus1db_per_sec", 0.0)
        peaks_minus0_1 = advanced.get("peaks_above_minus0_1db_per_sec", 0.0)
        loud_duty = advanced.get("peak_distribution_loud_percent", 0.0)

        lines.append(
            _format_table_row(
                [
                    group_name,
                    _format_number(peaks_minus3, 1),
                    _format_number(peaks_minus1, 1),
                    _format_number(peaks_minus0_1, 2),
                    _format_number(loud_duty, 4),
                ]
            )
        )

    lines.append("")

    # One sentence highlight per group
    for group_name in groups.keys():
        data = group_data.get(group_name, {})
        advanced = data.get("advanced", {})
        peaks_minus0_1 = advanced.get("peaks_above_minus0_1db_per_sec", 0.0)
        peaks_minus1 = advanced.get("peaks_above_minus1db_per_sec", 0.0)
        peaks_minus3 = advanced.get("peaks_above_minus3db_per_sec", 0.0)
        loud_duty = advanced.get("peak_distribution_loud_percent", 0.0)

        highlight = f"**{group_name}:** "
        if peaks_minus0_1 > 0.1:
            highlight += f"Frequent near-clipping peaks ({peaks_minus0_1:.2f}/s above -0.1dBFS) with "
        elif peaks_minus1 > 1.0:
            highlight += f"Moderate hot peaks ({peaks_minus1:.1f}/s above -1dBFS) with "
        elif peaks_minus3 > 10.0:
            highlight += f"Many loud peaks ({peaks_minus3:.1f}/s above -3dBFS) with "
        else:
            highlight += "Low peak activity with "
        highlight += f"{loud_duty:.4f}% duty cycle at loud levels (>-3dBFS)."
        lines.append(highlight)
        lines.append("")

    # Deep Dive Sections - replaced with links to separate files
    lines.append("## Deep Dive Sections")
    lines.append("")
    lines.append(
        "Detailed frequency-domain analysis for each channel group is available in separate reports:"
    )
    lines.append("")

    # Create report folder and generate deep dive files
    report_folder.mkdir(parents=True, exist_ok=True)

    deep_dive_links = []
    if _generate_lfe_deep_dive(track_dir, track_name, group_data, report_folder):
        deep_dive_links.append(
            "- [LFE Deep Dive](lfe_deep_dive.md) - Detailed analysis of the LFE channel"
        )
    if _generate_screen_deep_dive(track_dir, track_name, group_data, report_folder):
        deep_dive_links.append(
            "- [Screen Channel Deep Dive](screen_deep_dive.md) - Detailed analysis of Screen channels"
        )
    if _generate_surround_height_deep_dive(
        track_dir, track_name, group_data, report_folder
    ):
        deep_dive_links.append(
            "- [Surround+Height Channel Deep Dive](surround_height_deep_dive.md) - Detailed analysis of Surround and Height channels"
        )

    if deep_dive_links:
        lines.extend(deep_dive_links)
    else:
        lines.append("*No deep dive sections available for this track.*")

    lines.append("")

    # Appendix
    lines.append("## Appendix: Data Sources")
    lines.append("")
    lines.append("### CSV Sections Used")
    lines.append("")
    lines.append("- `[SUSTAINED_PEAKS_SUMMARY]`: Peak hold and recovery times")
    lines.append("- `[ADVANCED_STATISTICS]`: Peak rates and duty cycle")
    lines.append("- `[OCTAVE_BAND_ANALYSIS]`: Wideband and octave-band crest factors")
    lines.append("")

    lines.append("### Data Source Files")
    lines.append("")
    for group_name in groups.keys():
        data = group_data.get(group_name, {})
        channel_folder = data.get("channel_folder", "")
        if channel_folder:
            csv_rel_path = (
                f"../../analysis/{track_name}/{channel_folder}/analysis_results.csv"
            )
            lines.append(
                f"- **{group_name}:** Data from `{channel_folder}/analysis_results.csv`"
            )
        else:
            csv_rel_path = f"../../analysis/{track_name}/analysis_results.csv"
            lines.append(
                f"- **{group_name}:** Data from `analysis_results.csv` (mono track)"
            )
    lines.append("")

    lines.append("### Figure Paths")
    lines.append("")
    for group_name in groups.keys():
        data = group_data.get(group_name, {})
        channel_folder = data.get("channel_folder", "")
        if channel_folder:
            rel_path = f"{base_rel}/{channel_folder}/crest_factor.png"
            abs_path = (
                report_folder / base_rel / channel_folder / "crest_factor.png"
            ).resolve()
            if not abs_path.exists():
                logger.error(
                    f"Image path verification failed for {group_name} in {track_name}: "
                    f"Expected {abs_path} but file does not exist"
                )
                lines.append(f"- **{group_name}:** `{rel_path}` ⚠️ *FILE NOT FOUND*")
            else:
                lines.append(f"- **{group_name}:** `{rel_path}`")
        else:
            # Mono track - files in root
            rel_path = f"{base_rel}/crest_factor.png"
            abs_path = (report_folder / base_rel / "crest_factor.png").resolve()
            if not abs_path.exists():
                logger.error(
                    f"Image path verification failed for {group_name} (mono) in {track_name}: "
                    f"Expected {abs_path} but file does not exist"
                )
                lines.append(f"- **{group_name}:** `{rel_path}` ⚠️ *FILE NOT FOUND*")
            else:
                lines.append(f"- **{group_name}:** `{rel_path}`")
    if decay_plot_path.exists():
        rel_path = f"{base_rel}/peak_decay_groups.png"
        abs_path = (report_folder / base_rel / "peak_decay_groups.png").resolve()
        if not abs_path.exists():
            logger.error(
                f"Image path verification failed for decay plot in {track_name}: "
                f"Expected {abs_path} but file does not exist"
            )
            lines.append(f"- **Combined Decay Plot:** `{rel_path}` ⚠️ *FILE NOT FOUND*")
        else:
            lines.append(f"- **Combined Decay Plot:** `{rel_path}`")
    lines.append("")

    # Write report to folder
    main_report_path = report_folder / "analysis.md"
    main_report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to: {main_report_path}")
