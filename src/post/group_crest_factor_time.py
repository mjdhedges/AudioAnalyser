"""Generate group-based crest factor time plots."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from src.plotting_utils import add_calibrated_spl_axis

logger = logging.getLogger(__name__)


def _parse_time_domain_analysis(csv_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Parse TIME_DOMAIN_ANALYSIS section from CSV.

    Returns:
        Dictionary with keys: time_seconds, crest_factor_db, peak_level_dbfs,
        rms_level_dbfs, peak_level, rms_level or None if section not found
    """
    if not csv_path.exists():
        return None

    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("[TIME_DOMAIN_ANALYSIS]") + 1
    except ValueError:
        return None

    if start >= len(lines):
        return None

    # Skip header
    header = lines[start].split(",")
    key_to_idx = {k.strip(): i for i, k in enumerate(header)}

    time_seconds = []
    crest_factor_db = []
    peak_level_dbfs = []
    rms_level_dbfs = []
    peak_level = []
    rms_level = []

    for line in lines[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",")
        if len(parts) != len(header):
            continue

        try:
            time_seconds.append(float(parts[key_to_idx["time_seconds"]]))
            crest_factor_db.append(float(parts[key_to_idx["crest_factor_db"]]))
            peak_level_dbfs.append(float(parts[key_to_idx["peak_level_dbfs"]]))
            rms_level_dbfs.append(float(parts[key_to_idx["rms_level_dbfs"]]))
            if "peak_level" in key_to_idx:
                peak_level.append(float(parts[key_to_idx["peak_level"]]))
            if "rms_level" in key_to_idx:
                rms_level.append(float(parts[key_to_idx["rms_level"]]))
        except (KeyError, ValueError, IndexError):
            continue

    if not time_seconds:
        return None

    return {
        "time_seconds": np.array(time_seconds),
        "crest_factor_db": np.array(crest_factor_db),
        "peak_level_dbfs": np.array(peak_level_dbfs),
        "rms_level_dbfs": np.array(rms_level_dbfs),
        "peak_level": np.array(peak_level) if peak_level else None,
        "rms_level": np.array(rms_level) if rms_level else None,
    }


def _group_channels(track_dir: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """Group channels by their channel group (Screen, Surround+Height, LFE, All Channels).

    Returns:
        Dictionary mapping group name to list of (channel_folder, csv_path) tuples
    """
    # Channel name patterns
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
    # Surround and height channels: includes all channels from 5 onwards that aren't LFE or screen
    # This covers: SBL, SBR, SL, SR, and any height channels (9+)
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

    group_channels: Dict[str, List[Tuple[str, Path]]] = {
        "Screen": [],
        "Surround+Height": [],
        "LFE": [],
        "All Channels": [],
    }

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

        if folder in screen_names:
            group_channels["Screen"].append((folder, csv_path))
        elif folder in lfe_names:
            group_channels["LFE"].append((folder, csv_path))
        elif any(folder.startswith(prefix) for prefix in surround_prefixes):
            group_channels["Surround+Height"].append((folder, csv_path))
        else:
            # For non-cinema layouts (stereo, mono, etc.), add to "All Channels"
            group_channels["All Channels"].append((folder, csv_path))

    # Remove empty groups
    return {k: v for k, v in group_channels.items() if v}


def generate_group_crest_factor_time_plot(
    track_dir: Path, output_dir: Path
) -> Dict[str, Path]:
    """Generate crest factor time plots grouped by channel groups.

    Args:
        track_dir: Track output directory containing Channel X subfolders
        output_dir: Directory to save the plots (typically track_dir)

    Returns:
        Dictionary mapping group name to output plot path
    """
    track_name = track_dir.name
    logger.info(f"Generating group crest factor time plots for: {track_name}")

    # Group channels
    group_channels = _group_channels(track_dir)
    if not group_channels:
        logger.warning(f"No channel groups found for {track_name}")
        return {}

    output_paths: Dict[str, Path] = {}

    # Color palette for channels (distinct colors for up to 8 channels)
    channel_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
    ]

    # Generate a plot for each group
    for group_name, channels in group_channels.items():
        if not channels:
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Collect all crest factor values for y-axis scaling and statistics
        all_crest_factors = []
        all_peak_levels = []
        all_rms_levels = []

        # Plot each channel in the group
        for idx, (channel_folder, csv_path) in enumerate(channels):
            time_data = _parse_time_domain_analysis(csv_path)
            if not time_data:
                continue

            time_seconds = time_data["time_seconds"]
            crest_factor_db = time_data["crest_factor_db"]
            peak_level_dbfs = time_data["peak_level_dbfs"]
            rms_level_dbfs = time_data["rms_level_dbfs"]
            peak_level_linear = time_data.get("peak_level")
            rms_level_linear = time_data.get("rms_level")
            if peak_level_linear is None:
                peak_level_linear = np.power(10.0, peak_level_dbfs / 20.0)
            if rms_level_linear is None:
                rms_level_linear = np.power(10.0, rms_level_dbfs / 20.0)

            crest_factor_db_plot = np.copy(crest_factor_db)
            crest_factor_db_plot[~np.isfinite(crest_factor_db_plot)] = np.nan

            peak_level_dbfs_plot = np.copy(peak_level_dbfs)
            peak_level_dbfs_plot[peak_level_dbfs_plot == -np.inf] = -120
            peak_level_dbfs_plot[~np.isfinite(peak_level_dbfs_plot)] = -120

            rms_level_dbfs_plot = np.copy(rms_level_dbfs)
            rms_level_dbfs_plot[rms_level_dbfs_plot == -np.inf] = -120
            rms_level_dbfs_plot[~np.isfinite(rms_level_dbfs_plot)] = -120

            # Collect valid values for statistics
            valid_cf = crest_factor_db_plot[np.isfinite(crest_factor_db_plot)]
            if len(valid_cf) > 0:
                all_crest_factors.extend(valid_cf.tolist())

            valid_peak_lin = peak_level_linear[peak_level_linear > 0]
            if valid_peak_lin.size > 0:
                all_peak_levels.extend(valid_peak_lin.tolist())

            valid_rms_lin = rms_level_linear[rms_level_linear > 0]
            if valid_rms_lin.size > 0:
                all_rms_levels.extend(valid_rms_lin.tolist())

            # Get color for this channel (cycle through palette)
            color = channel_colors[idx % len(channel_colors)]

            # Extract channel name from folder (e.g., "Channel 1 Left" -> "Left")
            channel_label = (
                channel_folder.replace("Channel ", "").replace("Channel", "").strip()
            )
            if not channel_label:
                channel_label = channel_folder

            # Top plot: Crest Factor vs Time (original design - no color coding)
            ax1.plot(
                time_seconds,
                crest_factor_db_plot,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=channel_label,
            )

            # Bottom plot: Peak and RMS Levels vs Time
            ax2.plot(
                time_seconds,
                peak_level_dbfs_plot,
                color=color,
                linewidth=2,
                alpha=0.8,
                linestyle="-",
                label=f"{channel_label} Peak",
            )
            ax2.plot(
                time_seconds,
                rms_level_dbfs_plot,
                color=color,
                linewidth=2,
                alpha=0.8,
                linestyle="--",
                label=f"{channel_label} RMS",
            )

        # Format top plot
        ax1.set_ylabel("Crest Factor (dB)")
        ax1.set_title(f"Crest Factor vs Time - {track_name} - {group_name}")
        ax1.grid(True, alpha=0.3, which="major")
        ax1.grid(True, alpha=0.15, which="minor")
        # Add 1 dB minor steps
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1.legend(loc="best", fontsize=10)
        ax1.set_ylim([0, 30])

        # Add statistics text box for crest factor
        if all_crest_factors:
            # Average crest factor: ratio of highest peak to average RMS
            # This represents the average short-term level as a ratio to the highest peak
            avg_cf = 0.0
            if all_peak_levels and all_rms_levels:
                # all_peak_levels and all_rms_levels are already in linear scale
                peak_levels_linear = np.array([p for p in all_peak_levels if p > 0])
                rms_levels_linear = np.array([r for r in all_rms_levels if r > 0])
                if peak_levels_linear.size and rms_levels_linear.size:
                    highest_peak = np.max(peak_levels_linear)
                    avg_rms = np.mean(rms_levels_linear)
                    if highest_peak > 0 and avg_rms > 0:
                        avg_cf_linear = highest_peak / avg_rms
                        if avg_cf_linear > 0:
                            avg_cf = 20 * np.log10(avg_cf_linear)
            # Max and Min are the maximum and minimum instantaneous crest factors
            max_cf = np.max(all_crest_factors)
            min_cf = np.min(all_crest_factors)
            stats_text = (
                f"Ave. Crest Factor: {avg_cf:.1f} dB | "
                f"Max Crest Factor: {max_cf:.1f} dB | "
                f"Min Crest Factor: {min_cf:.1f} dB"
            )
            ax1.text(
                0.02,
                0.95,
                stats_text,
                transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="top",
                fontsize=9,
            )

        # Format bottom plot
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Level (dBFS)")
        ax2.set_title(f"Peak and RMS Levels vs Time - {track_name} - {group_name}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=9, ncol=2)
        level_ylim = (-40, 3)
        ax2.set_ylim(level_ylim)  # Standard format: +3dB at top, -40dB at bottom
        add_calibrated_spl_axis(ax2, level_ylim, is_lfe=(group_name == "LFE"))

        plt.tight_layout()

        # Save plot in group-specific folder
        # Create folder for this group (e.g., "Screen", "LFE", "Surround+Height")
        group_folder = output_dir / group_name
        group_folder.mkdir(parents=True, exist_ok=True)

        output_filename = "crest_factor_time.png"
        output_path = group_folder / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        output_paths[group_name] = output_path
        logger.info(f"Generated crest factor time plot for {group_name}: {output_path}")

    return output_paths
