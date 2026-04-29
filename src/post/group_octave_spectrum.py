"""Generate group-based octave spectrum plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _parse_octave_band_analysis(csv_path: Path) -> Optional[Dict]:
    """Parse OCTAVE_BAND_ANALYSIS section from CSV.

    Returns:
        Dictionary with keys: frequencies, max_amplitude_db, rms_db, crest_factor_db,
        track_peak_db, track_rms_db
        or None if section not found
    """
    if not csv_path.exists():
        return None

    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("[OCTAVE_BAND_ANALYSIS]") + 1
    except ValueError:
        return None

    if start >= len(lines):
        return None

    # Skip header
    header = lines[start].split(",")
    key_to_idx = {k.strip(): i for i, k in enumerate(header)}

    frequencies = []
    max_amplitude_db = []
    rms_db = []
    crest_factor_db = []
    track_peak_db = None
    track_rms_db = None

    for line in lines[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",")
        if len(parts) != len(header):
            continue

        try:
            freq_str = parts[key_to_idx["frequency_hz"]].strip()
            # Capture "Full Spectrum" entry for track totals
            if freq_str == "Full Spectrum":
                track_peak_db = float(parts[key_to_idx["max_amplitude_db"]])
                track_rms_db = float(parts[key_to_idx["rms_db"]])
                continue

            freq = float(freq_str)
            max_db = float(parts[key_to_idx["max_amplitude_db"]])
            rms_db_val = float(parts[key_to_idx["rms_db"]])
            cf_db = float(parts[key_to_idx["crest_factor_db"]])

            frequencies.append(freq)
            max_amplitude_db.append(max_db)
            rms_db.append(rms_db_val)
            crest_factor_db.append(cf_db)
        except (KeyError, ValueError, IndexError):
            continue

    if not frequencies:
        return None

    return {
        "frequencies": np.array(frequencies),
        "max_amplitude_db": np.array(max_amplitude_db),
        "rms_db": np.array(rms_db),
        "crest_factor_db": np.array(crest_factor_db),
        "track_peak_db": track_peak_db,
        "track_rms_db": track_rms_db,
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
        octave_data = _parse_octave_band_analysis(csv_path)
        if not octave_data:
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


def generate_group_octave_spectrum_plot(
    track_dir: Path, output_dir: Path
) -> Dict[str, Path]:
    """Generate octave spectrum plots grouped by channel groups.

    Args:
        track_dir: Track output directory containing Channel X subfolders
        output_dir: Directory to save the plots (typically track_dir)

    Returns:
        Dictionary mapping group name to output plot path
    """
    track_name = track_dir.name
    logger.info(f"Generating group octave spectrum plots for: {track_name}")

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

        # Create dual-figure plot: crest factor on top, peak/RMS on bottom
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot each channel in the group
        for idx, (channel_folder, csv_path) in enumerate(channels):
            octave_data = _parse_octave_band_analysis(csv_path)
            if not octave_data:
                continue

            frequencies = octave_data["frequencies"]
            max_amplitude_db = octave_data["max_amplitude_db"]
            rms_db = octave_data["rms_db"]
            crest_factor_db = octave_data["crest_factor_db"]

            # Replace -inf with very low value for plotting
            max_amplitude_db_plot = np.copy(max_amplitude_db)
            max_amplitude_db_plot[max_amplitude_db_plot == -np.inf] = -60
            max_amplitude_db_plot[~np.isfinite(max_amplitude_db_plot)] = -60

            rms_db_plot = np.copy(rms_db)
            rms_db_plot[rms_db_plot == -np.inf] = -60
            rms_db_plot[~np.isfinite(rms_db_plot)] = -60

            crest_factor_db_plot = np.copy(crest_factor_db)
            crest_factor_db_plot[~np.isfinite(crest_factor_db_plot)] = np.nan

            # Get color for this channel (cycle through palette)
            color = channel_colors[idx % len(channel_colors)]

            # Extract channel name from folder (e.g., "Channel 1 Left" -> "Left")
            channel_label = (
                channel_folder.replace("Channel ", "").replace("Channel", "").strip()
            )
            if not channel_label:
                channel_label = channel_folder

            # Top plot: Crest Factor
            ax1.semilogx(
                frequencies,
                crest_factor_db_plot,
                color=color,
                marker="o",
                linewidth=2,
                alpha=0.8,
                label=channel_label,
            )

            # Bottom plot: Peak and RMS levels
            ax2.semilogx(
                frequencies,
                max_amplitude_db_plot,
                color=color,
                marker="o",
                linewidth=2,
                alpha=0.8,
                label=f"{channel_label} Peak",
            )
            ax2.semilogx(
                frequencies,
                rms_db_plot,
                color=color,
                marker="s",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
                label=f"{channel_label} RMS",
            )

            # Add horizontal reference lines for track totals (full spectrum)
            # Use channel color with dotted style to match single-channel plot appearance
            if (
                octave_data.get("track_peak_db") is not None
                and octave_data.get("track_rms_db") is not None
            ):
                track_peak_db = octave_data["track_peak_db"]
                track_rms_db = octave_data["track_rms_db"]

                # Use channel color with dotted style (matching single-channel plot dotted style)
                ax2.axhline(
                    y=track_peak_db,
                    color=color,
                    linestyle=":",
                    linewidth=2,
                    alpha=0.7,
                    label=f"{channel_label} Track Peak ({track_peak_db:.1f} dBFS)",
                )
                ax2.axhline(
                    y=track_rms_db,
                    color=color,
                    linestyle=":",
                    linewidth=2,
                    alpha=0.7,
                    label=f"{channel_label} Track RMS ({track_rms_db:.1f} dBFS)",
                )

        # Format top plot (Crest Factor)
        ax1.set_ylabel("Crest Factor (dB)")
        ax1.set_title(f"Crest Factor by Octave Band - {track_name} - {group_name}")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=9)
        ax1.set_ylim([0, 30])

        # Format bottom plot (Peak and RMS)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude (dBFS)")
        ax2.set_title(f"Peak and RMS Levels (dBFS) - {track_name} - {group_name}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=9, ncol=2)
        ax2.set_ylim([-60, 3])

        # Add frequency labels to bottom plot (shared x-axis)
        ax2.set_xlim([7, 20000])
        ax2.set_xticks(
            [8, 16, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        )
        ax2.set_xticklabels(
            [
                "8",
                "16",
                "31.25",
                "62.5",
                "125",
                "250",
                "500",
                "1k",
                "2k",
                "4k",
                "8k",
                "16k",
            ]
        )

        plt.tight_layout()

        # Save plot in group-specific folder
        group_folder = output_dir / group_name
        group_folder.mkdir(parents=True, exist_ok=True)

        output_filename = "octave_spectrum.png"
        output_path = group_folder / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        output_paths[group_name] = output_path
        logger.info(f"Generated octave spectrum plot for {group_name}: {output_path}")

    return output_paths
