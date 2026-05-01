"""Generate deep dive octave band time plots for Screen and Surround+Height channels."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.audio_processor import AudioProcessor
from src.config import config
from src.octave_filter import OctaveBandFilter
from src.plotting_utils import add_calibrated_spl_axis
from src.time_domain_metrics import (
    FixedChunkTimeDomainCalculator,
    FixedWindowTimeDomainCalculator,
    SlowTimeDomainCalculator,
)

logger = logging.getLogger(__name__)


def _find_channel_folders(track_dir: Path, group_name: str) -> List[Tuple[str, Path]]:
    """Find channel folders for a given group.

    Args:
        track_dir: Track output directory
        group_name: "Screen" or "Surround+Height"

    Returns:
        List of (channel_folder, csv_path) tuples
    """
    screen_names = {
        "Channel 1 FL",
        "Channel 2 FR",
        "Channel 3 FC",
        "Channel 1 Front Left",
        "Channel 2 Front Right",
        "Channel 3 Front Center",
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

    channels = []
    for sub in track_dir.iterdir():
        if not sub.is_dir():
            continue
        csv_path = sub / "analysis_results.csv"
        if not csv_path.exists():
            continue

        folder = sub.name
        if group_name == "Screen" and folder in screen_names:
            channels.append((folder, csv_path))
        elif group_name == "Surround+Height" and any(
            folder.startswith(prefix) for prefix in surround_prefixes
        ):
            # Exclude LFE channels
            if "LFE" not in folder.upper():
                channels.append((folder, csv_path))

    return channels


def _find_original_track_path(track_dir: Path) -> Optional[Path]:
    """Find the original track file path from track directory name."""
    track_name = track_dir.name

    # Common locations to search
    search_paths = [
        Path("Tracks") / "Film" / f"{track_name}.mkv",
        Path("Tracks") / "Film" / f"{track_name}.mts",
        Path("Tracks") / "Film" / f"{track_name}.wav",
        Path("Tracks") / f"{track_name}.mkv",
        Path("Tracks") / f"{track_name}.mts",
        Path("Tracks") / f"{track_name}.wav",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def _get_channel_index_from_folder(
    folder_name: str, channel_layout: str
) -> Optional[int]:
    """Get channel index from folder name."""
    from src.channel_mapping import FFMPEG_CHANNEL_MAPS

    if channel_layout not in FFMPEG_CHANNEL_MAPS:
        return None

    channel_map = FFMPEG_CHANNEL_MAPS[channel_layout]

    # Extract channel name from folder (e.g., "Channel 1 FL" -> "FL")
    parts = folder_name.split()
    if len(parts) >= 3 and parts[0] == "Channel":
        try:
            channel_num = int(parts[1]) - 1  # Convert to 0-based index
            if 0 <= channel_num < len(channel_map):
                return channel_num
        except ValueError:
            pass

    # Try to match by name
    for idx, name in enumerate(channel_map):
        if name.upper() in folder_name.upper():
            return idx

    return None


def generate_channel_deep_dive_plot(
    track_dir: Path,
    group_name: str,
    output_dir: Path,
    original_track_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate deep dive octave band time plots for a channel group.

    Args:
        track_dir: Track output directory
        group_name: "Screen" or "Surround+Height"
        output_dir: Directory to save plots (e.g., track_dir / "Screen" or track_dir / "Surround+Height")
        original_track_path: Optional path to original audio file

    Returns:
        Path to first generated plot or None
    """
    track_name = track_dir.name

    # Find channels in this group
    channels = _find_channel_folders(track_dir, group_name)
    if not channels:
        logger.warning(f"No {group_name} channels found for {track_name}")
        return None

    # Find original track path if not provided
    if original_track_path is None:
        original_track_path = _find_original_track_path(track_dir)

    if original_track_path is None:
        logger.warning(f"Could not find original track file for {track_name}")
        return None

    # Load audio at its native source sample rate.
    audio_processor = AudioProcessor(enable_mkv_support=True)
    test_start = config.get("analysis.test_start_time", None)
    test_duration = config.get("analysis.test_duration", None)
    if test_start is not None or test_duration is not None:
        logger.info(
            "Using constrained analysis window for %s deep dive (start=%s, duration=%s)",
            group_name,
            test_start,
            test_duration,
        )
    try:
        logger.info(f"Loading audio from {original_track_path}")
        audio_data, sr = audio_processor.load_audio(
            original_track_path,
            start_time=test_start,
            duration=test_duration,
        )
        # Get channel info
        audio_info = audio_processor.get_audio_info(audio_data, sr)
        num_channels = audio_info["channels"]
        channel_layout = audio_info.get("channel_layout", "")
        logger.info(f"Loaded audio from {original_track_path}")
    except Exception as e:
        logger.error(f"Failed to load audio for {group_name} analysis: {e}")
        return None

    # Create octave bank
    octave_filter = OctaveBandFilter(
        sample_rate=sr,
        processing_mode=config.get("analysis.octave_filter_mode", "auto"),
        block_duration_seconds=config.get(
            "analysis.octave_fft_block_duration_seconds", 30.0
        ),
        max_memory_gb=config.get("analysis.octave_max_memory_gb", 4.0),
        include_low_residual_band=config.get(
            "analysis.octave_include_low_residual_band", True
        ),
        include_high_residual_band=config.get(
            "analysis.octave_include_high_residual_band", True
        ),
        low_residual_center_hz=config.get(
            "analysis.octave_low_residual_center_hz", 4.0
        ),
        high_residual_center_hz=config.get(
            "analysis.octave_high_residual_center_hz", 32000.0
        ),
    )

    configured_center_frequencies = config.get_octave_center_frequencies()
    center_frequencies = octave_filter.get_band_center_frequencies(
        configured_center_frequencies
    )
    # Use all octave bands (Full Spectrum is handled separately in the octave bank at column 0)
    octave_bands = center_frequencies

    time_domain_mode = config.get(
        "analysis.time_domain_crest_factor_mode", "fixed_window"
    )
    analysis_config = {
        "crest_factor_window_seconds": config.get(
            "analysis.crest_factor_window_seconds", 2.0
        ),
        "crest_factor_step_seconds": config.get(
            "analysis.crest_factor_step_seconds", 1.0
        ),
        "crest_factor_rms_floor_dbfs": config.get(
            "analysis.crest_factor_rms_floor_dbfs", -80.0
        ),
        "time_domain_slow_window_seconds": config.get(
            "analysis.time_domain_slow_window_seconds", 1.0
        ),
        "time_domain_slow_step_seconds": config.get(
            "analysis.time_domain_slow_step_seconds", 1.0
        ),
        "time_domain_slow_rms_tau_seconds": config.get(
            "analysis.time_domain_slow_rms_tau_seconds", 1.0
        ),
    }
    if time_domain_mode == "slow":
        chunk_duration = float(analysis_config["time_domain_slow_window_seconds"])
        step_seconds = float(analysis_config["time_domain_slow_step_seconds"])
    elif time_domain_mode == "fixed_chunk":
        chunk_duration = float(config.get("analysis.chunk_duration_seconds", 2.0))
        step_seconds = chunk_duration
    else:
        chunk_duration = float(analysis_config["crest_factor_window_seconds"])
        step_seconds = float(analysis_config["crest_factor_step_seconds"])
    chunk_samples = max(int(chunk_duration * sr), 1)
    step_samples = max(int(step_seconds * sr), 1)

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

    # Collect data for all channels first
    channel_data_dict: Dict[str, Dict] = {}

    # Extract all channels using AudioProcessor
    extracted_channels = audio_processor.extract_channels(audio_data)
    channel_idx_map = {ch_idx: ch_data for ch_data, ch_idx in extracted_channels}

    for channel_folder, csv_path in channels:
        # Get channel index
        channel_idx = _get_channel_index_from_folder(channel_folder, channel_layout)
        if channel_idx is None or channel_idx not in channel_idx_map:
            logger.warning(f"Could not determine channel index for {channel_folder}")
            continue

        channel_data = channel_idx_map[channel_idx]
        logger.info(f"Processing {channel_folder}: {len(channel_data)} samples")

        # Check minimum length for filter processing
        if len(channel_data) < 1000:
            logger.warning(
                f"Channel {channel_folder} too short for octave bank processing: {len(channel_data)} samples"
            )
            continue

        # Create octave bank for this channel
        logger.info(f"Creating octave bank for {channel_folder}...")
        try:
            octave_bank = octave_filter.create_octave_bank(
                channel_data,
                configured_center_frequencies,
            )
        except Exception as e:
            logger.error(f"Failed to create octave bank for {channel_folder}: {e}")
            continue

        num_samples = len(channel_data)
        num_complete_chunks = (num_samples - chunk_samples) // step_samples + 1

        if num_complete_chunks <= 0:
            logger.warning(
                f"Audio too short for {chunk_duration:.1f}-second block analysis: {num_samples} samples"
            )
            continue

        time_points = (
            np.arange(num_complete_chunks) * step_samples + chunk_samples
        ) / float(sr)

        # Process each octave band and store data
        channel_label = channel_folder.replace("Channel ", "").strip()
        channel_data_dict[channel_label] = {
            "time_points": time_points,
            "octave_bank": octave_bank,
            "channel_data": channel_data,
            "num_complete_chunks": num_complete_chunks,
        }

    if not channel_data_dict:
        logger.warning(f"No valid channels found for {group_name} analysis")
        return None

    # Use the first channel's time points (all should be the same)
    first_channel = list(channel_data_dict.values())[0]
    time_points = first_channel["time_points"]

    output_paths = []

    # Process each octave band - create one plot per frequency with all channels
    for band_idx, center_freq in enumerate(octave_bands):
        freq_label = (
            f"{center_freq:.1f} Hz" if center_freq < 100 else f"{center_freq:.0f} Hz"
        )

        # Create dual plot (matching LFE design - no color coding)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Collect data for all channels for this octave band
        channel_crest_data = {}
        channel_peak_data = {}
        channel_rms_data = {}

        for channel_label, channel_info in channel_data_dict.items():
            octave_bank = channel_info["octave_bank"]
            channel_data = channel_info["channel_data"]

            band_data = octave_bank[:, band_idx + 1]  # +1 to skip Full Spectrum

            if time_domain_mode == "slow":
                result = SlowTimeDomainCalculator(
                    peak_hold_tau_seconds=config.get(
                        "analysis.peak_hold_tau_seconds", 1.4
                    )
                ).compute(
                    band_data,
                    sample_rate=sr,
                    original_peak=1.0,
                    config=analysis_config,
                )
            elif time_domain_mode == "fixed_chunk":
                result = FixedChunkTimeDomainCalculator(
                    window_seconds=chunk_duration
                ).compute(
                    band_data,
                    sample_rate=sr,
                    original_peak=1.0,
                    config=analysis_config,
                )
            else:
                result = FixedWindowTimeDomainCalculator().compute(
                    band_data,
                    sample_rate=sr,
                    original_peak=1.0,
                    config=analysis_config,
                )

            crest_factor_db_plot = np.array(result.crest_factors_db)
            crest_factor_db_plot[~np.isfinite(crest_factor_db_plot)] = np.nan

            peak_level_dbfs_plot = np.array(result.peak_levels_dbfs)
            peak_level_dbfs_plot[peak_level_dbfs_plot == -np.inf] = -120
            peak_level_dbfs_plot[~np.isfinite(peak_level_dbfs_plot)] = -120

            rms_level_dbfs_plot = np.array(result.rms_levels_dbfs)
            rms_level_dbfs_plot[rms_level_dbfs_plot == -np.inf] = -120
            rms_level_dbfs_plot[~np.isfinite(rms_level_dbfs_plot)] = -120

            channel_time_points = result.time_points
            channel_crest_data[channel_label] = (
                channel_time_points,
                crest_factor_db_plot,
            )
            channel_peak_data[channel_label] = (
                channel_time_points,
                peak_level_dbfs_plot,
            )
            channel_rms_data[channel_label] = (channel_time_points, rms_level_dbfs_plot)

        if not channel_crest_data:
            continue

        # Plot crest factor for all channels (top)
        for idx, (channel_label, (ch_time, ch_crest)) in enumerate(
            channel_crest_data.items()
        ):
            color = channel_colors[idx % len(channel_colors)]
            ax1.plot(
                ch_time,
                ch_crest,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=f"{channel_label} Crest Factor",
            )

        ax1.set_xlim(time_points.min(), time_points.max())
        ax1.set_ylim([0, 30])
        ax1.set_ylabel("Crest Factor (dB)")
        ax1.set_title(
            f"{group_name} {freq_label} Octave Band Crest Factor Over Time - {track_name}"
        )
        ax1.grid(True, alpha=0.3, which="major")
        ax1.grid(True, alpha=0.15, which="minor")
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1.legend(loc="best", fontsize=9, ncol=2)

        # Plot peak and RMS for all channels (bottom)
        for idx, (channel_label, (ch_time, ch_peak)) in enumerate(
            channel_peak_data.items()
        ):
            color = channel_colors[idx % len(channel_colors)]
            ax2.plot(
                ch_time,
                ch_peak,
                color=color,
                marker="o",
                linewidth=2,
                alpha=0.8,
                label=f"{channel_label} Peak",
                markersize=3,
            )

        for idx, (channel_label, (ch_time, ch_rms)) in enumerate(
            channel_rms_data.items()
        ):
            color = channel_colors[idx % len(channel_colors)]
            ax2.plot(
                ch_time,
                ch_rms,
                color=color,
                marker="s",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
                label=f"{channel_label} RMS",
                markersize=3,
            )

        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Level (dBFS)")
        ax2.set_title(
            f"{group_name} {freq_label} Octave Band Peak and RMS Levels Over Time - {track_name}"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=9, ncol=2)
        level_ylim = (-40, 3)  # Standard format: +3dB at top, -40dB at bottom
        ax2.set_ylim(level_ylim)
        add_calibrated_spl_axis(ax2, level_ylim, is_lfe=False)

        plt.tight_layout()

        # Save plot with frequency in filename
        freq_str_filename = f"{center_freq:.1f}".replace(".", "_")
        freq_filename = (
            f"{group_name.lower().replace('+', '_')}_{freq_str_filename}Hz.png"
        )
        freq_output_path = output_dir / freq_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(freq_output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        output_paths.append(freq_output_path)
        logger.info(
            f"Generated {group_name} {freq_label} octave band time plot (all channels): {freq_output_path}"
        )

    if output_paths:
        return output_paths[0]
    else:
        logger.warning(f"No {group_name} octave band time plots were generated")
        return None
