"""Generate LFE octave band time plots for deep dive analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

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

# Target frequencies for LFE deep dive (includes 8 Hz sub-octave extension)
LFE_TARGET_FREQUENCIES = [8.0, 16.0, 31.25, 62.5, 125.0, 250.0]


def _find_lfe_channel(track_dir: Path) -> Optional[Tuple[str, Path]]:
    """Find the LFE channel folder and CSV path.

    Returns:
        Tuple of (channel_folder, csv_path) or None if not found
    """
    lfe_names = {
        "Channel 4 LFE",
        "Channel 5 LFE",
        "Channel 6 LFE",
        "Channel 4 Low Frequency Effects",
        "Channel 5 Low Frequency Effects",
    }

    for sub in track_dir.iterdir():
        if not sub.is_dir():
            continue
        if sub.name in lfe_names:
            csv_path = sub / "analysis_results.csv"
            if csv_path.exists():
                return (sub.name, csv_path)

    return None


def _find_original_track_path(track_dir: Path) -> Optional[Path]:
    """Try to find the original track file path.

    Looks for common audio file extensions in the Tracks directory.
    """
    track_name = track_dir.name

    # Common audio file extensions
    extensions = [".mts", ".mkv", ".wav", ".flac", ".mp3", ".m4a", ".aac"]

    # Look in common track directories
    possible_dirs = [
        Path("Tracks/Film"),
        Path("Tracks/Music"),
        Path("Tracks/Test Signal"),
        Path("Tracks"),
    ]

    for base_dir in possible_dirs:
        if not base_dir.exists():
            continue
        for ext in extensions:
            # Try exact match
            track_path = base_dir / f"{track_name}{ext}"
            if track_path.exists():
                return track_path
            # Try with common variations
            for variant in [
                track_name.replace(" (2018)", ""),
                track_name.split(" - ")[0],
            ]:
                track_path = base_dir / f"{variant}{ext}"
                if track_path.exists():
                    return track_path

    return None


def _get_octave_band_indices(
    center_frequencies: List[float], target_freqs: List[float]
) -> Dict[float, int]:
    """Get indices of target frequencies in center_frequencies list.

    Returns:
        Dictionary mapping target frequency to index in center_frequencies
    """
    indices = {}
    for target_freq in target_freqs:
        # Find closest match
        closest_idx = None
        min_diff = float("inf")
        for idx, freq in enumerate(center_frequencies):
            diff = abs(freq - target_freq)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx

        if closest_idx is not None and min_diff < 0.1:  # Allow small tolerance
            indices[target_freq] = closest_idx
        else:
            logger.warning(
                f"Could not find frequency {target_freq} Hz in center frequencies"
            )

    return indices


def generate_lfe_full_channel_plot(
    track_dir: Path, output_path: Path
) -> Optional[Path]:
    """Generate full LFE channel plot with color-coded crest factor.

    This is different from the regular LFE channel plot - it uses color coding
    to show peak levels, and is specifically for the LFE Deep Dive section.

    Args:
        track_dir: Track output directory
        output_path: Path to save the plot (should be in LFE subfolder)

    Returns:
        Path to saved plot or None if generation failed
    """
    track_name = track_dir.name
    logger.info(f"Generating full LFE channel plot for: {track_name}")

    # Find LFE channel
    lfe_info = _find_lfe_channel(track_dir)
    if not lfe_info:
        logger.warning(f"No LFE channel found for {track_name}")
        return None

    lfe_folder, lfe_csv = lfe_info

    # Parse time domain analysis from CSV
    from src.post.group_crest_factor_time import _parse_time_domain_analysis

    time_data = _parse_time_domain_analysis(lfe_csv)
    if not time_data:
        logger.warning(
            f"No time domain analysis data found for LFE channel in {track_name}"
        )
        return None

    time_seconds = time_data["time_seconds"]
    crest_factor_db = time_data["crest_factor_db"]
    peak_level_dbfs = time_data["peak_level_dbfs"]
    rms_level_dbfs = time_data["rms_level_dbfs"]

    crest_factor_db_plot = np.copy(crest_factor_db)
    crest_factor_db_plot[~np.isfinite(crest_factor_db_plot)] = np.nan

    peak_level_dbfs_plot = np.copy(peak_level_dbfs)
    peak_level_dbfs_plot[peak_level_dbfs_plot == -np.inf] = -120
    peak_level_dbfs_plot[~np.isfinite(peak_level_dbfs_plot)] = -120

    rms_level_dbfs_plot = np.copy(rms_level_dbfs)
    rms_level_dbfs_plot[rms_level_dbfs_plot == -np.inf] = -120
    rms_level_dbfs_plot[~np.isfinite(rms_level_dbfs_plot)] = -120

    # Create dual plot (original design)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top plot: Crest Factor vs Time
    ax1.plot(
        time_seconds,
        crest_factor_db_plot,
        "b-",
        linewidth=2,
        alpha=0.8,
        label="Crest Factor",
    )
    ax1.set_xlim(time_seconds.min(), time_seconds.max())
    ax1.set_ylim([0, max(30, np.max(crest_factor_db_plot) * 1.1)])
    ax1.set_ylabel("Crest Factor (dB)")
    ax1.set_title(f"LFE Channel Crest Factor Over Time - {track_name}")
    ax1.grid(True, alpha=0.3, which="major")
    ax1.grid(True, alpha=0.15, which="minor")
    # Add 1 dB minor steps
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.legend(loc="best", fontsize=10)

    # Bottom plot: Peak and RMS Levels vs Time
    ax2.plot(
        time_seconds,
        peak_level_dbfs_plot,
        color="#1f77b4",
        linewidth=2,
        alpha=0.8,
        label="Peak",
    )
    ax2.plot(
        time_seconds,
        rms_level_dbfs_plot,
        color="#1f77b4",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
        label="RMS",
    )
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Level (dBFS)")
    ax2.set_title(f"LFE Channel Peak and RMS Levels Over Time - {track_name}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=10)
    level_ylim = (-40, 3)  # Standard format: +3dB at top, -40dB at bottom
    ax2.set_ylim(level_ylim)
    add_calibrated_spl_axis(ax2, level_ylim, is_lfe=True)

    # Adjust layout manually for gridspec (tight_layout doesn't work well with gridspec)
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.3)

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated full LFE channel plot: {output_path}")
    return output_path


def generate_lfe_octave_time_plot(
    track_dir: Path, output_path: Path, original_track_path: Optional[Path] = None
) -> Optional[Path]:
    """Generate LFE octave band time plots for deep dive analysis.

    Args:
        track_dir: Track output directory
        output_path: Path to save the plot
        original_track_path: Optional path to original track file (if None, will try to find it)

    Returns:
        Path to saved plot or None if generation failed
    """
    track_name = track_dir.name
    logger.info(f"Generating LFE octave band time plots for: {track_name}")

    # Find LFE channel
    lfe_info = _find_lfe_channel(track_dir)
    if not lfe_info:
        logger.warning(f"No LFE channel found for {track_name}")
        return None

    lfe_folder, lfe_csv = lfe_info

    # Find original track path if not provided
    if original_track_path is None:
        original_track_path = _find_original_track_path(track_dir)

    if original_track_path is None or not original_track_path.exists():
        logger.warning(
            f"Could not find original track file for {track_name}. "
            "LFE octave band time plots require the original audio file."
        )
        return None

    # Load audio and extract LFE channel at the native source sample rate.
    audio_processor = AudioProcessor(enable_mkv_support=True)
    test_start = config.get("analysis.test_start_time", None)
    test_duration = config.get("analysis.test_duration", None)
    if test_start is not None or test_duration is not None:
        logger.info(
            "Using constrained analysis window for LFE deep dive (start=%s, duration=%s)",
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
        channel_layout = audio_info["channel_layout"]

        # Extract LFE channel (find index by name)
        from src.channel_mapping import FFMPEG_CHANNEL_MAPS, LFE

        lfe_channel_idx = None
        if channel_layout and channel_layout.lower() in FFMPEG_CHANNEL_MAPS:
            channel_names = FFMPEG_CHANNEL_MAPS[channel_layout.lower()]
            try:
                lfe_channel_idx = channel_names.index(LFE)
            except ValueError:
                pass

        # Fallback: try to find LFE in standard positions
        if lfe_channel_idx is None:
            # Common LFE positions: channel 3 in 3.1, channel 4 in 5.1/7.1
            if num_channels >= 4:
                # Check if channel 3 (0-indexed) might be LFE
                if num_channels == 4:  # 3.1
                    lfe_channel_idx = 3
                elif num_channels >= 6:  # 5.1 or higher
                    lfe_channel_idx = 3  # Channel 4 (0-indexed = 3) in 5.1/7.1

        if lfe_channel_idx is None or lfe_channel_idx >= num_channels:
            logger.warning(
                f"Could not determine LFE channel index for {track_name} (layout: {channel_layout}, channels: {num_channels})"
            )
            return None

        # Extract LFE channel
        channels = audio_processor.extract_channels(audio_data)
        lfe_channel_data = None
        for ch_data, ch_idx in channels:
            if ch_idx == lfe_channel_idx:
                lfe_channel_data = ch_data
                break

        if lfe_channel_data is None:
            logger.warning(f"Could not extract LFE channel data for {track_name}")
            return None

        logger.info(f"Extracted LFE channel data: {len(lfe_channel_data)} samples")

    except Exception as e:
        logger.error(f"Failed to load audio for LFE analysis: {e}")
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
    logger.info("Creating octave bank for LFE channel...")
    octave_bank = octave_filter.create_octave_bank(
        lfe_channel_data,
        configured_center_frequencies,
    )

    # Get indices for target frequencies
    freq_indices = _get_octave_band_indices(center_frequencies, LFE_TARGET_FREQUENCIES)
    if not freq_indices:
        logger.warning(
            f"Could not find target frequencies in octave bank for {track_name}"
        )
        return None

    # Calculate time-domain analysis for 1-second blocks
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
    num_samples = len(lfe_channel_data)
    num_complete_chunks = (num_samples - chunk_samples) // step_samples + 1

    if num_complete_chunks <= 0:
        logger.warning(
            f"Audio too short for {chunk_duration:.1f}-second block analysis: {num_samples} samples"
        )
        return None

    time_points = (
        np.arange(num_complete_chunks) * step_samples + chunk_samples
    ) / float(sr)

    # Calculate crest factor, peak, and RMS for each target frequency
    freq_data: Dict[float, Dict[str, np.ndarray]] = {}

    for target_freq, freq_idx in freq_indices.items():
        # Get octave band data (skip full spectrum at index 0)
        band_data = octave_bank[:, freq_idx + 1]

        if time_domain_mode == "slow":
            result = SlowTimeDomainCalculator(
                peak_hold_tau_seconds=config.get("analysis.peak_hold_tau_seconds", 1.4)
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

        freq_data[target_freq] = {
            "crest_factor_db": result.crest_factors_db,
            "peak_dbfs": np.where(
                np.isfinite(result.peak_levels_dbfs), result.peak_levels_dbfs, -120.0
            ),
            "rms_dbfs": np.where(
                np.isfinite(result.rms_levels_dbfs), result.rms_levels_dbfs, -120.0
            ),
        }

    # Generate separate plot for each frequency
    output_paths = []
    color = "#1f77b4"  # Blue for all plots

    for target_freq in LFE_TARGET_FREQUENCIES:
        if target_freq not in freq_data:
            logger.warning(f"Skipping {target_freq} Hz - no data available")
            continue

        data = freq_data[target_freq]
        freq_label = (
            f"{target_freq:.1f} Hz" if target_freq < 100 else f"{target_freq:.0f} Hz"
        )

        # Create dual plot (matching Full Spectrum style - no color coding)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot crest factor (top) - simple line plot (no color coding)
        ax1.plot(
            time_points,
            data["crest_factor_db"],
            color=color,
            linewidth=2,
            alpha=0.8,
            label="Crest Factor",
        )
        ax1.set_xlim(time_points.min(), time_points.max())
        ax1.set_ylim([0, 30])

        ax1.set_ylabel("Crest Factor (dB)")
        ax1.set_title(
            f"LFE {freq_label} Octave Band Crest Factor Over Time - {track_name}"
        )
        ax1.grid(True, alpha=0.3, which="major")
        ax1.grid(True, alpha=0.15, which="minor")
        # Add 1 dB minor steps
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1.legend(loc="best", fontsize=10)

        # Plot peak and RMS (bottom)
        ax2.plot(
            time_points,
            data["peak_dbfs"],
            color=color,
            marker="o",
            linewidth=2,
            alpha=0.8,
            label=f"Peak",
            markersize=3,
        )
        ax2.plot(
            time_points,
            data["rms_dbfs"],
            color=color,
            marker="s",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
            label=f"RMS",
            markersize=3,
        )

        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Level (dBFS)")
        ax2.set_title(
            f"LFE {freq_label} Octave Band Peak and RMS Levels Over Time - {track_name}"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=10)
        level_ylim = (-40, 3)  # Standard format: +3dB at top, -40dB at bottom
        ax2.set_ylim(level_ylim)
        add_calibrated_spl_axis(ax2, level_ylim, is_lfe=True)

        # Adjust layout manually for gridspec (tight_layout doesn't work well with gridspec)
        plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.3)

        # Save plot with frequency in filename
        # Replace decimal point in frequency with underscore, but keep .png extension
        freq_str = f"{target_freq:.1f}".replace(".", "_")
        freq_filename = f"lfe_octave_time_{freq_str}Hz.png"
        freq_output_path = output_path.parent / freq_filename
        plt.savefig(freq_output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        output_paths.append(freq_output_path)
        logger.info(
            f"Generated LFE octave band time plot for {freq_label}: {freq_output_path}"
        )

    if output_paths:
        # Return the first path for compatibility (or could return list)
        return output_paths[0]
    else:
        logger.warning("No LFE octave band time plots were generated")
        return None
