"""Report generator for analysis results."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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
        "focusing on octave bands at 16 Hz, 31.25 Hz, 62.5 Hz, 125 Hz, and 250 Hz. These frequencies represent "
        "the core LFE range plus the upper transition into the Screen band. The plots show crest factor and level "
        "behavior over time for each octave band, revealing how low-frequency content varies throughout the track."
    )
    lines.append("")
    
    lfe_data = group_data.get("LFE", {})
    lfe_channel_folder = lfe_data.get("channel_folder", "")
    
    if lfe_channel_folder:
        track_name_escaped = track_name.replace(" ", "%20")
        lfe_dir = track_dir / "LFE"
        
        # First, add the full LFE channel plot
        lfe_full_channel_path = lfe_dir / "lfe_full_channel.png"
        if lfe_full_channel_path.exists():
            rel_path = f"../../analysis/{track_name_escaped}/LFE/lfe_full_channel.png"
            lines.append("## LFE Channel (Full Spectrum)")
            lines.append("")
            lines.append(f"![LFE Full Channel Crest Factor Over Time]({rel_path})")
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
        lfe_frequencies = [16.0, 31.25, 62.5, 125.0, 250.0]
        found_plots = []
        
        for freq in lfe_frequencies:
            freq_str = f"{freq:.1f}".replace(".", "_")
            freq_filename = f"lfe_octave_time_{freq_str}Hz.png"
            freq_plot_path = lfe_dir / freq_filename
            
            if freq_plot_path.exists():
                freq_label = f"{freq:.1f} Hz" if freq < 100 else f"{freq:.0f} Hz"
                rel_path = f"../../analysis/{track_name_escaped}/LFE/{freq_filename}"
                lines.append(f"## {freq_label}")
                lines.append("")
                lines.append(f"![LFE {freq_label} Octave Band Time Analysis]({rel_path})")
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
        "focusing on octave bands across the full spectrum (16 Hz to 16 kHz). The plots show "
        "crest factor and level behavior over time for each octave band, revealing how frequency "
        "content varies throughout the track for the main screen channels."
    )
    lines.append("")
    
    track_name_escaped = track_name.replace(" ", "%20")
    screen_dir = track_dir / "Screen"
    
    # Get all octave band frequencies
    octave_frequencies = [16.0, 31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    found_plots = []
    
    for freq in octave_frequencies:
        freq_str_filename = f"{freq:.1f}".replace(".", "_")
        freq_filename = f"screen_{freq_str_filename}Hz.png"
        freq_plot_path = screen_dir / freq_filename
        freq_label = f"{freq:.1f} Hz" if freq < 100 else f"{freq:.0f} Hz"
        
        if freq_plot_path.exists():
            rel_path = f"../../analysis/{track_name_escaped}/Screen/{freq_filename}"
            lines.append(f"## {freq_label}")
            lines.append("")
            lines.append(f"![Screen {freq_label} Octave Band Time Analysis (All Channels)]({rel_path})")
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
        "focusing on octave bands across the full spectrum (16 Hz to 16 kHz). The plots show "
        "crest factor and level behavior over time for each octave band, revealing how frequency "
        "content varies throughout the track for the surround and height channels."
    )
    lines.append("")
    
    track_name_escaped = track_name.replace(" ", "%20")
    surround_dir = track_dir / "Surround+Height"
    
    # Get all octave band frequencies
    octave_frequencies = [16.0, 31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    found_plots = []
    
    for freq in octave_frequencies:
        freq_str_filename = f"{freq:.1f}".replace(".", "_")
        freq_filename = f"surround_height_{freq_str_filename}Hz.png"
        freq_plot_path = surround_dir / freq_filename
        freq_label = f"{freq:.1f} Hz" if freq < 100 else f"{freq:.0f} Hz"
        
        if freq_plot_path.exists():
            rel_path = f"../../analysis/{track_name_escaped}/Surround+Height/{freq_filename}"
            lines.append(f"## {freq_label}")
            lines.append("")
            lines.append(f"![Surround+Height {freq_label} Octave Band Time Analysis (All Channels)]({rel_path})")
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
            logger.warning(f"Missing analysis_results.csv for {channel_folder or 'root'}")
            continue

        sustained = _parse_sustained_peaks_summary(csv_path)
        advanced = _parse_advanced_stats(csv_path)
        octave = _parse_octave_band_analysis(csv_path)

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
        }

    # Generate markdown
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
        "This section provides statistical analysis of crest factor calculated over 2-second blocks "
        "across all channels within each group, showing the distribution and spread of dynamic range "
        "characteristics over time. The mean value represents the average crest factor across all 2-second "
        "blocks and should match the overall track crest factor."
    )
    lines.append("")
    lines.append(
        "**Percentile Definitions:** P90 (90th percentile) indicates that 90% of 2-second blocks have a crest factor "
        "at or below this value. P95 (95th percentile) indicates that 95% of 2-second blocks have a crest factor at or below this value."
    )
    lines.append("")
    
    # Collect all 2-second block crest factors from all channels in each group for statistical analysis
    from src.post.group_crest_factor_time import _parse_time_domain_analysis
    
    # Group channels similar to group_octave_spectrum
    screen_names = {
        "Channel 1 FL", "Channel 2 FR", "Channel 3 FC",
        "Channel 1 Front Left", "Channel 2 Front Right", "Channel 3 Front Center",
    }
    lfe_names = {
        "Channel 4 LFE", "Channel 5 LFE", "Channel 6 LFE",
        "Channel 4 Low Frequency Effects", "Channel 5 Low Frequency Effects",
    }
    surround_prefixes = {"Channel 5", "Channel 6", "Channel 7", "Channel 8", "Channel 9", "Channel 10", "Channel 11", "Channel 12"}
    
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
        
        # Extract all 2-second block crest factors
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
                ["Group", "Min (dB)", "P95 (dB)", "P90 (dB)", "Mean (dB)", "Median (dB)", "Max (dB)", "Std Dev (dB)"]
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
        "*Crest factor and level analysis over time. RMS levels are calculated over 2-second periods. "
        "The top plot shows crest factor (peak-to-RMS ratio) for each channel in the group. "
        "The bottom plot shows peak and RMS levels in dBFS, allowing comparison of dynamic range and level behavior across channels.*"
    )
    lines.append("")
    
    # Find group crest factor time plots in group-specific folders
    track_name_escaped = track_name.replace(" ", "%20")
    reports_dir = output_path.parent.parent  # Go up from reports/track_name/ to reports/
    
    for group_name in groups.keys():
        # Plots are now in group-specific folders: track_dir/group_name/crest_factor_time.png
        group_folder = track_dir / group_name
        plot_path = group_folder / "crest_factor_time.png"
        
        if plot_path.exists():
            # URL encode the group name for the path
            group_name_escaped = group_name.replace(" ", "%20").replace("+", "%2B")
            rel_path = f"../../analysis/{track_name_escaped}/{group_name_escaped}/crest_factor_time.png"
            abs_path_from_reports = (reports_dir / ".." / "analysis" / track_name / group_name / "crest_factor_time.png").resolve()
            if not abs_path_from_reports.exists():
                logger.error(
                    f"Image path verification failed for {group_name} in {track_name}: "
                    f"Expected {abs_path_from_reports} but file does not exist"
                )
                lines.append(f"*⚠️ *FILE NOT FOUND*: Crest factor time plot for {group_name}*")
            else:
                lines.append(f"### {group_name}")
                lines.append(f"![Crest Factor Over Time - {group_name}]({rel_path})")
                lines.append("")
        else:
            logger.warning(
                f"Crest factor time plot not found for {group_name} in {track_name} at {plot_path}"
            )
    
    # Check if any plots exist
    has_plots = False
    for group_name in groups.keys():
        group_folder = track_dir / group_name
        if (group_folder / "crest_factor_time.png").exists():
            has_plots = True
            break
    
    if not has_plots:
        lines.append("*Crest factor time plots not available*")
        lines.append("")

    # Group octave spectrum plots
    lines.append("## Octave Spectrum Analysis by Channel Group")
    lines.append("")
    lines.append(
        "*Frequency-domain analysis showing crest factor and amplitude levels across octave bands (16 Hz to 16 kHz). "
        "The top plot shows crest factor by octave band for each channel, indicating dynamic range at different frequencies. "
        "The bottom plot shows peak and RMS levels in dBFS for each octave band, revealing frequency content and energy distribution across the spectrum.*"
    )
    lines.append("")
    lines.append(
        "**Filter Processing:** Octave bands are calculated using 4th-order Linkwitz-Riley bandpass filters "
        "implemented in second-order sections (SOS) format for numerical stability. "
        "Each band follows the ISO 266:1997 / IEC 61260:1995 standard with bandwidth defined as "
        "lower frequency = center / √2 and upper frequency = center × √2, ensuring exactly one octave per band. "
        "The cascade complementary filter bank approach processes bands sequentially (non-overlapping) with zero-phase "
        "filtering (forward-backward via `filtfilt`), preserving phase relationships and ensuring bands sum to unity "
        "without ripple or interference between adjacent bands."
    )
    lines.append("")
    
    for group_name in groups.keys():
        # Plots are in group-specific folders: track_dir/group_name/octave_spectrum.png
        group_folder = track_dir / group_name
        plot_path = group_folder / "octave_spectrum.png"
        
        if plot_path.exists():
            # URL encode the group name for the path
            group_name_escaped = group_name.replace(" ", "%20").replace("+", "%2B")
            rel_path = f"../../analysis/{track_name_escaped}/{group_name_escaped}/octave_spectrum.png"
            abs_path_from_reports = (reports_dir / ".." / "analysis" / track_name / group_name / "octave_spectrum.png").resolve()
            if not abs_path_from_reports.exists():
                logger.error(
                    f"Image path verification failed for {group_name} octave spectrum in {track_name}: "
                    f"Expected {abs_path_from_reports} but file does not exist"
                )
                lines.append(f"*⚠️ *FILE NOT FOUND*: Octave spectrum plot for {group_name}*")
            else:
                lines.append(f"### {group_name}")
                lines.append(f"![Octave Spectrum - {group_name}]({rel_path})")
                lines.append("")
        else:
            logger.warning(
                f"Octave spectrum plot not found for {group_name} in {track_name} at {plot_path}"
            )
    
    # Check if any octave spectrum plots exist
    has_octave_plots = False
    for group_name in groups.keys():
        group_folder = track_dir / group_name
        if (group_folder / "octave_spectrum.png").exists():
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
    search_window_seconds = global_config.get('envelope_analysis.sustained_peaks_search_window_seconds', 5.0)
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
            _format_table_row(
                ["Metric", "Mean", "Median", "P90", "P95", "Max"]
            )
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
        # Use relative path from reports/track_name/ to analysis/
        # Escape spaces in track name for markdown
        track_name_escaped = track_name.replace(" ", "%20")
        rel_path = f"../../analysis/{track_name_escaped}/peak_decay_groups.png"
        # Verify the path exists relative to reports directory
        reports_dir = output_path.parent.parent  # Go up from reports/track_name/ to reports/
        abs_path_from_reports = reports_dir / ".." / "analysis" / track_name / "peak_decay_groups.png"
        abs_path_from_reports = abs_path_from_reports.resolve()
        if not abs_path_from_reports.exists():
            logger.error(
                f"Image path verification failed for {track_name}: "
                f"Expected {abs_path_from_reports} but file does not exist"
            )
            lines.append(f"*ERROR: Peak decay plot not found at expected path: {rel_path}*")
        else:
            lines.append(f"![Peak Decay Curves]({rel_path})")
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
    # output_path is reports/track_name/analysis.md, so parent is reports/track_name/
    report_folder = output_path.parent
    report_folder.mkdir(parents=True, exist_ok=True)
    
    deep_dive_links = []
    if _generate_lfe_deep_dive(track_dir, track_name, group_data, report_folder):
        deep_dive_links.append("- [LFE Deep Dive](lfe_deep_dive.md) - Detailed analysis of the LFE channel")
    if _generate_screen_deep_dive(track_dir, track_name, group_data, report_folder):
        deep_dive_links.append("- [Screen Channel Deep Dive](screen_deep_dive.md) - Detailed analysis of Screen channels")
    if _generate_surround_height_deep_dive(track_dir, track_name, group_data, report_folder):
        deep_dive_links.append("- [Surround+Height Channel Deep Dive](surround_height_deep_dive.md) - Detailed analysis of Surround and Height channels")
    
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
            csv_rel_path = f"../../analysis/{track_name}/{channel_folder}/analysis_results.csv"
            lines.append(
                f"- **{group_name}:** Data from `{channel_folder}/analysis_results.csv`"
            )
        else:
            csv_rel_path = f"../../analysis/{track_name}/analysis_results.csv"
            lines.append(f"- **{group_name}:** Data from `analysis_results.csv` (mono track)")
    lines.append("")

    lines.append("### Figure Paths")
    lines.append("")
    reports_dir = output_path.parent.parent  # Go up from reports/track_name/ to reports/
    for group_name in groups.keys():
        data = group_data.get(group_name, {})
        channel_folder = data.get("channel_folder", "")
        if channel_folder:
            track_name_escaped = track_name.replace(" ", "%20")
            channel_folder_escaped = channel_folder.replace(" ", "%20")
            rel_path = f"../../analysis/{track_name_escaped}/{channel_folder_escaped}/crest_factor.png"
            # Verify path exists
            abs_path = (reports_dir / ".." / "analysis" / track_name / channel_folder / "crest_factor.png").resolve()
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
            track_name_escaped = track_name.replace(" ", "%20")
            rel_path = f"../../analysis/{track_name_escaped}/crest_factor.png"
            abs_path = (reports_dir / ".." / "analysis" / track_name / "crest_factor.png").resolve()
            if not abs_path.exists():
                logger.error(
                    f"Image path verification failed for {group_name} (mono) in {track_name}: "
                    f"Expected {abs_path} but file does not exist"
                )
                lines.append(f"- **{group_name}:** `{rel_path}` ⚠️ *FILE NOT FOUND*")
            else:
                lines.append(f"- **{group_name}:** `{rel_path}`")
    if decay_plot_path.exists():
        track_name_escaped = track_name.replace(" ", "%20")
        rel_path = f"../../analysis/{track_name_escaped}/peak_decay_groups.png"
        abs_path = (reports_dir / ".." / "analysis" / track_name / "peak_decay_groups.png").resolve()
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
    # output_path is reports/track_name/analysis.md, so parent is reports/track_name/
    report_folder = output_path.parent
    report_folder.mkdir(parents=True, exist_ok=True)
    main_report_path = report_folder / "analysis.md"
    main_report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to: {main_report_path}")

