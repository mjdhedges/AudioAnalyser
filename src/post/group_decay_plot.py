from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _parse_sustained_summary(csv_path: Path) -> Dict[str, Dict[str, float]]:
    if not csv_path.exists():
        return {}
    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("[SUSTAINED_PEAKS_SUMMARY]") + 1
    except ValueError:
        return {}
    if start >= len(lines):
        return {}
    header = lines[start]
    cols = header.split(",")
    summaries: Dict[str, Dict[str, float]] = {}
    for line in lines[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",")
        if len(parts) != len(cols):
            continue
        row = dict(zip(cols, parts))
        freq = row.get("frequency_hz", "")
        if not freq:
            continue
        numeric: Dict[str, float] = {}
        for k, v in row.items():
            if k == "frequency_hz":
                continue
            try:
                numeric[k] = float(v)
            except ValueError:
                try:
                    numeric[k] = int(v)  # type: ignore[assignment]
                except ValueError:
                    pass
        summaries[freq] = numeric
    return summaries


def _parse_advanced_stats(csv_path: Path) -> Dict[str, float]:
    """Parse ADVANCED_STATISTICS section to get peak level."""
    if not csv_path.exists():
        return {}
    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("[ADVANCED_STATISTICS]") + 1
    except ValueError:
        return {}
    if start >= len(lines):
        return {}
    result: Dict[str, float] = {}
    for line in lines[start + 1 :]:
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


def _aggregate_group_times(rows: List[Dict[str, float]], use_worst: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate recovery times for a group.
    
    Args:
        rows: List of sustained peak summary dicts for channels in the group
        use_worst: If True, use p95 values (realistic worst case, avoids 2000ms cap). If False, use mean values.
    
    Returns:
        Tuple of (times_ms, levels_db) arrays
    """
    thresholds = [3, 6, 9, 12]
    times = []
    # Get search window limit from config (default 5 seconds = 5000ms)
    from src.config import config as global_config
    search_window_seconds = global_config.get('envelope_analysis.sustained_peaks_search_window_seconds', 5.0)
    search_window_limit = search_window_seconds * 1000.0  # Convert to milliseconds
    
    for th in thresholds:
        if use_worst:
            # Use p95 (95th percentile) for realistic worst case
            # This avoids the search window cap that max often hits
            key = f"t{th}_ms_p95"
        else:
            # Use mean for average behavior
            key = f"t{th}_ms_mean"
        vals = [r.get(key, 0.0) for r in rows if key in r and r.get(key, 0.0) > 0]
        if vals:
            # For worst case, take the maximum p95 across channels. For mean, average them.
            aggregated = max(vals) if use_worst else np.mean(vals)
            # Cap at search window limit (values at limit indicate "didn't recover")
            if aggregated >= search_window_limit * 0.99:  # Within 1% of limit
                # Try p90 as fallback if p95 is at limit
                fallback_key = f"t{th}_ms_p90"
                fallback_vals = [r.get(fallback_key, 0.0) for r in rows if fallback_key in r and r.get(fallback_key, 0.0) > 0]
                if fallback_vals:
                    aggregated = max(fallback_vals) if use_worst else np.mean(fallback_vals)
            times.append(aggregated)
        else:
            times.append(0.0)
    return np.array(times, dtype=float), -np.array(thresholds, dtype=float)


def _fit_logarithmic_decay(times_ms: np.ndarray, levels_db: np.ndarray) -> Tuple[float, float]:
    """Fit logarithmic decay curve to recovery time data.
    
    Fits logarithmic decay model: level(t) = -A * log(1 + t/tau)
    This produces a curved (non-linear) decay in dB space, matching typical audio behavior.
    
    Args:
        times_ms: Recovery times in milliseconds (measured data points)
        levels_db: Relative levels in dB (negative values: -3, -6, -9, -12)
    
    Returns:
        Tuple of (tau_ms, amplitude_db) where:
        - tau_ms: Time constant for logarithmic decay
        - amplitude_db: Amplitude parameter for the decay curve
    """
    if len(times_ms) < 2 or np.any(times_ms <= 0) or np.any(levels_db >= 0):
        return 50.0, 12.0  # Default values
    
    # Initial guess based on data
    # For logarithmic: level = -A * log(1 + t/tau)
    # At t=tau, level ≈ -A * log(2) ≈ -0.693*A
    # So A ≈ -level / 0.693 for a point near the middle
    mid_idx = len(times_ms) // 2
    A_guess = abs(levels_db[mid_idx]) / 0.693 if mid_idx < len(levels_db) else 12.0
    tau_guess = times_ms[mid_idx] if mid_idx < len(times_ms) else 100.0
    
    if SCIPY_AVAILABLE:
        try:
            # Ensure inputs are numpy arrays
            times_arr = np.asarray(times_ms, dtype=float)
            levels_arr = np.asarray(levels_db, dtype=float)
            
            # Fit logarithmic decay model: level(t) = -A * log(1 + t/tau)
            def log_decay_model(t: np.ndarray, tau: float, A: float) -> np.ndarray:
                """Logarithmic decay: level = -A * log(1 + t/tau)"""
                t_arr = np.asarray(t, dtype=float)
                return -A * np.log1p(t_arr / max(float(tau), 1e-3))
            
            popt, _ = curve_fit(
                log_decay_model, times_arr, levels_arr,
                p0=[float(tau_guess), float(A_guess)],
                bounds=([1.0, 1.0], [5000.0, 50.0]),
                maxfev=2000
            )
            tau_fit, A_fit = float(popt[0]), float(popt[1])
            return max(tau_fit, 1.0), max(A_fit, 1.0)
        except Exception as e:
            # Fallback to simple approximation
            logger.debug(f"Logarithmic curve fitting failed, using approximation: {e}")
            pass
    
    # Fallback: use approximation
    return max(tau_guess, 1.0), max(A_guess, 1.0)


def _synth_logarithmic_curve(tau_ms: float, amplitude_db: float, t_ms: np.ndarray) -> np.ndarray:
    """Generate logarithmic decay curve.
    
    Formula: level_db(t) = -amplitude_db * log(1 + t/tau_ms)
    
    This produces a curved (non-linear) decay in dB space:
    - At t=0: level = 0 dB (at peak)
    - As t increases: level decreases logarithmically
    - The curve is steeper initially and flattens out over time
    
    This matches typical audio decay behavior better than exponential.
    """
    return -amplitude_db * np.log1p(t_ms / max(tau_ms, 1e-3))


def generate_group_decay_plot(track_dir: Path, output_path: Path) -> Path:
    """Generate combined decay plot for channel groups using sustained summaries.
    
    Args:
        track_dir: Track output directory containing Channel X subfolders
        output_path: Destination image path
    
    Returns:
        Path to saved figure.
    """
    screen_names = {"Channel 1 FL", "Channel 2 FR", "Channel 3 FC"}
    surround_prefixes = {"Channel 5", "Channel 6", "Channel 7", "Channel 8"}
    lfe_names = {"Channel 4 LFE"}

    group_rows: Dict[str, List[Dict[str, float]]] = {"Screen": [], "Surround+Height": [], "LFE": [], "All Channels": []}
    group_peak_levels: Dict[str, List[float]] = {"Screen": [], "Surround+Height": [], "LFE": [], "All Channels": []}

    for sub in track_dir.iterdir():
        if not sub.is_dir():
            continue
        csv_path = sub / "analysis_results.csv"
        if not csv_path.exists():
            continue
        summaries = _parse_sustained_summary(csv_path)
        full = summaries.get("Full Spectrum")
        if not full:
            continue
        # Get peak level from advanced stats
        advanced = _parse_advanced_stats(csv_path)
        peak_dbfs = advanced.get("max_true_peak_dbfs", 0.0)
        
        folder = sub.name
        if folder in screen_names:
            group_rows["Screen"].append(full)
            group_peak_levels["Screen"].append(peak_dbfs)
        elif folder in lfe_names:
            group_rows["LFE"].append(full)
            group_peak_levels["LFE"].append(peak_dbfs)
        elif any(folder.startswith(prefix) for prefix in surround_prefixes):
            group_rows["Surround+Height"].append(full)
            group_peak_levels["Surround+Height"].append(peak_dbfs)
        else:
            # For non-cinema layouts (stereo, mono, etc.), add to "All Channels"
            group_rows["All Channels"].append(full)
            group_peak_levels["All Channels"].append(peak_dbfs)
    
    # Remove empty groups
    group_rows = {k: v for k, v in group_rows.items() if v}
    group_peak_levels = {k: v for k, v in group_peak_levels.items() if k in group_rows}

    plt.figure(figsize=(10, 6))
    colors = {"Screen": "#1f77b4", "Surround+Height": "#2ca02c", "LFE": "#d62728", "All Channels": "#ff7f0e"}

    # Determine max time needed for x-axis (use worst case for worst peak plot)
    max_time = 0
    for group, rows in group_rows.items():
        if not rows:
            continue
        times_ms, _ = _aggregate_group_times(rows, use_worst=True)
        if len(times_ms) > 0 and np.any(times_ms > 0):
            max_time = max(max_time, np.max(times_ms))
    
    # Set x-axis limit to accommodate data, with some padding
    x_max = max(200, int(max_time * 1.2)) if max_time > 0 else 200
    t_axis = np.linspace(0, x_max, 400)
    
    for group, rows in group_rows.items():
        if not rows:
            continue
        # Use worst case (max) values for "worst peak" visualization
        times_ms, levels_db = _aggregate_group_times(rows, use_worst=True)
        if np.all(times_ms == 0):
            logger.warning(f"No valid recovery time data for {group} group")
            continue
        # Filter out zero times for fitting
        valid_mask = times_ms > 0
        if not np.any(valid_mask):
            continue
        times_valid = times_ms[valid_mask]
        levels_valid = levels_db[valid_mask]
        
        # Fit logarithmic decay curve to the data points
        tau, amplitude = _fit_logarithmic_decay(times_valid, levels_valid)
        curve_db = _synth_logarithmic_curve(tau, amplitude, t_axis)
        
        # Get peak level for this group (worst channel's peak level)
        peak_levels = group_peak_levels.get(group, [])
        peak_dbfs = max(peak_levels) if peak_levels else 0.0
        peak_str = f"Peak: {peak_dbfs:.1f} dBFS" if peak_dbfs > -60 else "Peak: N/A"
        
        plt.plot(t_axis, curve_db, label=f"{group} (τ≈{tau:.1f} ms, {peak_str})", color=colors.get(group, None), linewidth=2)
        plt.scatter(times_valid, levels_valid, color=colors.get(group, None), marker="o", s=50, zorder=5)

    plt.xlabel("Time after peak (ms)")
    plt.ylabel("Relative level (dB)")
    plt.title("Worst Case (P95) Peak Decay by Channel Group")
    
    # Set y-axis with 3dB major intervals and 1dB minor marks
    plt.ylim([-20, 0])
    # Major ticks at 3dB intervals
    major_ticks = np.arange(-21, 1, 3)  # -21, -18, -15, -12, -9, -6, -3, 0
    # Minor ticks at 1dB intervals
    minor_ticks = np.arange(-20, 1, 1)  # -20, -19, ..., -1, 0
    plt.yticks(major_ticks)
    plt.yticks(minor_ticks, minor=True)
    
    plt.grid(True, alpha=0.3, which='major', linewidth=1.0)
    plt.grid(True, alpha=0.15, which='minor', linewidth=0.5)
    plt.legend()
    plt.xlim([0, x_max])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path



