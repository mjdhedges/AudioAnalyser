"""
Visualization module for Music Analyser.

This module handles all plotting and visualization functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

# Use Agg backend for non-interactive plotting (faster, no GUI)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from src.plotting_utils import add_calibrated_spl_axis
from src.signal_metrics import compute_slow_rms_envelope, max_abs_over_window

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Handles all plotting and visualization operations."""

    def __init__(self, sample_rate: int = 44100, original_peak: float = 1.0, dpi: int = 300) -> None:
        """Initialize the plot generator.
        
        Args:
            sample_rate: Sample rate for audio processing
            original_peak: Original peak level before normalization (for dBFS calculation)
            dpi: DPI for plot output (lower for faster batch processing)
        """
        self.sample_rate = sample_rate
        self.original_peak = original_peak
        self.dpi = dpi

    def _format_title(self, base_title: str, track_name: Optional[str] = None, 
                     channel_name: Optional[str] = None) -> str:
        """Format plot title with track and channel information.
        
        Args:
            base_title: Base title for the plot
            track_name: Optional track name to include
            channel_name: Optional channel name to include
            
        Returns:
            Formatted title string
        """
        if track_name and channel_name:
            return f"{base_title} - {track_name} - {channel_name}"
        elif track_name:
            return f"{base_title} - {track_name}"
        elif channel_name:
            return f"{base_title} - {channel_name}"
        else:
            return base_title

    def create_octave_spectrum_plot(self, analysis_results: Dict, 
                                  output_path: Optional[str] = None,
                                  time_analysis: Optional[Dict] = None,
                                  chunk_octave_analysis: Optional[Dict] = None,
                                  track_name: Optional[str] = None,
                                  channel_name: Optional[str] = None) -> None:
        """Create octave spectrum plot similar to MATLAB's semilogx plot.
        
        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
            time_analysis: Optional time-domain analysis results for chunk comparison
            chunk_octave_analysis: Optional pre-computed octave analysis for extreme chunks
            track_name: Optional track name for title
            channel_name: Optional channel name for title
        """
        logger.info("Creating octave spectrum plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        center_freqs = analysis_results["center_frequencies"]
        statistics = analysis_results["statistics"]
        
        # Extract data for plotting
        max_db_values = []
        rms_db_values = []
        
        for freq in center_freqs:
            freq_label = f"{freq:.3f}" if freq > 0 else "Full Spectrum"
            if freq_label in statistics:
                max_db = statistics[freq_label]["max_amplitude_db"]
                rms_db = statistics[freq_label]["rms_db"]
                
                # Handle NaN and infinite values
                max_db_values.append(max_db if np.isfinite(max_db) else -np.inf)
                rms_db_values.append(rms_db if np.isfinite(rms_db) else -np.inf)
            else:
                max_db_values.append(-np.inf)
                rms_db_values.append(-np.inf)

        # Convert to numpy arrays for easier handling
        max_db_values = np.array(max_db_values)
        rms_db_values = np.array(rms_db_values)
        
        # Replace -inf with a very low value for plotting
        max_db_values[max_db_values == -np.inf] = -60
        rms_db_values[rms_db_values == -np.inf] = -60

        # Plot the data (skip full spectrum at index 0)
        plot_freqs = center_freqs[1:]  # Skip full spectrum (0 Hz)
        plot_max_db = max_db_values[1:]  # Skip full spectrum data
        plot_rms_db = rms_db_values[1:]  # Skip full spectrum data
        
        ax.semilogx(plot_freqs, plot_max_db, 'b-o', label='Max Peak (dBFS)', linewidth=2)
        ax.semilogx(plot_freqs, plot_rms_db, 'r-s', label='RMS (dBFS)', linewidth=2)
        
        # Add horizontal reference lines for track totals
        track_peak_db = statistics["Full Spectrum"]["max_amplitude_db"]
        track_rms_db = statistics["Full Spectrum"]["rms_db"]
        
        ax.axhline(y=track_peak_db, color='blue', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'Track Peak ({track_peak_db:.1f} dBFS)')
        ax.axhline(y=track_rms_db, color='red', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'Track RMS ({track_rms_db:.1f} dBFS)')
        
        # Add extreme crest factor chunk analysis if available
        if time_analysis is not None and chunk_octave_analysis is not None:
            self._add_extreme_chunk_analysis(ax, time_analysis, chunk_octave_analysis, plot_freqs)
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (dBFS)')
        ax.set_title(self._format_title('Octave Band Analysis - Peak and RMS Levels (dBFS)', 
                                       track_name, channel_name))
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([15, 20000])
        ax.set_ylim([-60, 3])
        
        # Add frequency labels
        ax.set_xticks([16, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_xticklabels(['16', '31.25', '62.5', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_path}")
        
        plt.close(fig)

    def _add_extreme_chunk_analysis(self, ax, time_analysis: Dict, chunk_octave_analysis: Dict, 
                                   center_freqs: List[float]) -> None:
        """Add analysis of extreme crest factor chunks to the octave spectrum plot.
        
        Args:
            ax: Matplotlib axis to plot on
            time_analysis: Time-domain analysis results
            chunk_octave_analysis: Pre-computed octave analysis for extreme chunks
            center_freqs: Center frequencies for octave bands
        """
        # Use pre-computed chunk analysis data
        min_chunk_data = chunk_octave_analysis.get("min_chunk")
        max_chunk_data = chunk_octave_analysis.get("max_chunk")
        
        if min_chunk_data is not None:
            min_analysis = min_chunk_data["analysis"]
            min_time = min_chunk_data["time"]
            min_crest_db = min_chunk_data["crest_factor_db"]
            
            # Extract RMS and peak levels for octave bands (center_freqs now excludes full spectrum)
            min_rms_db = []
            min_peak_db = []
            
            for freq in center_freqs:  # center_freqs now only contains actual frequency bands
                freq_key = f"{freq:.3f}"
                if freq_key in min_analysis["statistics"]:
                    min_rms_db.append(min_analysis["statistics"][freq_key]["rms_db"])
                    min_peak_db.append(min_analysis["statistics"][freq_key]["max_amplitude_db"])
                else:
                    min_rms_db.append(-60)
                    min_peak_db.append(-60)
            
            # Get full spectrum peak for legend
            full_spectrum_peak = min_analysis["statistics"]["Full Spectrum"]["max_amplitude_db"]
            
            # Plot minimum crest factor chunk levels
            ax.semilogx(center_freqs, min_rms_db, 'g--', linewidth=1.5, alpha=0.6,
                       label=f'Min Crest RMS ({min_crest_db:.1f} dB @ {min_time:.0f}s)')
            ax.semilogx(center_freqs, min_peak_db, 'g:', linewidth=1.5, alpha=0.6,
                       label=f'Min Crest Peak ({full_spectrum_peak:.1f} dBFS @ {min_time:.0f}s)')
        
        if max_chunk_data is not None:
            max_analysis = max_chunk_data["analysis"]
            max_time = max_chunk_data["time"]
            max_crest_db = max_chunk_data["crest_factor_db"]
            
            # Extract RMS and peak levels for octave bands (center_freqs now excludes full spectrum)
            max_rms_db = []
            max_peak_db = []
            
            for freq in center_freqs:  # center_freqs now only contains actual frequency bands
                freq_key = f"{freq:.3f}"
                if freq_key in max_analysis["statistics"]:
                    max_rms_db.append(max_analysis["statistics"][freq_key]["rms_db"])
                    max_peak_db.append(max_analysis["statistics"][freq_key]["max_amplitude_db"])
                else:
                    max_rms_db.append(-60)
                    max_peak_db.append(-60)
            
            # Get full spectrum peak for legend
            full_spectrum_peak = max_analysis["statistics"]["Full Spectrum"]["max_amplitude_db"]
            
            # Plot maximum crest factor chunk levels
            ax.semilogx(center_freqs, max_rms_db, 'm--', linewidth=1.5, alpha=0.6,
                       label=f'Max Crest RMS ({max_crest_db:.1f} dB @ {max_time:.0f}s)')
            ax.semilogx(center_freqs, max_peak_db, 'm:', linewidth=1.5, alpha=0.6,
                       label=f'Max Crest Peak ({full_spectrum_peak:.1f} dBFS @ {max_time:.0f}s)')

    def create_histogram_plots(self, analysis_results: Dict, 
                             output_dir: Optional[str] = None,
                             octave_bank: Optional[np.ndarray] = None,
                             track_name: Optional[str] = None,
                             channel_name: Optional[str] = None) -> None:
        """Create histogram plots for each octave band.
        
        Args:
            analysis_results: Results from octave band analysis
            output_dir: Optional directory to save plots
            octave_bank: Octave bank array to slice from (required if band_data not in results)
        """
        logger.info("Creating histogram plots...")
        
        # Get band_data or use octave_bank
        band_data = analysis_results.get("band_data")
        if band_data is None and octave_bank is not None:
            # Reconstruct from octave_bank
            center_freqs = analysis_results["center_frequencies"]
            extended_freqs = [0] + center_freqs if center_freqs[0] != 0 else center_freqs
            num_bands = len(extended_freqs)
            # Create a band_data-like structure
            band_data = {}
            for i in range(num_bands):
                freq_label = f"{extended_freqs[i]:.3f}" if i > 0 else "Full Spectrum"
                band_data[freq_label] = octave_bank[:, i]
        elif band_data is None:
            logger.error("band_data not in results and octave_bank not provided!")
            return
        
        num_bands = len(band_data)
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (freq_label, signal) in enumerate(band_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Clean the signal data - remove NaN and infinite values
            clean_signal = signal[~np.isnan(signal) & ~np.isinf(signal)]
            
            if len(clean_signal) == 0:
                # If no valid data, show empty plot with message
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{freq_label} Hz (No Data)')
            else:
                # Create histogram with valid data - use 51 bins (odd) for 0-centered bin
                try:
                    ax.hist(clean_signal, bins=51, alpha=0.7, density=True, range=(-1, 1))
                    ax.set_title(f'{freq_label} Hz')
                except Exception as e:
                    logger.warning(f"Could not create histogram for {freq_label}: {e}")
                    ax.text(0.5, 0.5, 'Plot Error', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{freq_label} Hz (Error)')
            
            # Standardize X-axis scale for all plots
            ax.set_xlim(-1, 1)
            ax.set_xlabel('Amplitude')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(band_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(self._format_title('Amplitude Distribution by Octave Band', 
                                       track_name, channel_name), fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "histograms.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Histograms saved to: {output_path}")
        
        plt.close(fig)

    def create_histogram_plots_log_db(self, analysis_results: Dict, 
                                     output_dir: Optional[str] = None,
                                     config: Optional[Dict] = None,
                                     octave_bank: Optional[np.ndarray] = None,
                                     track_name: Optional[str] = None,
                                     channel_name: Optional[str] = None) -> None:
        """Create histogram plots for each octave band with log dB X-axis.
        
        Args:
            analysis_results: Results from octave band analysis
            output_dir: Optional directory to save plots
            config: Optional configuration dictionary
            octave_bank: Octave bank array to slice from (required if band_data not in results)
        """
        logger.info("Creating log dB histogram plots...")
        
        # Get band_data or use octave_bank
        band_data = analysis_results.get("band_data")
        if band_data is None and octave_bank is not None:
            # Reconstruct from octave_bank
            center_freqs = analysis_results["center_frequencies"]
            extended_freqs = [0] + center_freqs if center_freqs[0] != 0 else center_freqs
            num_bands = len(extended_freqs)
            # Create a band_data-like structure
            band_data = {}
            for i in range(num_bands):
                freq_label = f"{extended_freqs[i]:.3f}" if i > 0 else "Full Spectrum"
                band_data[freq_label] = octave_bank[:, i]
        elif band_data is None:
            logger.error("band_data not in results and octave_bank not provided!")
            return
        
        num_bands = len(band_data)
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        # Define dB range and noise floor from config
        noise_floor_db = config.get("log_histogram_noise_floor_db", -60) if config else -60
        max_db = config.get("log_histogram_max_db", 0) if config else 0
        max_bin_size_db = config.get("log_histogram_max_bin_size_db", 3.0) if config else 3.0
        
        # Calculate number of bins to ensure bin size is no larger than max_bin_size_db
        db_range = max_db - noise_floor_db
        min_bins = int(np.ceil(db_range / max_bin_size_db))
        
        logger.info(f"Log histogram range: {noise_floor_db} to {max_db} dBFS with {min_bins} bins (max bin size: {max_bin_size_db} dB)")
        
        for i, (freq_label, signal) in enumerate(band_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Clean the signal data - remove NaN and infinite values
            clean_signal = signal[~np.isnan(signal) & ~np.isinf(signal)]
            
            if len(clean_signal) == 0:
                # If no valid data, show empty plot with message
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{freq_label} Hz (No Data)')
            else:
                # Convert to dBFS, handling zero and negative values
                try:
                    # Take absolute value and scale by original peak for dBFS
                    abs_signal = np.abs(clean_signal) * self.original_peak
                    # Replace zeros with very small value to avoid log(0)
                    abs_signal[abs_signal == 0] = 10**(noise_floor_db/20)
                    # Convert to dBFS
                    signal_db = 20 * np.log10(abs_signal)
                    
                    # Create logarithmically spaced bins in dB
                    # Use calculated number of bins to ensure bin size is no larger than max_bin_size_db
                    db_bins = np.linspace(noise_floor_db, max_db, min_bins)
                    
                    # Create histogram
                    ax.hist(signal_db, bins=db_bins, alpha=0.7, density=True)
                    ax.set_title(f'{freq_label} Hz (Log dB)')
                    
                except Exception as e:
                    logger.warning(f"Could not create log dB histogram for {freq_label}: {e}")
                    ax.text(0.5, 0.5, 'Plot Error', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{freq_label} Hz (Error)')
            
            # Standardize X-axis scale for all plots
            ax.set_xlim(noise_floor_db, max_db)
            ax.set_xlabel('Amplitude (dBFS)')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(band_data), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(self._format_title('Amplitude Distribution by Octave Band (Log dBFS Scale)', 
                                       track_name, channel_name), fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "histograms_log_db.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Log dB histograms saved to: {output_path}")
        
        plt.close(fig)

    def create_crest_factor_time_plot(self, time_analysis: Dict,
                                    output_path: Optional[str] = None,
                                    track_name: Optional[str] = None,
                                    channel_name: Optional[str] = None) -> None:
        """Create crest factor vs time plot.
        
        Args:
            time_analysis: Results from analyze_crest_factor_over_time
            output_path: Optional path to save the plot
        """
        logger.info("Creating crest factor vs time plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        time_points = time_analysis["time_points"]
        crest_factors_db = time_analysis["crest_factors_db"]
        peak_levels_dbfs = time_analysis["peak_levels_dbfs"]
        rms_levels_dbfs = time_analysis["rms_levels_dbfs"]
        
        # Replace -inf with very low value for plotting
        crest_factors_db_plot = np.copy(crest_factors_db)
        crest_factors_db_plot[crest_factors_db_plot == -np.inf] = -60
        
        peak_levels_dbfs_plot = np.copy(peak_levels_dbfs)
        peak_levels_dbfs_plot[peak_levels_dbfs_plot == -np.inf] = -120
        
        rms_levels_dbfs_plot = np.copy(rms_levels_dbfs)
        rms_levels_dbfs_plot[rms_levels_dbfs_plot == -np.inf] = -120
        
        # Top plot: Crest Factor vs Time (color-coded by peak level)
        # Create line segments for color mapping
        points = np.array([time_points, crest_factors_db_plot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Normalize peak levels for colormap: -10 dBFS (green) to 0 dBFS (red)
        # Green for low peaks (safe), Red for high peaks near 0 dBFS (stressful)
        peak_for_colormap = np.clip(peak_levels_dbfs_plot, -60, 1.0)
        # Map: -10 dBFS and below = green (0), 0 dBFS = red (1)
        peak_normalized = np.clip((peak_for_colormap + 10) / 10, 0, 1)
        
        # Use midpoint of each segment for smoother color transitions
        # This creates a continuous fade between points
        segment_peak_values = (peak_normalized[:-1] + peak_normalized[1:]) / 2
        
        # Create custom colormap: green to yellow to red (traffic light)
        colors_list = ['#00ff00', '#ffff00', '#ff0000']  # Green, Yellow, Red
        n_bins = 256  # Higher resolution for smoother transitions
        cmap = LinearSegmentedColormap.from_list('traffic_light', colors_list, N=n_bins)
        
        # Create LineCollection with color mapping
        # Use antialiased=True for smoother rendering
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1),
                            linewidth=2, alpha=0.8, antialiased=True)
        lc.set_array(segment_peak_values)
        line = ax1.add_collection(lc)
        ax1.set_xlim(time_points.min(), time_points.max())
        ax1.set_ylim([0, max(30, np.max(crest_factors_db_plot) * 1.1)])
        
        ax1.set_ylabel('Crest Factor (dB)')
        ax1.set_title(self._format_title('Crest Factor vs Time (Color: Peak Level)', track_name, channel_name))
        ax1.grid(True, alpha=0.3, which='major')
        ax1.grid(True, alpha=0.15, which='minor')
        # Add 1 dB minor steps
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
        
        # Add colorbar at the bottom to avoid X-axis distortion
        cbar = plt.colorbar(line, ax=ax1, orientation='horizontal', pad=0.15)
        cbar.set_label('Peak Level (dBFS)', labelpad=10)

        # Bottom plot: Peak and RMS Levels vs Time
        channel_name_normalized = (channel_name or "").upper()
        is_lfe_channel = "LFE" in channel_name_normalized

        ax2.plot(time_points, peak_levels_dbfs_plot, 'b-', linewidth=2, label='Peak Level')
        ax2.plot(time_points, rms_levels_dbfs_plot, 'r-', linewidth=2, label='RMS Level')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Level (dBFS)')
        ax2.set_title(self._format_title('Peak and RMS Levels vs Time', track_name, channel_name))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        level_ylim = (-40, 3)
        ax2.set_ylim(level_ylim)
        add_calibrated_spl_axis(ax2, level_ylim, is_lfe=is_lfe_channel)
        
        # Add overall statistics as text
        avg_crest_db = np.mean(crest_factors_db_plot)
        max_crest_db = np.max(crest_factors_db_plot)
        min_crest_db = np.min(crest_factors_db_plot)
        
        stats_text = f'Avg: {avg_crest_db:.1f} dB | Max: {max_crest_db:.1f} dB | Min: {min_crest_db:.1f} dB'
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Crest factor time plot saved to: {output_path}")
        
        plt.close(fig)

    def create_octave_crest_factor_time_plot(self, octave_bank: np.ndarray,
                                           time_analysis: Dict, 
                                           center_frequencies: List[float],
                                           output_path: Optional[str] = None,
                                           track_name: Optional[str] = None,
                                           channel_name: Optional[str] = None) -> None:
        """Create plot showing crest factor over time for all octave bands.
        
        Args:
            octave_bank: Pre-computed octave bank (full track) - CRITICAL for performance
            time_analysis: Time-domain analysis results with chunk data
            center_frequencies: List of center frequencies for octave bands
            output_path: Optional path to save the plot
        """
        logger.info("Creating octave band crest factor vs time plot...")
        
        # Extract data from time analysis
        time_points = time_analysis["time_points"]
        chunk_duration = 2.0  # seconds
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        # Use provided center frequencies (no need to create filter instance)
        center_freqs = center_frequencies
        
        # Initialize storage for octave band crest factors over time
        octave_crest_factors = {}
        
        # VECTORIZATION OPTIMIZATION: Process all chunks for each frequency at once
        # This replaces nested loops with vectorized operations (10-20x faster)
        window_samples = max(int(self.sample_rate * 1.0), 1)

        for freq_idx, freq in enumerate(center_freqs):
            # Get all samples for this frequency band (skip full spectrum at index 0)
            band_all = octave_bank[:, freq_idx + 1]
            slow_rms_env = compute_slow_rms_envelope(band_all, self.sample_rate)
            
            # Calculate how many complete chunks we can make
            num_complete_chunks = len(band_all) // chunk_samples
            
            if num_complete_chunks > 0:
                crest_values = np.zeros(num_complete_chunks, dtype=np.float64)
                for chunk_idx in range(num_complete_chunks):
                    start_idx = chunk_idx * chunk_samples
                    end_idx = start_idx + chunk_samples
                    chunk = band_all[start_idx:end_idx]
                    peak = max_abs_over_window(chunk, window_samples)
                    center_idx = min(end_idx - 1, start_idx + chunk_samples // 2)
                    rms = slow_rms_env[center_idx] if slow_rms_env.size else 0.0
                    crest = peak / rms if rms > 0 else 1.0
                    crest_values[chunk_idx] = max(crest, 1.0)
                
                crest_db = 20 * np.log10(crest_values, where=crest_values > 0, out=np.zeros_like(crest_values))
                crest_db = np.where(np.isfinite(crest_db), crest_db, 0.0)
                
                # Pad with zeros if we have incomplete chunks
                if len(crest_db) < len(time_points):
                    padding = np.zeros(len(time_points) - len(crest_db))
                    crest_db = np.concatenate([crest_db, padding])
                
                octave_crest_factors[freq] = crest_db.tolist()
            else:
                # No complete chunks - fill with zeros
                octave_crest_factors[freq] = [0.0] * len(time_points)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define colors for different frequency bands
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot each octave band
        for i, freq in enumerate(center_freqs):
            crest_values = octave_crest_factors[freq]
            color = colors[i % len(colors)]
            
            # Format frequency label
            if freq >= 1000:
                freq_label = f"{freq/1000:.0f}k Hz" if freq % 1000 == 0 else f"{freq/1000:.1f}k Hz"
            else:
                freq_label = f"{freq:.0f} Hz" if freq == int(freq) else f"{freq:.1f} Hz"
            
            ax.plot(time_points, crest_values, color=color, linewidth=1.5, 
                   label=freq_label, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Crest Factor (dB)')
        ax.set_title(self._format_title('Octave Band Crest Factor vs Time', 
                                       track_name, channel_name))
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set fixed Y-axis limits for consistency across tracks
        ax.set_ylim([0, 40])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Octave band crest factor time plot saved to: {output_path}")
        
        plt.close(fig)

    def create_pattern_envelope_plots(self, envelope_statistics: Dict,
                                      center_frequencies: List[float],
                                      output_dir: Optional[str] = None,
                                      config: Optional[Dict] = None,
                                      track_name: Optional[str] = None,
                                      channel_name: Optional[str] = None) -> None:
        """Create plots showing top N envelopes from repeating patterns for each band.
        
        Args:
            envelope_statistics: Results from analyze_envelope_statistics
            center_frequencies: List of center frequencies
            output_dir: Optional directory to save plots
            config: Optional configuration dictionary
        """
        logger.info("Creating pattern envelope plots...")
        
        # Create subdirectory for pattern envelope plots
        if output_dir:
            pattern_dir = Path(output_dir) / "pattern_envelopes"
            pattern_dir.mkdir(parents=True, exist_ok=True)
        else:
            pattern_dir = None
        
        if config is None:
            from src.config import config as global_config
            config = global_config.get('envelope_analysis', {})
        
        # Pattern plots show configurable number of envelopes
        num_envelopes = config.get('envelope_plots_num_pattern_envelopes', 10)
        num_wavelengths = config.get('envelope_plots_num_wavelengths', 50)
        fallback_window_ms = config.get('envelope_plots_window_ms', 200.0)
        ylim_min = config.get('envelope_plots_ylim_min', -30)
        ylim_max = config.get('envelope_plots_ylim_max', 0)
        is_lfe_channel = "LFE" in (channel_name or "").upper()
        
        extended_frequencies = [0] + center_frequencies
        
        for band_idx, freq in enumerate(extended_frequencies):
            freq_label = f"{freq:.3f}" if freq > 0 else "Full Spectrum"
            band_data = envelope_statistics.get(freq_label)
            
            if band_data is None:
                continue
            
            pattern_analysis = band_data.get("pattern_analysis", {})
            patterns_detected = pattern_analysis.get("patterns_detected", 0)
            
            if patterns_detected == 0:
                continue
            
            rms_envelope_db = band_data.get("rms_envelope_db")
            rms_envelope_time = band_data.get("rms_envelope_time")
            
            if rms_envelope_db is None or rms_envelope_time is None:
                continue
            
            # Calculate window size based on center frequency (in wavelengths)
            # For Full Spectrum (freq = 0), use fallback absolute window
            # IMPORTANT: Use same window size as pattern analysis stored windows
            window_was_capped = band_data.get("window_was_capped", False)
            if freq > 0:
                # Window in seconds = num_wavelengths / frequency
                window_seconds = num_wavelengths / freq
                window_ms = window_seconds * 1000.0
                
                # Check if this frequency would exceed pattern analysis cap
                max_pattern_window_ms = 500.0
                if window_ms > max_pattern_window_ms:
                    window_was_capped = True
            else:
                # For Full Spectrum, match pattern analysis window (fallback_window_ms * 2)
                # This ensures stored windows from pattern analysis match plotting window
                window_ms = fallback_window_ms * 2
            
            window_samples = int(window_ms * self.sample_rate / 1000)
            half_window = window_samples // 2
            
            # Get independent envelope peak indices to exclude (mutual exclusivity)
            independent_peak_indices = set()
            worst_case_envelopes = band_data.get("worst_case_envelopes", [])
            for env in worst_case_envelopes:
                peak_idx = env.get("peak_idx")
                if peak_idx is not None:
                    independent_peak_indices.add(peak_idx)
            
            # Collect all envelopes from patterns (excluding independent peaks)
            pattern_envelopes = []
            
            for pattern_num in range(1, patterns_detected + 1):
                pattern_key = f"pattern_{pattern_num}"
                if pattern_key not in pattern_analysis:
                    continue
                
                pattern = pattern_analysis[pattern_key]
                peak_indices = pattern.get("peak_indices", [])
                peak_times = pattern.get("peak_times_seconds", [])
                
                # Use stored envelope windows if available (from pattern analysis)
                stored_windows = pattern.get("envelope_windows", None)
                stored_time_windows = pattern.get("time_windows_ms", None)
                
                # Extract envelope around each peak in this pattern
                for idx, (peak_idx, peak_time) in enumerate(zip(peak_indices, peak_times)):
                    # Ensure peak_idx is within bounds
                    if peak_idx < 0 or peak_idx >= len(rms_envelope_db):
                        continue
                    
                    # Exclude peaks that are in independent analysis (mutual exclusivity)
                    if peak_idx in independent_peak_indices:
                        continue
                    
                    # Reuse stored window if available, otherwise extract
                    if stored_windows is not None and idx < len(stored_windows) and stored_windows[idx] is not None:
                        envelope_window = stored_windows[idx]
                        time_relative_ms = stored_time_windows[idx]
                    else:
                        # Fallback: extract window (shouldn't happen if pattern analysis stored windows)
                        start_idx = max(0, peak_idx - half_window)
                        end_idx = min(len(rms_envelope_db), peak_idx + half_window)
                        
                        if end_idx - start_idx < window_samples // 2:
                            continue
                        
                        envelope_window = rms_envelope_db[start_idx:end_idx]
                        time_window = rms_envelope_time[start_idx:end_idx]
                        
                        # Convert time to relative (ms from peak)
                        peak_time_idx = peak_idx - start_idx
                        if peak_time_idx < 0 or peak_time_idx >= len(time_window):
                            continue
                        
                        time_relative_ms = (time_window - time_window[peak_time_idx]) * 1000.0
                    
                    peak_value_db = rms_envelope_db[peak_idx]
                    
                    pattern_envelopes.append({
                        "envelope": envelope_window,
                        "time_ms": time_relative_ms,
                        "peak_value_db": peak_value_db,
                        "peak_time_seconds": peak_time,
                        "pattern_num": pattern_num,
                        "peak_idx": peak_idx
                    })
            
            if len(pattern_envelopes) == 0:
                continue
            
            # Sort by peak value and take top N
            pattern_envelopes.sort(key=lambda x: x["peak_value_db"], reverse=True)
            top_envelopes = pattern_envelopes[:num_envelopes]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for idx, env_data in enumerate(top_envelopes):
                color = colors[idx % len(colors)]
                label = f"Pattern {env_data['pattern_num']} - Peak: {env_data['peak_value_db']:.1f} dBFS @ {env_data['peak_time_seconds']:.1f}s"
                
                ax.plot(env_data["time_ms"], env_data["envelope"], 
                       color=color, linewidth=2, label=label, alpha=0.8)
                
                # Mark peak (should be at time 0)
                ax.plot(0, env_data["peak_value_db"],
                       'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
            
            ax.set_xlabel('Time (ms relative to peak)')
            ax.set_ylabel('RMS Level (dBFS)')
            
            # Check if window was capped and add note to title
            window_was_capped = band_data.get("window_was_capped", False)
            base_title = f'Top {len(top_envelopes)} Pattern Envelopes - {freq_label}'
            if window_was_capped and freq > 0:
                expected_window_ms = (num_wavelengths / freq) * 1000.0
                base_title += f'\n(Window capped at 500ms, {num_wavelengths}λ would be {expected_window_ms:.0f}ms)'
            
            title = self._format_title(base_title, track_name, channel_name)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            level_ylim = (ylim_min, ylim_max)
            ax.set_ylim(level_ylim)
            add_calibrated_spl_axis(ax, level_ylim, is_lfe=is_lfe_channel)
            ax.set_xlim([-window_ms/2, window_ms/2])
            
            plt.tight_layout()
            
            if pattern_dir:
                output_path = pattern_dir / f"pattern_envelopes_{freq_label.replace(' ', '_').replace('.', '_')}.png"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Pattern envelope plot saved: {output_path.name}")
            
            plt.close(fig)

    def create_independent_envelope_plots(self, envelope_statistics: Dict,
                                         center_frequencies: List[float],
                                         output_dir: Optional[str] = None,
                                         config: Optional[Dict] = None,
                                         track_name: Optional[str] = None,
                                         channel_name: Optional[str] = None) -> None:
        """Create plots showing top N independent (non-repeating) envelopes for each band.
        
        Args:
            envelope_statistics: Results from analyze_envelope_statistics
            center_frequencies: List of center frequencies
            output_dir: Optional directory to save plots
            config: Optional configuration dictionary
        """
        logger.info("Creating independent envelope plots...")
        
        # Create subdirectory for independent envelope plots
        if output_dir:
            independent_dir = Path(output_dir) / "independent_envelopes"
            independent_dir.mkdir(parents=True, exist_ok=True)
        else:
            independent_dir = None
        
        if config is None:
            from src.config import config as global_config
            config = global_config.get('envelope_analysis', {})
        
        # Independent plots show configurable number of worst envelopes
        num_envelopes = config.get('envelope_plots_num_independent_envelopes', 3)
        num_wavelengths = config.get('envelope_plots_num_wavelengths', 50)
        fallback_window_ms = config.get('envelope_plots_window_ms', 200.0)
        ylim_min = config.get('envelope_plots_ylim_min', -30)
        ylim_max = config.get('envelope_plots_ylim_max', 0)
        is_lfe_channel = "LFE" in (channel_name or "").upper()
        
        extended_frequencies = [0] + center_frequencies
        
        for band_idx, freq in enumerate(extended_frequencies):
            freq_label = f"{freq:.3f}" if freq > 0 else "Full Spectrum"
            band_data = envelope_statistics.get(freq_label)
            
            if band_data is None:
                continue
            
            worst_case_envelopes = band_data.get("worst_case_envelopes", [])
            
            if len(worst_case_envelopes) == 0:
                continue
            
            rms_envelope_db = band_data.get("rms_envelope_db")
            rms_envelope_time = band_data.get("rms_envelope_time")
            
            if rms_envelope_db is None or rms_envelope_time is None:
                continue
            
            # Calculate window size based on center frequency (in wavelengths)
            # For Full Spectrum (freq = 0), use fallback absolute window
            # IMPORTANT: Use same window size as worst-case analysis stored windows
            window_was_capped = band_data.get("window_was_capped", False)
            if freq > 0:
                # Window in seconds = num_wavelengths / frequency
                window_seconds = num_wavelengths / freq
                window_ms = window_seconds * 1000.0
                
                # Check if this frequency would exceed pattern analysis cap
                max_pattern_window_ms = 500.0
                if window_ms > max_pattern_window_ms:
                    window_was_capped = True
            else:
                # For Full Spectrum, match worst-case analysis window (fallback_window_ms * 2)
                # This ensures stored windows from worst-case analysis match plotting window
                window_ms = fallback_window_ms * 2
            
            window_samples = int(window_ms * self.sample_rate / 1000)
            half_window = window_samples // 2
            
            # Take top N independent envelopes (ensure we get worst 3)
            top_envelopes = worst_case_envelopes[:min(num_envelopes, len(worst_case_envelopes))]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for idx, envelope_data in enumerate(top_envelopes):
                color = colors[idx % len(colors)]
                
                peak_time_seconds = envelope_data["peak_time_seconds"]
                peak_idx = int(peak_time_seconds * self.sample_rate)
                peak_value_db = envelope_data["peak_value_db"]
                
                # Reuse stored envelope window if available, otherwise extract
                if "envelope_window" in envelope_data and envelope_data["envelope_window"] is not None:
                    envelope_window = envelope_data["envelope_window"]
                    time_relative_ms = envelope_data["time_window_ms"]
                else:
                    # Fallback: extract window (shouldn't happen if worst-case analysis stored windows)
                    start_idx = max(0, peak_idx - half_window)
                    end_idx = min(len(rms_envelope_db), peak_idx + half_window)
                    
                    if end_idx - start_idx < window_samples // 2:
                        continue
                    
                    envelope_window = rms_envelope_db[start_idx:end_idx]
                    time_window = rms_envelope_time[start_idx:end_idx]
                    
                    # Convert time to relative (ms from peak)
                    peak_time_idx = peak_idx - start_idx
                    time_relative_ms = (time_window - time_window[peak_time_idx]) * 1000.0
                
                label = (f"Rank {envelope_data['rank']} - Peak: {peak_value_db:.1f} dBFS @ "
                        f"{peak_time_seconds:.1f}s")
                
                ax.plot(time_relative_ms, envelope_window, 
                       color=color, linewidth=2, label=label, alpha=0.8)
                
                # Mark peak
                ax.plot(0, peak_value_db, 'o', color=color, markersize=8, 
                       markeredgecolor='black', markeredgewidth=1)
                
                # Mark decay thresholds
                decay_times = envelope_data.get("decay_times", {})
                for decay_db in [-3, -6, -9, -12]:
                    decay_key = f"decay_{abs(decay_db)}db_ms"
                    if decay_key in decay_times:
                        decay_time_ms = decay_times[decay_key]
                        decay_reached = decay_times.get(f"decay_{abs(decay_db)}db_reached", False)
                        if decay_reached and decay_time_ms <= window_ms / 2:
                            decay_level = peak_value_db + decay_db
                            ax.plot(decay_time_ms, decay_level, 'x', color=color, 
                                   markersize=6, alpha=0.7)
            
            ax.set_xlabel('Time (ms relative to peak)')
            ax.set_ylabel('RMS Level (dBFS)')
            
            # Check if window was capped and add note to title
            base_title = f'Top {len(top_envelopes)} Independent Envelopes - {freq_label}'
            if window_was_capped and freq > 0:
                expected_window_ms = (num_wavelengths / freq) * 1000.0
                base_title += f'\n(Window capped at 500ms, {num_wavelengths}λ would be {expected_window_ms:.0f}ms)'
            
            title = self._format_title(base_title, track_name, channel_name)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            level_ylim = (ylim_min, ylim_max)
            ax.set_ylim(level_ylim)
            add_calibrated_spl_axis(ax, level_ylim, is_lfe=is_lfe_channel)
            ax.set_xlim([-window_ms/2, window_ms/2])
            
            plt.tight_layout()
            
            if independent_dir:
                output_path = independent_dir / f"independent_envelopes_{freq_label.replace(' ', '_').replace('.', '_')}.png"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Independent envelope plot saved: {output_path.name}")
            
            plt.close(fig)

    def create_crest_factor_plot(self, analysis_results: Dict, 
                               output_path: Optional[str] = None,
                               time_analysis: Optional[Dict] = None,
                               chunk_octave_analysis: Optional[Dict] = None,
                               track_name: Optional[str] = None,
                               channel_name: Optional[str] = None) -> None:
        """Create crest factor plot showing peak-to-RMS ratio for each octave band.
        
        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
            time_analysis: Optional time-domain analysis results for chunk markers
            chunk_octave_analysis: Optional pre-computed octave analysis for extreme chunks
        """
        logger.info("Creating crest factor plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        center_freqs = analysis_results["center_frequencies"]
        statistics = analysis_results["statistics"]
        
        # Extract crest factor data for plotting
        crest_factor_db_values = []
        
        for freq in center_freqs:
            freq_label = f"{freq:.3f}" if freq > 0 else "Full Spectrum"
            if freq_label in statistics:
                crest_factor_db = statistics[freq_label]["crest_factor_db"]
                
                # Handle NaN and infinite values
                crest_factor_db_values.append(crest_factor_db if np.isfinite(crest_factor_db) else -np.inf)
            else:
                crest_factor_db_values.append(-np.inf)

        # Convert to numpy arrays for easier handling
        crest_factor_db_values = np.array(crest_factor_db_values)
        
        # Replace any remaining -inf with minimum valid crest factor (0 dB)
        crest_factor_db_values[crest_factor_db_values == -np.inf] = 0.0
        # Ensure no crest factor is below 0 dB (physically impossible)
        crest_factor_db_values = np.maximum(crest_factor_db_values, 0.0)
        
        # Get the full spectrum (average) crest factor for reference line
        full_spectrum_crest_db = statistics["Full Spectrum"]["crest_factor_db"]
        
        # Plot dB crest factor (skip full spectrum at index 0)
        plot_freqs = center_freqs[1:]  # Skip full spectrum (0 Hz)
        plot_crest_db = crest_factor_db_values[1:]  # Skip full spectrum data
        
        ax.semilogx(plot_freqs, plot_crest_db, 'b-o', 
                   label='Crest Factor (dB)', linewidth=2, markersize=6)
        
        # Add horizontal reference line for average crest factor
        ax.axhline(y=full_spectrum_crest_db, color='r', linestyle='--', linewidth=2, 
                  label=f'Track Average ({full_spectrum_crest_db:.1f} dB)')
        
        # Add octave band crest factors for min/max chunks if pre-computed data is available
        if chunk_octave_analysis is not None:
            min_chunk_data = chunk_octave_analysis.get("min_chunk")
            max_chunk_data = chunk_octave_analysis.get("max_chunk")
            
            if min_chunk_data is not None:
                min_analysis = min_chunk_data["analysis"]
                min_time = min_chunk_data["time"]
                min_crest_db = min_chunk_data["crest_factor_db"]
                
                # Extract crest factors for each octave band (skip full spectrum)
                min_octave_crest_db = []
                for freq in plot_freqs:  # Use plot_freqs which excludes full spectrum
                    freq_key = f"{freq:.3f}"
                    if freq_key in min_analysis["statistics"]:
                        crest_db = min_analysis["statistics"][freq_key].get("crest_factor_db", 0.0)
                        min_octave_crest_db.append(crest_db if np.isfinite(crest_db) else 0.0)
                    else:
                        min_octave_crest_db.append(0.0)
                
                # Plot min chunk octave band crest factors
                ax.semilogx(plot_freqs, min_octave_crest_db, 'g--', linewidth=1.5, alpha=0.8,
                           label=f'Min Crest Chunk Octaves ({min_crest_db:.1f} dB @ {min_time:.0f}s)')
            
            if max_chunk_data is not None:
                max_analysis = max_chunk_data["analysis"]
                max_time = max_chunk_data["time"]
                max_crest_db = max_chunk_data["crest_factor_db"]
                
                # Extract crest factors for each octave band (skip full spectrum)
                max_octave_crest_db = []
                for freq in plot_freqs:  # Use plot_freqs which excludes full spectrum
                    freq_key = f"{freq:.3f}"
                    if freq_key in max_analysis["statistics"]:
                        crest_db = max_analysis["statistics"][freq_key].get("crest_factor_db", 0.0)
                        max_octave_crest_db.append(crest_db if np.isfinite(crest_db) else 0.0)
                    else:
                        max_octave_crest_db.append(0.0)
                
                # Plot max chunk octave band crest factors
                ax.semilogx(plot_freqs, max_octave_crest_db, 'm--', linewidth=1.5, alpha=0.8,
                           label=f'Max Crest Chunk Octaves ({max_crest_db:.1f} dB @ {max_time:.0f}s)')
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Crest Factor (dB)')
        ax.set_title(self._format_title('Crest Factor by Octave Band', 
                                       track_name, channel_name))
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([15, 20000])
        ax.set_ylim([0, 30])
        
        # Add frequency labels
        ax.set_xticks([16, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_xticklabels(['16', '31.25', '62.5', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Crest factor plot saved to: {output_path}")
        
        plt.close(fig)

