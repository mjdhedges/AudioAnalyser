"""
Analysis and visualization module for Music Analyser.

This module handles octave band analysis, statistical calculations, and visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MusicAnalyzer:
    """Main class for music analysis operations."""

    def __init__(self, sample_rate: int = 44100, original_peak: float = 1.0) -> None:
        """Initialize the music analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
            original_peak: Original peak level before normalization (for dBFS calculation)
        """
        self.sample_rate = sample_rate
        self.original_peak = original_peak

    def analyze_octave_bands(self, octave_bank: np.ndarray, 
                           center_frequencies: List[float]) -> Dict:
        """Perform comprehensive octave band analysis.
        
        Args:
            octave_bank: Octave bank array with filtered signals
            center_frequencies: List of center frequencies
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting octave band analysis...")
        
        num_bands = octave_bank.shape[1]
        
        # Create extended center frequencies list to match octave bank
        extended_frequencies = [0] + center_frequencies  # Add 0 for full spectrum
        
        results = {
            "band_data": {},
            "statistics": {},
            "center_frequencies": extended_frequencies
        }

        # Analyze each band
        for i in range(num_bands):
            band_signal = octave_bank[:, i]
            freq_label = f"{extended_frequencies[i]:.3f}" if i > 0 else "Full Spectrum"
            
            band_stats = self._calculate_band_statistics(band_signal)
            results["band_data"][freq_label] = band_signal
            results["statistics"][freq_label] = band_stats

        logger.info("Octave band analysis complete")
        return results

    def _calculate_band_statistics(self, signal: np.ndarray) -> Dict:
        """Calculate statistics for a single frequency band.
        
        Args:
            signal: Audio signal for the band
            
        Returns:
            Dictionary with statistical measures
        """
        # Basic statistics
        max_val = np.max(np.abs(signal))
        # Calculate dBFS relative to original track's full scale
        max_dbfs = 20 * np.log10(max_val * self.original_peak) if max_val > 0 else -np.inf
        
        rms_val = np.sqrt(np.mean(signal**2))
        # Calculate RMS in dBFS relative to original track's full scale  
        rms_dbfs = 20 * np.log10(rms_val * self.original_peak) if rms_val > 0 else -np.inf
        
        # Dynamic range
        dynamic_range = rms_val / max_val if max_val > 0 else 0
        dynamic_range_db = 20 * np.log10(dynamic_range) if dynamic_range > 0 else -np.inf
        
        # Crest factor (peak to RMS ratio)
        # Crest factor must be >= 1 (0 dB) since peak >= RMS always
        if rms_val > 0 and max_val > 0:
            crest_factor = max_val / rms_val
            # Ensure crest factor is at least 1.0 (0 dB)
            crest_factor = max(crest_factor, 1.0)
            crest_factor_db = 20 * np.log10(crest_factor)
        else:
            # No meaningful signal - set to minimum valid crest factor
            crest_factor = 1.0
            crest_factor_db = 0.0
        
        # Additional statistics
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        # Percentiles for distribution analysis
        percentiles = np.percentile(np.abs(signal), [10, 25, 50, 75, 90, 95, 99])
        
        stats = {
            "max_amplitude": max_val,
            "max_amplitude_db": max_dbfs,  # Now in dBFS
            "rms": rms_val,
            "rms_db": rms_dbfs,  # Now in dBFS
            "dynamic_range": dynamic_range,
            "dynamic_range_db": dynamic_range_db,
            "crest_factor": crest_factor,
            "crest_factor_db": crest_factor_db,
            "mean": mean_val,
            "std": std_val,
            "percentiles": {
                "p10": percentiles[0],
                "p25": percentiles[1],
                "p50": percentiles[2],
                "p75": percentiles[3],
                "p90": percentiles[4],
                "p95": percentiles[5],
                "p99": percentiles[6],
            }
        }
        
        return stats

    def create_octave_spectrum_plot(self, analysis_results: Dict, 
                                  output_path: Optional[str] = None,
                                  time_analysis: Optional[Dict] = None,
                                  audio_data: Optional[np.ndarray] = None) -> None:
        """Create octave spectrum plot similar to MATLAB's semilogx plot.
        
        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
            time_analysis: Optional time-domain analysis results for chunk comparison
            audio_data: Optional original audio data for chunk analysis
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

        # Plot the data
        ax.semilogx(center_freqs, max_db_values, 'b-o', label='Max Peak (dBFS)', linewidth=2)
        ax.semilogx(center_freqs, rms_db_values, 'r-s', label='RMS (dBFS)', linewidth=2)
        
        # Add horizontal reference lines for track totals
        track_peak_db = statistics["Full Spectrum"]["max_amplitude_db"]
        track_rms_db = statistics["Full Spectrum"]["rms_db"]
        
        ax.axhline(y=track_peak_db, color='blue', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'Track Peak ({track_peak_db:.1f} dBFS)')
        ax.axhline(y=track_rms_db, color='red', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'Track RMS ({track_rms_db:.1f} dBFS)')
        
        # Add extreme crest factor chunk analysis if available
        if time_analysis is not None and audio_data is not None:
            self._add_extreme_chunk_analysis(ax, time_analysis, audio_data, center_freqs)
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (dBFS)')
        ax.set_title('Octave Band Analysis - Peak and RMS Levels (dBFS)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([20, 20000])
        ax.set_ylim([-60, 3])
        
        # Add frequency labels
        ax.set_xticks([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_xticklabels(['31.25', '62.5', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_path}")
        
        plt.close(fig)

    def _add_extreme_chunk_analysis(self, ax, time_analysis: Dict, audio_data: np.ndarray, 
                                   center_freqs: List[float]) -> None:
        """Add analysis of extreme crest factor chunks to the octave spectrum plot.
        
        Args:
            ax: Matplotlib axis to plot on
            time_analysis: Time-domain analysis results
            audio_data: Original audio data
            center_freqs: Center frequencies for octave bands
        """
        from src.octave_filter import OctaveBandFilter
        
        # Find chunks with extreme crest factors
        crest_factors_db = time_analysis["crest_factors_db"]
        valid_indices = np.isfinite(crest_factors_db)
        
        if not np.any(valid_indices):
            return
            
        valid_crest_db = crest_factors_db[valid_indices]
        valid_time_points = time_analysis["time_points"][valid_indices]
        
        # Find min and max crest factor indices
        min_idx = np.argmin(valid_crest_db)
        max_idx = np.argmax(valid_crest_db)
        
        # Get the actual indices in the original arrays
        valid_indices_array = np.where(valid_indices)[0]
        min_chunk_idx = valid_indices_array[min_idx]
        max_chunk_idx = valid_indices_array[max_idx]
        
        # Extract chunks from audio data
        chunk_duration = time_analysis["chunk_duration"]
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        # Create octave filter for chunk analysis
        octave_filter = OctaveBandFilter(self.sample_rate)
        
        # Analyze minimum crest factor chunk
        min_start = min_chunk_idx * chunk_samples
        min_end = min_start + chunk_samples
        min_chunk = audio_data[min_start:min_end] if min_end <= len(audio_data) else audio_data[min_start:]
        
        if len(min_chunk) > 0:
            min_octave_bank = octave_filter.create_octave_bank(min_chunk, center_freqs[1:])  # Skip full spectrum
            min_analysis = self.analyze_octave_bands(min_octave_bank, center_freqs[1:])
            
            # Get both RMS and peak levels for octave bands
            min_rms_db = [min_analysis["statistics"][f"{freq:.3f}"]["rms_db"] 
                         for freq in center_freqs[1:]]
            min_peak_db = [min_analysis["statistics"][f"{freq:.3f}"]["max_amplitude_db"] 
                          for freq in center_freqs[1:]]
            
            # Calculate RMS and peak for the full chunk
            min_chunk_rms = np.sqrt(np.mean(min_chunk**2))
            min_chunk_peak = np.max(np.abs(min_chunk))
            min_chunk_rms_dbfs = 20 * np.log10(min_chunk_rms * self.original_peak) if min_chunk_rms > 0 else -np.inf
            min_chunk_peak_dbfs = 20 * np.log10(min_chunk_peak * self.original_peak) if min_chunk_peak > 0 else -np.inf
            
            min_rms_db = [min_chunk_rms_dbfs] + min_rms_db  # Add full spectrum RMS
            min_peak_db = [min_chunk_peak_dbfs] + min_peak_db  # Add full spectrum peak
            
            # Plot minimum crest factor chunk levels
            min_time = valid_time_points[min_idx]
            ax.semilogx(center_freqs, min_rms_db, 'g--', linewidth=1.5, alpha=0.6,
                       label=f'Min Crest RMS ({valid_crest_db[min_idx]:.1f} dB @ {min_time:.0f}s)')
            ax.semilogx(center_freqs, min_peak_db, 'g:', linewidth=1.5, alpha=0.6,
                       label=f'Min Crest Peak ({min_chunk_peak_dbfs:.1f} dBFS @ {min_time:.0f}s)')
        
        # Analyze maximum crest factor chunk
        max_start = max_chunk_idx * chunk_samples
        max_end = max_start + chunk_samples
        max_chunk = audio_data[max_start:max_end] if max_end <= len(audio_data) else audio_data[max_start:]
        
        if len(max_chunk) > 0:
            max_octave_bank = octave_filter.create_octave_bank(max_chunk, center_freqs[1:])  # Skip full spectrum
            max_analysis = self.analyze_octave_bands(max_octave_bank, center_freqs[1:])
            
            # Get both RMS and peak levels for octave bands
            max_rms_db = [max_analysis["statistics"][f"{freq:.3f}"]["rms_db"] 
                         for freq in center_freqs[1:]]
            max_peak_db = [max_analysis["statistics"][f"{freq:.3f}"]["max_amplitude_db"] 
                          for freq in center_freqs[1:]]
            
            # Calculate RMS and peak for the full chunk
            max_chunk_rms = np.sqrt(np.mean(max_chunk**2))
            max_chunk_peak = np.max(np.abs(max_chunk))
            max_chunk_rms_dbfs = 20 * np.log10(max_chunk_rms * self.original_peak) if max_chunk_rms > 0 else -np.inf
            max_chunk_peak_dbfs = 20 * np.log10(max_chunk_peak * self.original_peak) if max_chunk_peak > 0 else -np.inf
            
            max_rms_db = [max_chunk_rms_dbfs] + max_rms_db  # Add full spectrum RMS
            max_peak_db = [max_chunk_peak_dbfs] + max_peak_db  # Add full spectrum peak
            
            # Plot maximum crest factor chunk levels
            max_time = valid_time_points[max_idx]
            ax.semilogx(center_freqs, max_rms_db, 'm--', linewidth=1.5, alpha=0.6,
                       label=f'Max Crest RMS ({valid_crest_db[max_idx]:.1f} dB @ {max_time:.0f}s)')
            ax.semilogx(center_freqs, max_peak_db, 'm:', linewidth=1.5, alpha=0.6,
                       label=f'Max Crest Peak ({max_chunk_peak_dbfs:.1f} dBFS @ {max_time:.0f}s)')

    def create_histogram_plots(self, analysis_results: Dict, 
                             output_dir: Optional[str] = None) -> None:
        """Create histogram plots for each octave band.
        
        Args:
            analysis_results: Results from octave band analysis
            output_dir: Optional directory to save plots
        """
        logger.info("Creating histogram plots...")
        
        band_data = analysis_results["band_data"]
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
        
        plt.suptitle('Amplitude Distribution by Octave Band', fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "histograms.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Histograms saved to: {output_path}")
        
        plt.close(fig)

    def create_histogram_plots_log_db(self, analysis_results: Dict, 
                                     output_dir: Optional[str] = None) -> None:
        """Create histogram plots for each octave band with log dB X-axis.
        
        Args:
            analysis_results: Results from octave band analysis
            output_dir: Optional directory to save plots
        """
        logger.info("Creating log dB histogram plots...")
        
        band_data = analysis_results["band_data"]
        num_bands = len(band_data)
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        # Define dB range and noise floor
        noise_floor_db = -120  # dB below full scale
        max_db = 0  # 0 dBFS
        
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
                    # Use 51 bins (odd) for better 0-centered distribution
                    db_bins = np.linspace(noise_floor_db, max_db, 51)
                    
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
        
        plt.suptitle('Amplitude Distribution by Octave Band (Log dBFS Scale)', fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "histograms_log_db.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Log dB histograms saved to: {output_path}")
        
        plt.close(fig)

    def analyze_crest_factor_over_time(self, audio_data: np.ndarray, 
                                      chunk_duration: float = 2.0) -> Dict:
        """Analyze crest factor over time using sliding windows.
        
        Args:
            audio_data: Input audio signal (normalized)
            chunk_duration: Duration of each analysis chunk in seconds
            
        Returns:
            Dictionary with time-domain crest factor analysis
        """
        logger.info(f"Analyzing crest factor over time with {chunk_duration}s chunks...")
        
        # Calculate chunk size in samples
        chunk_samples = int(chunk_duration * self.sample_rate)
        total_samples = len(audio_data)
        
        # Calculate number of chunks
        num_chunks = total_samples // chunk_samples
        
        # Initialize arrays for results
        time_points = []
        crest_factors = []
        crest_factors_db = []
        peak_levels = []
        rms_levels = []
        peak_levels_dbfs = []
        rms_levels_dbfs = []
        
        for i in range(num_chunks):
            # Extract chunk
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples
            chunk = audio_data[start_idx:end_idx]
            
            # Calculate time point (center of chunk)
            time_point = (start_idx + end_idx) / 2 / self.sample_rate
            time_points.append(time_point)
            
            # Calculate peak and RMS for this chunk
            peak_val = np.max(np.abs(chunk))
            rms_val = np.sqrt(np.mean(chunk**2))
            
            # Calculate crest factor
            crest_factor = peak_val / rms_val if rms_val > 0 else 0
            crest_factor_db = 20 * np.log10(crest_factor) if crest_factor > 0 else -np.inf
            
            # Calculate dBFS values using original peak
            peak_dbfs = 20 * np.log10(peak_val * self.original_peak) if peak_val > 0 else -np.inf
            rms_dbfs = 20 * np.log10(rms_val * self.original_peak) if rms_val > 0 else -np.inf
            
            # Store results
            crest_factors.append(crest_factor)
            crest_factors_db.append(crest_factor_db)
            peak_levels.append(peak_val)
            rms_levels.append(rms_val)
            peak_levels_dbfs.append(peak_dbfs)
            rms_levels_dbfs.append(rms_dbfs)
        
        results = {
            "time_points": np.array(time_points),
            "crest_factors": np.array(crest_factors),
            "crest_factors_db": np.array(crest_factors_db),
            "peak_levels": np.array(peak_levels),
            "rms_levels": np.array(rms_levels),
            "peak_levels_dbfs": np.array(peak_levels_dbfs),
            "rms_levels_dbfs": np.array(rms_levels_dbfs),
            "chunk_duration": chunk_duration,
            "num_chunks": num_chunks,
            "total_duration": total_samples / self.sample_rate
        }
        
        logger.info(f"Time-domain analysis complete: {num_chunks} chunks over {results['total_duration']:.1f}s")
        return results

    def create_crest_factor_time_plot(self, time_analysis: Dict, 
                                     output_path: Optional[str] = None) -> None:
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
        
        # Top plot: Crest Factor vs Time
        ax1.plot(time_points, crest_factors_db_plot, 'g-', linewidth=2, label='Crest Factor')
        ax1.set_ylabel('Crest Factor (dB)')
        ax1.set_title('Crest Factor vs Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, max(30, np.max(crest_factors_db_plot) * 1.1)])
        
        # Bottom plot: Peak and RMS Levels vs Time
        ax2.plot(time_points, peak_levels_dbfs_plot, 'b-', linewidth=2, label='Peak Level')
        ax2.plot(time_points, rms_levels_dbfs_plot, 'r-', linewidth=2, label='RMS Level')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Level (dBFS)')
        ax2.set_title('Peak and RMS Levels vs Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([-60, 5])
        
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
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Crest factor time plot saved to: {output_path}")
        
        plt.close(fig)

    def create_octave_crest_factor_time_plot(self, audio_data: np.ndarray,
                                           time_analysis: Dict, 
                                           output_path: Optional[str] = None) -> None:
        """Create plot showing crest factor over time for all octave bands.
        
        Args:
            audio_data: Original audio data for chunk analysis
            time_analysis: Time-domain analysis results with chunk data
            output_path: Optional path to save the plot
        """
        logger.info("Creating octave band crest factor vs time plot...")
        
        # Extract data from time analysis
        time_points = time_analysis["time_points"]
        chunk_duration = 2.0  # seconds
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        # Get octave band center frequencies
        from src.octave_filter import OctaveBandFilter
        octave_filter = OctaveBandFilter(sample_rate=self.sample_rate)
        center_freqs = octave_filter.OCTAVE_CENTER_FREQUENCIES
        
        # Initialize storage for octave band crest factors over time
        octave_crest_factors = {freq: [] for freq in center_freqs}
        
        # Process each chunk to get octave band crest factors
        for i, time_point in enumerate(time_points):
            # Calculate chunk boundaries
            start_sample = i * chunk_samples
            end_sample = start_sample + chunk_samples
            
            # Get chunk from original audio
            chunk = audio_data[start_sample:end_sample] if end_sample <= len(audio_data) else audio_data[start_sample:]
            
            if len(chunk) > 0:
                # Create octave bank for this chunk
                chunk_octave_bank = octave_filter.create_octave_bank(chunk)
                chunk_analysis = self.analyze_octave_bands(chunk_octave_bank, center_freqs)
                
                # Extract crest factors for each octave band
                for freq in center_freqs:
                    freq_key = f"{freq:.3f}"
                    if freq_key in chunk_analysis["statistics"]:
                        crest_db = chunk_analysis["statistics"][freq_key].get("crest_factor_db", -np.inf)
                        octave_crest_factors[freq].append(crest_db if np.isfinite(crest_db) else 0.0)
                    else:
                        octave_crest_factors[freq].append(0.0)
            else:
                # Empty chunk - add default values
                for freq in center_freqs:
                    octave_crest_factors[freq].append(0.0)
        
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
        ax.set_title('Octave Band Crest Factor vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set reasonable Y-axis limits
        all_values = []
        for freq_values in octave_crest_factors.values():
            all_values.extend([v for v in freq_values if v >= 0])
        
        if all_values:
            y_min = min(all_values) - 1
            y_max = max(all_values) + 2
            # Ensure Y-axis never goes below 0 dB (physically impossible for crest factor)
            ax.set_ylim([max(y_min, 0), min(y_max, 35)])
        else:
            ax.set_ylim([0, 30])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Octave band crest factor time plot saved to: {output_path}")
        
        plt.close(fig)

    def export_analysis_results(self, analysis_results: Dict, 
                              output_path: str) -> None:
        """Export analysis results to CSV file.
        
        Args:
            analysis_results: Results from octave band analysis
            output_path: Path to save CSV file
        """
        logger.info(f"Exporting analysis results to: {output_path}")
        
        # Prepare data for export
        export_data = []
        
        for freq_label, stats in analysis_results["statistics"].items():
            row = {
                "frequency_hz": freq_label,
                "max_amplitude": stats["max_amplitude"],
                "max_amplitude_db": stats["max_amplitude_db"],
                "rms": stats["rms"],
                "rms_db": stats["rms_db"],
                "dynamic_range": stats["dynamic_range"],
                "dynamic_range_db": stats["dynamic_range_db"],
                "crest_factor": stats["crest_factor"],
                "crest_factor_db": stats["crest_factor_db"],
                "mean": stats["mean"],
                "std": stats["std"],
            }
            
            # Add percentiles
            for p_name, p_value in stats["percentiles"].items():
                row[p_name] = p_value
            
            export_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        
        logger.info("Analysis results exported successfully")

    def export_comprehensive_results(self, analysis_results: Dict, time_analysis: Dict,
                                    track_metadata: Dict, output_path: str) -> None:
        """Export comprehensive analysis results including all data to CSV file.
        
        Args:
            analysis_results: Results from octave band analysis
            time_analysis: Results from time-domain crest factor analysis
            track_metadata: Track information and metadata
            output_path: Path to save CSV file
        """
        logger.info(f"Exporting comprehensive analysis results to: {output_path}")
        
        # Create a comprehensive CSV with multiple sections
        with open(output_path, 'w', newline='') as csvfile:
            csvfile.write("# Music Analyser - Comprehensive Analysis Results\n")
            csvfile.write(f"# Generated: {track_metadata['analysis_date']}\n")
            csvfile.write("\n")
            
            # Section 1: Track Metadata
            csvfile.write("[TRACK_METADATA]\n")
            csvfile.write("parameter,value\n")
            for key, value in track_metadata.items():
                csvfile.write(f"{key},{value}\n")
            csvfile.write("\n")
            
            # Section 2: Octave Band Analysis
            csvfile.write("[OCTAVE_BAND_ANALYSIS]\n")
            csvfile.write("frequency_hz,max_amplitude,max_amplitude_db,rms,rms_db,dynamic_range,dynamic_range_db,crest_factor,crest_factor_db,mean,std,p10,p25,p50,p75,p90,p95,p99\n")
            
            statistics = analysis_results["statistics"]
            for freq_label, stats in statistics.items():
                row_data = [
                    freq_label,
                    stats["max_amplitude"],
                    stats["max_amplitude_db"],
                    stats["rms"],
                    stats["rms_db"],
                    stats["dynamic_range"],
                    stats["dynamic_range_db"],
                    stats["crest_factor"],
                    stats["crest_factor_db"],
                    stats["mean"],
                    stats["std"]
                ]
                
                # Add percentiles
                for p_name in ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
                    row_data.append(stats["percentiles"][p_name])
                
                # Write row
                csvfile.write(",".join(map(str, row_data)) + "\n")
            
            csvfile.write("\n")
            
            # Section 3: Time Domain Analysis
            csvfile.write("[TIME_DOMAIN_ANALYSIS]\n")
            csvfile.write("time_seconds,crest_factor,crest_factor_db,peak_level,rms_level,peak_level_dbfs,rms_level_dbfs\n")
            
            for i in range(len(time_analysis["time_points"])):
                row_data = [
                    time_analysis["time_points"][i],
                    time_analysis["crest_factors"][i],
                    time_analysis["crest_factors_db"][i],
                    time_analysis["peak_levels"][i],
                    time_analysis["rms_levels"][i],
                    time_analysis["peak_levels_dbfs"][i],
                    time_analysis["rms_levels_dbfs"][i]
                ]
                csvfile.write(",".join(map(str, row_data)) + "\n")
            
            csvfile.write("\n")
            
            # Section 4: Time Domain Summary Statistics
            csvfile.write("[TIME_DOMAIN_SUMMARY]\n")
            csvfile.write("parameter,value\n")
            
            # Calculate summary statistics for time domain data
            valid_crest_db = time_analysis["crest_factors_db"][np.isfinite(time_analysis["crest_factors_db"])]
            valid_peak_dbfs = time_analysis["peak_levels_dbfs"][np.isfinite(time_analysis["peak_levels_dbfs"])]
            valid_rms_dbfs = time_analysis["rms_levels_dbfs"][np.isfinite(time_analysis["rms_levels_dbfs"])]
            
            summary_stats = {
                "chunk_duration_seconds": time_analysis["chunk_duration"],
                "total_chunks": time_analysis["num_chunks"],
                "crest_factor_mean_db": np.mean(valid_crest_db) if len(valid_crest_db) > 0 else np.nan,
                "crest_factor_std_db": np.std(valid_crest_db) if len(valid_crest_db) > 0 else np.nan,
                "crest_factor_min_db": np.min(valid_crest_db) if len(valid_crest_db) > 0 else np.nan,
                "crest_factor_max_db": np.max(valid_crest_db) if len(valid_crest_db) > 0 else np.nan,
                "peak_level_mean_dbfs": np.mean(valid_peak_dbfs) if len(valid_peak_dbfs) > 0 else np.nan,
                "peak_level_std_dbfs": np.std(valid_peak_dbfs) if len(valid_peak_dbfs) > 0 else np.nan,
                "rms_level_mean_dbfs": np.mean(valid_rms_dbfs) if len(valid_rms_dbfs) > 0 else np.nan,
                "rms_level_std_dbfs": np.std(valid_rms_dbfs) if len(valid_rms_dbfs) > 0 else np.nan
            }
            
            for key, value in summary_stats.items():
                csvfile.write(f"{key},{value}\n")
        
        logger.info("Comprehensive analysis results exported successfully")

    def create_crest_factor_plot(self, analysis_results: Dict, 
                               output_path: Optional[str] = None,
                               time_analysis: Optional[Dict] = None,
                               audio_data: Optional[np.ndarray] = None) -> None:
        """Create crest factor plot showing peak-to-RMS ratio for each octave band.
        
        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
            time_analysis: Optional time-domain analysis results for chunk markers
            audio_data: Optional audio data for calculating chunk octave band crest factors
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
        
        # Plot dB crest factor
        ax.semilogx(center_freqs, crest_factor_db_values, 'b-o', 
                   label='Crest Factor (dB)', linewidth=2, markersize=6)
        
        # Add horizontal reference line for average crest factor
        ax.axhline(y=full_spectrum_crest_db, color='r', linestyle='--', linewidth=2, 
                  label=f'Track Average ({full_spectrum_crest_db:.1f} dB)')
        
        # Add octave band crest factors for min/max chunks if time analysis is provided
        if time_analysis is not None:
            crest_factors_db = time_analysis["crest_factors_db"]
            time_points = time_analysis["time_points"]
            
            # Find valid (finite) crest factor values
            valid_mask = np.isfinite(crest_factors_db)
            if np.any(valid_mask):
                valid_crest_db = np.array(crest_factors_db)[valid_mask]
                valid_time_points = np.array(time_points)[valid_mask]
                
                # Find min and max crest factor chunks
                min_idx = np.argmin(valid_crest_db)
                max_idx = np.argmax(valid_crest_db)
                
                min_time = valid_time_points[min_idx]
                max_time = valid_time_points[max_idx]
                min_crest = valid_crest_db[min_idx]
                max_crest = valid_crest_db[max_idx]
                
                # Calculate octave band crest factors for these chunks
                chunk_duration = 2.0  # seconds
                chunk_samples = int(chunk_duration * self.sample_rate)
                
                # Get the actual chunk indices from the original time analysis
                min_chunk_idx = np.where(valid_mask)[0][min_idx]
                max_chunk_idx = np.where(valid_mask)[0][max_idx]
                
                # Calculate octave band crest factors for chunks if audio data is provided
                if audio_data is not None:
                    from src.octave_filter import OctaveBandFilter
                    
                    # Create octave filter for chunk analysis
                    octave_filter = OctaveBandFilter(sample_rate=self.sample_rate)
                    
                    # Analyze minimum crest factor chunk
                    min_start = min_chunk_idx * chunk_samples
                    min_end = min_start + chunk_samples
                    min_chunk = audio_data[min_start:min_end] if min_end <= len(audio_data) else audio_data[min_start:]
                    
                    if len(min_chunk) > 0:
                        min_octave_bank = octave_filter.create_octave_bank(min_chunk)
                        min_analysis = self.analyze_octave_bands(min_octave_bank, center_freqs)
                        min_octave_crest_db = []
                        
                        for freq in center_freqs:
                            freq_key = f"{freq:.3f}"
                            if freq_key in min_analysis["statistics"]:
                                crest_db = min_analysis["statistics"][freq_key].get("crest_factor_db", -np.inf)
                                min_octave_crest_db.append(crest_db if np.isfinite(crest_db) else 0.0)
                            else:
                                min_octave_crest_db.append(0.0)
                        
                        # Plot min chunk octave band crest factors
                        ax.semilogx(center_freqs, min_octave_crest_db, 'g--', linewidth=1.5, alpha=0.8,
                                   label=f'Min Crest Chunk Octaves ({min_crest:.1f} dB @ {min_time:.0f}s)')
                    
                    # Analyze maximum crest factor chunk
                    max_start = max_chunk_idx * chunk_samples
                    max_end = max_start + chunk_samples
                    max_chunk = audio_data[max_start:max_end] if max_end <= len(audio_data) else audio_data[max_start:]
                    
                    if len(max_chunk) > 0:
                        max_octave_bank = octave_filter.create_octave_bank(max_chunk)
                        max_analysis = self.analyze_octave_bands(max_octave_bank, center_freqs)
                        max_octave_crest_db = []
                        
                        for freq in center_freqs:
                            freq_key = f"{freq:.3f}"
                            if freq_key in max_analysis["statistics"]:
                                crest_db = max_analysis["statistics"][freq_key].get("crest_factor_db", -np.inf)
                                max_octave_crest_db.append(crest_db if np.isfinite(crest_db) else 0.0)
                            else:
                                max_octave_crest_db.append(0.0)
                        
                        # Plot max chunk octave band crest factors
                        ax.semilogx(center_freqs, max_octave_crest_db, 'm--', linewidth=1.5, alpha=0.8,
                                   label=f'Max Crest Chunk Octaves ({max_crest:.1f} dB @ {max_time:.0f}s)')
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Crest Factor (dB)')
        ax.set_xlim([20, 20000])
        ax.set_ylim([0, max(30, np.max(crest_factor_db_values) * 1.1)])
        ax.grid(True, alpha=0.3)
        
        # Add frequency labels
        ax.set_xticks([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_xticklabels(['31.25', '62.5', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
        
        # Title and legend
        ax.set_title('Octave Band Analysis - Crest Factor (Peak/RMS Ratio)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Crest factor plot saved to: {output_path}")
        
        plt.close(fig)
