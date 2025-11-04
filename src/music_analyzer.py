"""
Analysis and visualization module for Music Analyser.

This module handles octave band analysis, statistical calculations, and visualization.
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
import pandas as pd
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)


class MusicAnalyzer:
    """Main class for music analysis operations."""

    def __init__(self, sample_rate: int = 44100, original_peak: float = 1.0, dpi: int = 300) -> None:
        """Initialize the music analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
            original_peak: Original peak level before normalization (for dBFS calculation)
            dpi: DPI for plot output (lower for faster batch processing)
        """
        self.sample_rate = sample_rate
        self.original_peak = original_peak
        self.dpi = dpi

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
            # band_data removed - causes 500MB+ memory duplication per track!
            "statistics": {},
            "center_frequencies": extended_frequencies
        }

        # Analyze each band
        for i in range(num_bands):
            band_signal = octave_bank[:, i]
            freq_label = f"{extended_frequencies[i]:.3f}" if i > 0 else "Full Spectrum"
            
            band_stats = self._calculate_band_statistics(band_signal)
            # Don't store band_data - creates massive memory duplication!
            # Instead, histogram functions can slice from octave_bank directly
            # results["band_data"][freq_label] = band_signal
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
                                  chunk_octave_analysis: Optional[Dict] = None) -> None:
        """Create octave spectrum plot similar to MATLAB's semilogx plot.
        
        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
            time_analysis: Optional time-domain analysis results for chunk comparison
            chunk_octave_analysis: Optional pre-computed octave analysis for extreme chunks
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
        ax.set_title('Octave Band Analysis - Peak and RMS Levels (dBFS)')
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
                             octave_bank: Optional[np.ndarray] = None) -> None:
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
        
        plt.suptitle('Amplitude Distribution by Octave Band', fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "histograms.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Histograms saved to: {output_path}")
        
        plt.close(fig)

    def create_histogram_plots_log_db(self, analysis_results: Dict, 
                                     output_dir: Optional[str] = None,
                                     config: Optional[Dict] = None,
                                     octave_bank: Optional[np.ndarray] = None) -> None:
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
        
        plt.suptitle('Amplitude Distribution by Octave Band (Log dBFS Scale)', fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "histograms_log_db.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Log dB histograms saved to: {output_path}")
        
        plt.close(fig)

    def analyze_comprehensive(self, audio_data: np.ndarray, octave_bank: np.ndarray,
                            center_frequencies: List[float], chunk_duration: float = 2.0) -> Dict:
        """Perform comprehensive analysis including time-domain and chunk-specific octave analysis.
        
        Args:
            audio_data: Original audio data
            octave_bank: Pre-computed octave bank
            center_frequencies: List of center frequencies
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Performing comprehensive analysis...")
        
        # Main octave band analysis
        main_analysis = self.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Time-domain analysis
        time_analysis = self._analyze_time_domain_chunks(audio_data, chunk_duration)
        
        # Chunk-specific octave analysis for min/max crest factor chunks
        chunk_octave_analysis = self._analyze_extreme_chunks_octave_bands(
            octave_bank, time_analysis, center_frequencies, chunk_duration
        )
        
        return {
            "main_analysis": main_analysis,
            "time_analysis": time_analysis,
            "chunk_octave_analysis": chunk_octave_analysis
        }

    def _analyze_time_domain_chunks(self, audio_data: np.ndarray, 
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

    def analyze_crest_factor_over_time(self, audio_data: np.ndarray, 
                                     chunk_duration: float = 2.0) -> Dict:
        """Analyze crest factor over time using sliding windows.
        
        This is a convenience method that calls the internal _analyze_time_domain_chunks.
        """
        return self._analyze_time_domain_chunks(audio_data, chunk_duration)

    def _analyze_extreme_chunks_octave_bands(self, octave_bank: np.ndarray,
                                           time_analysis: Dict,
                                           center_frequencies: List[float],
                                           chunk_duration: float) -> Dict:
        """Analyze octave bands for min/max crest factor chunks.
        
        Args:
            octave_bank: Pre-computed octave bank (full track) - CRITICAL for performance
            time_analysis: Time-domain analysis results
            center_frequencies: List of center frequencies
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            Dictionary with octave analysis for extreme chunks
        """
        crest_factors_db = time_analysis["crest_factors_db"]
        time_points = time_analysis["time_points"]
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        # Find valid (finite) crest factor values
        valid_mask = np.isfinite(crest_factors_db)
        if not np.any(valid_mask):
            return {"min_chunk": None, "max_chunk": None}
        
        valid_crest_db = np.array(crest_factors_db)[valid_mask]
        valid_time_points = np.array(time_points)[valid_mask]
        
        # Find min and max crest factor chunks
        min_idx = np.argmin(valid_crest_db)
        max_idx = np.argmax(valid_crest_db)
        
        # Get the actual chunk indices from the original time analysis
        min_chunk_idx = np.where(valid_mask)[0][min_idx]
        max_chunk_idx = np.where(valid_mask)[0][max_idx]
        
        results = {}
        
        # Analyze minimum crest factor chunk using pre-computed octave bank
        min_start = min_chunk_idx * chunk_samples
        min_end = min_start + chunk_samples
        if min_end > octave_bank.shape[0]:
            min_end = octave_bank.shape[0]
        
        if min_start < min_end:
            # Slice pre-computed octave bank
            min_chunk_octave = octave_bank[min_start:min_end, :]
            min_analysis = self.analyze_octave_bands(min_chunk_octave, center_frequencies)
            results["min_chunk"] = {
                "analysis": min_analysis,
                "time": valid_time_points[min_idx],
                "crest_factor_db": valid_crest_db[min_idx],
                "chunk_idx": min_chunk_idx
            }
        else:
            results["min_chunk"] = None
        
        # Analyze maximum crest factor chunk using pre-computed octave bank
        max_start = max_chunk_idx * chunk_samples
        max_end = max_start + chunk_samples
        if max_end > octave_bank.shape[0]:
            max_end = octave_bank.shape[0]
        
        if max_start < max_end:
            # Slice pre-computed octave bank
            max_chunk_octave = octave_bank[max_start:max_end, :]
            max_analysis = self.analyze_octave_bands(max_chunk_octave, center_frequencies)
            results["max_chunk"] = {
                "analysis": max_analysis,
                "time": valid_time_points[max_idx],
                "crest_factor_db": valid_crest_db[max_idx],
                "chunk_idx": max_chunk_idx
            }
        else:
            results["max_chunk"] = None
        
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
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Crest factor time plot saved to: {output_path}")
        
        plt.close(fig)

    def create_octave_crest_factor_time_plot(self, octave_bank: np.ndarray,
                                           time_analysis: Dict, 
                                           center_frequencies: List[float],
                                           output_path: Optional[str] = None) -> None:
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
        for freq_idx, freq in enumerate(center_freqs):
            # Get all samples for this frequency band (skip full spectrum at index 0)
            band_all = octave_bank[:, freq_idx + 1]
            
            # Calculate how many complete chunks we can make
            num_complete_chunks = len(band_all) // chunk_samples
            
            if num_complete_chunks > 0:
                # Reshape into chunks: (num_chunks, chunk_samples)
                # This creates a view that allows vectorized computation
                band_reshaped = band_all[:num_complete_chunks * chunk_samples].reshape(
                    num_complete_chunks, chunk_samples
                )
                
                # Vectorized RMS calculation for all chunks simultaneously
                rms_vals = np.sqrt(np.mean(band_reshaped**2, axis=1))
                
                # Vectorized peak calculation for all chunks simultaneously
                peak_vals = np.max(np.abs(band_reshaped), axis=1)
                
                # Vectorized crest factor calculation
                # Avoid division by zero and ensure crest factor >= 1.0
                crest_factors = np.divide(
                    peak_vals, rms_vals, 
                    out=np.ones_like(peak_vals), 
                    where=(rms_vals > 0)
                )
                crest_factors = np.maximum(crest_factors, 1.0)
                
                # Convert to dB
                crest_db = 20 * np.log10(crest_factors)
                
                # Handle infinite values
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
        ax.set_title('Octave Band Crest Factor vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set fixed Y-axis limits for consistency across tracks
        ax.set_ylim([0, 40])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Octave band crest factor time plot saved to: {output_path}")
        
        plt.close(fig)

    def _calculate_rms_envelope(self, signal: np.ndarray, 
                                window_ms: float) -> np.ndarray:
        """Calculate RMS envelope using efficient FFT convolution.
        
        This method uses scipy's fftconvolve for optimal performance with large
        windows. The RMS envelope can be reused for both plotting and statistics.
        
        Args:
            signal: Audio signal array (normalized)
            window_ms: Window size in milliseconds for RMS calculation
            
        Returns:
            RMS envelope array in linear scale (same length as input)
        """
        window_samples = int(window_ms * self.sample_rate / 1000)
        
        if window_samples < 1:
            window_samples = 1
        
        if window_samples >= len(signal):
            # Window larger than signal - return single RMS value
            rms_val = np.sqrt(np.mean(signal**2))
            return np.full_like(signal, rms_val)
        
        # Create rectangular window
        window = np.ones(window_samples, dtype=signal.dtype) / window_samples
        
        # Calculate RMS envelope using FFT convolution (faster for large windows)
        signal_squared = signal ** 2
        rms_squared = sp_signal.fftconvolve(signal_squared, window, mode='same')
        
        # Ensure non-negative (handle numerical precision issues)
        rms_squared = np.maximum(rms_squared, 0.0)
        
        # Calculate RMS
        rms_envelope = np.sqrt(rms_squared)
        
        return rms_envelope

    def _find_attack_time(self, envelope_db: np.ndarray, peak_idx: int,
                         peak_value_db: float, attack_threshold_db: float) -> float:
        """Find attack time using vectorized backward search.
        
        Args:
            envelope_db: RMS envelope in dBFS
            peak_idx: Index of peak sample
            peak_value_db: Peak value in dBFS
            attack_threshold_db: Threshold below peak to detect attack start
            
        Returns:
            Attack time in milliseconds
        """
        threshold = peak_value_db - attack_threshold_db
        
        # Search backward from peak using vectorized boolean indexing
        search_start = max(0, peak_idx - int(10 * self.sample_rate))  # Limit search to 10s
        search_region = envelope_db[search_start:peak_idx + 1]
        
        # Find first sample below threshold (searching backward)
        below_threshold = search_region < threshold
        below_indices = np.where(below_threshold)[0]
        
        if len(below_indices) > 0:
            # Last index before peak that's below threshold
            attack_start_idx = search_start + below_indices[-1] + 1
        else:
            # Attack started at beginning of search region
            attack_start_idx = search_start
        
        attack_samples = peak_idx - attack_start_idx
        attack_time_ms = (attack_samples / self.sample_rate) * 1000.0
        
        return max(0.0, attack_time_ms)

    def _find_peak_hold_time(self, envelope_db: np.ndarray, peak_idx: int,
                            peak_value_db: float, hold_threshold_db: float) -> float:
        """Find peak hold time using vectorized forward/backward search.
        
        Args:
            envelope_db: RMS envelope in dBFS
            peak_idx: Index of peak sample
            peak_value_db: Peak value in dBFS
            hold_threshold_db: Threshold for peak hold (within this of peak)
            
        Returns:
            Peak hold time in milliseconds
        """
        threshold = peak_value_db - hold_threshold_db
        
        # Search backward from peak
        search_back = max(0, peak_idx - int(5 * self.sample_rate))  # Limit to 5s
        back_region = envelope_db[search_back:peak_idx + 1]
        back_mask = back_region >= threshold
        back_indices = np.where(back_mask)[0]
        
        if len(back_indices) > 0:
            hold_start_idx = search_back + back_indices[0]
        else:
            hold_start_idx = peak_idx
        
        # Search forward from peak
        search_forward = min(len(envelope_db), peak_idx + int(5 * self.sample_rate))
        forward_region = envelope_db[peak_idx:search_forward]
        forward_mask = forward_region >= threshold
        forward_indices = np.where(forward_mask)[0]
        
        if len(forward_indices) > 0:
            hold_end_idx = peak_idx + forward_indices[-1]
        else:
            hold_end_idx = peak_idx
        
        hold_samples = hold_end_idx - hold_start_idx
        hold_time_ms = (hold_samples / self.sample_rate) * 1000.0
        
        return max(0.0, hold_time_ms)

    def _find_decay_times(self, envelope_db: np.ndarray, peak_idx: int,
                         peak_value_db: float,
                         decay_thresholds_db: List[float]) -> Dict[str, float]:
        """Find decay times to multiple thresholds in single pass.
        
        Args:
            envelope_db: RMS envelope in dBFS
            peak_idx: Index of peak sample
            peak_value_db: Peak value in dBFS
            decay_thresholds_db: List of decay thresholds in dB (negative values)
            
        Returns:
            Dictionary with decay times in milliseconds and reach flags
        """
        # Limit search to reasonable distance (e.g., 30 seconds)
        search_end = min(len(envelope_db), peak_idx + int(30 * self.sample_rate))
        decay_region = envelope_db[peak_idx:search_end]
        
        # Calculate absolute thresholds for each decay level
        absolute_thresholds = [peak_value_db + threshold_db for threshold_db in decay_thresholds_db]
        
        decay_times = {}
        
        # Single pass: find first sample below each threshold
        for i, (threshold_db, abs_threshold) in enumerate(zip(decay_thresholds_db, absolute_thresholds)):
            # Vectorized search: find first sample <= threshold
            below_mask = decay_region <= abs_threshold
            below_indices = np.where(below_mask)[0]
            
            if len(below_indices) > 0:
                decay_samples = below_indices[0]
                decay_time_ms = (decay_samples / self.sample_rate) * 1000.0
                decay_reached = True
            else:
                # Decay never reached - use track duration as fallback
                decay_time_ms = ((search_end - peak_idx) / self.sample_rate) * 1000.0
                decay_reached = False
            
            decay_times[f"decay_{abs(int(threshold_db))}db_ms"] = decay_time_ms
            decay_times[f"decay_{abs(int(threshold_db))}db_reached"] = decay_reached
        
        return decay_times

    def _analyze_worst_case_envelopes(self, envelope_db: np.ndarray,
                                     peak_indices: np.ndarray,
                                     peak_values_db: np.ndarray,
                                     num_envelopes: int,
                                     sort_by: str,
                                     attack_threshold_db: float,
                                     peak_hold_threshold_db: float,
                                     decay_thresholds_db: List[float],
                                     exclude_peak_indices: Optional[set] = None) -> List[Dict]:
        """Analyze worst-case envelopes efficiently using partition-based selection.
        
        Args:
            envelope_db: RMS envelope in dBFS
            peak_indices: Array of peak sample indices
            peak_values_db: Array of peak values in dBFS
            num_envelopes: Number of worst-case envelopes to find
            sort_by: Sorting criterion ("peak_value", "decay_time", "energy")
            attack_threshold_db: Attack detection threshold
            peak_hold_threshold_db: Peak hold threshold
            decay_thresholds_db: Decay thresholds
            exclude_peak_indices: Optional set of peak indices to exclude (e.g., pattern peaks)
            
        Returns:
            List of worst-case envelope dictionaries with characteristics
        """
        if len(peak_indices) == 0:
            return []
        
        # Filter out excluded peaks (non-repeating only)
        if exclude_peak_indices is not None:
            exclude_set = exclude_peak_indices
            filtered_mask = np.array([idx not in exclude_set for idx in peak_indices])
            peak_indices = peak_indices[filtered_mask]
            peak_values_db = peak_values_db[filtered_mask]
        
        if len(peak_indices) == 0:
            return []
        
        num_envelopes = min(num_envelopes, len(peak_indices))
        
        # Select top N peaks based on criterion
        if sort_by == "peak_value":
            # Sort by peak value (highest first)
            if num_envelopes < len(peak_indices):
                # Use partition for efficiency (faster than full sort for small N)
                top_indices = np.argpartition(peak_values_db, -num_envelopes)[-num_envelopes:]
                top_indices = top_indices[np.argsort(peak_values_db[top_indices])[::-1]]
            else:
                top_indices = np.argsort(peak_values_db)[::-1]
        elif sort_by == "decay_time":
            # This requires calculating decay first - for now, use peak value
            # TODO: Implement decay_time sorting if needed
            top_indices = np.argsort(peak_values_db)[::-1][:num_envelopes]
        else:  # "energy" or default
            # Use peak value as proxy for energy
            top_indices = np.argsort(peak_values_db)[::-1][:num_envelopes]
        
        worst_case_envelopes = []
        
        for rank, idx in enumerate(top_indices[:num_envelopes], 1):
            peak_idx = peak_indices[idx]
            peak_value_db = peak_values_db[idx]
            peak_time_seconds = peak_idx / self.sample_rate
            
            # Calculate envelope characteristics
            attack_time_ms = self._find_attack_time(
                envelope_db, peak_idx, peak_value_db, attack_threshold_db
            )
            
            peak_hold_time_ms = self._find_peak_hold_time(
                envelope_db, peak_idx, peak_value_db, peak_hold_threshold_db
            )
            
            decay_times = self._find_decay_times(
                envelope_db, peak_idx, peak_value_db, decay_thresholds_db
            )
            
            envelope_data = {
                "rank": rank,
                "peak_value_db": float(peak_value_db),
                "peak_time_seconds": float(peak_time_seconds),
                "attack_time_ms": float(attack_time_ms),
                "peak_hold_time_ms": float(peak_hold_time_ms),
                "decay_times": decay_times
            }
            
            worst_case_envelopes.append(envelope_data)
        
        return worst_case_envelopes

    def _compare_envelope_shapes(self, pattern1: np.ndarray,
                                 pattern2: np.ndarray) -> float:
        """Compare two envelope patterns using correlation.
        
        Patterns are normalized to 0-1 range before comparison.
        
        Args:
            pattern1: First envelope pattern
            pattern2: Second envelope pattern
            
        Returns:
            Correlation coefficient (0-1, higher = more similar)
        """
        # Normalize both patterns to 0-1 range
        p1_min, p1_max = np.min(pattern1), np.max(pattern1)
        p2_min, p2_max = np.min(pattern2), np.max(pattern2)
        
        if p1_max - p1_min < 1e-10 or p2_max - p2_min < 1e-10:
            # Constant patterns - return 1.0 if both constant, 0.0 otherwise
            return 1.0 if abs(p1_max - p2_max) < 1e-10 else 0.0
        
        p1_norm = (pattern1 - p1_min) / (p1_max - p1_min)
        p2_norm = (pattern2 - p2_min) / (p2_max - p2_min)
        
        # Ensure same length (interpolate if needed)
        if len(p1_norm) != len(p2_norm):
            from scipy.interpolate import interp1d
            min_len = min(len(p1_norm), len(p2_norm))
            x1 = np.linspace(0, 1, len(p1_norm))
            x2 = np.linspace(0, 1, len(p2_norm))
            x_common = np.linspace(0, 1, min_len)
            
            f1 = interp1d(x1, p1_norm, kind='linear', bounds_error=False, fill_value='extrapolate')
            f2 = interp1d(x2, p2_norm, kind='linear', bounds_error=False, fill_value='extrapolate')
            
            p1_norm = f1(x_common)
            p2_norm = f2(x_common)
        
        # Calculate correlation
        correlation = np.corrcoef(p1_norm, p2_norm)[0, 1]
        
        # Handle NaN (shouldn't happen, but safety check)
        if np.isnan(correlation):
            return 0.0
        
        return float(correlation)

    def _analyze_repeating_patterns(self, envelope_db: np.ndarray,
                                   peak_indices: np.ndarray,
                                   peak_values_db: np.ndarray,
                                   min_repetitions: int,
                                   max_patterns: int,
                                   similarity_threshold: float,
                                   window_ms: float) -> Dict:
        """Analyze repeating patterns using optimized correlation matching.
        
        Args:
            envelope_db: RMS envelope in dBFS
            peak_indices: Array of peak sample indices
            peak_values_db: Array of peak values in dBFS
            min_repetitions: Minimum number of repetitions to consider a pattern
            max_patterns: Maximum number of patterns to detect
            similarity_threshold: Correlation threshold for pattern matching
            window_ms: Window size for pattern extraction
            
        Returns:
            Dictionary with pattern analysis results
        """
        if len(peak_indices) < min_repetitions:
            return {"patterns_detected": 0}
        
        window_samples = int(window_ms * self.sample_rate / 1000)
        half_window = window_samples // 2
        
        # Extract envelope windows around all peaks (vectorized)
        patterns = []
        valid_peak_indices = []
        
        for peak_idx in peak_indices:
            start_idx = max(0, peak_idx - half_window)
            end_idx = min(len(envelope_db), peak_idx + half_window)
            
            if end_idx - start_idx >= window_samples // 2:  # Minimum pattern size
                pattern = envelope_db[start_idx:end_idx]
                patterns.append(pattern)
                valid_peak_indices.append(peak_idx)
        
        if len(patterns) < min_repetitions:
            return {"patterns_detected": 0}
        
        # Normalize patterns to 0-1 for comparison
        normalized_patterns = []
        for pattern in patterns:
            p_min, p_max = np.min(pattern), np.max(pattern)
            if p_max - p_min > 1e-10:
                normalized = (pattern - p_min) / (p_max - p_min)
            else:
                normalized = np.zeros_like(pattern)
            normalized_patterns.append(normalized)
        
        # Find pattern groups using correlation
        pattern_groups = []
        used_indices = set()
        
        for i, pattern in enumerate(normalized_patterns):
            if i in used_indices:
                continue
            
            # Find similar patterns
            group = [i]
            for j in range(i + 1, len(normalized_patterns)):
                if j in used_indices:
                    continue
                
                correlation = np.corrcoef(pattern, normalized_patterns[j])[0, 1]
                if not np.isnan(correlation) and correlation >= similarity_threshold:
                    group.append(j)
                    used_indices.add(j)
            
            if len(group) >= min_repetitions:
                pattern_groups.append(group)
                used_indices.add(i)
        
        if len(pattern_groups) == 0:
            return {"patterns_detected": 0}
        
        # Sort groups by size (largest first) and take top max_patterns
        pattern_groups.sort(key=len, reverse=True)
        pattern_groups = pattern_groups[:max_patterns]
        
        result = {"patterns_detected": len(pattern_groups)}
        
        # Analyze each pattern group
        for pattern_num, group in enumerate(pattern_groups, 1):
            group_peak_indices = [valid_peak_indices[i] for i in group]
            group_peak_times = [idx / self.sample_rate for idx in group_peak_indices]
            
            # Calculate inter-peak intervals
            intervals = np.diff(sorted(group_peak_times))
            
            if len(intervals) == 0:
                continue
            
            mean_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))
            median_interval = float(np.median(intervals))
            min_interval = float(np.min(intervals))
            max_interval = float(np.max(intervals))
            
            # Calculate coefficient of variation for regularity
            cv = std_interval / mean_interval if mean_interval > 0 else np.inf
            pattern_regularity_score = float(1.0 / (1.0 + cv))
            
            # Classify confidence
            if cv < 0.1:
                confidence = "high"
            elif cv < 0.3:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Calculate BPM
            beats_per_minute = 60.0 / mean_interval if mean_interval > 0 else 0.0
            
            result[f"pattern_{pattern_num}"] = {
                "num_repetitions": len(group),
                "mean_interval_seconds": mean_interval,
                "std_interval_seconds": std_interval,
                "median_interval_seconds": median_interval,
                "min_interval_seconds": min_interval,
                "max_interval_seconds": max_interval,
                "pattern_regularity_score": pattern_regularity_score,
                "pattern_confidence": confidence,
                "beats_per_minute": beats_per_minute,
                "peak_times_seconds": sorted(group_peak_times),
                "peak_indices": group_peak_indices  # Store peak indices for filtering
            }
        
        return result

    def analyze_envelope_statistics(self, octave_bank: np.ndarray,
                                    center_frequencies: List[float],
                                    config: Optional[Dict] = None) -> Dict:
        """Analyze envelope characteristics and repeating patterns for each octave band.
        
        This method performs two types of analysis:
        1. Worst-case envelope analysis: Finds top N worst envelopes (for one-off events)
        2. Pattern analysis: Detects repeating patterns with configurable minimum repetitions
        
        The RMS envelope is calculated ONCE per band and reused for all analysis to maximize
        efficiency.
        
        Args:
            octave_bank: Pre-computed octave bank (full track) - shape (samples, bands)
            center_frequencies: List of center frequencies for octave bands
            config: Optional configuration dictionary (defaults from config.toml if None)
            
        Returns:
            Dictionary with envelope statistics per band
        """
        logger.info("Starting envelope statistics analysis...")
        
        # Get configuration with defaults
        if config is None:
            from src.config import config as global_config
            config = global_config.get('envelope_analysis', {})
        
        window_ms = config.get('rms_envelope_window_ms', 10.0)
        min_height_db = config.get('peak_detection_min_height_db', -40.0)
        min_distance_ms = config.get('peak_detection_min_distance_ms', 50.0)
        worst_case_num = config.get('worst_case_num_envelopes', 1)
        worst_case_sort_by = config.get('worst_case_sort_by', 'peak_value')
        pattern_min_reps = config.get('pattern_min_repetitions', 3)
        pattern_max_patterns = config.get('pattern_max_patterns_per_band', 1)
        pattern_similarity = config.get('pattern_similarity_threshold', 0.85)
        attack_threshold_db = config.get('attack_threshold_db', -20.0)
        peak_hold_threshold_db = config.get('peak_hold_threshold_db', -1.0)
        decay_thresholds_db = config.get('decay_thresholds_db', [-3.0, -6.0, -9.0, -12.0])
        
        # Calculate minimum distance in samples
        min_distance_samples = int(min_distance_ms * self.sample_rate / 1000)
        
        # Extended frequencies list (includes Full Spectrum at index 0)
        extended_frequencies = [0] + center_frequencies
        num_bands = octave_bank.shape[1]
        
        results = {}
        
        # Process each band
        for band_idx in range(num_bands):
            freq = extended_frequencies[band_idx]
            freq_label = f"{freq:.3f}" if freq > 0 else "Full Spectrum"
            
            logger.info(f"Analyzing envelope statistics for {freq_label}...")
            
            # Extract band signal
            band_signal = octave_bank[:, band_idx]
            
            # Calculate RMS envelope ONCE (reused for all analysis)
            rms_envelope_linear = self._calculate_rms_envelope(band_signal, window_ms)
            
            # Convert to dBFS
            rms_envelope_db = 20 * np.log10(
                rms_envelope_linear * self.original_peak + 1e-10
            )
            
            # Replace -inf with very low value for peak detection
            rms_envelope_db_clean = np.copy(rms_envelope_db)
            rms_envelope_db_clean[rms_envelope_db_clean == -np.inf] = -120.0
            
            # Detect peaks using scipy
            peak_indices, peak_properties = sp_signal.find_peaks(
                rms_envelope_db_clean,
                height=min_height_db,
                distance=min_distance_samples
            )
            
            peak_values_db = rms_envelope_db_clean[peak_indices]
            
            band_results = {}
            
            # Worst-case analysis (if enabled)
            if worst_case_num > 0 and len(peak_indices) > 0:
                worst_case_envelopes = self._analyze_worst_case_envelopes(
                    rms_envelope_db,
                    peak_indices,
                    peak_values_db,
                    worst_case_num,
                    worst_case_sort_by,
                    attack_threshold_db,
                    peak_hold_threshold_db,
                    decay_thresholds_db
                )
                band_results["worst_case_envelopes"] = worst_case_envelopes
            else:
                band_results["worst_case_envelopes"] = []
            
            # Pattern analysis (if enabled) - run FIRST to identify pattern peaks
            pattern_peak_indices = set()
            if pattern_max_patterns > 0 and len(peak_indices) >= pattern_min_reps:
                pattern_analysis = self._analyze_repeating_patterns(
                    rms_envelope_db,
                    peak_indices,
                    peak_values_db,
                    pattern_min_reps,
                    pattern_max_patterns,
                    pattern_similarity,
                    window_ms * 2  # Use 2x window for pattern extraction
                )
                band_results["pattern_analysis"] = pattern_analysis
                
                # Collect all peak indices that are part of patterns
                patterns_detected = pattern_analysis.get("patterns_detected", 0)
                for pattern_num in range(1, patterns_detected + 1):
                    pattern_key = f"pattern_{pattern_num}"
                    if pattern_key in pattern_analysis:
                        pattern_peak_indices.update(pattern_analysis[pattern_key].get("peak_indices", []))
            else:
                band_results["pattern_analysis"] = {"patterns_detected": 0}
            
            # Worst-case analysis (if enabled) - exclude pattern peaks (non-repeating only)
            if worst_case_num > 0 and len(peak_indices) > 0:
                worst_case_envelopes = self._analyze_worst_case_envelopes(
                    rms_envelope_db,
                    peak_indices,
                    peak_values_db,
                    worst_case_num,
                    worst_case_sort_by,
                    attack_threshold_db,
                    peak_hold_threshold_db,
                    decay_thresholds_db,
                    exclude_peak_indices=pattern_peak_indices if pattern_peak_indices else None
                )
                band_results["worst_case_envelopes"] = worst_case_envelopes
            else:
                band_results["worst_case_envelopes"] = []
            
            # Store RMS envelope for visualization (needed for plotting)
            band_results["rms_envelope_db"] = rms_envelope_db
            band_results["rms_envelope_time"] = np.arange(len(rms_envelope_db)) / self.sample_rate
            
            results[freq_label] = band_results
            
            # Memory cleanup (keep rms_envelope_db for visualization)
            del rms_envelope_linear, rms_envelope_db_clean
        
        logger.info("Envelope statistics analysis complete")
        return results

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

    def _calculate_advanced_statistics(self, audio_data: np.ndarray, 
                                      analysis_results: Dict, 
                                      time_analysis: Dict) -> Dict:
        """Calculate advanced statistics for track dynamics and quality assessment.
        
        Args:
            audio_data: Raw audio data
            analysis_results: Octave band analysis results
            time_analysis: Time-domain analysis results
            
        Returns:
            Dictionary containing advanced statistics
        """
        logger.info("Calculating advanced statistics...")
        
        stats = {}
        duration = len(audio_data) / self.sample_rate
        
        # Convert audio to dBFS for peak analysis
        audio_dbfs = 20 * np.log10(np.abs(audio_data) * self.original_peak + 1e-10)
        
        # 1. CLIPPING & PEAK STATISTICS
        hot_peaks = np.sum(audio_dbfs > -1.0)  # Near-clipping events
        clip_events = np.sum(audio_dbfs > -0.1)  # Actual clipping events
        peak_saturation = np.sum(audio_dbfs > -3.0)  # Heavily compressed regions
        
        stats.update({
            "hot_peaks_rate_per_sec": hot_peaks / duration,
            "clip_events_rate_per_sec": clip_events / duration,
            "peak_saturation_percent": (peak_saturation / len(audio_data)) * 100,
            "max_true_peak_dbfs": np.max(audio_dbfs),
            "peak_density_above_minus3db": peak_saturation / len(audio_data)
        })
        
        # 2. FREQUENCY-SPECIFIC DYNAMICS
        center_freqs = analysis_results["center_frequencies"][1:]  # Skip full spectrum
        octave_crest_factors = []
        octave_rms_levels = []
        
        for freq in center_freqs:
            freq_key = f"{freq:.3f}"
            if freq_key in analysis_results["statistics"]:
                crest_db = analysis_results["statistics"][freq_key]["crest_factor_db"]
                rms_db = analysis_results["statistics"][freq_key]["rms_db"]
                octave_crest_factors.append(crest_db if np.isfinite(crest_db) else 0.0)
                octave_rms_levels.append(rms_db if np.isfinite(rms_db) else -60.0)
        
        if octave_crest_factors:
            stats.update({
                "frequency_crest_factor_variance": np.var(octave_crest_factors),
                "frequency_crest_factor_range": np.max(octave_crest_factors) - np.min(octave_crest_factors),
                "bass_vs_treble_dynamics_ratio": np.mean(octave_crest_factors[:3]) / max(np.mean(octave_crest_factors[-3:]), 0.1),
                "frequency_balance_std": np.std(octave_rms_levels),
                "most_dynamic_frequency_hz": center_freqs[np.argmax(octave_crest_factors)],
                "least_dynamic_frequency_hz": center_freqs[np.argmin(octave_crest_factors)]
            })
        
        # 3. TEMPORAL DYNAMICS
        time_crest_factors = time_analysis["crest_factors_db"]
        time_rms_levels = time_analysis["rms_levels_dbfs"]
        time_peak_levels = time_analysis["peak_levels_dbfs"]
        
        # Calculate transient density (sudden level changes)
        rms_changes = np.abs(np.diff(time_rms_levels))
        significant_changes = np.sum(rms_changes > 3.0)  # Changes > 3dB
        
        stats.update({
            "dynamic_consistency_std": np.std(time_crest_factors),
            "transient_density_per_sec": significant_changes / duration,
            "loudness_stability_std": np.std(time_rms_levels),
            "peak_level_variance": np.var(time_peak_levels),
            "temporal_crest_factor_range": np.max(time_crest_factors) - np.min(time_crest_factors)
        })
        
        # 4. MASTERING QUALITY INDICATORS
        full_spectrum_stats = analysis_results["statistics"]["Full Spectrum"]
        true_peak_to_rms_ratio = full_spectrum_stats["max_amplitude_db"] - full_spectrum_stats["rms_db"]
        
        # Calculate spectral centroid (brightness indicator)
        # Use energy in each octave band weighted by frequency
        octave_energies = []
        for freq in center_freqs:
            freq_key = f"{freq:.3f}"
            if freq_key in analysis_results["statistics"]:
                rms_linear = analysis_results["statistics"][freq_key]["rms"]
                energy = rms_linear ** 2
                octave_energies.append(energy)
            else:
                octave_energies.append(0.0)
        
        total_energy = sum(octave_energies)
        if total_energy > 0:
            spectral_centroid = sum(freq * energy for freq, energy in zip(center_freqs, octave_energies)) / total_energy
            bass_energy_ratio = sum(octave_energies[:3]) / total_energy  # 31.25, 62.5, 125 Hz
        else:
            spectral_centroid = 0.0
            bass_energy_ratio = 0.0
        
        stats.update({
            "true_peak_to_rms_ratio_db": true_peak_to_rms_ratio,
            "spectral_centroid_hz": spectral_centroid,
            "bass_energy_ratio": bass_energy_ratio,
            "treble_energy_ratio": sum(octave_energies[-3:]) / max(total_energy, 1e-10),  # 4k, 8k, 16k Hz
            "mid_energy_ratio": sum(octave_energies[3:7]) / max(total_energy, 1e-10)  # 250Hz - 2kHz
        })
        
        # 5. PEAK DISTRIBUTION ANALYSIS
        # Analyze distribution of peaks in different dBFS ranges
        peak_ranges = [
            (-0.1, 0.0, "clipping"),
            (-1.0, -0.1, "hot"),
            (-3.0, -1.0, "loud"),
            (-6.0, -3.0, "moderate"),
            (-12.0, -6.0, "quiet"),
            (-60.0, -12.0, "very_quiet")
        ]
        
        for min_db, max_db, range_name in peak_ranges:
            count = np.sum((audio_dbfs >= min_db) & (audio_dbfs < max_db))
            stats[f"peak_distribution_{range_name}_percent"] = (count / len(audio_data)) * 100
        
        logger.info(f"Advanced statistics calculated: {len(stats)} metrics")
        return stats

    def export_comprehensive_results(self, analysis_results: Dict, time_analysis: Dict,
                                   track_metadata: Dict, output_path: str,
                                   chunk_octave_analysis: Optional[Dict] = None,
                                   audio_data: Optional[np.ndarray] = None,
                                   envelope_statistics: Optional[Dict] = None) -> None:
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
            
            # Section 2: Advanced Statistics
            if audio_data is not None:
                csvfile.write("[ADVANCED_STATISTICS]\n")
                csvfile.write("parameter,value,description\n")
                
                advanced_stats = self._calculate_advanced_statistics(audio_data, analysis_results, time_analysis)
                
                # Define descriptions for each statistic
                stat_descriptions = {
                    "hot_peaks_rate_per_sec": "Near-clipping events (>-1dBFS) per second",
                    "clip_events_rate_per_sec": "Actual clipping events (>-0.1dBFS) per second", 
                    "peak_saturation_percent": "Percentage of samples above -3dBFS (heavily compressed)",
                    "max_true_peak_dbfs": "Maximum true peak level in dBFS",
                    "peak_density_above_minus3db": "Fraction of samples above -3dBFS",
                    "frequency_crest_factor_variance": "Variance of crest factors across frequency bands",
                    "frequency_crest_factor_range": "Range of crest factors across frequency bands (dB)",
                    "bass_vs_treble_dynamics_ratio": "Ratio of bass dynamics to treble dynamics",
                    "frequency_balance_std": "Standard deviation of RMS levels across frequencies",
                    "most_dynamic_frequency_hz": "Frequency band with highest crest factor",
                    "least_dynamic_frequency_hz": "Frequency band with lowest crest factor",
                    "dynamic_consistency_std": "Standard deviation of crest factor over time",
                    "transient_density_per_sec": "Rate of significant level changes (>3dB) per second",
                    "loudness_stability_std": "Standard deviation of RMS level over time",
                    "peak_level_variance": "Variance of peak levels over time",
                    "temporal_crest_factor_range": "Range of crest factors over time (dB)",
                    "true_peak_to_rms_ratio_db": "Overall dynamic range (peak to RMS ratio)",
                    "spectral_centroid_hz": "Spectral centroid (brightness indicator)",
                    "bass_energy_ratio": "Fraction of energy in bass frequencies (31-125Hz)",
                    "treble_energy_ratio": "Fraction of energy in treble frequencies (4-16kHz)",
                    "mid_energy_ratio": "Fraction of energy in mid frequencies (250Hz-2kHz)",
                    "peak_distribution_clipping_percent": "Percentage of samples in clipping range (-0.1 to 0dBFS)",
                    "peak_distribution_hot_percent": "Percentage of samples in hot range (-1 to -0.1dBFS)",
                    "peak_distribution_loud_percent": "Percentage of samples in loud range (-3 to -1dBFS)",
                    "peak_distribution_moderate_percent": "Percentage of samples in moderate range (-6 to -3dBFS)",
                    "peak_distribution_quiet_percent": "Percentage of samples in quiet range (-12 to -6dBFS)",
                    "peak_distribution_very_quiet_percent": "Percentage of samples in very quiet range (<-12dBFS)"
                }
                
                for key, value in advanced_stats.items():
                    description = stat_descriptions.get(key, "Advanced statistic")
                    csvfile.write(f"{key},{value},{description}\n")
                csvfile.write("\n")
            
            # Section 3: Octave Band Analysis
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
            
            # Section 4: Time Domain Analysis
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
            
            # Section 5: Time Domain Summary Statistics
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
            
            # Add histogram data if available
            if audio_data is not None:
                csvfile.write("\n[HISTOGRAM_DATA]\n")
                csvfile.write("frequency_hz,bin_center,bin_count,bin_density\n")
                
                center_freqs = analysis_results["center_frequencies"]
                statistics = analysis_results["statistics"]
                
                # PERFORMANCE FIX: Use cached band_data instead of recreating octave bank
                band_data = analysis_results.get("band_data", {})
                
                for freq in center_freqs:
                    freq_key = f"{freq:.3f}" if freq > 0 else "Full Spectrum"
                    if freq_key in statistics:
                        # Get the filtered signal for this frequency band from cached data
                        signal = band_data.get(freq_key, audio_data)
                        
                        # Remove DC component and very small values
                        clean_signal = signal[np.abs(signal) > 1e-10]
                        
                        if len(clean_signal) > 0:
                            # Linear histogram
                            hist, bin_edges = np.histogram(clean_signal, bins=51, range=(-1, 1), density=True)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            
                            for bin_center, count, density in zip(bin_centers, hist, hist):
                                csvfile.write(f"{freq},{bin_center},{count},{density}\n")
            
            # Note: Octave crest factor time data is not included in CSV export
            # as it would require recalculating octave analysis for every chunk,
            # which defeats our efficiency optimization. This data is available
            # in the octave_crest_factor_time.png plot.
            
            # Add min/max chunk octave analysis if available
            if chunk_octave_analysis is not None:
                csvfile.write("\n[EXTREME_CHUNKS_OCTAVE_ANALYSIS]\n")
                csvfile.write("chunk_type,time_seconds,crest_factor_db,frequency_hz,max_amplitude_db,rms_db,chunk_crest_factor_db\n")
                
                min_chunk_data = chunk_octave_analysis.get("min_chunk")
                max_chunk_data = chunk_octave_analysis.get("max_chunk")
                
                if min_chunk_data is not None:
                    min_analysis = min_chunk_data["analysis"]
                    min_time = min_chunk_data["time"]
                    min_crest_db = min_chunk_data["crest_factor_db"]
                    
                    center_freqs = analysis_results["center_frequencies"]
                    for freq in center_freqs:
                        freq_key = f"{freq:.3f}"
                        if freq_key in min_analysis["statistics"]:
                            stats = min_analysis["statistics"][freq_key]
                            csvfile.write(f"min_crest,{min_time},{min_crest_db},{freq},"
                                        f"{stats['max_amplitude_db']},{stats['rms_db']},{stats['crest_factor_db']}\n")
                
                if max_chunk_data is not None:
                    max_analysis = max_chunk_data["analysis"]
                    max_time = max_chunk_data["time"]
                    max_crest_db = max_chunk_data["crest_factor_db"]
                    
                    center_freqs = analysis_results["center_frequencies"]
                    for freq in center_freqs:
                        freq_key = f"{freq:.3f}"
                        if freq_key in max_analysis["statistics"]:
                            stats = max_analysis["statistics"][freq_key]
                            csvfile.write(f"max_crest,{max_time},{max_crest_db},{freq},"
                                        f"{stats['max_amplitude_db']},{stats['rms_db']},{stats['crest_factor_db']}\n")
            
            # Add envelope statistics if available
            if envelope_statistics is not None:
                csvfile.write("\n[ENVELOPE_STATISTICS]\n")
                csvfile.write("frequency_hz,analysis_type,rank,peak_value_db,peak_time_seconds,attack_time_ms,peak_hold_time_ms,decay_3db_ms,decay_6db_ms,decay_9db_ms,decay_12db_ms,decay_12db_reached\n")
                
                # Export worst-case envelopes
                for freq_label, band_data in envelope_statistics.items():
                    worst_case = band_data.get("worst_case_envelopes", [])
                    for envelope in worst_case:
                        decay = envelope.get("decay_times", {})
                        csvfile.write(
                            f"{freq_label},worst_case,{envelope['rank']},"
                            f"{envelope['peak_value_db']},{envelope['peak_time_seconds']},"
                            f"{envelope['attack_time_ms']},{envelope['peak_hold_time_ms']},"
                            f"{decay.get('decay_3db_ms', '')},{decay.get('decay_6db_ms', '')},"
                            f"{decay.get('decay_9db_ms', '')},{decay.get('decay_12db_ms', '')},"
                            f"{decay.get('decay_12db_reached', False)}\n"
                        )
                
                # Export pattern analysis summary
                csvfile.write("\n[ENVELOPE_PATTERN_ANALYSIS]\n")
                csvfile.write("frequency_hz,pattern_num,num_repetitions,mean_interval_seconds,std_interval_seconds,median_interval_seconds,min_interval_seconds,max_interval_seconds,pattern_regularity_score,pattern_confidence,beats_per_minute\n")
                
                for freq_label, band_data in envelope_statistics.items():
                    pattern_analysis = band_data.get("pattern_analysis", {})
                    patterns_detected = pattern_analysis.get("patterns_detected", 0)
                    
                    for pattern_num in range(1, patterns_detected + 1):
                        pattern_key = f"pattern_{pattern_num}"
                        if pattern_key in pattern_analysis:
                            pattern = pattern_analysis[pattern_key]
                            csvfile.write(
                                f"{freq_label},{pattern_num},{pattern['num_repetitions']},"
                                f"{pattern['mean_interval_seconds']},{pattern['std_interval_seconds']},"
                                f"{pattern['median_interval_seconds']},{pattern['min_interval_seconds']},"
                                f"{pattern['max_interval_seconds']},{pattern['pattern_regularity_score']},"
                                f"{pattern['pattern_confidence']},{pattern['beats_per_minute']}\n"
                            )
        
        logger.info("Comprehensive analysis results exported successfully")

    def create_crest_factor_plot(self, analysis_results: Dict, 
                               output_path: Optional[str] = None,
                               time_analysis: Optional[Dict] = None,
                               chunk_octave_analysis: Optional[Dict] = None) -> None:
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
        ax.set_xlim([15, 20000])
        ax.set_ylim([0, max(30, np.max(crest_factor_db_values) * 1.1)])
        ax.grid(True, alpha=0.3)
        
        # Add frequency labels
        ax.set_xticks([16, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_xticklabels(['16', '31.25', '62.5', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
        
        # Title and legend
        ax.set_title('Octave Band Analysis - Crest Factor (Peak/RMS Ratio)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Crest factor plot saved to: {output_path}")
        
        plt.close(fig)
