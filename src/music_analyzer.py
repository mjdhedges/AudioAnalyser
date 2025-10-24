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

    def __init__(self, sample_rate: int = 44100) -> None:
        """Initialize the music analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate

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
        results = {
            "band_data": {},
            "statistics": {},
            "center_frequencies": center_frequencies
        }

        # Analyze each band
        for i in range(num_bands):
            band_signal = octave_bank[:, i]
            freq_label = f"{center_frequencies[i]:.3f}" if i > 0 else "Full Spectrum"
            
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
        max_db = 20 * np.log10(max_val) if max_val > 0 else -np.inf
        
        rms_val = np.sqrt(np.mean(signal**2))
        rms_db = 20 * np.log10(rms_val) if rms_val > 0 else -np.inf
        
        # Dynamic range
        dynamic_range = rms_val / max_val if max_val > 0 else 0
        dynamic_range_db = 20 * np.log10(dynamic_range) if dynamic_range > 0 else -np.inf
        
        # Additional statistics
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        # Percentiles for distribution analysis
        percentiles = np.percentile(np.abs(signal), [10, 25, 50, 75, 90, 95, 99])
        
        stats = {
            "max_amplitude": max_val,
            "max_amplitude_db": max_db,
            "rms": rms_val,
            "rms_db": rms_db,
            "dynamic_range": dynamic_range,
            "dynamic_range_db": dynamic_range_db,
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
                                  output_path: Optional[str] = None) -> None:
        """Create octave spectrum plot similar to MATLAB's semilogx plot.
        
        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
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
                max_db_values.append(statistics[freq_label]["max_amplitude_db"])
                rms_db_values.append(statistics[freq_label]["rms_db"])
            else:
                max_db_values.append(-np.inf)
                rms_db_values.append(-np.inf)

        # Plot the data
        ax.semilogx(center_freqs, max_db_values, 'b-o', label='Max Peak (dB)', linewidth=2)
        ax.semilogx(center_freqs, rms_db_values, 'r-s', label='RMS (dB)', linewidth=2)
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (dB)')
        ax.set_title('Octave Band Analysis - Peak and RMS Levels')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([20, 20000])
        ax.set_ylim([-40, 0])
        
        # Add frequency labels
        ax.set_xticks([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ax.set_xticklabels(['31.25', '62.5', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_path}")
        
        plt.show()

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
            
            # Create histogram
            ax.hist(signal, bins=50, alpha=0.7, density=True)
            ax.set_title(f'{freq_label} Hz')
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
        
        plt.show()

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
