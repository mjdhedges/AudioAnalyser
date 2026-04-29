"""
Analysis and visualization module for Audio Analyser.

This module handles octave band analysis, statistical calculations, and visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

# Use Agg backend for non-interactive plotting (faster, no GUI)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from src.data_export import DataExporter
from src.envelope_analyzer import EnvelopeAnalyzer
from src.visualization import PlotGenerator
from src.time_domain_metrics import (
    FixedChunkTimeDomainCalculator,
    FixedWindowTimeDomainCalculator,
    SlowTimeDomainCalculator,
    TimeDomainCrestFactorResult,
    compute_whole_interval_crest_factor,
)

logger = logging.getLogger(__name__)


class MusicAnalyzer:
    """Main class for audio analysis operations."""

    def __init__(
        self,
        sample_rate: int = 44100,
        original_peak: float = 1.0,
        dpi: int = 300,
        peak_hold_tau: float = 1.0,
        time_domain_crest_factor_mode: str = "fixed_window",
        analysis_config: Optional[Dict] = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            sample_rate: Sample rate for audio processing
            original_peak: Original peak level before normalization (for dBFS calculation)
            dpi: DPI for plot output (lower for faster batch processing)
            peak_hold_tau: Time constant (seconds) for the crest-factor peak-hold envelope
        """
        self.sample_rate = sample_rate
        self.original_peak = original_peak
        self.dpi = dpi
        self.peak_hold_tau = peak_hold_tau
        self.time_domain_mode = time_domain_crest_factor_mode
        self._analysis_config: Dict[str, object] = dict(analysis_config or {})

        # Use composition for better separation of concerns
        self.plot_generator = PlotGenerator(
            sample_rate=sample_rate,
            original_peak=original_peak,
            dpi=dpi,
            peak_hold_tau=peak_hold_tau,
        )
        self.envelope_analyzer = EnvelopeAnalyzer(
            sample_rate=sample_rate, original_peak=original_peak
        )
        self.data_exporter = DataExporter(
            sample_rate=sample_rate, original_peak=original_peak
        )

    def analyze_octave_bands(
        self, octave_bank: np.ndarray, center_frequencies: List[float]
    ) -> Dict:
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
            "center_frequencies": extended_frequencies,
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
        interval_crest = compute_whole_interval_crest_factor(
            signal,
            original_peak=self.original_peak,
        )
        max_val = interval_crest.peak_level
        rms_val = interval_crest.rms_level
        max_dbfs = interval_crest.peak_level_dbfs
        rms_dbfs = interval_crest.rms_level_dbfs
        dynamic_range = rms_val / max_val if max_val > 0 else np.nan
        dynamic_range_db = 20 * np.log10(dynamic_range) if dynamic_range > 0 else np.nan

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
            "crest_factor": interval_crest.crest_factor,
            "crest_factor_db": interval_crest.crest_factor_db,
            "is_valid_crest_factor": interval_crest.is_valid_crest_factor,
            "crest_factor_method": interval_crest.crest_factor_method,
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
            },
        }

        return stats

    def create_octave_spectrum_plot(
        self,
        analysis_results: Dict,
        output_path: Optional[str] = None,
        time_analysis: Optional[Dict] = None,
        chunk_octave_analysis: Optional[Dict] = None,
    ) -> None:
        """Create octave spectrum plot similar to MATLAB's semilogx plot.

        Delegates to PlotGenerator for actual plotting.

        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
            time_analysis: Optional time-domain analysis results for chunk comparison
            chunk_octave_analysis: Optional pre-computed octave analysis for extreme chunks
        """
        self.plot_generator.create_octave_spectrum_plot(
            analysis_results, output_path, time_analysis, chunk_octave_analysis
        )

    def _add_extreme_chunk_analysis(
        self,
        ax,
        time_analysis: Dict,
        chunk_octave_analysis: Dict,
        center_freqs: List[float],
    ) -> None:
        """Add analysis of extreme crest factor chunks to the octave spectrum plot.

        Delegates to PlotGenerator for actual plotting.

        Args:
            ax: Matplotlib axis to plot on
            time_analysis: Time-domain analysis results
            chunk_octave_analysis: Pre-computed octave analysis for extreme chunks
            center_freqs: Center frequencies for octave bands
        """
        self.plot_generator._add_extreme_chunk_analysis(
            ax, time_analysis, chunk_octave_analysis, center_freqs
        )

    def create_histogram_plots(
        self,
        analysis_results: Dict,
        output_dir: Optional[str] = None,
        octave_bank: Optional[np.ndarray] = None,
    ) -> None:
        """Create histogram plots for each octave band.

        Delegates to PlotGenerator for actual plotting.

        Args:
            analysis_results: Results from octave band analysis
            output_dir: Optional directory to save plots
            octave_bank: Octave bank array to slice from (required if band_data not in results)
        """
        self.plot_generator.create_histogram_plots(
            analysis_results, output_dir, octave_bank
        )

    def create_histogram_plots_log_db(
        self,
        analysis_results: Dict,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
        octave_bank: Optional[np.ndarray] = None,
    ) -> None:
        """Create histogram plots for each octave band with log dB X-axis.

        Delegates to PlotGenerator for actual plotting.

        Args:
            analysis_results: Results from octave band analysis
            output_dir: Optional directory to save plots
            config: Optional configuration dictionary
            octave_bank: Octave bank array to slice from (required if band_data not in results)
        """
        self.plot_generator.create_histogram_plots_log_db(
            analysis_results, output_dir, config, octave_bank
        )

    def analyze_comprehensive(
        self,
        audio_data: np.ndarray,
        octave_bank: np.ndarray,
        center_frequencies: List[float],
        chunk_duration: float = 2.0,
    ) -> Dict:
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
            audio_data, octave_bank, center_frequencies, chunk_duration
        )

        return {
            "main_analysis": main_analysis,
            "time_analysis": time_analysis,
            "chunk_octave_analysis": chunk_octave_analysis,
        }

    def _analyze_time_domain_chunks(
        self,
        audio_data: np.ndarray,
        chunk_duration: float = 2.0,
    ) -> Dict:
        """Analyze crest factor over time using selected method.

        Args:
            audio_data: Input audio signal (normalized)
            chunk_duration: Duration of each analysis chunk in seconds

        Returns:
            Dictionary with time-domain crest factor analysis
        """
        mode = str(getattr(self, "time_domain_mode", "slow"))
        if mode not in {"fixed_window", "slow", "fixed_chunk"}:
            mode = "fixed_window"

        if mode == "fixed_window":
            calc = FixedWindowTimeDomainCalculator()
            result = calc.compute(
                audio_data,
                sample_rate=self.sample_rate,
                original_peak=self.original_peak,
                config=self._analysis_config,
            )
        elif mode == "fixed_chunk":
            calc = FixedChunkTimeDomainCalculator(window_seconds=float(chunk_duration))
            result = calc.compute(
                audio_data,
                sample_rate=self.sample_rate,
                original_peak=self.original_peak,
                config=self._analysis_config,
            )
        else:
            calc = SlowTimeDomainCalculator(peak_hold_tau_seconds=self.peak_hold_tau)
            result = calc.compute(
                audio_data,
                sample_rate=self.sample_rate,
                original_peak=self.original_peak,
                config=self._analysis_config,
            )

        output = self._time_domain_result_to_dict(result)
        if result.mode == "fixed_window":
            output["crest_factor_rms_floor_dbfs"] = float(
                self._analysis_config.get("crest_factor_rms_floor_dbfs", -80.0) or -80.0
            )
        return output

    @staticmethod
    def _time_domain_result_to_dict(result: TimeDomainCrestFactorResult) -> Dict:
        """Convert structured time-domain results to legacy dict format."""
        crest_factors_db = np.asarray(result.crest_factors_db)
        return {
            "time_points": np.asarray(result.time_points),
            "crest_factors": np.asarray(result.crest_factors),
            "crest_factors_db": crest_factors_db,
            "peak_levels": np.asarray(result.peak_levels),
            "rms_levels": np.asarray(result.rms_levels),
            "peak_levels_dbfs": np.asarray(result.peak_levels_dbfs),
            "rms_levels_dbfs": np.asarray(result.rms_levels_dbfs),
            "is_valid_crest_factor": np.isfinite(crest_factors_db),
            "chunk_duration": float(result.chunk_duration),
            "time_step_seconds": float(result.time_step_seconds),
            "num_chunks": int(result.num_chunks),
            "total_duration": float(result.total_duration),
            "time_domain_mode": result.mode,
            "time_domain_rms_method": result.rms_method,
            "time_domain_peak_method": result.peak_method,
            "crest_factor_method": (
                "fixed_window_peak_rms"
                if result.mode == "fixed_window"
                else f"{result.mode}_peak_rms"
            ),
        }

    def analyze_crest_factor_over_time(
        self, audio_data: np.ndarray, chunk_duration: float = 2.0
    ) -> Dict:
        """Analyze crest factor over time using sliding windows.

        This is a convenience method that calls the internal _analyze_time_domain_chunks.
        """
        return self._analyze_time_domain_chunks(audio_data, chunk_duration)

    def _analyze_extreme_chunks_octave_bands(
        self,
        audio_data: np.ndarray,
        octave_bank: np.ndarray,
        center_frequencies: List[float],
        chunk_duration: float,
    ) -> Dict:
        """Analyze octave bands for min/max crest factor chunks.

        Args:
            octave_bank: Pre-computed octave bank (full track) - CRITICAL for performance
            time_analysis: Time-domain analysis results
            center_frequencies: List of center frequencies
            chunk_duration: Duration of each chunk in seconds

        Returns:
            Dictionary with octave analysis for extreme chunks
        """
        # Extreme chunks are always selected from fixed windows (chunk_duration),
        # regardless of the time-domain crest factor mode used for plotting/export.
        fixed = FixedChunkTimeDomainCalculator(
            window_seconds=float(chunk_duration)
        ).compute(
            audio_data,
            sample_rate=self.sample_rate,
            original_peak=self.original_peak,
            config=self._analysis_config,
        )
        crest_factors_db = fixed.crest_factors_db
        time_points = fixed.time_points
        chunk_samples = max(int(float(chunk_duration) * self.sample_rate), 1)

        # Find valid (finite) crest factor values
        valid_mask = np.isfinite(crest_factors_db)
        if not np.any(valid_mask):
            return {"min_chunk": None, "max_chunk": None}

        valid_crest_db = np.array(crest_factors_db)[valid_mask]
        valid_time_points = np.array(time_points)[valid_mask]

        # Find min and max crest factor chunks
        min_idx = np.argmin(valid_crest_db)
        max_idx = np.argmax(valid_crest_db)

        # Chunk indices for fixed windows are the same as array indices
        min_chunk_idx = int(np.where(valid_mask)[0][min_idx])
        max_chunk_idx = int(np.where(valid_mask)[0][max_idx])

        results = {}

        # Analyze minimum crest factor chunk using pre-computed octave bank
        min_start = min_chunk_idx * chunk_samples
        min_end = min_start + chunk_samples
        if min_end > octave_bank.shape[0]:
            min_end = octave_bank.shape[0]

        if min_start < min_end:
            # Slice pre-computed octave bank
            min_chunk_octave = octave_bank[min_start:min_end, :]
            min_analysis = self.analyze_octave_bands(
                min_chunk_octave, center_frequencies
            )
            results["min_chunk"] = {
                "analysis": min_analysis,
                "time": valid_time_points[min_idx],
                "crest_factor_db": valid_crest_db[min_idx],
                "chunk_idx": min_chunk_idx,
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
            max_analysis = self.analyze_octave_bands(
                max_chunk_octave, center_frequencies
            )
            results["max_chunk"] = {
                "analysis": max_analysis,
                "time": valid_time_points[max_idx],
                "crest_factor_db": valid_crest_db[max_idx],
                "chunk_idx": max_chunk_idx,
            }
        else:
            results["max_chunk"] = None

        return results

    def create_crest_factor_time_plot(
        self, time_analysis: Dict, output_path: Optional[str] = None
    ) -> None:
        """Create crest factor vs time plot.

        Delegates to PlotGenerator for actual plotting.

        Args:
            time_analysis: Results from analyze_crest_factor_over_time
            output_path: Optional path to save the plot
        """
        self.plot_generator.create_crest_factor_time_plot(time_analysis, output_path)

    def create_octave_crest_factor_time_plot(
        self,
        octave_bank: np.ndarray,
        time_analysis: Dict,
        center_frequencies: List[float],
        output_path: Optional[str] = None,
    ) -> None:
        """Create plot showing crest factor over time for all octave bands.

        Delegates to PlotGenerator for actual plotting.

        Args:
            octave_bank: Pre-computed octave bank (full track) - CRITICAL for performance
            time_analysis: Time-domain analysis results with chunk data
            center_frequencies: List of center frequencies for octave bands
            output_path: Optional path to save the plot
        """
        self.plot_generator.create_octave_crest_factor_time_plot(
            octave_bank, time_analysis, center_frequencies, output_path
        )

    def _calculate_peak_envelope(
        self,
        signal: np.ndarray,
        center_freq: float = 0.0,
        wavelength_multiplier: float = 1.0,
        fallback_window_ms: float = 10.0,
    ) -> np.ndarray:
        """Calculate peak envelope using standard peak follower with wavelength-based timing.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            signal: Audio signal array (normalized)
            center_freq: Center frequency of the octave band (Hz). Use 0 for Full Spectrum.
            wavelength_multiplier: Multiplier for attack/release time (default 1.0 = 1x wavelength)
            fallback_window_ms: Fallback window size in ms for Full Spectrum (center_freq=0)

        Returns:
            Peak envelope array in linear scale (same length as input, maintains peak levels)
        """
        return self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq, wavelength_multiplier, fallback_window_ms
        )

    def _calculate_rms_envelope(
        self,
        signal: np.ndarray,
        center_freq: float = 0.0,
        method: str = "peak_envelope",
        wavelength_multiplier: float = 1.0,
        fallback_window_ms: float = 10.0,
    ) -> np.ndarray:
        """Calculate envelope using peak envelope follower (primary method).

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            signal: Audio signal array (normalized)
            center_freq: Center frequency of the octave band (Hz). Use 0 for Full Spectrum.
            method: Envelope detection method ('peak_envelope' or 'rms_window' for legacy)
            wavelength_multiplier: Multiplier for attack/release time (default 1.0 = 1x wavelength)
            fallback_window_ms: Fallback window size in ms for Full Spectrum (center_freq=0)

        Returns:
            Envelope array in linear scale (same length as input)
        """
        return self.envelope_analyzer.calculate_rms_envelope(
            signal, center_freq, method, wavelength_multiplier, fallback_window_ms
        )

    def _find_attack_time(
        self,
        envelope_db: np.ndarray,
        peak_idx: int,
        peak_value_db: float,
        attack_threshold_db: float,
    ) -> float:
        """Find attack time using vectorized backward search.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            envelope_db: Peak envelope in dBFS
            peak_idx: Index of peak sample
            peak_value_db: Peak value in dBFS
            attack_threshold_db: Threshold below peak to detect attack start

        Returns:
            Attack time in milliseconds
        """
        return self.envelope_analyzer.find_attack_time(
            envelope_db, peak_idx, peak_value_db, attack_threshold_db
        )

    def _find_peak_hold_time(
        self,
        envelope_db: np.ndarray,
        peak_idx: int,
        peak_value_db: float,
        hold_threshold_db: float,
    ) -> float:
        """Find peak hold time using vectorized forward/backward search.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            envelope_db: Peak envelope in dBFS
            peak_idx: Index of peak sample
            peak_value_db: Peak value in dBFS
            hold_threshold_db: Threshold for peak hold (within this of peak)

        Returns:
            Peak hold time in milliseconds
        """
        return self.envelope_analyzer.find_peak_hold_time(
            envelope_db, peak_idx, peak_value_db, hold_threshold_db
        )

    def _find_decay_times(
        self,
        envelope_db: np.ndarray,
        peak_idx: int,
        peak_value_db: float,
        decay_thresholds_db: List[float],
    ) -> Dict[str, float]:
        """Find decay times to multiple thresholds in single pass.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            envelope_db: Peak envelope in dBFS
            peak_idx: Index of peak sample
            peak_value_db: Peak value in dBFS
            decay_thresholds_db: List of decay thresholds in dB (negative values)

        Returns:
            Dictionary with decay times in milliseconds and reach flags
        """
        return self.envelope_analyzer.find_decay_times(
            envelope_db, peak_idx, peak_value_db, decay_thresholds_db
        )

    def _analyze_worst_case_envelopes(
        self,
        envelope_db: np.ndarray,
        peak_indices: np.ndarray,
        peak_values_db: np.ndarray,
        num_envelopes: int,
        sort_by: str,
        attack_threshold_db: float,
        peak_hold_threshold_db: float,
        decay_thresholds_db: List[float],
        exclude_peak_indices: Optional[set] = None,
        window_ms: Optional[float] = None,
    ) -> List[Dict]:
        """Analyze worst-case envelopes efficiently using partition-based selection.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            envelope_db: Peak envelope in dBFS
            peak_indices: Array of peak sample indices
            peak_values_db: Array of peak values in dBFS
            num_envelopes: Number of worst-case envelopes to find
            sort_by: Sorting criterion ("peak_value", "decay_time", "energy")
            attack_threshold_db: Attack detection threshold
            peak_hold_threshold_db: Peak hold threshold
            decay_thresholds_db: Decay thresholds
            exclude_peak_indices: Optional set of peak indices to exclude (e.g., pattern peaks)
            window_ms: Optional window size in ms for extracting envelope windows

        Returns:
            List of worst-case envelope dictionaries with characteristics
        """
        return self.envelope_analyzer.analyze_worst_case_envelopes(
            envelope_db,
            peak_indices,
            peak_values_db,
            num_envelopes,
            sort_by,
            attack_threshold_db,
            peak_hold_threshold_db,
            decay_thresholds_db,
            exclude_peak_indices,
            window_ms,
        )

    def _compare_envelope_shapes(
        self, pattern1: np.ndarray, pattern2: np.ndarray
    ) -> float:
        """Compare two envelope patterns using correlation.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            pattern1: First envelope pattern
            pattern2: Second envelope pattern

        Returns:
            Correlation coefficient (0-1, higher = more similar)
        """
        return self.envelope_analyzer.compare_envelope_shapes(pattern1, pattern2)

    def _analyze_repeating_patterns(
        self,
        envelope_db: np.ndarray,
        peak_indices: np.ndarray,
        peak_values_db: np.ndarray,
        min_repetitions: int,
        max_patterns: int,
        similarity_threshold: float,
        window_ms: float,
    ) -> Dict:
        """Analyze repeating patterns using optimized correlation matching.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            envelope_db: Peak envelope in dBFS
            peak_indices: Array of peak sample indices
            peak_values_db: Array of peak values in dBFS
            min_repetitions: Minimum number of repetitions to consider a pattern
            max_patterns: Maximum number of patterns to detect
            similarity_threshold: Correlation threshold for pattern matching
            window_ms: Window size for pattern extraction

        Returns:
            Dictionary with pattern analysis results
        """
        return self.envelope_analyzer.analyze_repeating_patterns(
            envelope_db,
            peak_indices,
            peak_values_db,
            min_repetitions,
            max_patterns,
            similarity_threshold,
            window_ms,
        )

    def analyze_envelope_statistics(
        self,
        octave_bank: np.ndarray,
        center_frequencies: List[float],
        config: Optional[Dict] = None,
    ) -> Dict:
        """Analyze envelope characteristics and repeating patterns for each octave band.

        Delegates to EnvelopeAnalyzer for actual calculation.

        Args:
            octave_bank: Pre-computed octave bank (full track) - shape (samples, bands)
            center_frequencies: List of center frequencies for octave bands
            config: Optional configuration dictionary (defaults from config.toml if None)

        Returns:
            Dictionary with envelope statistics per band
        """
        return self.envelope_analyzer.analyze_envelope_statistics(
            octave_bank, center_frequencies, config
        )

    def create_pattern_envelope_plots(
        self,
        envelope_statistics: Dict,
        center_frequencies: List[float],
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """Create plots showing top N envelopes from repeating patterns for each band.

        Delegates to PlotGenerator for actual plotting.

        Args:
            envelope_statistics: Results from analyze_envelope_statistics
            center_frequencies: List of center frequencies
            output_dir: Optional directory to save plots
            config: Optional configuration dictionary
        """
        self.plot_generator.create_pattern_envelope_plots(
            envelope_statistics, center_frequencies, output_dir, config
        )

    def create_independent_envelope_plots(
        self,
        envelope_statistics: Dict,
        center_frequencies: List[float],
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """Create plots showing top N independent (non-repeating) envelopes for each band.

        Delegates to PlotGenerator for actual plotting.

        Args:
            envelope_statistics: Results from analyze_envelope_statistics
            center_frequencies: List of center frequencies
            output_dir: Optional directory to save plots
            config: Optional configuration dictionary
        """
        self.plot_generator.create_independent_envelope_plots(
            envelope_statistics, center_frequencies, output_dir, config
        )

    def export_analysis_results(self, analysis_results: Dict, output_path: str) -> None:
        """Export analysis results to CSV file.

        Delegates to DataExporter for actual export.

        Args:
            analysis_results: Results from octave band analysis
            output_path: Path to save CSV file
        """
        self.data_exporter.export_analysis_results(analysis_results, output_path)

    def _calculate_advanced_statistics(
        self, audio_data: np.ndarray, analysis_results: Dict, time_analysis: Dict
    ) -> Dict:
        """Calculate advanced statistics for track dynamics and quality assessment.

        Delegates to DataExporter for actual calculation.

        Args:
            audio_data: Raw audio data
            analysis_results: Octave band analysis results
            time_analysis: Time-domain analysis results

        Returns:
            Dictionary containing advanced statistics
        """
        return self.data_exporter.calculate_advanced_statistics(
            audio_data, analysis_results, time_analysis
        )

    def export_comprehensive_results(
        self,
        analysis_results: Dict,
        time_analysis: Dict,
        track_metadata: Dict,
        output_path: str,
        chunk_octave_analysis: Optional[Dict] = None,
        audio_data: Optional[np.ndarray] = None,
        envelope_statistics: Optional[Dict] = None,
    ) -> None:
        """Export comprehensive analysis results including all data to CSV file.

        Delegates to DataExporter for actual export.

        Args:
            analysis_results: Results from octave band analysis
            time_analysis: Results from time-domain crest factor analysis
            track_metadata: Track information and metadata
            output_path: Path to save CSV file
            chunk_octave_analysis: Optional pre-computed octave analysis for extreme chunks
            audio_data: Optional raw audio data for advanced statistics
            envelope_statistics: Optional envelope statistics for export
        """
        self.data_exporter.export_comprehensive_results(
            analysis_results,
            time_analysis,
            track_metadata,
            output_path,
            chunk_octave_analysis,
            audio_data,
            envelope_statistics,
        )

    def create_crest_factor_plot(
        self,
        analysis_results: Dict,
        output_path: Optional[str] = None,
        time_analysis: Optional[Dict] = None,
        chunk_octave_analysis: Optional[Dict] = None,
    ) -> None:
        """Create crest factor plot showing peak-to-RMS ratio for each octave band.

        Delegates to PlotGenerator for actual plotting.

        Args:
            analysis_results: Results from octave band analysis
            output_path: Optional path to save the plot
            time_analysis: Optional time-domain analysis results for chunk markers
            chunk_octave_analysis: Optional pre-computed octave analysis for extreme chunks
        """
        self.plot_generator.create_crest_factor_plot(
            analysis_results, output_path, time_analysis, chunk_octave_analysis
        )
