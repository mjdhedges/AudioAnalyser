"""
Data export module for Music Analyser.

This module handles CSV export functionality for analysis results.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataExporter:
    """Handles data export operations."""

    def __init__(self, sample_rate: int = 44100, original_peak: float = 1.0) -> None:
        """Initialize the data exporter.
        
        Args:
            sample_rate: Sample rate for audio processing
            original_peak: Original peak level before normalization (for dBFS calculation)
        """
        self.sample_rate = sample_rate
        self.original_peak = original_peak

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

    def calculate_advanced_statistics(self, audio_data: np.ndarray, 
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
        
        # 1. PEAK EVENT STATISTICS (count discrete peaks, not samples)
        # Use envelope to detect discrete peak events
        from scipy import signal as sp_signal
        
        # Calculate a simple peak envelope for peak detection
        rectified = np.abs(audio_data)
        envelope = np.zeros_like(rectified)
        envelope[0] = rectified[0]
        # Simple peak follower with 10ms release time
        release_samples = int(0.01 * self.sample_rate)  # 10ms
        alpha = 1.0 / release_samples
        for i in range(1, len(rectified)):
            envelope[i] = max(rectified[i], envelope[i-1] * (1 - alpha))
        
        envelope_dbfs = 20 * np.log10(envelope * self.original_peak + 1e-10)
        envelope_dbfs_clean = np.copy(envelope_dbfs)
        envelope_dbfs_clean[envelope_dbfs_clean == -np.inf] = -120.0
        
        # Detect peaks with minimum distance (avoid counting same peak multiple times)
        min_distance_samples = int(0.05 * self.sample_rate)  # 50ms minimum between peaks
        peak_indices, peak_properties = sp_signal.find_peaks(
            envelope_dbfs_clean,
            height=-40.0,  # Minimum peak height
            distance=min_distance_samples
        )
        peak_values_db = envelope_dbfs_clean[peak_indices]
        
        # Count peaks at different thresholds
        peaks_above_minus3db = np.sum(peak_values_db > -3.0)
        peaks_above_minus1db = np.sum(peak_values_db > -1.0)
        peaks_above_minus0_1db = np.sum(peak_values_db > -0.1)
        
        # Legacy sample-based metrics (keep for backward compatibility)
        clip_events_samples = np.sum(audio_dbfs > -0.1)  # Actual clipping events (samples)
        peak_saturation = np.sum(audio_dbfs > -3.0)  # Heavily compressed regions (samples)
        
        stats.update({
            "peaks_above_minus3db_count": int(peaks_above_minus3db),
            "peaks_above_minus3db_per_sec": peaks_above_minus3db / duration,
            "peaks_above_minus1db_count": int(peaks_above_minus1db),
            "peaks_above_minus1db_per_sec": peaks_above_minus1db / duration,
            "peaks_above_minus0_1db_count": int(peaks_above_minus0_1db),
            "peaks_above_minus0_1db_per_sec": peaks_above_minus0_1db / duration,
            # Legacy metrics (kept for compatibility, but deprecated)
            "hot_peaks_rate_per_sec": peaks_above_minus1db / duration,  # Use peak count instead
            "clip_events_rate_per_sec": clip_events_samples / duration,
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
            chunk_octave_analysis: Optional pre-computed octave analysis for extreme chunks
            audio_data: Optional raw audio data for advanced statistics
            envelope_statistics: Optional envelope statistics for export
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
                
                advanced_stats = self.calculate_advanced_statistics(audio_data, analysis_results, time_analysis)
                
                # Define descriptions for each statistic
                stat_descriptions = {
                    "peaks_above_minus3db_count": "Number of discrete peaks above -3dBFS",
                    "peaks_above_minus3db_per_sec": "Peaks above -3dBFS per second",
                    "peaks_above_minus1db_count": "Number of discrete peaks above -1dBFS",
                    "peaks_above_minus1db_per_sec": "Peaks above -1dBFS per second",
                    "peaks_above_minus0_1db_count": "Number of discrete peaks above -0.1dBFS",
                    "peaks_above_minus0_1db_per_sec": "Peaks above -0.1dBFS per second",
                    "hot_peaks_rate_per_sec": "Near-clipping events (>-1dBFS) per second (deprecated - use peaks_above_minus1db_per_sec)",
                    "clip_events_rate_per_sec": "Actual clipping events (>-0.1dBFS) per second (sample-based, deprecated)", 
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
                "time_domain_mode": time_analysis.get("time_domain_mode", "unknown"),
                "time_domain_time_step_seconds": time_analysis.get("time_step_seconds", np.nan),
                "time_domain_rms_method": time_analysis.get("time_domain_rms_method", "unknown"),
                "time_domain_peak_method": time_analysis.get("time_domain_peak_method", "unknown"),
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
                
                # Export sustained peaks summary
                csvfile.write("\n[SUSTAINED_PEAKS_SUMMARY]\n")
                # Header: frequency_hz,n_peaks,hold_ms_mean,hold_ms_median,hold_ms_p90,hold_ms_p95,hold_ms_max,t3_ms_mean,...
                headers = [
                    "frequency_hz",
                    "n_peaks",
                    "hold_ms_mean","hold_ms_median","hold_ms_p90","hold_ms_p95","hold_ms_max",
                    "t3_ms_mean","t3_ms_median","t3_ms_p90","t3_ms_p95","t3_ms_max",
                    "t6_ms_mean","t6_ms_median","t6_ms_p90","t6_ms_p95","t6_ms_max",
                    "t9_ms_mean","t9_ms_median","t9_ms_p90","t9_ms_p95","t9_ms_max",
                    "t12_ms_mean","t12_ms_median","t12_ms_p90","t12_ms_p95","t12_ms_max",
                ]
                csvfile.write(",".join(headers) + "\n")

                for freq_label, band_data in envelope_statistics.items():
                    sust = band_data.get("sustained_peaks_summary", {})
                    def get_stats(d, k):
                        s = d.get(k, {})
                        return [s.get("mean", 0.0), s.get("median", 0.0), s.get("p90", 0.0), s.get("p95", 0.0), s.get("max", 0.0)]
                    row = [freq_label, sust.get("n_peaks", 0)]
                    row += get_stats(sust, "hold_ms")
                    row += get_stats(sust, "t3_ms")
                    row += get_stats(sust, "t6_ms")
                    row += get_stats(sust, "t9_ms")
                    row += get_stats(sust, "t12_ms")
                    csvfile.write(",".join(map(str, row)) + "\n")

                # Optionally export sustained peaks events if present
                has_events = any("sustained_peaks_events" in band for band in envelope_statistics.values())
                if has_events:
                    csvfile.write("\n[SUSTAINED_PEAKS_EVENTS]\n")
                    csvfile.write("frequency_hz,peak_time_seconds,peak_value_db,hold_ms,t3_ms,t6_ms,t9_ms,t12_ms\n")
                    for freq_label, band_data in envelope_statistics.items():
                        events = band_data.get("sustained_peaks_events", [])
                        for e in events:
                            csvfile.write(
                                f"{freq_label},{e.get('peak_time_seconds', 0.0)},{e.get('peak_value_db', 0.0)},"
                                f"{e.get('hold_ms', 0.0)},{e.get('t3_ms', 0.0)},{e.get('t6_ms', 0.0)},"
                                f"{e.get('t9_ms', 0.0)},{e.get('t12_ms', 0.0)}\n"
                            )
        
        logger.info("Comprehensive analysis results exported successfully")

