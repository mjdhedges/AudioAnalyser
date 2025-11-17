"""
Envelope analysis module for Music Analyser.

This module handles envelope processing, peak detection, and pattern analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np
from scipy import signal as sp_signal

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _calculate_peak_envelope_core(
    rectified: np.ndarray, decay_multiplier: float, release_coeff: float
) -> np.ndarray:
    """Core peak envelope calculation with Numba JIT compilation.
    
    This function implements the peak follower algorithm with exponential decay.
    JIT compilation provides significant speedup (10-50x) over pure Python loops.
    
    Args:
        rectified: Rectified (absolute value) audio signal
        decay_multiplier: Multiplier for exponential decay (1 - release_coeff)
        release_coeff: Release coefficient for exponential smoothing
        
    Returns:
        Peak envelope array
    """
    envelope = np.zeros_like(rectified)
    envelope[0] = rectified[0]
    
    for i in range(1, len(rectified)):
        # Compute exponential decay: envelope[i-1] * decay_multiplier + rectified[i] * release_coeff
        decayed = envelope[i - 1] * decay_multiplier + rectified[i] * release_coeff
        # Take maximum to ensure instantaneous rise when signal exceeds decayed value
        envelope[i] = max(rectified[i], decayed)
    
    return envelope


class EnvelopeAnalyzer:
    """Handles envelope analysis operations."""

    def __init__(self, sample_rate: int = 44100, original_peak: float = 1.0) -> None:
        """Initialize the envelope analyzer.
        
        Args:
            sample_rate: Sample rate for audio processing
            original_peak: Original peak level before normalization (for dBFS calculation)
        """
        self.sample_rate = sample_rate
        self.original_peak = original_peak

    def calculate_peak_envelope(self, signal: np.ndarray,
                                 center_freq: float = 0.0,
                                 wavelength_multiplier: float = 1.0,
                                 fallback_window_ms: float = 10.0) -> np.ndarray:
        """Calculate peak envelope using standard peak follower with wavelength-based timing.
        
        This method implements a peak envelope follower that creates a line linking
        the peak of every wave. It uses exponential attack and release times based on
        the wavelength (period) of the center frequency.
        
        Attack time is 0ms (instantaneous rise to catch peaks immediately), while release time
        follows the full wavelength for smooth decay. Release time is scaled by wavelength_multiplier.
        
        Args:
            signal: Audio signal array (normalized)
            center_freq: Center frequency of the octave band (Hz). Use 0 for Full Spectrum.
            wavelength_multiplier: Multiplier for attack/release time (default 1.0 = 1x wavelength)
            fallback_window_ms: Fallback window size in ms for Full Spectrum (center_freq=0)
            
        Returns:
            Peak envelope array in linear scale (same length as input, maintains peak levels)
        """
        # Rectify signal (take absolute value)
        rectified = np.abs(signal)
        
        # Calculate period and attack/release time in samples
        if center_freq > 0:
            # Calculate period = 1 / frequency (seconds)
            period = 1.0 / center_freq
            # Convert to samples: period_samples = (wavelength_multiplier × period) × sample_rate
            period_samples = (wavelength_multiplier * period) * self.sample_rate
            # Ensure minimum of 1 sample
            period_samples = max(1.0, period_samples)
        else:
            # Full Spectrum (0 Hz) - use fallback window as period
            period_samples = (fallback_window_ms / 1000.0) * self.sample_rate
            period_samples = max(1.0, period_samples)
        
        # Calculate release coefficient for exponential decay
        # Attack: 0ms (instantaneous rise to catch peaks immediately)
        # Release: wavelength-based decay for smooth release
        # Using standard exponential: coeff = 1 - exp(-1 / time_constant)
        release_time_samples = period_samples  # Wavelength-based release
        release_coeff = 1.0 - np.exp(-1.0 / release_time_samples)
        
        # OPTIMIZED IMPLEMENTATION: Use Numba JIT-compiled function for peak follower
        # This provides 10-50x speedup over pure Python loops while maintaining
        # exact same algorithm behavior
        # Pre-compute the decay multiplier
        decay_multiplier = 1.0 - release_coeff
        
        # Use JIT-compiled core function if available, otherwise fall back to Python
        if NUMBA_AVAILABLE:
            envelope = _calculate_peak_envelope_core(rectified, decay_multiplier, release_coeff)
        else:
            # Fallback to Python implementation if Numba not available
            envelope = np.zeros_like(rectified)
            envelope[0] = rectified[0]
            for i in range(1, len(rectified)):
                decayed = envelope[i - 1] * decay_multiplier + rectified[i] * release_coeff
                envelope[i] = max(rectified[i], decayed)
        
        return envelope

    def calculate_rms_envelope(self, signal: np.ndarray,
                                center_freq: float = 0.0,
                                method: str = 'peak_envelope',
                                wavelength_multiplier: float = 1.0,
                                fallback_window_ms: float = 10.0) -> np.ndarray:
        """Calculate envelope using peak envelope follower (primary method).
        
        This method uses a standard peak envelope follower with wavelength-based
        attack and release times. This creates a line linking peaks of every wave
        and maintains correct peak levels.
        
        Args:
            signal: Audio signal array (normalized)
            center_freq: Center frequency of the octave band (Hz). Use 0 for Full Spectrum.
            method: Envelope detection method ('peak_envelope' or 'rms_window' for legacy)
            wavelength_multiplier: Multiplier for attack/release time (default 1.0 = 1x wavelength)
            fallback_window_ms: Fallback window size in ms for Full Spectrum (center_freq=0)
            
        Returns:
            Envelope array in linear scale (same length as input)
        """
        if method == 'peak_envelope':
            # Primary method: Peak envelope follower
            return self.calculate_peak_envelope(
                signal,
                center_freq=center_freq,
                wavelength_multiplier=wavelength_multiplier,
                fallback_window_ms=fallback_window_ms
            )
        else:
            # Legacy RMS window method (for backwards compatibility / Full Spectrum fallback)
            window_samples = int(fallback_window_ms * self.sample_rate / 1000)
            
            if window_samples < 1:
                window_samples = 1
            
            if window_samples >= len(signal):
                rms_val = np.sqrt(np.mean(signal**2))
                return np.full_like(signal, rms_val)
            
            window = np.ones(window_samples, dtype=signal.dtype) / window_samples
            signal_squared = signal ** 2
            rms_squared = sp_signal.fftconvolve(signal_squared, window, mode='same')
            rms_squared = np.maximum(rms_squared, 0.0)
            rms_envelope = np.sqrt(rms_squared)
            
            return rms_envelope

    def find_attack_time(self, envelope_db: np.ndarray, peak_idx: int,
                         peak_value_db: float, attack_threshold_db: float) -> float:
        """Find attack time using vectorized backward search.
        
        Args:
            envelope_db: Peak envelope in dBFS
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

    def find_peak_hold_time(self, envelope_db: np.ndarray, peak_idx: int,
                            peak_value_db: float, hold_threshold_db: float) -> float:
        """Find peak hold time using vectorized forward/backward search.
        
        Args:
            envelope_db: Peak envelope in dBFS
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

    def find_decay_times(self, envelope_db: np.ndarray, peak_idx: int,
                         peak_value_db: float,
                         decay_thresholds_db: List[float]) -> Dict[str, float]:
        """Find decay times to multiple thresholds in single pass.
        
        Args:
            envelope_db: Peak envelope in dBFS
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

    def analyze_worst_case_envelopes(self, envelope_db: np.ndarray,
                                     peak_indices: np.ndarray,
                                     peak_values_db: np.ndarray,
                                     num_envelopes: int,
                                     sort_by: str,
                                     attack_threshold_db: float,
                                     peak_hold_threshold_db: float,
                                     decay_thresholds_db: List[float],
                                     exclude_peak_indices: Optional[Set[int]] = None,
                                     window_ms: Optional[float] = None) -> List[Dict]:
        """Analyze worst-case envelopes efficiently using partition-based selection.
        
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
            attack_time_ms = self.find_attack_time(
                envelope_db, peak_idx, peak_value_db, attack_threshold_db
            )
            
            peak_hold_time_ms = self.find_peak_hold_time(
                envelope_db, peak_idx, peak_value_db, peak_hold_threshold_db
            )
            
            decay_times = self.find_decay_times(
                envelope_db, peak_idx, peak_value_db, decay_thresholds_db
            )
            
            # Extract envelope window for plotting if window_ms provided
            envelope_window = None
            time_window_ms = None
            if window_ms is not None:
                window_samples = int(window_ms * self.sample_rate / 1000)
                half_window = window_samples // 2
                start_idx = max(0, peak_idx - half_window)
                end_idx = min(len(envelope_db), peak_idx + half_window)
                if end_idx - start_idx >= window_samples // 2:
                    envelope_window = envelope_db[start_idx:end_idx]
                    # Convert to relative time (ms from peak)
                    peak_time_idx = peak_idx - start_idx
                    time_window_ms = ((np.arange(len(envelope_window)) - peak_time_idx) / 
                                     self.sample_rate * 1000.0)
            
            envelope_data = {
                "rank": rank,
                "peak_value_db": float(peak_value_db),
                "peak_time_seconds": float(peak_time_seconds),
                "peak_idx": int(peak_idx),  # Store for mutual exclusivity checking
                "attack_time_ms": float(attack_time_ms),
                "peak_hold_time_ms": float(peak_hold_time_ms),
                "decay_times": decay_times,
                "envelope_window": envelope_window,  # Store for plotting reuse
                "time_window_ms": time_window_ms  # Store relative time window
            }
            
            worst_case_envelopes.append(envelope_data)
        
        return worst_case_envelopes

    def compare_envelope_shapes(self, pattern1: np.ndarray,
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
        # Check for constant patterns (zero variance) which cause division by zero
        if np.std(p1_norm) == 0 or np.std(p2_norm) == 0:
            # If both patterns are constant and identical, return 1.0
            if np.allclose(p1_norm, p2_norm):
                return 1.0
            # Otherwise, cannot compute correlation
            return 0.0
        
        correlation = np.corrcoef(p1_norm, p2_norm)[0, 1]
        
        # Handle NaN/Inf (shouldn't happen, but safety check)
        if np.isnan(correlation) or np.isinf(correlation):
            return 0.0
        
        return float(correlation)

    def analyze_repeating_patterns(self, envelope_db: np.ndarray,
                                   peak_indices: np.ndarray,
                                   peak_values_db: np.ndarray,
                                   min_repetitions: int,
                                   max_patterns: int,
                                   similarity_threshold: float,
                                   window_ms: float) -> Dict:
        """Analyze repeating patterns using optimized correlation matching.
        
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
        
        # VECTORIZED: Normalize patterns to 0-1 for comparison
        normalized_patterns = []
        for pattern in patterns:
            p_min, p_max = np.min(pattern), np.max(pattern)
            if p_max - p_min > 1e-10:
                normalized = (pattern - p_min) / (p_max - p_min)
            else:
                normalized = np.zeros_like(pattern)
            normalized_patterns.append(normalized)
        
        # Find pattern groups using correlation
        # Optimize: Limit comparisons for performance with large patterns
        # For very large patterns, sample down for comparison or limit number of peaks
        max_peaks_for_pattern_matching = 500  # Limit peaks analyzed to prevent O(n²) explosion
        
        if len(normalized_patterns) > max_peaks_for_pattern_matching:
            # For bands with many peaks, sample the top peaks by value
            peak_values_sorted_indices = np.argsort(peak_values_db)[::-1]
            selected_indices = set(peak_values_sorted_indices[:max_peaks_for_pattern_matching])
            normalized_patterns_filtered = [(i, p) for i, p in enumerate(normalized_patterns) 
                                          if i in selected_indices]
            patterns_to_compare = [p for _, p in normalized_patterns_filtered]
            indices_mapping = {new_idx: orig_idx for new_idx, (orig_idx, _) 
                              in enumerate(normalized_patterns_filtered)}
        else:
            patterns_to_compare = normalized_patterns
            indices_mapping = {i: i for i in range(len(normalized_patterns))}
        
        # VECTORIZED PATTERN MATCHING: Compute correlation matrix for all pattern pairs
        # This replaces the nested loop with vectorized operations
        num_patterns = len(patterns_to_compare)
        if num_patterns == 0:
            return {"patterns_detected": 0}
        
        # For patterns of different lengths, we need to handle them individually
        # But we can still vectorize the comparison by batching similar-length patterns
        # For now, use optimized batch correlation where possible
        pattern_groups = []
        used_indices = set()
        
        # OPTIMIZED: Use vectorized correlation computation
        # Group patterns by length to enable batch processing
        length_groups = {}
        for idx, pattern in enumerate(patterns_to_compare):
            length = len(pattern)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((idx, pattern))
        
        # Process each length group with vectorized operations
        for length, group_patterns in length_groups.items():
            if len(group_patterns) < 2:
                continue
            
            group_indices = [idx for idx, _ in group_patterns]
            group_arrays = np.array([p for _, p in group_patterns])
            
            # Compute correlation matrix for this length group
            # Normalize each pattern (already normalized, but ensure same length)
            # Use np.corrcoef for batch correlation computation
            if len(group_arrays) > 1:
                # Flatten patterns and compute correlation matrix
                # corrcoef expects (n_vars, n_obs) format
                corr_matrix = np.corrcoef(group_arrays)
                
                # Find similar patterns using vectorized operations
                for i, (orig_i, _) in enumerate(group_patterns):
                    if orig_i in used_indices:
                        continue
                    
                    # Find all patterns with correlation >= threshold
                    similar_mask = (corr_matrix[i, :] >= similarity_threshold) & \
                                   (np.arange(len(group_patterns)) > i)
                    similar_indices = np.where(similar_mask)[0]
                    
                    if len(similar_indices) > 0:
                        group = [orig_i]
                        for j in similar_indices:
                            orig_j = group_indices[j]
                            if orig_j not in used_indices:
                                group.append(orig_j)
                                used_indices.add(orig_j)
                        
                        if len(group) >= min_repetitions:
                            pattern_groups.append(group)
                            used_indices.add(orig_i)
        
        # Handle cross-length comparisons (patterns of different lengths)
        # These still need individual comparison but we've reduced the O(n²) work
        for i, pattern in enumerate(patterns_to_compare):
            orig_i = indices_mapping[i]
            if orig_i in used_indices:
                continue
            
            # Only compare with patterns we haven't processed yet
            group = [orig_i]
            for j in range(i + 1, len(patterns_to_compare)):
                orig_j = indices_mapping[j]
                if orig_j in used_indices:
                    continue
                
                # Check if patterns are same length (already processed above)
                if len(pattern) == len(patterns_to_compare[j]):
                    continue
                
                # Use comparison method for different-length patterns
                correlation = self.compare_envelope_shapes(pattern, patterns_to_compare[j])
                if correlation >= similarity_threshold:
                    group.append(orig_j)
                    used_indices.add(orig_j)
            
            if len(group) >= min_repetitions:
                pattern_groups.append(group)
                used_indices.add(orig_i)
        
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
            
            # Sort by time for consistent ordering
            sorted_pairs = sorted(zip(group_peak_times, group_peak_indices))
            group_peak_times = [t for t, _ in sorted_pairs]
            group_peak_indices = [idx for _, idx in sorted_pairs]
            
            # Calculate inter-peak intervals
            intervals = np.diff(group_peak_times)
            
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
            
            # Store envelope windows for plotting (extract once, reuse in plots)
            envelope_windows = []
            time_windows_ms = []
            for peak_idx in group_peak_indices:
                start_idx = max(0, peak_idx - half_window)
                end_idx = min(len(envelope_db), peak_idx + half_window)
                if end_idx - start_idx >= window_samples // 2:
                    window_data = envelope_db[start_idx:end_idx]
                    # Convert to relative time (ms from peak)
                    peak_time_idx = peak_idx - start_idx
                    time_relative_ms = ((np.arange(len(window_data)) - peak_time_idx) / 
                                       self.sample_rate * 1000.0)
                    envelope_windows.append(window_data)
                    time_windows_ms.append(time_relative_ms)
                else:
                    envelope_windows.append(None)
                    time_windows_ms.append(None)
            
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
                "peak_times_seconds": group_peak_times,
                "peak_indices": group_peak_indices,  # Store peak indices for filtering
                "envelope_windows": envelope_windows,  # Store extracted windows for plotting
                "time_windows_ms": time_windows_ms  # Store relative time windows
            }
        
        return result

    def analyze_envelope_statistics(self, octave_bank: np.ndarray,
                                    center_frequencies: List[float],
                                    config: Optional[Dict] = None) -> Dict:
        """Analyze envelope characteristics and repeating patterns for each octave band.
        
        This method performs two types of analysis:
        1. Worst-case envelope analysis: Finds top N worst envelopes (for one-off events)
        2. Pattern analysis: Detects repeating patterns with configurable minimum repetitions
        
        The peak envelope is calculated ONCE per band and reused for all analysis to maximize
        efficiency. The peak envelope creates a line linking peaks of every wave cycle.
        
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
        
        # Envelope calculation method
        envelope_method = config.get('envelope_method', 'peak_envelope')
        wavelength_multiplier = config.get('envelope_wavelength_multiplier', 1.0)
        fallback_window_ms = config.get('rms_envelope_window_ms', 10.0)
        
        # Peak detection and analysis parameters
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
            
            # Calculate envelope ONCE (reused for all analysis)
            # Use peak envelope follower method (band-relative, wavelength-based)
            rms_envelope_linear = self.calculate_rms_envelope(
                band_signal,
                center_freq=freq,
                method=envelope_method,
                wavelength_multiplier=wavelength_multiplier,
                fallback_window_ms=fallback_window_ms
            )
            
            # Convert to dBFS
            # Ensure envelope values are positive (handle any numerical issues)
            rms_envelope_linear = np.maximum(rms_envelope_linear, 1e-10)
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
            
            # Pattern analysis (if enabled) - run FIRST to identify pattern peaks
            pattern_peak_indices = set()
            window_was_capped = False  # Track if window was capped for plotting
            
            if pattern_max_patterns > 0 and len(peak_indices) >= pattern_min_reps:
                # Calculate pattern extraction window based on center frequency
                # Get num_wavelengths from config (should match plotting)
                num_wavelengths = config.get('envelope_plots_num_wavelengths', 50)
                
                if freq > 0:
                    # Use frequency-relative window (in wavelengths)
                    period = 1.0 / freq
                    pattern_window_ms = (num_wavelengths * period) * 1000.0
                    
                    # Limit maximum window size to prevent performance issues
                    # For very low frequencies, cap at reasonable limit (500ms)
                    max_pattern_window_ms = 500.0
                    if pattern_window_ms > max_pattern_window_ms:
                        window_was_capped = True
                        pattern_window_ms = max_pattern_window_ms
                else:
                    # Full Spectrum fallback
                    pattern_window_ms = fallback_window_ms * 2
                
                pattern_analysis = self.analyze_repeating_patterns(
                    rms_envelope_db,
                    peak_indices,
                    peak_values_db,
                    pattern_min_reps,
                    pattern_max_patterns,
                    pattern_similarity,
                    pattern_window_ms
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
            # Always analyze at least 3 worst-case envelopes for visualization
            worst_case_num_vis = max(worst_case_num, 3)
            if worst_case_num_vis > 0 and len(peak_indices) > 0:
                # Calculate plotting window for worst-case envelopes (same as pattern analysis)
                if freq > 0:
                    num_wavelengths = config.get('envelope_plots_num_wavelengths', 50)
                    period = 1.0 / freq
                    worst_case_window_ms = (num_wavelengths * period) * 1000.0
                    max_pattern_window_ms = 500.0
                    worst_case_window_ms = min(worst_case_window_ms, max_pattern_window_ms)
                else:
                    # For Full Spectrum, use same window as pattern analysis (fallback_window_ms * 2)
                    # This ensures consistent window sizes between analysis and plotting
                    worst_case_window_ms = fallback_window_ms * 2
                
                worst_case_envelopes = self.analyze_worst_case_envelopes(
                    rms_envelope_db,
                    peak_indices,
                    peak_values_db,
                    worst_case_num_vis,
                    worst_case_sort_by,
                    attack_threshold_db,
                    peak_hold_threshold_db,
                    decay_thresholds_db,
                    exclude_peak_indices=pattern_peak_indices if pattern_peak_indices else None,
                    window_ms=worst_case_window_ms
                )
                band_results["worst_case_envelopes"] = worst_case_envelopes
            else:
                band_results["worst_case_envelopes"] = []
            
            # Sustained peak hold and recovery analysis (relative thresholds by default)
            sustained_enable = config.get('sustained_peaks_enable', True)
            sustained_min_peak_dbfs = config.get('sustained_peaks_min_peak_dbfs', -3.0)
            sustained_thresholds_db = config.get('sustained_peaks_thresholds_db', [-3.0, -6.0, -9.0, -12.0])
            sustained_relative = config.get('sustained_peaks_relative', True)
            export_events = config.get('sustained_peaks_export_events', False)

            sustained_summary = {}
            sustained_events = []
            if sustained_enable and len(peak_indices) > 0:
                # Qualify peaks >= min_peak threshold (absolute dBFS)
                qualifying_mask = peak_values_db >= sustained_min_peak_dbfs
                qualifying_indices = peak_indices[qualifying_mask]
                qualifying_values = peak_values_db[qualifying_mask]

                if len(qualifying_indices) > 0:
                    # Compute hold time and recovery times to thresholds for each qualifying peak
                    # Limit search window to cap runtime (configurable, default 5 seconds)
                    search_window_seconds = config.get('sustained_peaks_search_window_seconds', 5.0)
                    logger.info(
                        f"Sustained peaks analysis: Using search window of {search_window_seconds:.1f} seconds "
                        f"({int(search_window_seconds * 1000)}ms) for recovery time calculations"
                    )
                    max_search_samples = int(search_window_seconds * self.sample_rate)
                    epsilon_db = 1.0  # Hold time threshold: 1 dB below peak

                    for p_idx, p_val in zip(qualifying_indices, qualifying_values):
                        # Hold: continuous samples within 1 dB of peak value from peak forward
                        # Find the FIRST index where signal drops below threshold (not the last where it's above)
                        end_lim = min(len(rms_envelope_db), p_idx + max_search_samples)
                        forward_region = rms_envelope_db[p_idx:end_lim]
                        threshold = p_val - epsilon_db
                        # Find first index where signal drops below threshold
                        below = forward_region < threshold
                        if np.any(below):
                            # First drop below threshold
                            first_below = np.where(below)[0][0]
                            hold_samples = first_below
                        else:
                            # Signal stayed above threshold for entire search window
                            hold_samples = len(forward_region)
                        hold_ms = (hold_samples / self.sample_rate) * 1000.0

                        # Recovery times to thresholds (relative to peak or absolute)
                        decay_region = rms_envelope_db[p_idx:end_lim]
                        recovery_times = {}
                        for th in sustained_thresholds_db:
                            target = p_val + th if sustained_relative else th
                            # Find first sample where signal is at or below target threshold
                            below = np.where(decay_region <= target)[0]
                            if len(below) > 0:
                                # Recovery time found: convert sample offset to milliseconds
                                t_ms = (below[0] / self.sample_rate) * 1000.0
                            else:
                                # Signal didn't recover within search window: cap at window limit
                                t_ms = (end_lim - p_idx) / self.sample_rate * 1000.0
                            recovery_times[f"t{abs(int(th))}_ms"] = float(t_ms)

                        sustained_events.append({
                            "peak_time_seconds": float(p_idx / self.sample_rate),
                            "peak_value_db": float(p_val),
                            "hold_ms": float(hold_ms),
                            **recovery_times
                        })

                    # Aggregate summary statistics
                    def _agg(arr: np.ndarray) -> Dict[str, float]:
                        if arr.size == 0:
                            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
                        return {
                            "mean": float(np.mean(arr)),
                            "median": float(np.median(arr)),
                            "p90": float(np.percentile(arr, 90)),
                            "p95": float(np.percentile(arr, 95)),
                            "max": float(np.max(arr)),
                        }

                    holds = np.array([e["hold_ms"] for e in sustained_events], dtype=float)
                    sustained_summary["hold_ms"] = _agg(holds)
                    for th in sustained_thresholds_db:
                        key = f"t{abs(int(th))}_ms"
                        vals = np.array([e[key] for e in sustained_events], dtype=float)
                        sustained_summary[key] = _agg(vals)
                else:
                    # No qualifying peaks
                    sustained_summary["hold_ms"] = {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
                    for th in sustained_thresholds_db:
                        sustained_summary[f"t{abs(int(th))}_ms"] = {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}

            band_results["sustained_peaks_summary"] = {
                "n_peaks": int(len(sustained_events)),
                **sustained_summary
            }
            if export_events:
                band_results["sustained_peaks_events"] = sustained_events
            
            # Store peak envelope for visualization (needed for plotting)
            band_results["rms_envelope_db"] = rms_envelope_db  # Variable name kept for compatibility
            band_results["rms_envelope_time"] = np.arange(len(rms_envelope_db)) / self.sample_rate
            band_results["window_was_capped"] = window_was_capped  # Store cap status for plotting
            
            results[freq_label] = band_results
            
            # Memory cleanup (keep rms_envelope_db for visualization)
            del rms_envelope_linear, rms_envelope_db_clean
        
        logger.info("Envelope statistics analysis complete")
        return results

