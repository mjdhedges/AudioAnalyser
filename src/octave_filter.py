"""
Octave band filtering module for Music Analyser.

This module implements octave band filtering functionality similar to MATLAB's octdsgn function.
"""

from __future__ import annotations

import logging
from typing import List, Tuple
import multiprocessing

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def _filter_worker_sos(args: Tuple[np.ndarray, float, int, int]) -> Tuple[float, np.ndarray]:
    """Worker function for parallel octave band filtering.
    
    This module-level function is used for multiprocessing as it can be pickled.
    
    Args:
        args: Tuple of (audio_data, center_freq, sample_rate, filter_order)
        
    Returns:
        Tuple of (center_freq, filtered_signal)
    """
    audio_data, center_freq, sample_rate, filter_order = args
    
    try:
        # ISO 266:1997 / IEC 61260:1995 standard octave band calculation
        # ISO defines: lower = center / sqrt(2), upper = center * sqrt(2)
        # This ensures upper/lower = 2.0 (exactly one octave)
        low_freq = center_freq / np.sqrt(2)
        high_freq = center_freq * np.sqrt(2)
        
        # Ensure frequencies are within valid range
        nyquist = sample_rate / 2
        low_freq = max(low_freq, 1.0)
        high_freq = min(high_freq, nyquist * 0.99)
        
        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Always use 4th order for consistent filtering across all frequencies
        actual_filter_order = 4
        
        # Ensure normalized frequencies are within valid range
        low_norm = max(low_norm, 1e-6)
        high_norm = min(high_norm, 0.99)
        
        # Design and apply filter using SOS format for numerical stability
        sos = signal.butter(actual_filter_order, [low_norm, high_norm], btype='band', output='sos')
        filtered_signal = signal.sosfiltfilt(sos, audio_data)
        
        return (center_freq, filtered_signal)
    
    except Exception as e:
        logger.error(f"Error in filter worker for {center_freq}Hz: {e}")
        # Return zeros on error
        return (center_freq, np.zeros_like(audio_data))


class OctaveBandFilter:
    """Handles octave band filtering operations."""

    # Standard octave band center frequencies (Hz)
    # Added 16Hz for cinema/LFE signal analysis
    OCTAVE_CENTER_FREQUENCIES = [
        16.0, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000
    ]

    def __init__(self, sample_rate: int = 44100, use_linkwitz_riley: bool = False,
                 normalize_overlap: bool = True, use_cascade: bool = False) -> None:
        """Initialize the octave band filter.
        
        Args:
            sample_rate: Sample rate of the audio signal
            use_linkwitz_riley: If True, use Linkwitz-Riley filters instead of Butterworth.
                               LR filters sum flat with proper phase alignment when -6dB points are aligned at crossovers.
            normalize_overlap: If True, normalize band gains to eliminate ripple from overlapping bands.
                               When bands overlap, their summed response can exceed unity, causing ripple.
                               Normalization ensures bands sum to flat frequency response.
            use_cascade: If True, use cascade complementary filter bank approach (sequential extraction).
                        Preserves phase relationships and ensures natural summing to unity.
                        Requires use_linkwitz_riley=True for proper operation.
        """
        self.sample_rate = sample_rate
        self.use_linkwitz_riley = use_linkwitz_riley
        self.normalize_overlap = normalize_overlap
        self.use_cascade = use_cascade
        if use_cascade and not use_linkwitz_riley:
            logger.warning("Cascade mode requires Linkwitz-Riley filters. Enabling use_linkwitz_riley=True")
            self.use_linkwitz_riley = True
        self._filter_cache = {}  # Cache for filter coefficients (b, a)
        self._sos_cache = {}  # Cache for SOS filter coefficients
        self._crossover_frequencies = None  # Cached crossover frequencies for LR alignment
        self._normalization_factors = {}  # Cache for per-band normalization factors

    def design_octave_filter(self, center_freq: float, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Design an octave band filter.
        
        This function replicates MATLAB's octdsgn functionality.
        
        Args:
            center_freq: Center frequency of the octave band
            order: Filter order (1 for octave, 3 for third-octave)
            
        Returns:
            Tuple of (b, a) filter coefficients
        """
        # OPTIMIZATION: Check cache first to avoid redundant filter design
        cache_key = (center_freq, order, self.sample_rate, self.use_linkwitz_riley)
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]
        
        # Calculate bandwidth based on order
        if order == 1:
            # Octave band: bandwidth = center_freq / sqrt(2)
            bandwidth = center_freq / np.sqrt(2)
        elif order == 3:
            # Third-octave band: bandwidth = center_freq / 2^(1/6)
            bandwidth = center_freq / (2**(1/6))
        else:
            raise ValueError(f"Unsupported filter order: {order}")

        # Calculate filter frequencies
        low_freq = center_freq - bandwidth / 2
        high_freq = center_freq + bandwidth / 2

        # Ensure frequencies are within valid range
        nyquist = self.sample_rate / 2
        low_freq = max(low_freq, 1.0)  # Minimum frequency
        high_freq = min(high_freq, nyquist * 0.99)  # Below Nyquist

        # Design bandpass filter
        filter_order = 4
        
        if self.use_linkwitz_riley:
            logger.debug(f"Using 4th order Linkwitz-Riley filter for: {center_freq}Hz")
            # Linkwitz-Riley filters: adjust bandwidth to align -6dB points at crossovers
            # This ensures proper summing when bands are recombined
            adjusted_low_freq, adjusted_high_freq = self._calculate_lr_crossover_bandwidth(
                center_freq, self.OCTAVE_CENTER_FREQUENCIES
            )
            b, a = self._design_linkwitz_riley_bandpass(adjusted_low_freq, adjusted_high_freq, filter_order)
            # Update frequencies for logging
            low_freq = adjusted_low_freq
            high_freq = adjusted_high_freq
        else:
            # Design Butterworth bandpass filter
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            
            logger.debug(f"Using 4th order Butterworth filter for: {center_freq}Hz")
            
            # Ensure normalized frequencies are within valid range for scipy.signal.butter
            low_norm = max(low_norm, 1e-6)  # Minimum normalized frequency
            high_norm = min(high_norm, 0.99)  # Maximum normalized frequency
            
            b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')

        logger.debug(f"Designed octave filter: {center_freq}Hz, "
                    f"bandwidth: {bandwidth:.2f}Hz, "
                    f"range: {low_freq:.2f}-{high_freq:.2f}Hz")

        # Cache the filter coefficients for future use
        self._filter_cache[cache_key] = (b, a)
        
        return b, a

    def design_octave_filter_sos(self, center_freq: float, order: int = 1) -> np.ndarray:
        """Design an octave band filter using SOS (second-order sections) format.
        
        This method produces more numerically stable filters, especially for
        low frequencies and long signals. Recommended for production use.
        
        When use_linkwitz_riley is True, designs complementary LR bandpass filters
        with crossover-based alignment for flat summing. Otherwise uses Butterworth.
        
        When normalize_overlap is enabled with Butterworth, adjusts bandwidths to use 
        geometric mean crossover points, reducing ripple from overlapping bands.
        
        Args:
            center_freq: Center frequency of the octave band
            order: Filter order (1 for octave, 3 for third-octave)
            
        Returns:
            SOS (second-order sections) filter coefficients (Nx6 array)
        """
        if self.use_linkwitz_riley:
            # Use complementary Linkwitz-Riley design with crossover alignment
            return self._design_lr_complementary_bandpass_sos(center_freq, order)
        
        # Calculate filter frequencies for Butterworth design
        if self.normalize_overlap and center_freq in self.OCTAVE_CENTER_FREQUENCIES:
            # Use geometric mean crossover points for better summing
            sorted_freqs = sorted([f for f in self.OCTAVE_CENTER_FREQUENCIES if f > 0])
            idx = sorted_freqs.index(center_freq)
            
            # Lower crossover: geometric mean with previous band
            if idx > 0:
                low_freq = np.sqrt(sorted_freqs[idx - 1] * center_freq)
            else:
                # First band: use ISO standard calculation
                if order == 1:
                    low_freq = center_freq / np.sqrt(2)
                elif order == 3:
                    bandwidth = center_freq / (2**(1/6))
                    low_freq = center_freq - bandwidth / 2
                else:
                    raise ValueError(f"Unsupported filter order: {order}")
            
            # Higher crossover: geometric mean with next band
            if idx < len(sorted_freqs) - 1:
                high_freq = np.sqrt(center_freq * sorted_freqs[idx + 1])
            else:
                # Last band: use ISO standard calculation
                if order == 1:
                    high_freq = center_freq * np.sqrt(2)
                elif order == 3:
                    bandwidth = center_freq / (2**(1/6))
                    high_freq = center_freq + bandwidth / 2
                else:
                    raise ValueError(f"Unsupported filter order: {order}")
        else:
            # ISO 266:1997 / IEC 61260:1995 standard octave band calculation
            # ISO defines: lower = center / sqrt(2), upper = center * sqrt(2)
            # This ensures upper/lower = 2.0 (exactly one octave)
            if order == 1:
                # Octave band: ISO standard
                low_freq = center_freq / np.sqrt(2)
                high_freq = center_freq * np.sqrt(2)
            elif order == 3:
                # Third-octave band: bandwidth = center_freq / 2^(1/6)
                bandwidth = center_freq / (2**(1/6))
                low_freq = center_freq - bandwidth / 2
                high_freq = center_freq + bandwidth / 2
            else:
                raise ValueError(f"Unsupported filter order: {order}")

        # Ensure frequencies are within valid range
        nyquist = self.sample_rate / 2
        low_freq = max(low_freq, 1.0)
        high_freq = min(high_freq, nyquist * 0.99)

        # Normalize frequencies
        low_norm = max(low_freq / nyquist, 1e-6)
        high_norm = min(high_freq / nyquist, 0.99)
        
        # Design Butterworth bandpass filter directly in SOS format
        # This avoids numerical issues when converting from b,a to SOS
        # Use 4th order for consistency
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        
        return sos

    def _calculate_complementary_crossovers(self, center_freq: float) -> Tuple[float, float]:
        """Calculate complementary crossover frequencies for adjacent bands.
        
        For complementary filter design, crossover frequencies are calculated as
        geometric mean of adjacent center frequencies. At these crossovers, 
        adjacent filters should each contribute -6 dB (magnitude 0.5), summing to 0 dB.
        
        Args:
            center_freq: Center frequency of current band
            
        Returns:
            Tuple of (lower_crossover_freq, higher_crossover_freq) in Hz
        """
        sorted_freqs = sorted([f for f in self.OCTAVE_CENTER_FREQUENCIES if f > 0])
        
        if center_freq not in sorted_freqs:
            # Fallback to ISO standard if not in standard frequencies
            lower_crossover = center_freq / np.sqrt(2)
            higher_crossover = center_freq * np.sqrt(2)
        else:
            idx = sorted_freqs.index(center_freq)
            
            # Lower crossover: geometric mean with previous band
            if idx > 0:
                lower_crossover = np.sqrt(sorted_freqs[idx - 1] * center_freq)
            else:
                # First band: use ISO standard lower bound
                lower_crossover = center_freq / np.sqrt(2)
            
            # Higher crossover: geometric mean with next band
            if idx < len(sorted_freqs) - 1:
                higher_crossover = np.sqrt(center_freq * sorted_freqs[idx + 1])
            else:
                # Last band: use ISO standard upper bound
                higher_crossover = center_freq * np.sqrt(2)
        
        # Ensure frequencies are within valid range
        nyquist = self.sample_rate / 2
        lower_crossover = max(lower_crossover, 1.0)
        higher_crossover = min(higher_crossover, nyquist * 0.99)
        
        return lower_crossover, higher_crossover
    
    def _design_lr_complementary_bandpass_sos(self, center_freq: float, order: int = 1) -> np.ndarray:
        """Design complementary bandpass filter with -6 dB alignment at crossovers.
        
        The key insight: For complementary summing, we need each filter to be at -6 dB
        (magnitude 0.5) at crossover frequencies, so they sum to 1.0 (0 dB).
        
        Butterworth filters are -3 dB at their cutoff frequencies. To get -6 dB at
        the crossover, we need to adjust the filter bandwidths so the -6 dB points
        align with the crossover frequencies.
        
        For a 4th order Butterworth bandpass:
        - -3 dB points are at the specified cutoff frequencies
        - -6 dB points are further from center (bandwidth is wider)
        - We calculate what cutoff frequencies give us -6 dB at the crossovers
        
        Args:
            center_freq: Center frequency of the octave band
            order: Filter order (1 for octave, 3 for third-octave) - not used for LR
            
        Returns:
            SOS (second-order sections) filter coefficients (Nx6 array)
        """
        # Calculate complementary crossover frequencies (where we want -6 dB)
        lower_crossover, higher_crossover = self._calculate_complementary_crossovers(center_freq)
        
        nyquist = self.sample_rate / 2
        
        # For 4th order Butterworth, the relationship between -3 dB and -6 dB points
        # is approximately: f_6db = f_3db * (1 ± 0.189) for bandpass
        # More precisely, we need to solve: |H(f)| = 0.5 at crossover frequency
        # This requires iterative adjustment or analytical solution
        
        # Simplified approach: adjust cutoff frequencies to achieve -6 dB at crossovers
        # For Butterworth filters, we can approximate by widening/narrowing bandwidth
        
        # Test frequencies to find -6 dB points
        test_freqs_low = np.logspace(np.log10(lower_crossover * 0.5), np.log10(lower_crossover * 1.5), 200)
        test_freqs_high = np.logspace(np.log10(higher_crossover * 0.5), np.log10(higher_crossover * 1.5), 200)
        
        # Find filter cutoffs that give -6 dB at crossover frequencies
        # Start with crossover frequencies as initial guess
        lower_cutoff = lower_crossover
        higher_cutoff = higher_crossover
        
        # Iteratively adjust to get -6 dB at crossovers
        for iteration in range(5):  # Limit iterations
            lower_norm = max(lower_cutoff / nyquist, 1e-6)
            higher_norm = min(higher_cutoff / nyquist, 0.99)
            
            sos_test = signal.butter(4, [lower_norm, higher_norm], btype='band', output='sos')
            
            # Check magnitude at crossover frequencies
            w_low, h_low = signal.sosfreqz(sos_test, worN=[lower_crossover], fs=self.sample_rate)
            w_high, h_high = signal.sosfreqz(sos_test, worN=[higher_crossover], fs=self.sample_rate)
            
            mag_low_db = 20 * np.log10(np.abs(h_low[0]) + 1e-10)
            mag_high_db = 20 * np.log10(np.abs(h_high[0]) + 1e-10)
            
            target_db = -6.0
            error_low = mag_low_db - target_db
            error_high = mag_high_db - target_db
            
            # If close enough, break
            if abs(error_low) < 0.1 and abs(error_high) < 0.1:
                break
            
            # Adjust cutoffs: if magnitude too high, widen bandwidth; if too low, narrow
            # For low crossover: if mag > -6 dB, increase lower cutoff (narrow low side)
            # For high crossover: if mag > -6 dB, decrease higher cutoff (narrow high side)
            if error_low > 0:  # Magnitude too high (closer to 0 dB)
                lower_cutoff *= 1.05  # Move cutoff further from center
            else:
                lower_cutoff *= 0.95  # Move cutoff closer to center
            
            if error_high > 0:  # Magnitude too high
                higher_cutoff *= 0.95  # Move cutoff closer to center
            else:
                higher_cutoff *= 1.05  # Move cutoff further from center
            
            # Ensure cutoffs are reasonable
            lower_cutoff = max(lower_cutoff, 1.0)
            higher_cutoff = min(higher_cutoff, nyquist * 0.99)
        
        # Final design with adjusted cutoffs
        lower_norm = max(lower_cutoff / nyquist, 1e-6)
        higher_norm = min(higher_cutoff / nyquist, 0.99)
        
        sos_band = signal.butter(4, [lower_norm, higher_norm], btype='band', output='sos')
        
        return sos_band
    
    def _calculate_lr_crossover_bandwidth(self, center_freq: float, 
                                         all_center_freqs: List[float]) -> Tuple[float, float]:
        """Calculate adjusted bandwidth for Linkwitz-Riley filters with -6dB crossover alignment.
        
        For LR filters to sum correctly, adjacent bands must align at -6dB points.
        The crossover frequency between bands is the geometric mean of center frequencies.
        Each filter should be -6dB at its crossover points.
        
        Args:
            center_freq: Center frequency of current band
            all_center_freqs: List of all center frequencies to calculate crossovers
            
        Returns:
            Tuple of (adjusted_low_freq, adjusted_high_freq) with -6dB points at crossovers
        """
        # Calculate crossover frequencies with adjacent bands
        sorted_freqs = sorted([f for f in all_center_freqs if f > 0])
        idx = sorted_freqs.index(center_freq)
        
        # Lower crossover: geometric mean with previous band
        if idx > 0:
            lower_crossover = np.sqrt(sorted_freqs[idx - 1] * center_freq)
        else:
            # First band: use standard calculation
            bandwidth = center_freq / np.sqrt(2)
            lower_crossover = center_freq - bandwidth / 2
        
        # Higher crossover: geometric mean with next band
        if idx < len(sorted_freqs) - 1:
            higher_crossover = np.sqrt(center_freq * sorted_freqs[idx + 1])
        else:
            # Last band: use standard calculation
            bandwidth = center_freq / np.sqrt(2)
            higher_crossover = center_freq + bandwidth / 2
        
        # Ensure frequencies are within valid range
        nyquist = self.sample_rate / 2
        lower_crossover = max(lower_crossover, 1.0)
        higher_crossover = min(higher_crossover, nyquist * 0.99)
        
        logger.debug(f"LR crossover alignment for {center_freq}Hz: "
                    f"{lower_crossover:.2f}-{higher_crossover:.2f}Hz (crossover at -6dB)")
        
        return lower_crossover, higher_crossover
    
    def _design_linkwitz_riley_bandpass(self, low_freq: float, high_freq: float, 
                                        order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Design a Linkwitz-Riley bandpass filter.
        
        Linkwitz-Riley filters are created by cascading Butterworth filters.
        A 4th order LR filter = two 2nd order Butterworth filters cascaded.
        LR filters sum flat when used as complementary pairs, preserving phase alignment.
        
        For bandpass, we cascade LR lowpass and LR highpass filters.
        
        Args:
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            order: Filter order (must be even, typically 4)
            
        Returns:
            Tuple of (b, a) filter coefficients for bandpass
        """
        if order % 2 != 0:
            raise ValueError("Linkwitz-Riley filter order must be even")
        
        nyquist = self.sample_rate / 2
        
        # Normalize frequencies
        low_norm = max(low_freq / nyquist, 1e-6)
        high_norm = min(high_freq / nyquist, 0.99)
        
        # Half order for cascading (4th order LR = two 2nd order Butterworth)
        half_order = order // 2
        
        # Design 2nd order Butterworth filters
        b_lp_bw, a_lp_bw = signal.butter(half_order, high_norm, btype='low')
        b_hp_bw, a_hp_bw = signal.butter(half_order, low_norm, btype='high')
        
        # Create LR filters by cascading Butterworth filters twice
        # 4th order LR LP = 2nd order BW LP cascaded twice
        b_lp_lr1, a_lp_lr1 = self._cascade_filters(b_lp_bw, a_lp_bw, b_lp_bw, a_lp_bw)
        # 4th order LR HP = 2nd order BW HP cascaded twice  
        b_hp_lr1, a_hp_lr1 = self._cascade_filters(b_hp_bw, a_hp_bw, b_hp_bw, a_hp_bw)
        
        # Combine LP and HP to create bandpass
        # Apply HP first, then LP
        b_band, a_band = self._cascade_filters(b_hp_lr1, a_hp_lr1, b_lp_lr1, a_lp_lr1)
        
        # Check for stability and fix if needed
        # Check if poles are inside unit circle
        poles = np.roots(a_band)
        max_pole_magnitude = np.max(np.abs(poles))
        
        if max_pole_magnitude >= 1.0:
            logger.warning(f"Linkwitz-Riley filter may be unstable (max pole magnitude: {max_pole_magnitude:.6f})")
            # Normalize to ensure stability
            # Scale poles slightly inward if needed
            if max_pole_magnitude > 1.001:
                logger.error(f"Linkwitz-Riley filter unstable for {low_freq}-{high_freq}Hz, falling back to Butterworth")
                # Fallback to Butterworth
                low_norm = max(low_freq / nyquist, 1e-6)
                high_norm = min(high_freq / nyquist, 0.99)
                return signal.butter(order, [low_norm, high_norm], btype='band')
        
        return b_band, a_band
    
    def _cascade_filters(self, b1: np.ndarray, a1: np.ndarray, 
                        b2: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cascade two filters by convolving their coefficients.
        
        Args:
            b1, a1: Coefficients of first filter
            b2, a2: Coefficients of second filter
            
        Returns:
            Tuple of (b_cascade, a_cascade) filter coefficients
        """
        # Convolve numerator and denominator polynomials
        b_cascade = np.convolve(b1, b2)
        a_cascade = np.convolve(a1, a2)
        
        return b_cascade, a_cascade
    
    def _design_lr_highpass_sos(self, crossover_freq: float, order: int = 4) -> np.ndarray:
        """Design Linkwitz-Riley high-pass filter in SOS format.
        
        LR filters are created by cascading Butterworth filters:
        - 4th order LR = two 2nd order Butterworth filters cascaded
        - At crossover frequency, LR HP is -6 dB (magnitude 0.5)
        
        Args:
            crossover_freq: Crossover frequency in Hz
            order: Filter order (must be even, typically 4)
            
        Returns:
            SOS (second-order sections) filter coefficients (Nx6 array)
        """
        if order % 2 != 0:
            raise ValueError("Linkwitz-Riley filter order must be even")
        
        nyquist = self.sample_rate / 2
        crossover_norm = max(crossover_freq / nyquist, 1e-6)
        crossover_norm = min(crossover_norm, 0.99)
        
        # Design 2nd order Butterworth HP
        half_order = order // 2
        sos_hp_bw = signal.butter(half_order, crossover_norm, btype='high', output='sos')
        
        # Cascade two 2nd order sections to get LR4
        b_hp_bw, a_hp_bw = signal.sos2tf(sos_hp_bw)
        b_hp_lr, a_hp_lr = self._cascade_filters(b_hp_bw, a_hp_bw, b_hp_bw, a_hp_bw)
        
        # Convert back to SOS format for stability
        sos_hp_lr = signal.tf2sos(b_hp_lr, a_hp_lr)
        
        return sos_hp_lr
    
    def _design_lr_lowpass_sos(self, crossover_freq: float, order: int = 4) -> np.ndarray:
        """Design Linkwitz-Riley low-pass filter in SOS format.
        
        LR filters are created by cascading Butterworth filters:
        - 4th order LR = two 2nd order Butterworth filters cascaded
        - At crossover frequency, LR LP is -6 dB (magnitude 0.5)
        
        Args:
            crossover_freq: Crossover frequency in Hz
            order: Filter order (must be even, typically 4)
            
        Returns:
            SOS (second-order sections) filter coefficients (Nx6 array)
        """
        if order % 2 != 0:
            raise ValueError("Linkwitz-Riley filter order must be even")
        
        nyquist = self.sample_rate / 2
        crossover_norm = max(crossover_freq / nyquist, 1e-6)
        crossover_norm = min(crossover_norm, 0.99)
        
        # Design 2nd order Butterworth LP
        half_order = order // 2
        sos_lp_bw = signal.butter(half_order, crossover_norm, btype='low', output='sos')
        
        # Cascade two 2nd order sections to get LR4
        b_lp_bw, a_lp_bw = signal.sos2tf(sos_lp_bw)
        b_lp_lr, a_lp_lr = self._cascade_filters(b_lp_bw, a_lp_bw, b_lp_bw, a_lp_bw)
        
        # Convert back to SOS format for stability
        sos_lp_lr = signal.tf2sos(b_lp_lr, a_lp_lr)
        
        return sos_lp_lr
    
    def create_octave_bank_cascade(self, audio_data: np.ndarray,
                                   center_frequencies: List[float] = None) -> np.ndarray:
        """Create octave bank using cascade complementary filter approach.
        
        Bands are extracted from the FULL signal using complementary bandpass filters
        designed to partition the spectrum into non-overlapping bands. All bands use
        the same signal path (full signal), preserving phase relationships and ensuring
        linear phase (via filtfilt) for crest factor preservation.
        
        Each band is extracted as: LP(lower_crossover) - LP(upper_crossover)
        This creates non-overlapping partitions that naturally sum closer to unity.
        
        This approach avoids error accumulation from sequential filtering while
        maintaining phase coherence through the use of filtfilt (zero-phase).
        
        Args:
            audio_data: Input audio signal
            center_frequencies: List of center frequencies (default: standard octaves)
            
        Returns:
            Octave bank array with original signal in first column, then extracted bands
        """
        if center_frequencies is None:
            center_frequencies = self.OCTAVE_CENTER_FREQUENCIES
        
        logger.info("Creating octave bank using cascade complementary filter approach...")
        
        sorted_freqs = sorted([f for f in center_frequencies if f > 0])
        filtered_signals = [audio_data]  # First column: full spectrum
        
        # Extract each band from the FULL signal to avoid error accumulation
        # All bands use the same signal path, preserving phase coherence
        for i in range(len(sorted_freqs)):
            fc = sorted_freqs[i]
            
            if fc >= self.sample_rate / 2:
                # Skip frequencies above Nyquist
                filtered_signals.append(np.zeros_like(audio_data))
                continue
            
            # Calculate crossover frequencies for this band
            if i > 0:
                lower_crossover = np.sqrt(sorted_freqs[i - 1] * fc)
            else:
                # First band: use ISO standard lower bound
                lower_crossover = fc / np.sqrt(2)
            
            if i < len(sorted_freqs) - 1:
                upper_crossover = np.sqrt(fc * sorted_freqs[i + 1])
            else:
                # Last band: everything above lower crossover
                cache_key_lp = f"lr_lp_{lower_crossover:.2f}"
                if cache_key_lp not in self._sos_cache:
                    sos_lp = self._design_lr_lowpass_sos(lower_crossover, order=4)
                    self._sos_cache[cache_key_lp] = sos_lp
                else:
                    sos_lp = self._sos_cache[cache_key_lp]
                
                # Extract final band: signal above lower crossover from FULL signal
                # This is the complement of LP(lower_crossover)
                low_pass_signal = signal.sosfiltfilt(sos_lp, audio_data)
                extracted_band = audio_data - low_pass_signal
                filtered_signals.append(extracted_band)
                logger.debug(f"Extracted final band {fc:.2f} Hz (f > {lower_crossover:.2f} Hz)")
                break
            
            # Extract band as frequency range between lower and upper crossover
            # Applied to FULL signal to preserve phase coherence and avoid error accumulation
            # Band = LP(lower) - LP(upper) from full signal
            
            # Design LP filters at both crossovers
            cache_key_lp_low = f"lr_lp_{lower_crossover:.2f}"
            cache_key_lp_high = f"lr_lp_{upper_crossover:.2f}"
            
            if cache_key_lp_low not in self._sos_cache:
                sos_lp_low = self._design_lr_lowpass_sos(lower_crossover, order=4)
                self._sos_cache[cache_key_lp_low] = sos_lp_low
            else:
                sos_lp_low = self._sos_cache[cache_key_lp_low]
            
            if cache_key_lp_high not in self._sos_cache:
                sos_lp_high = self._design_lr_lowpass_sos(upper_crossover, order=4)
                self._sos_cache[cache_key_lp_high] = sos_lp_high
            else:
                sos_lp_high = self._sos_cache[cache_key_lp_high]
            
            # Extract band from FULL signal: frequency range between crossovers
            # This creates a non-overlapping partition without error accumulation
            band_low = signal.sosfiltfilt(sos_lp_low, audio_data)
            band_high = signal.sosfiltfilt(sos_lp_high, audio_data)
            extracted_band = band_low - band_high
            filtered_signals.append(extracted_band)
            
            logger.debug(f"Extracted band {fc:.2f} Hz (range: {lower_crossover:.2f}-{upper_crossover:.2f} Hz)")
        
        # Stack all signals into octave bank array
        octave_bank = np.column_stack(filtered_signals)
        
        logger.info(f"Created octave bank with {octave_bank.shape[1]} bands using cascade method")
        return octave_bank

    def apply_octave_filter(self, audio_data: np.ndarray, center_freq: float, 
                           order: int = 1) -> np.ndarray:
        """Apply octave band filter to audio data.
        
        Uses second-order sections (SOS) format for numerical stability,
        especially important for long signals and low frequencies.
        
        Args:
            audio_data: Input audio signal
            center_freq: Center frequency of the octave band
            order: Filter order
            
        Returns:
            Filtered audio signal
        """
        # Use direct SOS design for better numerical stability
        # This avoids conversion errors from b,a to SOS format
        sos = self.design_octave_filter_sos(center_freq, order)
        
        # Use sosfiltfilt which is more numerically stable than filtfilt
        # It processes filters in stable second-order sections
        filtered_audio = signal.sosfiltfilt(sos, audio_data)
        
        logger.debug(f"Applied octave filter at {center_freq}Hz")
        return filtered_audio

    def _calculate_normalization_factors(self, center_frequencies: List[float]) -> dict[float, float]:
        """Calculate frequency-dependent normalization factors for each band.
        
        When octave bands overlap, their summed magnitude response can exceed unity,
        causing ripple. This method computes normalization factors so that bands
        sum to a flat frequency response (0 dB gain).
        
        Args:
            center_frequencies: List of center frequencies to analyze
            
        Returns:
            Dictionary mapping center frequency to normalization factor
        """
        if not self.normalize_overlap:
            return {freq: 1.0 for freq in center_frequencies}
        
        # Use a representative set of test frequencies for analysis
        # Focus on range where bands overlap significantly
        test_freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
        nyquist = self.sample_rate / 2
        
        # Calculate frequency response for each band
        band_responses = {}
        for fc in center_frequencies:
            if fc >= self.sample_rate / 2:
                continue
            
            # Design filter in SOS format
            sos = self.design_octave_filter_sos(fc, order=1)
            w, h = signal.sosfreqz(sos, worN=test_freqs, fs=self.sample_rate)
            band_responses[fc] = np.abs(h)
        
        # Calculate summed response at each frequency
        summed_response = np.zeros(len(test_freqs))
        for fc, response in band_responses.items():
            summed_response += response
        
        # Calculate normalization factors: scale each band so sum = 1.0
        # Use average scaling factor weighted by band's contribution
        normalization_factors = {}
        for fc, response in band_responses.items():
            # At frequencies where this band contributes, calculate what scale factor
            # is needed for the summed response to equal 1.0
            band_mask = response > 0.01  # Only consider frequencies where band has significant gain
            if np.any(band_mask):
                # Calculate scale factor: at each frequency, scale = 1.0 / summed_response
                # Use geometric mean of scale factors across the band's passband
                scale_factors = np.ones_like(test_freqs)
                scale_factors[band_mask] = 1.0 / (summed_response[band_mask] + 1e-10)
                # Use median to avoid outliers from very low summed gain
                normalization_factors[fc] = float(np.median(scale_factors[band_mask]))
            else:
                normalization_factors[fc] = 1.0
        
        logger.debug(f"Calculated normalization factors: {normalization_factors}")
        return normalization_factors

    def create_octave_bank(self, audio_data: np.ndarray, 
                          center_frequencies: List[float] = None) -> np.ndarray:
        """Create octave bank with filtered signals.
        
        Uses cascade complementary filter approach if use_cascade=True,
        otherwise uses parallel filtering approach.
        
        Args:
            audio_data: Input audio signal
            center_frequencies: List of center frequencies (default: standard octaves)
            
        Returns:
            Octave bank array with original signal in first column
        """
        if self.use_cascade:
            return self.create_octave_bank_cascade(audio_data, center_frequencies)
        
        if center_frequencies is None:
            center_frequencies = self.OCTAVE_CENTER_FREQUENCIES

        logger.info("Creating octave bank...")
        
        # OPTIMIZATION: Build list of signals and stack once instead of repeated column_stack
        # This is more efficient and reduces memory allocations
        filtered_signals = [audio_data]
        
        # Add filtered signals for each center frequency
        # When normalize_overlap is True, geometric mean crossover points are used in filter design
        for freq in center_frequencies:
            if freq < self.sample_rate / 2:  # Check Nyquist limit
                filtered_signal = self.apply_octave_filter(audio_data, freq)
                filtered_signals.append(filtered_signal)
                logger.info(f"Added octave band: {freq}Hz")
            else:
                logger.warning(f"Skipping frequency {freq}Hz (above Nyquist limit)")

        # Stack all signals into octave bank array
        octave_bank = np.column_stack(filtered_signals)
        
        logger.info(f"Octave bank created with {octave_bank.shape[1]} bands")
        return octave_bank

    def create_octave_bank_parallel(self, audio_data: np.ndarray, 
                                    center_frequencies: List[float] = None,
                                    num_workers: int = None) -> np.ndarray:
        """Create octave bank using parallel processing for improved performance.
        
        Args:
            audio_data: Input audio signal
            center_frequencies: List of center frequencies (default: standard octaves)
            num_workers: Number of parallel workers (default: min of freq count and CPU count)
            
        Returns:
            Octave bank array with original signal in first column
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        if center_frequencies is None:
            center_frequencies = self.OCTAVE_CENTER_FREQUENCIES
        
        logger.info("Creating octave bank with parallel processing...")
        
        # Determine optimal number of workers
        available_cores = multiprocessing.cpu_count()
        if num_workers is None:
            num_workers = min(len(center_frequencies), available_cores)
        
        logger.info(f"Using {num_workers} parallel workers (out of {available_cores} available cores)")
        
        # Start with original signal
        filtered_signals = [audio_data]
        
        # Filter out frequencies above Nyquist
        valid_frequencies = [f for f in center_frequencies if f < self.sample_rate / 2]
        
        if len(valid_frequencies) == 0:
            logger.warning("No valid frequencies below Nyquist limit")
            return np.column_stack(filtered_signals)
        
        # Process frequencies in parallel
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all filtering tasks
                futures = {
                    executor.submit(
                        _filter_worker_sos,
                        (audio_data, freq, self.sample_rate, 4)
                    ): freq
                    for freq in valid_frequencies
                }
                
                # Collect results as they complete
                results = {}
                for future in as_completed(futures):
                    freq = futures[future]
                    try:
                        _, filtered_signal = future.result(timeout=300)  # 5 min timeout per task
                        results[freq] = filtered_signal
                        logger.info(f"Completed octave band: {freq}Hz")
                    except Exception as e:
                        logger.error(f"Failed to process {freq}Hz: {e}")
                        # Use zeros as fallback
                        results[freq] = np.zeros_like(audio_data)
            
            # Add filtered signals in the correct order
            for freq in valid_frequencies:
                filtered_signals.append(results.get(freq, np.zeros_like(audio_data)))
                
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Falling back to sequential processing...")
            # Fallback to sequential processing
            return self.create_octave_bank(audio_data, center_frequencies)
        
        # Stack all signals into octave bank array
        octave_bank = np.column_stack(filtered_signals)
        
        logger.info(f"Octave bank created with {octave_bank.shape[1]} bands (parallel)")
        return octave_bank

    def get_octave_analysis(self, octave_bank: np.ndarray) -> dict:
        """Perform analysis on octave bank data.
        
        Args:
            octave_bank: Octave bank array
            
        Returns:
            Dictionary with analysis results
        """
        num_bands = octave_bank.shape[1]
        
        # Initialize arrays for analysis
        max_values = np.zeros(num_bands)
        max_values_db = np.zeros(num_bands)
        rms_values = np.zeros(num_bands)
        rms_values_db = np.zeros(num_bands)
        dynamic_range = np.zeros(num_bands)
        dynamic_range_db = np.zeros(num_bands)

        logger.info("Performing octave band analysis...")

        for n in range(num_bands):
            band_signal = octave_bank[:, n]
            
            # Calculate maximum signal peak
            max_values[n] = np.max(np.abs(band_signal))
            max_values_db[n] = 20 * np.log10(max_values[n]) if max_values[n] > 0 else -np.inf
            
            # Calculate RMS
            rms_values[n] = np.sqrt(np.mean(band_signal**2))
            rms_values_db[n] = 20 * np.log10(rms_values[n]) if rms_values[n] > 0 else -np.inf
            
            # Calculate dynamic range
            if max_values[n] > 0:
                dynamic_range[n] = rms_values[n] / max_values[n]
                dynamic_range_db[n] = 20 * np.log10(dynamic_range[n])
            else:
                dynamic_range[n] = 0
                dynamic_range_db[n] = -np.inf

        analysis_results = {
            "max_values": max_values,
            "max_values_db": max_values_db,
            "rms_values": rms_values,
            "rms_values_db": rms_values_db,
            "dynamic_range": dynamic_range,
            "dynamic_range_db": dynamic_range_db,
            "center_frequencies": [0] + self.OCTAVE_CENTER_FREQUENCIES[:num_bands-1]
        }

        logger.info("Octave band analysis complete")
        return analysis_results
