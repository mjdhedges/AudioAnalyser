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


def _filter_worker(args: Tuple[np.ndarray, float, int, int]) -> Tuple[float, np.ndarray]:
    """Worker function for parallel octave band filtering.
    
    This module-level function is used for multiprocessing as it can be pickled.
    
    Args:
        args: Tuple of (audio_data, center_freq, sample_rate, filter_order)
        
    Returns:
        Tuple of (center_freq, filtered_signal)
    """
    audio_data, center_freq, sample_rate, filter_order = args
    
    try:
        # Calculate bandwidth for octave band
        bandwidth = center_freq / np.sqrt(2)
        
        # Calculate filter frequencies
        low_freq = center_freq - bandwidth / 2
        high_freq = center_freq + bandwidth / 2
        
        # Ensure frequencies are within valid range
        nyquist = sample_rate / 2
        low_freq = max(low_freq, 1.0)
        high_freq = min(high_freq, nyquist * 0.99)
        
        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Adjust filter order for low frequencies
        if low_norm < 0.01:
            actual_filter_order = 2
        else:
            actual_filter_order = filter_order
        
        # Ensure normalized frequencies are within valid range
        low_norm = max(low_norm, 1e-6)
        high_norm = min(high_norm, 0.99)
        
        # Design and apply filter
        b, a = signal.butter(actual_filter_order, [low_norm, high_norm], btype='band')
        filtered_signal = signal.filtfilt(b, a, audio_data)
        
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

    def __init__(self, sample_rate: int = 44100) -> None:
        """Initialize the octave band filter.
        
        Args:
            sample_rate: Sample rate of the audio signal
        """
        self.sample_rate = sample_rate
        self._filter_cache = {}  # Cache for filter coefficients (b, a)

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
        cache_key = (center_freq, order, self.sample_rate)
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

        # Design Butterworth bandpass filter
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Check for very low frequencies that might cause numerical issues
        if low_norm < 0.01:  # Less than 1% of Nyquist frequency
            # Use lower order filter for very low frequencies to avoid numerical instability
            filter_order = 2
            logger.debug(f"Using 2nd order filter for low frequency band: {center_freq}Hz")
        else:
            # Use 4th order Butterworth filter for good performance
            filter_order = 4
            
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

    def apply_octave_filter(self, audio_data: np.ndarray, center_freq: float, 
                           order: int = 1) -> np.ndarray:
        """Apply octave band filter to audio data.
        
        Args:
            audio_data: Input audio signal
            center_freq: Center frequency of the octave band
            order: Filter order
            
        Returns:
            Filtered audio signal
        """
        b, a = self.design_octave_filter(center_freq, order)
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        logger.debug(f"Applied octave filter at {center_freq}Hz")
        return filtered_audio

    def create_octave_bank(self, audio_data: np.ndarray, 
                          center_frequencies: List[float] = None) -> np.ndarray:
        """Create octave bank with filtered signals.
        
        Args:
            audio_data: Input audio signal
            center_frequencies: List of center frequencies (default: standard octaves)
            
        Returns:
            Octave bank array with original signal in first column
        """
        if center_frequencies is None:
            center_frequencies = self.OCTAVE_CENTER_FREQUENCIES

        logger.info("Creating octave bank...")
        
        # OPTIMIZATION: Build list of signals and stack once instead of repeated column_stack
        # This is more efficient and reduces memory allocations
        filtered_signals = [audio_data]
        
        # Add filtered signals for each center frequency
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
                        _filter_worker,
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
