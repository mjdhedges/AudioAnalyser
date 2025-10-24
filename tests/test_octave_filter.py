"""
Tests for the octave filter module.
"""

import pytest
import numpy as np

from src.octave_filter import OctaveBandFilter


class TestOctaveBandFilter:
    """Test cases for OctaveBandFilter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.filter = OctaveBandFilter(sample_rate=44100)

    def test_init(self):
        """Test OctaveBandFilter initialization."""
        assert self.filter.sample_rate == 44100
        assert len(self.filter.OCTAVE_CENTER_FREQUENCIES) == 10

    def test_design_octave_filter(self):
        """Test octave filter design."""
        center_freq = 1000.0
        b, a = self.filter.design_octave_filter(center_freq)
        
        # Check that coefficients are returned
        assert len(b) > 0
        assert len(a) > 0
        assert len(b) == len(a)

    def test_design_octave_filter_invalid_order(self):
        """Test octave filter design with invalid order."""
        with pytest.raises(ValueError):
            self.filter.design_octave_filter(1000.0, order=2)

    def test_apply_octave_filter(self):
        """Test applying octave filter."""
        # Create test signal
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_signal = np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave
        
        # Apply filter
        filtered_signal = self.filter.apply_octave_filter(test_signal, 1000.0)
        
        # Check that filtered signal has same length
        assert len(filtered_signal) == len(test_signal)
        assert isinstance(filtered_signal, np.ndarray)

    def test_create_octave_bank(self):
        """Test creating octave bank."""
        # Create test signal
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_signal = np.sin(2 * np.pi * 1000 * t)
        
        # Create octave bank
        octave_bank = self.filter.create_octave_bank(test_signal)
        
        # Check dimensions
        assert octave_bank.shape[0] == len(test_signal)
        assert octave_bank.shape[1] == 11  # Original + 10 octave bands
        
        # Check that first column is original signal
        np.testing.assert_array_equal(octave_bank[:, 0], test_signal)

    def test_create_octave_bank_custom_frequencies(self):
        """Test creating octave bank with custom frequencies."""
        # Create test signal
        test_signal = np.random.randn(1000)
        custom_frequencies = [125, 250, 500]
        
        # Create octave bank
        octave_bank = self.filter.create_octave_bank(test_signal, custom_frequencies)
        
        # Check dimensions
        assert octave_bank.shape[0] == len(test_signal)
        assert octave_bank.shape[1] == 4  # Original + 3 custom bands

    def test_get_octave_analysis(self):
        """Test octave band analysis."""
        # Create test octave bank
        test_signal = np.random.randn(1000)
        octave_bank = self.filter.create_octave_bank(test_signal)
        
        # Perform analysis
        analysis = self.filter.get_octave_analysis(octave_bank)
        
        # Check analysis results
        assert "max_values" in analysis
        assert "max_values_db" in analysis
        assert "rms_values" in analysis
        assert "rms_values_db" in analysis
        assert "dynamic_range" in analysis
        assert "dynamic_range_db" in analysis
        assert "center_frequencies" in analysis
        
        # Check array lengths
        num_bands = octave_bank.shape[1]
        assert len(analysis["max_values"]) == num_bands
        assert len(analysis["rms_values"]) == num_bands
        assert len(analysis["dynamic_range"]) == num_bands
