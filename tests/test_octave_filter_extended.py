"""
Extended tests for octave filter module.

Tests filter design, caching, error handling, and advanced features.
"""

import pytest
import numpy as np
from pathlib import Path

from src.octave_filter import OctaveBandFilter, _filter_worker_sos


class TestOctaveFilterExtended:
    """Extended test cases for OctaveBandFilter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_rate = 44100
        self.filter = OctaveBandFilter(sample_rate=self.sample_rate)

    def test_design_octave_filter_caching(self):
        """Test that filter design results are cached."""
        center_freq = 1000.0
        
        # First call - should design filter
        b1, a1 = self.filter.design_octave_filter(center_freq)
        
        # Second call - should return cached result
        b2, a2 = self.filter.design_octave_filter(center_freq)
        
        # Results should be identical (same object reference)
        assert b1 is b2
        assert a1 is a2
        
        # Cache should contain the filter
        cache_key = (center_freq, 1, self.sample_rate, False)
        assert cache_key in self.filter._filter_cache

    def test_design_octave_filter_invalid_order(self):
        """Test that invalid filter order raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported filter order"):
            self.filter.design_octave_filter(1000.0, order=2)

    def test_design_octave_filter_third_octave(self):
        """Test third-octave filter design."""
        center_freq = 1000.0
        b, a = self.filter.design_octave_filter(center_freq, order=3)
        
        assert b is not None
        assert a is not None
        assert len(b) > 0
        assert len(a) > 0

    def test_design_octave_filter_sos_caching(self):
        """Test that SOS filter design results are cached (when using LR)."""
        # Use Linkwitz-Riley to test caching (LR uses cache)
        filter_lr = OctaveBandFilter(
            sample_rate=self.sample_rate,
            use_linkwitz_riley=True
        )
        
        center_freq = 1000.0
        
        # First call
        sos1 = filter_lr.design_octave_filter_sos(center_freq)
        
        # Second call - should return cached result (if LR caches)
        sos2 = filter_lr.design_octave_filter_sos(center_freq)
        
        # Results should be identical
        np.testing.assert_array_equal(sos1, sos2)

    def test_design_octave_filter_sos_invalid_order(self):
        """Test that invalid order in SOS design raises ValueError for Butterworth."""
        # Create filter with normalize_overlap=False to test else branch
        filter_no_norm = OctaveBandFilter(
            sample_rate=self.sample_rate,
            normalize_overlap=False
        )
        
        # Test standard Butterworth path (raises error in else branch)
        with pytest.raises(ValueError, match="Unsupported filter order"):
            filter_no_norm.design_octave_filter_sos(1000.0, order=2)
        
        # Test with normalize_overlap enabled and freq in standard list
        # For middle bands, order is not checked in normalize branch
        # So test with first band (16Hz) which checks order
        filter_norm = OctaveBandFilter(
            sample_rate=self.sample_rate,
            normalize_overlap=True
        )
        
        # Use first frequency in OCTAVE_CENTER_FREQUENCIES to test normalize branch error
        with pytest.raises(ValueError, match="Unsupported filter order"):
            filter_norm.design_octave_filter_sos(16.0, order=2)  # First band checks order
        
        # Note: Linkwitz-Riley ignores order parameter, so it won't raise error
        # This is expected behavior - LR always uses 4th order filters

    def test_design_octave_filter_linkwitz_riley(self):
        """Test Linkwitz-Riley filter design."""
        filter_lr = OctaveBandFilter(
            sample_rate=self.sample_rate,
            use_linkwitz_riley=True
        )
        
        center_freq = 1000.0
        b, a = filter_lr.design_octave_filter(center_freq)
        
        assert b is not None
        assert a is not None

    def test_design_octave_filter_sos_linkwitz_riley(self):
        """Test Linkwitz-Riley SOS filter design."""
        filter_lr = OctaveBandFilter(
            sample_rate=self.sample_rate,
            use_linkwitz_riley=True
        )
        
        center_freq = 1000.0
        sos = filter_lr.design_octave_filter_sos(center_freq)
        
        assert sos is not None
        assert sos.ndim == 2
        assert sos.shape[1] == 6  # SOS format is Nx6

    def test_design_octave_filter_sos_normalize_overlap(self):
        """Test SOS filter design with overlap normalization."""
        filter_norm = OctaveBandFilter(
            sample_rate=self.sample_rate,
            normalize_overlap=True
        )
        
        center_freq = 1000.0
        sos = filter_norm.design_octave_filter_sos(center_freq)
        
        assert sos is not None
        assert sos.ndim == 2

    def test_apply_octave_filter(self):
        """Test applying octave filter to audio data."""
        # Create test signal
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave
        
        center_freq = 1000.0
        filtered = self.filter.apply_octave_filter(audio_data, center_freq)
        
        assert filtered is not None
        assert len(filtered) == len(audio_data)
        assert filtered.dtype == audio_data.dtype

    def test_apply_octave_filter_sos(self):
        """Test applying octave filter (always uses SOS internally)."""
        # Create test signal
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        center_freq = 1000.0
        # apply_octave_filter always uses SOS internally
        filtered = self.filter.apply_octave_filter(audio_data, center_freq)
        
        assert filtered is not None
        assert len(filtered) == len(audio_data)

    def test_create_octave_bank(self):
        """Test creating octave bank with all bands."""
        # Create test signal
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        octave_bank = self.filter.create_octave_bank(audio_data)
        
        assert octave_bank is not None
        assert octave_bank.ndim == 2
        assert octave_bank.shape[0] == len(audio_data)
        # Should have bands + full spectrum
        assert octave_bank.shape[1] == len(self.filter.OCTAVE_CENTER_FREQUENCIES) + 1

    def test_create_octave_bank_custom_frequencies(self):
        """Test creating octave bank with custom frequencies."""
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        custom_freqs = [125.0, 250.0, 500.0, 1000.0]
        octave_bank = self.filter.create_octave_bank(audio_data, center_frequencies=custom_freqs)
        
        assert octave_bank.shape[1] == len(custom_freqs) + 1  # +1 for full spectrum

    def test_create_octave_bank_cascade_mode(self):
        """Test creating octave bank in cascade mode."""
        filter_cascade = OctaveBandFilter(
            sample_rate=self.sample_rate,
            use_linkwitz_riley=True,
            use_cascade=True
        )
        
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        octave_bank = filter_cascade.create_octave_bank(audio_data)
        
        assert octave_bank is not None
        assert octave_bank.ndim == 2

    def test_create_octave_bank_parallel(self):
        """Test creating octave bank with parallel processing."""
        filter_parallel = OctaveBandFilter(sample_rate=self.sample_rate)
        
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        octave_bank = filter_parallel.create_octave_bank_parallel(audio_data)
        
        assert octave_bank is not None
        assert octave_bank.ndim == 2

    def test_get_octave_analysis(self):
        """Test getting octave analysis from octave bank."""
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        octave_bank = self.filter.create_octave_bank(audio_data)
        analysis = self.filter.get_octave_analysis(octave_bank)
        
        # Check expected keys
        assert "max_values" in analysis
        assert "max_values_db" in analysis
        assert "rms_values" in analysis
        assert "rms_values_db" in analysis
        assert "dynamic_range" in analysis
        assert "dynamic_range_db" in analysis
        assert "center_frequencies" in analysis
        
        # Check array lengths match number of bands
        num_bands = octave_bank.shape[1]
        assert len(analysis["max_values"]) == num_bands
        assert len(analysis["rms_values"]) == num_bands

    def test_cascade_mode_auto_enables_linkwitz_riley(self):
        """Test that cascade mode automatically enables Linkwitz-Riley."""
        filter_cascade = OctaveBandFilter(
            sample_rate=self.sample_rate,
            use_linkwitz_riley=False,
            use_cascade=True
        )
        
        # Should automatically enable Linkwitz-Riley
        assert filter_cascade.use_linkwitz_riley is True

    def test_filter_worker_sos(self):
        """Test the filter worker function for parallel processing."""
        duration = 0.1
        audio_data = np.random.randn(int(self.sample_rate * duration))
        center_freq = 1000.0
        
        args = (audio_data, center_freq, self.sample_rate, 4)
        result_freq, filtered = _filter_worker_sos(args)
        
        assert result_freq == center_freq
        assert filtered is not None
        assert len(filtered) == len(audio_data)

    def test_filter_worker_sos_error_handling(self):
        """Test filter worker handles errors gracefully."""
        # Invalid frequency that will cause error
        audio_data = np.random.randn(100)
        center_freq = -100.0  # Invalid frequency
        
        args = (audio_data, center_freq, self.sample_rate, 4)
        result_freq, filtered = _filter_worker_sos(args)
        
        # Should return zeros on error
        assert result_freq == center_freq
        assert np.all(filtered == 0)

    def test_frequency_edge_cases(self):
        """Test filter design with edge case frequencies."""
        # Very low frequency
        b1, a1 = self.filter.design_octave_filter(16.0)
        assert b1 is not None
        
        # Very high frequency (near Nyquist)
        nyquist = self.sample_rate / 2
        high_freq = nyquist * 0.9
        b2, a2 = self.filter.design_octave_filter(high_freq)
        assert b2 is not None
        
        # Frequency at Nyquist limit
        limit_freq = nyquist * 0.95
        b3, a3 = self.filter.design_octave_filter(limit_freq)
        assert b3 is not None

    def test_normalize_overlap_calculation(self):
        """Test normalization factor calculation."""
        filter_norm = OctaveBandFilter(
            sample_rate=self.sample_rate,
            normalize_overlap=True
        )
        
        center_freqs = [125.0, 250.0, 500.0, 1000.0]
        factors = filter_norm._calculate_normalization_factors(center_freqs)
        
        assert isinstance(factors, dict)
        # All factors should be positive
        for freq, factor in factors.items():
            assert factor > 0

    def test_cascade_filters(self):
        """Test cascading two filters."""
        b1, a1 = self.filter.design_octave_filter(500.0)
        b2, a2 = self.filter.design_octave_filter(1000.0)
        
        b_cascade, a_cascade = self.filter._cascade_filters(b1, a1, b2, a2)
        
        assert b_cascade is not None
        assert a_cascade is not None

    def test_linkwitz_riley_highpass_lowpass(self):
        """Test Linkwitz-Riley highpass and lowpass design."""
        filter_lr = OctaveBandFilter(
            sample_rate=self.sample_rate,
            use_linkwitz_riley=True
        )
        
        crossover = 1000.0
        
        # Test highpass
        sos_hp = filter_lr._design_lr_highpass_sos(crossover)
        assert sos_hp is not None
        assert sos_hp.ndim == 2
        
        # Test lowpass
        sos_lp = filter_lr._design_lr_lowpass_sos(crossover)
        assert sos_lp is not None
        assert sos_lp.ndim == 2

    def test_calculate_complementary_crossovers(self):
        """Test complementary crossover calculation."""
        filter_lr = OctaveBandFilter(
            sample_rate=self.sample_rate,
            use_linkwitz_riley=True
        )
        
        center_freq = 1000.0
        low_crossover, high_crossover = filter_lr._calculate_complementary_crossovers(center_freq)
        
        assert low_crossover < center_freq
        assert high_crossover > center_freq
        assert low_crossover > 0
        assert high_crossover < self.sample_rate / 2

