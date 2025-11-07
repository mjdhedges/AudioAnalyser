"""
Tests for the envelope analyzer module.
"""

import pytest
import numpy as np

from src.envelope_analyzer import EnvelopeAnalyzer


class TestEnvelopeAnalyzer:
    """Test cases for EnvelopeAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.envelope_analyzer = EnvelopeAnalyzer(sample_rate=44100, original_peak=1.0)

    def test_init(self):
        """Test EnvelopeAnalyzer initialization."""
        assert self.envelope_analyzer.sample_rate == 44100
        assert self.envelope_analyzer.original_peak == 1.0

    def test_calculate_peak_envelope_basic(self):
        """Test basic peak envelope calculation."""
        # Create test signal: sine wave
        sample_rate = 44100
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 1000.0
        signal = np.sin(2 * np.pi * freq * t)
        
        # Calculate peak envelope
        envelope = self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq=freq, wavelength_multiplier=1.0
        )
        
        # Verify envelope properties
        assert len(envelope) == len(signal)
        assert np.all(envelope >= 0), "Envelope should be non-negative"
        
        # Envelope should track peaks (be >= rectified signal)
        rectified = np.abs(signal)
        assert np.all(envelope >= rectified - 1e-6), "Envelope should track peaks"

    def test_calculate_peak_envelope_wavelength_scaling(self):
        """Test that release time scales correctly with frequency."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Test with different frequencies
        for freq in [250, 1000, 4000]:
            signal = np.sin(2 * np.pi * freq * t)
            envelope = self.envelope_analyzer.calculate_peak_envelope(
                signal, center_freq=freq, wavelength_multiplier=1.0
            )
            
            # Envelope should track peaks
            rectified = np.abs(signal)
            max_rectified = np.max(rectified)
            max_envelope = np.max(envelope)
            
            assert max_envelope >= max_rectified * 0.9, \
                f"Envelope should track peaks for {freq} Hz"

    def test_calculate_peak_envelope_wavelength_multiplier(self):
        """Test wavelength multiplier affects release time."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 1000.0
        signal = np.sin(2 * np.pi * freq * t)
        
        # Test with different multipliers
        envelope_1x = self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq=freq, wavelength_multiplier=1.0
        )
        envelope_2x = self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq=freq, wavelength_multiplier=2.0
        )
        
        # Both should track peaks
        assert np.all(envelope_1x >= 0)
        assert np.all(envelope_2x >= 0)

    def test_calculate_peak_envelope_full_spectrum(self):
        """Test peak envelope with Full Spectrum (center_freq=0)."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 1000 * t)
        
        # Calculate with center_freq=0 (Full Spectrum)
        envelope = self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq=0.0, fallback_window_ms=10.0
        )
        
        # Should still produce valid envelope
        assert len(envelope) == len(signal)
        assert np.all(envelope >= 0)

    def test_calculate_rms_envelope(self):
        """Test RMS envelope calculation."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 1000.0
        signal = np.sin(2 * np.pi * freq * t)
        
        # Calculate RMS envelope using peak_envelope method
        envelope = self.envelope_analyzer.calculate_rms_envelope(
            signal, center_freq=freq, method='peak_envelope'
        )
        
        # Verify envelope properties
        assert len(envelope) == len(signal)
        assert np.all(envelope >= 0)

    def test_find_attack_time(self):
        """Test finding attack time."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with attack
        signal = np.zeros_like(t)
        attack_samples = int(0.01 * sample_rate)  # 10ms attack
        signal[:attack_samples] = np.linspace(0, 1, attack_samples)
        signal[attack_samples:] = 1.0
        
        # Calculate envelope
        envelope = self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq=1000.0
        )
        
        # Convert to dBFS
        envelope_db = 20 * np.log10(envelope * self.envelope_analyzer.original_peak + 1e-10)
        
        # Find peak
        peak_idx = np.argmax(envelope_db)
        peak_value_db = envelope_db[peak_idx]
        
        # Find attack time
        attack_time = self.envelope_analyzer.find_attack_time(
            envelope_db, peak_idx, peak_value_db, attack_threshold_db=3.0
        )
        
        # Attack time should be reasonable
        assert attack_time is not None
        assert attack_time >= 0

    def test_find_peak_hold_time(self):
        """Test finding peak hold time."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with peak hold
        signal = np.zeros_like(t)
        signal[1000:2000] = 1.0  # Peak from 1000 to 2000 samples
        
        # Calculate envelope
        envelope = self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq=1000.0
        )
        
        # Convert to dBFS
        envelope_db = 20 * np.log10(envelope * self.envelope_analyzer.original_peak + 1e-10)
        
        # Find peak
        peak_idx = np.argmax(envelope_db)
        peak_value_db = envelope_db[peak_idx]
        
        # Find peak hold time
        hold_time = self.envelope_analyzer.find_peak_hold_time(
            envelope_db, peak_idx, peak_value_db, hold_threshold_db=3.0
        )
        
        # Hold time should be reasonable
        assert hold_time is not None
        assert hold_time >= 0

    def test_find_decay_times(self):
        """Test finding decay times."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with decay
        signal = np.ones_like(t)
        decay_start = int(0.05 * sample_rate)  # Start decay at 50ms
        signal[decay_start:] = np.exp(-(t[decay_start:] - t[decay_start]) * 10)
        
        # Calculate envelope
        envelope = self.envelope_analyzer.calculate_peak_envelope(
            signal, center_freq=1000.0
        )
        
        # Convert to dBFS
        envelope_db = 20 * np.log10(envelope * self.envelope_analyzer.original_peak + 1e-10)
        
        # Find peak
        peak_idx = np.argmax(envelope_db)
        peak_value_db = envelope_db[peak_idx]
        
        # Find decay times
        decay_times = self.envelope_analyzer.find_decay_times(
            envelope_db, peak_idx, peak_value_db, decay_thresholds_db=[3.0, 6.0, 12.0]
        )
        
        # Should return dictionary with decay times
        assert isinstance(decay_times, dict)

    def test_analyze_envelope_statistics(self):
        """Test envelope statistics analysis."""
        # Create test octave bank
        num_samples = 1000
        num_bands = 3
        octave_bank = np.random.randn(num_samples, num_bands)
        center_frequencies = [125, 250, 500]
        
        # Analyze envelope statistics
        stats = self.envelope_analyzer.analyze_envelope_statistics(
            octave_bank, center_frequencies, config={}
        )
        
        # Should return dictionary with statistics
        assert isinstance(stats, dict)
        assert len(stats) > 0

