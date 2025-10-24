"""
Tests for the audio processor module.
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

from src.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor(sample_rate=44100)

    def test_init(self):
        """Test AudioProcessor initialization."""
        assert self.processor.sample_rate == 44100
        
        # Test with custom sample rate
        custom_processor = AudioProcessor(sample_rate=48000)
        assert custom_processor.sample_rate == 48000

    def test_load_audio_nonexistent_file(self):
        """Test loading nonexistent audio file."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_audio("nonexistent.wav")

    def test_load_audio_valid_file(self):
        """Test loading valid audio file."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Generate test audio
            sample_rate = 44100
            duration = 0.1  # 100ms
            frequency = 440  # A4 note
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Write as stereo
            stereo_data = np.column_stack([audio_data, audio_data])
            sf.write(tmp_file.name, stereo_data, sample_rate)
            
            try:
                # Load the file
                loaded_audio, sr = self.processor.load_audio(tmp_file.name)
                
                # Check results
                assert sr == 44100
                assert len(loaded_audio) == len(audio_data)
                assert np.allclose(loaded_audio, audio_data, atol=1e-6)
                
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_stereo_to_mono_already_mono(self):
        """Test stereo_to_mono with already mono audio."""
        mono_audio = np.array([0.1, 0.2, 0.3, 0.4])
        result = self.processor.stereo_to_mono(mono_audio)
        np.testing.assert_array_equal(result, mono_audio)

    def test_stereo_to_mono_conversion(self):
        """Test stereo_to_mono conversion."""
        stereo_audio = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        expected_mono = np.array([0.15, 0.35, 0.55])  # Average of channels
        result = self.processor.stereo_to_mono(stereo_audio)
        np.testing.assert_array_equal(result, expected_mono)

    def test_normalize_audio(self):
        """Test audio normalization."""
        # Test with audio that needs normalization
        audio_data = np.array([0.5, -0.3, 0.8, -0.2])
        normalized = self.processor.normalize_audio(audio_data)
        
        # Check that max absolute value is 1.0
        assert np.max(np.abs(normalized)) == 1.0
        
        # Test with already normalized audio
        already_normalized = np.array([0.5, -0.3, 0.8, -0.2])
        result = self.processor.normalize_audio(already_normalized)
        np.testing.assert_array_equal(result, already_normalized)

    def test_get_audio_info(self):
        """Test getting audio information."""
        # Create test audio
        sample_rate = 44100
        duration = 1.0
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        info = self.processor.get_audio_info(audio_data, sample_rate)
        
        # Check basic info
        assert info["duration_seconds"] == duration
        assert info["sample_rate"] == sample_rate
        assert info["channels"] == 1
        assert info["samples"] == len(audio_data)
        
        # Check calculated values
        assert info["max_amplitude"] == 0.5
        assert abs(info["rms"] - 0.5/np.sqrt(2)) < 1e-6  # RMS of sine wave
