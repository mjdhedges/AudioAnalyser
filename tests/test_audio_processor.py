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
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Generate test audio
            sample_rate = 44100
            duration = 0.1  # 100ms
            frequency = 440  # A4 note
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Write as mono
            sf.write(tmp_path, audio_data, sample_rate)
            
            # Load the file
            loaded_audio, sr = self.processor.load_audio(tmp_path)
            
            # Check results
            assert sr == 44100
            assert len(loaded_audio) == len(audio_data)
            assert np.allclose(loaded_audio, audio_data, atol=1e-4)
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)

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
        np.testing.assert_allclose(result, expected_mono, atol=1e-10)

    def test_normalize_audio(self):
        """Test audio normalization."""
        # Test with audio that needs normalization
        audio_data = np.array([0.5, -0.3, 0.8, -0.2])
        normalized = self.processor.normalize_audio(audio_data)
        
        # Check that max absolute value is 1.0
        assert np.max(np.abs(normalized)) == 1.0
        
        # Test with already normalized audio (max abs value = 1.0)
        already_normalized = np.array([0.5, -0.3, 1.0, -0.2])
        result = self.processor.normalize_audio(already_normalized)
        np.testing.assert_allclose(result, already_normalized, atol=1e-10)

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
        assert info["channel_layout"] == "mono"
        assert info["samples"] == len(audio_data)
        
        # Check calculated values (allow for floating point precision)
        assert abs(info["max_amplitude"] - 0.5) < 1e-6
        assert abs(info["rms"] - 0.5/np.sqrt(2)) < 1e-6  # RMS of sine wave

    def test_load_audio_multichannel(self):
        """Test loading multi-channel audio file."""
        # Create temporary multi-channel audio file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            sample_rate = 44100
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create 5.1 surround (6 channels)
            ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)  # FL
            ch2 = 0.4 * np.sin(2 * np.pi * 440 * t)  # FR
            ch3 = 0.3 * np.sin(2 * np.pi * 440 * t)  # FC
            ch4 = 0.2 * np.sin(2 * np.pi * 440 * t)  # LFE
            ch5 = 0.25 * np.sin(2 * np.pi * 440 * t)  # SL
            ch6 = 0.25 * np.sin(2 * np.pi * 440 * t)  # SR
            
            multichannel_data = np.column_stack([ch1, ch2, ch3, ch4, ch5, ch6])
            sf.write(tmp_path, multichannel_data, sample_rate)
            
            # Load the file
            loaded_audio, sr = self.processor.load_audio(tmp_path)
            
            # Check results - should preserve multi-channel shape
            assert sr == 44100
            assert loaded_audio.ndim == 2
            assert loaded_audio.shape[0] == len(ch1)  # samples
            assert loaded_audio.shape[1] == 6  # channels
            
            # Check channel data is preserved (allow for resampling/precision differences)
            assert np.allclose(loaded_audio[:, 0], ch1, atol=1e-4)
            assert np.allclose(loaded_audio[:, 1], ch2, atol=1e-4)
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_get_audio_info_stereo(self):
        """Test getting audio information for stereo audio."""
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create stereo audio
        left_channel = 0.5 * np.sin(2 * np.pi * 440 * t)
        right_channel = 0.3 * np.sin(2 * np.pi * 440 * t)
        stereo_audio = np.column_stack([left_channel, right_channel])
        
        info = self.processor.get_audio_info(stereo_audio, sample_rate)
        
        assert info["channels"] == 2
        assert info["channel_layout"] == "stereo"
        assert info["duration_seconds"] == duration
        assert info["sample_rate"] == sample_rate

    def test_get_audio_info_multichannel(self):
        """Test getting audio information for multi-channel audio."""
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create 5.1 surround (6 channels)
        multichannel_audio = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.4 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 440 * t),
            0.2 * np.sin(2 * np.pi * 440 * t),
            0.25 * np.sin(2 * np.pi * 440 * t),
            0.25 * np.sin(2 * np.pi * 440 * t),
        ])
        
        info = self.processor.get_audio_info(multichannel_audio, sample_rate)
        
        assert info["channels"] == 6
        assert info["channel_layout"] == "multi-channel"
        assert info["duration_seconds"] == duration

    def test_extract_channels_mono(self):
        """Test extracting channels from mono audio."""
        mono_audio = np.array([0.1, 0.2, 0.3, 0.4])
        channels = self.processor.extract_channels(mono_audio)
        
        assert len(channels) == 1
        channel_data, channel_idx = channels[0]
        assert channel_idx == 0
        np.testing.assert_array_equal(channel_data, mono_audio)

    def test_extract_channels_stereo(self):
        """Test extracting channels from stereo audio."""
        stereo_audio = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        channels = self.processor.extract_channels(stereo_audio)
        
        assert len(channels) == 2
        
        # Check first channel (left)
        ch0_data, ch0_idx = channels[0]
        assert ch0_idx == 0
        np.testing.assert_array_equal(ch0_data, stereo_audio[:, 0])
        
        # Check second channel (right)
        ch1_data, ch1_idx = channels[1]
        assert ch1_idx == 1
        np.testing.assert_array_equal(ch1_data, stereo_audio[:, 1])

    def test_extract_channels_multichannel(self):
        """Test extracting channels from multi-channel audio."""
        # Create 5.1 surround (6 channels)
        multichannel_audio = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        ])
        channels = self.processor.extract_channels(multichannel_audio)
        
        assert len(channels) == 6
        
        # Check each channel
        for i, (channel_data, channel_idx) in enumerate(channels):
            assert channel_idx == i
            np.testing.assert_array_equal(channel_data, multichannel_audio[:, i])

    def test_normalize_audio_multichannel(self):
        """Test normalizing multi-channel audio (global normalization)."""
        # Create multi-channel audio with different channel levels
        multichannel_audio = np.array([
            [0.5, 0.3, 0.2],
            [0.8, 0.4, 0.1],
            [0.2, 0.6, 0.9]  # Max is 0.9
        ])
        
        normalized = self.processor.normalize_audio(multichannel_audio)
        
        # Check that max absolute value across all channels is 1.0
        assert np.max(np.abs(normalized)) == 1.0
        
        # Check that relative channel levels are preserved
        original_ratio = multichannel_audio[1, 0] / multichannel_audio[1, 1]
        normalized_ratio = normalized[1, 0] / normalized[1, 1]
        assert abs(original_ratio - normalized_ratio) < 1e-10

    def test_is_mkv_file(self):
        """Test MKV file detection."""
        assert self.processor._is_mkv_file(Path("test.mkv")) is True
        assert self.processor._is_mkv_file(Path("test.MKV")) is True
        assert self.processor._is_mkv_file(Path("test.wav")) is False
        assert self.processor._is_mkv_file(Path("test.flac")) is False

    def test_mkv_support_enabled(self):
        """Test that MKV support can be enabled/disabled."""
        processor_with_mkv = AudioProcessor(sample_rate=44100, enable_mkv_support=True)
        assert processor_with_mkv.enable_mkv_support is True
        
        processor_without_mkv = AudioProcessor(sample_rate=44100, enable_mkv_support=False)
        assert processor_without_mkv.enable_mkv_support is False
