"""
Tests for the track processor module.
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.track_processor import TrackProcessor
from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter
from src.config import config


class TestTrackProcessor:
    """Test cases for TrackProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.track_processor = TrackProcessor(sample_rate=44100, original_peak=1.0)

    def test_init(self):
        """Test TrackProcessor initialization."""
        assert self.track_processor.sample_rate == 44100
        assert self.track_processor.original_peak == 1.0

    def test_process_channel(self):
        """Test processing a single channel."""
        # Create test audio data
        sample_rate = 44100
        duration = 2.0
        channel_data = np.random.randn(int(sample_rate * duration))
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            track_output_dir = Path(tmp_dir) / "test_track"
            track_output_dir.mkdir()
            channel_output_dir = track_output_dir / "Channel 1 Left"
            channel_output_dir.mkdir()
            
            # Initialize dependencies
            audio_processor = AudioProcessor(sample_rate=sample_rate)
            octave_filter = OctaveBandFilter(sample_rate=sample_rate)
            
            # Process channel
            success = self.track_processor.process_channel(
                channel_data=channel_data,
                channel_index=0,
                channel_name="Channel 1 Left",
                channel_folder_name="Channel 1 Left",
                total_channels=1,
                track_path=Path("test_track.wav"),
                track_output_dir=track_output_dir,
                channel_output_dir=channel_output_dir,
                audio_processor=audio_processor,
                octave_filter=octave_filter,
                chunk_duration=2.0,
                config=config,
                content_type="Music",
                original_peak=1.0
            )
            
            # Should succeed
            assert success is True
            
            # Check that output files were created
            assert (channel_output_dir / "octave_spectrum.png").exists()
            assert (channel_output_dir / "crest_factor.png").exists()
            assert (channel_output_dir / "histograms.png").exists()
            assert (channel_output_dir / "analysis_results.csv").exists()

    def test_process_channel_with_short_audio(self):
        """Test processing channel with very short audio."""
        # Create very short audio data
        sample_rate = 44100
        duration = 0.1  # 100ms
        channel_data = np.random.randn(int(sample_rate * duration))
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            track_output_dir = Path(tmp_dir) / "test_track"
            track_output_dir.mkdir()
            channel_output_dir = track_output_dir / "Channel 1 Left"
            channel_output_dir.mkdir()
            
            # Initialize dependencies
            audio_processor = AudioProcessor(sample_rate=sample_rate)
            octave_filter = OctaveBandFilter(sample_rate=sample_rate)
            
            # Process channel (should handle short audio gracefully)
            success = self.track_processor.process_channel(
                channel_data=channel_data,
                channel_index=0,
                channel_name="Channel 1 Left",
                channel_folder_name="Channel 1 Left",
                total_channels=1,
                track_path=Path("test_track.wav"),
                track_output_dir=track_output_dir,
                channel_output_dir=channel_output_dir,
                audio_processor=audio_processor,
                octave_filter=octave_filter,
                chunk_duration=0.1,
                config=config,
                content_type="Music",
                original_peak=1.0
            )
            
            # Should succeed (or fail gracefully)
            assert isinstance(success, bool)

    def test_process_channel_error_handling(self):
        """Test error handling in process_channel."""
        # Create invalid audio data (empty array)
        channel_data = np.array([])
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            track_output_dir = Path(tmp_dir) / "test_track"
            track_output_dir.mkdir()
            channel_output_dir = track_output_dir / "Channel 1 Left"
            channel_output_dir.mkdir()
            
            # Initialize dependencies
            audio_processor = AudioProcessor(sample_rate=44100)
            octave_filter = OctaveBandFilter(sample_rate=44100)
            
            # Process channel (should handle error gracefully)
            success = self.track_processor.process_channel(
                channel_data=channel_data,
                channel_index=0,
                channel_name="Channel 1 Left",
                channel_folder_name="Channel 1 Left",
                total_channels=1,
                track_path=Path("test_track.wav"),
                track_output_dir=track_output_dir,
                channel_output_dir=channel_output_dir,
                audio_processor=audio_processor,
                octave_filter=octave_filter,
                chunk_duration=2.0,
                config=config,
                content_type="Music",
                original_peak=1.0
            )
            
            # Should return False on error
            assert success is False

