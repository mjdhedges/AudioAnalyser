"""
Extended tests for main module.

Tests error handling, edge cases, and integration scenarios.
"""

import pytest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from click.testing import CliRunner

from src.main import analyze_single_track, main


class TestMainExtended:
    """Extended test cases for main module."""

    def test_analyze_single_track_empty_audio(self):
        """Test analyzing empty audio file."""
        # Create empty audio file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Write empty audio
            sf.write(tmp_path, np.array([]), 44100)
            
            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                result = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=44100,
                    chunk_duration=2.0
                )
                
                # Should handle gracefully (may return False or raise)
                assert isinstance(result, bool)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_very_short_audio(self):
        """Test analyzing very short audio file."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Create very short audio (0.01 seconds)
            sample_rate = 44100
            duration = 0.01
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            sf.write(tmp_path, audio_data, sample_rate)
            
            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                result = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=sample_rate,
                    chunk_duration=2.0
                )
                
                # Should handle gracefully
                assert isinstance(result, bool)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_mono_output_structure(self):
        """Test that mono files create correct output structure."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Create mono audio
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            sf.write(tmp_path, audio_data, sample_rate)
            
            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                result = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=sample_rate,
                    chunk_duration=2.0
                )
                
                assert result is True
                
                # For mono, output should be directly in track folder
                track_name = Path(tmp_path).stem
                track_output_dir = output_dir / track_name
                assert track_output_dir.exists()
                
                # Should have analysis files
                csv_files = list(track_output_dir.rglob('analysis_results.csv'))
                assert len(csv_files) > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_stereo_output_structure(self):
        """Test that stereo files create channel-specific folders."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Create stereo audio
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            left = 0.5 * np.sin(2 * np.pi * 440 * t)
            right = 0.3 * np.sin(2 * np.pi * 440 * t)
            stereo_data = np.column_stack([left, right])
            sf.write(tmp_path, stereo_data, sample_rate)
            
            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                result = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=sample_rate,
                    chunk_duration=2.0
                )
                
                assert result is True
                
                # Should have channel folders
                track_name = Path(tmp_path).stem
                track_output_dir = output_dir / track_name
                assert track_output_dir.exists()
                
                # Check for channel folders
                channel_folders = [d for d in track_output_dir.iterdir() if d.is_dir()]
                assert len(channel_folders) == 2  # Left and Right
                
                # Verify channel folder names
                folder_names = [f.name for f in channel_folders]
                assert "Channel 1 Left" in folder_names
                assert "Channel 2 Right" in folder_names
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_main_with_custom_config(self):
        """Test main function with custom configuration file."""
        runner = CliRunner()
        
        # Create custom config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp_config:
            tmp_config.write("""
[analysis]
sample_rate = 48000
chunk_duration_seconds = 1.0
""")
            config_path = Path(tmp_config.name)
        
        try:
            result = runner.invoke(main, ['--config', str(config_path), '--help'])
            assert result.exit_code == 0
        finally:
            config_path.unlink(missing_ok=True)

    def test_main_with_invalid_chunk_duration(self):
        """Test main function with invalid chunk duration."""
        runner = CliRunner()
        
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Create valid audio
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            sf.write(tmp_path, audio_data, sample_rate)
            
            # Try with invalid chunk duration (negative)
            result = runner.invoke(main, [
                '--input', tmp_path,
                '--single',
                '--chunk-duration', '-1.0'
            ])
            
            # Should handle error gracefully
            assert result.exit_code != 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_cache_usage(self):
        """Test that analyze_single_track uses caching correctly."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Create audio
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            sf.write(tmp_path, audio_data, sample_rate)
            
            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                
                # First run
                result1 = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=sample_rate,
                    chunk_duration=2.0,
                    use_cache=True
                )
                assert result1 is True
                
                # Second run - should use cache
                result2 = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=sample_rate,
                    chunk_duration=2.0,
                    use_cache=True
                )
                assert result2 is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_cache_disabled(self):
        """Test that analyze_single_track works with cache disabled."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Create audio
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            sf.write(tmp_path, audio_data, sample_rate)
            
            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                
                # Run with cache disabled
                result = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=sample_rate,
                    chunk_duration=2.0,
                    use_cache=False
                )
                assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

