"""
Tests for the main module.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

from src.main import main, analyze_single_track


def test_main_help():
    """Test that main function shows help when called without arguments."""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'Music Analyser' in result.output
    assert '--tracks-dir' in result.output


def test_main_with_nonexistent_file():
    """Test that main function fails gracefully with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(main, ['--input', 'nonexistent.wav', '--single'])
    assert result.exit_code == 1


def test_main_with_nonexistent_tracks_dir():
    """Test that main function fails gracefully with nonexistent tracks directory."""
    runner = CliRunner()
    result = runner.invoke(main, ['--tracks-dir', 'nonexistent_dir'])
    assert result.exit_code == 1


def test_analyze_single_track():
    """Test analyze_single_track function."""
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        # Generate a simple sine wave
        sample_rate = 44100
        duration = 0.1  # 100ms
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Write as stereo
        stereo_data = np.column_stack([audio_data, audio_data])
        sf.write(tmp_file.name, stereo_data, sample_rate)
        
        try:
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                
                # Test the function
                result = analyze_single_track(Path(tmp_file.name), output_dir, sample_rate)
                
                # Check that it completed successfully
                assert result is True
                
                # Check that output directory was created
                track_name = Path(tmp_file.name).stem
                track_output_dir = output_dir / track_name
                assert track_output_dir.exists()
                assert (track_output_dir / 'analysis_results.csv').exists()
                assert (track_output_dir / 'octave_spectrum.png').exists()
                assert (track_output_dir / 'histograms.png').exists()
                
        finally:
            # Clean up
            Path(tmp_file.name).unlink(missing_ok=True)


def test_main_with_valid_audio_file():
    """Test that main function works with a valid audio file."""
    runner = CliRunner()
    
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        # Generate a simple sine wave
        sample_rate = 44100
        duration = 0.1  # 100ms
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Write as stereo
        stereo_data = np.column_stack([audio_data, audio_data])
        sf.write(tmp_file.name, stereo_data, sample_rate)
        
        try:
            # Run the main function in single file mode
            result = runner.invoke(main, [
                '--input', tmp_file.name,
                '--output-dir', 'test_output',
                '--single'
            ])
            
            # Check that it completed successfully
            assert result.exit_code == 0
            assert 'Analysis complete!' in result.output
            
            # Check that output directory was created
            output_dir = Path('test_output')
            track_name = Path(tmp_file.name).stem
            track_output_dir = output_dir / track_name
            assert track_output_dir.exists()
            assert (track_output_dir / 'analysis_results.csv').exists()
            
        finally:
            # Clean up
            Path(tmp_file.name).unlink(missing_ok=True)
            import shutil
            shutil.rmtree('test_output', ignore_errors=True)
