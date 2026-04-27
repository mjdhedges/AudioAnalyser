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
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Audio Analyser" in result.output
    assert "--tracks-dir" in result.output


def test_main_with_nonexistent_file():
    """Test that main function fails gracefully with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(main, ["--input", "nonexistent.wav", "--single"])
    # Click returns 2 for usage errors (invalid file), 1 for application errors
    assert result.exit_code in (1, 2)


def test_main_with_nonexistent_tracks_dir():
    """Test that main function fails gracefully with nonexistent tracks directory."""
    runner = CliRunner()
    result = runner.invoke(main, ["--tracks-dir", "nonexistent_dir"])
    # Click returns 2 for usage errors (invalid directory), 1 for application errors
    assert result.exit_code in (1, 2)


def test_analyze_single_track():
    """Test analyze_single_track function."""
    # Create a temporary audio file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        # Generate a simple sine wave (longer duration for analysis)
        sample_rate = 44100
        duration = 2.0  # 2 seconds - long enough for analysis
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

        # Write as stereo
        stereo_data = np.column_stack([audio_data, audio_data])
        sf.write(tmp_path, stereo_data, sample_rate)

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_output:
            output_dir = Path(tmp_output)
            chunk_duration = 2.0  # Required parameter

            # Test the function
            result = analyze_single_track(
                Path(tmp_path), output_dir, sample_rate, chunk_duration
            )

            # Check that it completed successfully
            success, elapsed_seconds = result
            assert success is True
            assert elapsed_seconds >= 0

            track_name = Path(tmp_path).stem

            bundle_dir = output_dir / f"{track_name}.aaresults"
            assert (bundle_dir / "manifest.json").exists()
            assert not (output_dir / track_name).exists()
            assert not list(output_dir.rglob("analysis_results.csv"))

    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_main_with_valid_audio_file():
    """Test that main function works with a valid audio file."""
    runner = CliRunner()

    # Create a temporary audio file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        # Generate a simple sine wave (longer duration for analysis)
        sample_rate = 44100
        duration = 2.0  # 2 seconds - long enough for analysis
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

        # Write as mono (simpler for testing)
        sf.write(tmp_path, audio_data, sample_rate)

        # Run the main function in single file mode
        result = runner.invoke(
            main, ["--input", tmp_path, "--output-dir", "test_output", "--single"]
        )

        # Check that it completed successfully
        assert result.exit_code == 0

        output_dir = Path("test_output")
        track_name = Path(tmp_path).stem

        bundle_dir = output_dir / f"{track_name}.aaresults"
        assert (bundle_dir / "manifest.json").exists()
        assert not (output_dir / track_name).exists()
        assert not list(output_dir.rglob("analysis_results.csv"))

    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        import shutil

        shutil.rmtree("test_output", ignore_errors=True)
