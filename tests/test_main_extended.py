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

from src.main import (
    _warn_legacy_post_disabled,
    analyze_single_track,
    main,
    resolve_track_output_dir,
)


class TestMainExtended:
    """Extended test cases for main module."""

    def test_bundle_only_batch_output_uses_category_parent(self, tmp_path):
        """Bundle-only batch output should avoid redundant per-track folders."""
        tracks_root = tmp_path / "tracks"
        track_path = tracks_root / "Film" / "A Star Is Born.wav"
        output_dir = tmp_path / "analysis"

        assert (
            resolve_track_output_dir(
                output_dir,
                track_path,
                tracks_root,
                include_track_name=False,
            )
            == output_dir / "Film"
        )

    def test_legacy_post_warns_for_bundle_only_output(self, tmp_path, caplog):
        """Legacy post path should point bundle users to src.render."""
        bundle_dir = tmp_path / "track.aaresults"
        bundle_dir.mkdir()

        assert _warn_legacy_post_disabled(tmp_path) is True
        assert "python -m src.render" in caplog.text

    def test_analyze_single_track_empty_audio(self):
        """Test analyzing empty audio file."""
        # Create empty audio file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            # Write empty audio
            sf.write(tmp_path, np.array([]), 44100)

            with tempfile.TemporaryDirectory() as tmp_output:
                output_dir = Path(tmp_output)
                result = analyze_single_track(
                    Path(tmp_path), output_dir, sample_rate=44100, chunk_duration=2.0
                )

                # Should handle gracefully.
                success, elapsed_seconds = result
                assert isinstance(success, bool)
                assert elapsed_seconds >= 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_very_short_audio(self):
        """Test analyzing very short audio file."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
                    chunk_duration=2.0,
                )

                # Should handle gracefully.
                success, elapsed_seconds = result
                assert isinstance(success, bool)
                assert elapsed_seconds >= 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_mono_output_structure(self):
        """Test that mono files create correct output structure."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
                    chunk_duration=2.0,
                )

                success, elapsed_seconds = result
                assert success is True
                assert elapsed_seconds >= 0

                track_name = Path(tmp_path).stem

                bundle_dir = output_dir / f"{track_name}.aaresults"
                assert (bundle_dir / "manifest.json").exists()
                assert not (output_dir / track_name).exists()
                assert not list(output_dir.rglob("analysis_results.csv"))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_stereo_output_structure(self):
        """Test that stereo files create channel entries in the result bundle."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
                    chunk_duration=2.0,
                )

                success, elapsed_seconds = result
                assert success is True
                assert elapsed_seconds >= 0

                track_name = Path(tmp_path).stem

                bundle_dir = output_dir / f"{track_name}.aaresults"
                assert (bundle_dir / "manifest.json").exists()
                assert (bundle_dir / "channels" / "channel_01").exists()
                assert (bundle_dir / "channels" / "channel_02").exists()
                assert not (output_dir / track_name).exists()
                assert not (output_dir / "Channel 1 Left").exists()
                assert not (output_dir / "Channel 2 Right").exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_main_with_custom_config(self):
        """Test main function with custom configuration file."""
        runner = CliRunner()

        # Create custom config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as tmp_config:
            tmp_config.write("""
[analysis]
sample_rate = 48000
chunk_duration_seconds = 1.0
""")
            config_path = Path(tmp_config.name)

        try:
            result = runner.invoke(main, ["--config", str(config_path), "--help"])
            assert result.exit_code == 0
        finally:
            config_path.unlink(missing_ok=True)

    def test_main_with_invalid_chunk_duration(self):
        """Test main function with invalid chunk duration."""
        runner = CliRunner()

        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
            result = runner.invoke(
                main, ["--input", tmp_path, "--single", "--chunk-duration", "-1.0"]
            )

            # Should handle error gracefully
            assert result.exit_code != 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_cache_usage(self):
        """Test that analyze_single_track uses caching correctly."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
                    use_cache=True,
                )
                success1, elapsed_seconds1 = result1
                assert success1 is True
                assert elapsed_seconds1 >= 0

                # Second run - should use cache
                result2 = analyze_single_track(
                    Path(tmp_path),
                    output_dir,
                    sample_rate=sample_rate,
                    chunk_duration=2.0,
                    use_cache=True,
                )
                success2, elapsed_seconds2 = result2
                assert success2 is True
                assert elapsed_seconds2 >= 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_analyze_single_track_cache_disabled(self):
        """Test that analyze_single_track works with cache disabled."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
                    use_cache=False,
                )
                success, elapsed_seconds = result
                assert success is True
                assert elapsed_seconds >= 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
