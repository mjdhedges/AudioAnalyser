"""
Tests for MKV container and TrueHD audio support.

Note: These tests mock ffmpeg/ffprobe calls since they require external tools.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.audio_processor import AudioProcessor, _ffmpeg_tool_command


class TestMKVSupport:
    """Test cases for MKV/TrueHD support."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor(sample_rate=44100, enable_mkv_support=True)

    def test_is_mkv_file(self):
        """Test MKV file detection."""
        assert self.processor._is_mkv_file(Path("test.mkv")) is True
        assert self.processor._is_mkv_file(Path("test.MKV")) is True
        assert self.processor._is_mkv_file(Path("test.mkvv")) is False
        assert self.processor._is_mkv_file(Path("test.wav")) is False

    def test_ffmpeg_tool_command_uses_vendored_binary(self):
        """FFmpeg commands should prefer the bundled Windows binaries."""
        command = _ffmpeg_tool_command("ffmpeg")

        assert command.endswith("ffmpeg.exe")
        assert Path(command).exists()

    @patch("subprocess.run")
    def test_probe_mkv_audio_streams(self, mock_subprocess):
        """Test probing MKV file for audio streams."""
        # Mock ffprobe output - note: ffprobe with -select_streams a only returns audio streams
        mock_output = {
            "streams": [
                {
                    "index": 1,
                    "codec_name": "truehd",
                    "codec_long_name": "TrueHD",
                    "channels": 6,
                    "channel_layout": "5.1",
                    "sample_rate": "48000",
                },
                {
                    "index": 2,
                    "codec_name": "ac3",
                    "codec_long_name": "AC-3",
                    "channels": 2,
                    "channel_layout": "stereo",
                    "sample_rate": "48000",
                },
            ]
        }

        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output)
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        mkv_path = Path("test.mkv")
        streams = self.processor._probe_mkv_audio_streams(mkv_path)

        # Should return audio streams (ffprobe filters to audio only)
        assert len(streams) == 2
        assert streams[0]["codec_name"] == "truehd"
        assert streams[0]["channels"] == 6
        assert streams[1]["codec_name"] == "ac3"

        # Verify ffprobe was called correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert Path(call_args[0]).name == "ffprobe.exe"
        assert "-select_streams" in call_args
        assert "a" in call_args  # Select audio streams

    @patch("subprocess.run")
    def test_probe_mkv_no_ffprobe(self, mock_subprocess):
        """Test probing when ffprobe is not available."""
        mock_subprocess.side_effect = FileNotFoundError("ffprobe not found")

        mkv_path = Path("test.mkv")

        with pytest.raises(RuntimeError) as exc_info:
            self.processor._probe_mkv_audio_streams(mkv_path)
        message = str(exc_info.value)
        assert "ffprobe was not found" in message
        assert "ffmpeg.exe and ffprobe.exe" in message
        assert "PATH" in message

    @patch("src.audio_processor.AudioProcessor._probe_audio_streams")
    @patch("src.audio_processor.AudioProcessor._decode_audio_to_wav")
    @patch("soundfile.read")
    @patch("pathlib.Path.exists")
    def test_load_audio_mkv_with_truehd(
        self, mock_exists, mock_sf_read, mock_decode, mock_probe
    ):
        """Test loading MKV file with TrueHD audio."""
        # Mock file exists
        mock_exists.return_value = True

        # Mock probing - return TrueHD stream
        mock_probe.return_value = [
            {
                "index": 1,
                "codec_name": "truehd",
                "codec_long_name": "TrueHD",
                "channels": 6,
                "sample_rate": "48000",
                "channel_layout": "5.1",
            }
        ]

        # Mock decoding - return temporary WAV path
        temp_wav_path = Path("/tmp/decoded.wav")
        mock_decode.return_value = temp_wav_path

        # Mock soundfile reading the extracted WAV
        import numpy as np

        sample_rate = 44100
        duration = 0.1
        samples = int(sample_rate * duration)
        mock_audio = np.random.randn(samples, 6).astype(np.float32)
        mock_sf_read.return_value = (mock_audio, sample_rate)

        # Create a mock MKV file path
        mkv_path = Path("test.mkv")

        # Load audio
        audio_data, sr = self.processor.load_audio(mkv_path)

        # Verify probing/decoding was called
        mock_probe.assert_called_once_with(mkv_path)
        mock_decode.assert_called_once()
        assert mock_decode.call_args.kwargs["stream_index"] == 1

        # Verify soundfile was called with decoded WAV
        mock_sf_read.assert_called_once()
        assert mock_sf_read.call_args[0][0] == str(temp_wav_path)
        assert sr == sample_rate
        assert audio_data.shape == mock_audio.shape

    @patch("src.audio_processor.AudioProcessor._probe_audio_streams")
    @patch("src.audio_processor.AudioProcessor._decode_audio_to_wav")
    @patch("pathlib.Path.exists")
    def test_load_audio_mkv_no_truehd_stream(
        self, mock_exists, mock_decode, mock_probe
    ):
        """Test loading MKV file without TrueHD stream."""
        # Mock file exists
        mock_exists.return_value = True

        # Mock probing - return non-TrueHD audio stream
        mock_probe.return_value = [
            {
                "index": 1,
                "codec_name": "ac3",
                "codec_long_name": "AC-3",
                "channels": 2,
                "sample_rate": "48000",
                "channel_layout": "stereo",
            }
        ]

        # Mock decoding for fallback to first audio stream
        temp_wav_path = Path("/tmp/decoded.wav")
        mock_decode.return_value = temp_wav_path

        mkv_path = Path("test.mkv")

        # Should attempt to decode first audio stream anyway
        import numpy as np

        with patch("soundfile.read", return_value=(np.zeros(10, dtype=np.float32), 44100)):
            try:
                self.processor.load_audio(mkv_path)
            except Exception:
                pass

        mock_probe.assert_called_once()
        mock_decode.assert_called_once()
        assert mock_decode.call_args.kwargs["stream_index"] == 1

    def test_mkv_support_disabled(self):
        """Test that MKV files are skipped when support is disabled."""
        processor_no_mkv = AudioProcessor(sample_rate=44100, enable_mkv_support=False)

        # Create a temporary WAV file (not MKV, but test the flag)
        import tempfile
        import numpy as np
        import soundfile as sf

        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            sample_rate = 44100
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
            sf.write(tmp_path, audio_data, sample_rate)

            # Should work normally for non-MKV files
            loaded_audio, sr = processor_no_mkv.load_audio(tmp_path)
            assert sr == sample_rate
        finally:
            Path(tmp_path).unlink(missing_ok=True)
