"""
Audio processing module for Audio Analyser.

This module handles audio file loading, preprocessing, and basic audio operations.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _bundled_tool_path(tool_name: str) -> Optional[Path]:
    """Return a bundled FFmpeg tool path when available."""
    executable_name = f"{tool_name}.exe" if sys.platform == "win32" else tool_name
    search_roots = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        search_roots.append(Path(frozen_root))
    search_roots.append(Path(__file__).resolve().parents[1])

    for root in search_roots:
        candidate = root / "vendor" / "ffmpeg" / "bin" / executable_name
        if candidate.exists():
            return candidate
    return None


def _ffmpeg_tool_command(tool_name: str) -> str:
    """Return a bundled FFmpeg executable path or fall back to PATH lookup."""
    bundled = _bundled_tool_path(tool_name)
    return str(bundled) if bundled is not None else tool_name


class AudioProcessor:
    """Handles audio file processing and preprocessing operations."""

    def __init__(
        self, sample_rate: int = 44100, enable_mkv_support: bool = True
    ) -> None:
        """Initialize the audio processor.

        Args:
            sample_rate: Legacy default retained for compatibility. Decoding
                preserves the source stream's native sample rate.
            enable_mkv_support: Enable container (MKV/MTS/M2TS) audio support via ffmpeg
        """
        self.sample_rate = sample_rate
        self.enable_mkv_support = enable_mkv_support
        self._last_channel_layout: Optional[str] = (
            None  # Store channel layout from ffprobe
        )

    def _is_mkv_file(self, file_path: Path) -> bool:
        """Check if file is an MKV container.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .mkv extension, False otherwise
        """
        return file_path.suffix.lower() == ".mkv"

    def _is_mts_file(self, file_path: Path) -> bool:
        """Check if file is an MTS container.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .mts extension, False otherwise
        """
        return file_path.suffix.lower() == ".mts"

    def _probe_audio_streams(self, file_path: Path) -> List[dict]:
        """Probe audio file to get stream information.

        Uses ffprobe to extract stream metadata including codec and channel information.
        Works for both MKV and MTS files.

        Args:
            file_path: Path to the audio file

        Returns:
            List of dictionaries containing stream information:
            - index: Stream index
            - codec_name: Audio codec (e.g., 'truehd', 'ac3', 'pcm_s24le')
            - codec_long_name: Full codec name
            - channels: Number of audio channels
            - channel_layout: Channel layout string
            - sample_rate: Sample rate in Hz

        Raises:
            RuntimeError: If ffprobe is not available or fails to probe file
        """
        try:
            cmd = [
                _ffmpeg_tool_command("ffprobe"),
                "-v",
                "error",
                "-select_streams",
                "a",  # Select audio streams only
                "-show_entries",
                "stream=index,codec_name,codec_long_name,channels,channel_layout,sample_rate",
                "-of",
                "json",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            probe_data = json.loads(result.stdout)
            streams = probe_data.get("streams", [])

            logger.info(f"Found {len(streams)} audio stream(s) in file")
            for stream in streams:
                logger.info(
                    f"  Stream {stream.get('index')}: {stream.get('codec_name')} "
                    f"({stream.get('channels')} channels, {stream.get('sample_rate')} Hz, "
                    f"layout: {stream.get('channel_layout', 'unknown')})"
                )

            return streams

        except FileNotFoundError:
            raise RuntimeError(
                "ffprobe was not found. MKV/MTS/TrueHD support requires the "
                "ffmpeg tools package, including both ffmpeg.exe and ffprobe.exe. "
                "Install ffmpeg from https://ffmpeg.org/download.html, add its "
                "bin folder to PATH, then restart Audio Analyser or your terminal."
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e.stderr}")
            raise RuntimeError(f"Failed to probe file: {e.stderr}")

    # Backward-compatibility aliases (older tests/code patched these names)
    def _probe_mkv_audio_streams(self, mkv_path: Path) -> List[dict]:
        """Alias for probing MKV audio streams (compat)."""
        return self._probe_audio_streams(mkv_path)

    def _decode_audio_to_wav(
        self,
        input_path: Path,
        stream_index: Optional[int] = None,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Path:
        """Decode an audio file/container stream to a temporary float32 WAV.

        This uses ffmpeg for decoding for maximum format support (mp3/aac/m4a/etc)
        and for container audio (mkv/mts/m2ts). The resulting WAV preserves the
        decoded stream's native sample rate and is loaded via soundfile to
        preserve multi-channel shapes.

        Args:
            input_path: Source audio path (file or container).
            stream_index: Optional global stream index to decode (ffprobe "index").
                If None, decodes the first audio stream (a:0).
            start_time: Optional start time in seconds (best-effort trim).
            duration: Optional duration in seconds (best-effort trim).

        Returns:
            Path to a temporary WAV file.

        Raises:
            RuntimeError: If ffmpeg is missing or decoding fails.
        """
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = Path(temp_wav.name)
        temp_wav.close()

        try:
            cmd: list[str] = [_ffmpeg_tool_command("ffmpeg")]

            # Best-effort trimming for test mode. Using -ss/-t before -i is fast but
            # not sample-accurate; that's fine for test sections and avoids decoding
            # full files just to discard.
            if start_time is not None and start_time > 0:
                cmd += ["-ss", str(float(start_time))]
            if duration is not None and duration > 0:
                cmd += ["-t", str(float(duration))]

            cmd += [
                "-i",
                str(input_path),
                "-vn",
            ]

            if stream_index is not None:
                cmd += ["-map", f"0:{int(stream_index)}"]
            else:
                cmd += ["-map", "0:a:0"]

            # Decode to float32 PCM WAV while preserving the source sample rate.
            # Forcing resampling here can create intersample overs above 0 dBFS,
            # which is not appropriate for source sample-peak time traces.
            cmd += [
                "-c:a",
                "pcm_f32le",
                "-y",
                str(temp_wav_path),
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return temp_wav_path

        except FileNotFoundError:
            if temp_wav_path.exists():
                temp_wav_path.unlink()
            raise RuntimeError(
                "ffmpeg was not found. Audio decoding requires the ffmpeg tools "
                "package (ffmpeg.exe and ffprobe.exe). Install ffmpeg and ensure it "
                "is on PATH, then restart Audio Analyser or your terminal."
            )
        except subprocess.CalledProcessError as e:
            if temp_wav_path.exists():
                temp_wav_path.unlink()
            logger.error("ffmpeg decode failed: %s", e.stderr)
            raise RuntimeError(f"Failed to decode audio via ffmpeg: {e.stderr}")

    def load_audio(
        self,
        file_path: str | Path,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate.

        Preserves multi-channel audio. Returns shape (samples,) for mono or (samples, channels) for multi-channel.
        Uses ffmpeg decoding for broad format and container support.

        Args:
            file_path: Path to the audio file
            start_time: Optional start time in seconds (for testing - analyze only a section)
            duration: Optional duration in seconds (for testing - limit analysis length)

        Returns:
            Tuple of (audio_data, sample_rate)
            - audio_data: numpy array with shape (samples,) for mono or (samples, channels) for multi-channel
            - sample_rate: Sample rate in Hz

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the audio file format is not supported
            RuntimeError: If ffmpeg decode fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        temp_wav_path: Optional[Path] = None
        try:
            stream_index: Optional[int] = None
            if self.enable_mkv_support and (
                self._is_mkv_file(file_path) or self._is_mts_file(file_path)
            ):
                streams = self._probe_audio_streams(file_path)
                if not streams:
                    raise ValueError("No audio streams found in container file")

                # Prefer TrueHD when present (and not Atmos metadata), else fall back
                # to the first audio stream.
                chosen = streams[0]
                for stream in streams:
                    codec = str(stream.get("codec_name", "")).lower()
                    if codec == "truehd":
                        codec_long = str(stream.get("codec_long_name", "")).lower()
                        if "atmos" not in codec_long:
                            chosen = stream
                            break

                stream_index = int(chosen.get("index", 0))
                self._last_channel_layout = chosen.get("channel_layout", None)
                if self._last_channel_layout:
                    logger.info("Detected channel layout: %s", self._last_channel_layout)

            # Decode everything through ffmpeg for consistent format support.
            temp_wav_path = self._decode_audio_to_wav(
                file_path,
                stream_index=stream_index,
                start_time=start_time,
                duration=duration,
            )

            # Load audio file preserving multi-channel audio
            # The decoded WAV is float32 at the source stream sample rate.
            audio_data, sr = sf.read(str(temp_wav_path), dtype=np.float32, always_2d=False)

            # Ensure consistent shape: (samples,) for mono, (samples, channels) for multi-channel
            if audio_data.ndim == 1:
                logger.info(f"Loaded audio file: {file_path} (mono)")
            else:
                logger.info(
                    f"Loaded audio file: {file_path} ({audio_data.shape[1]} channels)"
                )

            original_duration = len(audio_data) / sr
            logger.info(
                f"Sample rate: {sr} Hz, Duration: {original_duration:.2f}s, dtype: {audio_data.dtype}"
            )

            # Trim audio to specified section if requested (for testing)
            if start_time is not None or duration is not None:
                start_sample = int(start_time * sr) if start_time is not None else 0
                if duration is not None:
                    end_sample = start_sample + int(duration * sr)
                else:
                    end_sample = len(audio_data)

                # Clamp to valid range
                start_sample = max(0, min(start_sample, len(audio_data)))
                end_sample = max(start_sample, min(end_sample, len(audio_data)))

                if audio_data.ndim == 1:
                    audio_data = audio_data[start_sample:end_sample]
                else:
                    audio_data = audio_data[start_sample:end_sample, :]

                trimmed_duration = len(audio_data) / sr
                if duration is not None:
                    logger.info(
                        f"Trimmed audio: {start_time or 0:.2f}s to {(start_time or 0) + duration:.2f}s "
                        f"(duration: {trimmed_duration:.2f}s)"
                    )
                else:
                    logger.info(
                        f"Trimmed audio: from {start_time:.2f}s to end "
                        f"(duration: {trimmed_duration:.2f}s)"
                    )

            return audio_data, sr

        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise ValueError(f"Failed to load audio file: {e}")
        finally:
            # Clean up temporary WAV file created for ffmpeg decoding
            if temp_wav_path and temp_wav_path.exists():
                try:
                    temp_wav_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_wav_path}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temporary file {temp_wav_path}: {cleanup_error}"
                    )

    def stereo_to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono by averaging channels.

        Args:
            audio_data: Input audio data (can be mono or stereo)

        Returns:
            Mono audio data
        """
        if audio_data.ndim == 1:
            # Already mono
            return audio_data
        elif audio_data.ndim == 2:
            # Stereo - average channels
            mono_audio = np.mean(audio_data, axis=1)
            logger.info("Converted stereo to mono")
            return mono_audio
        else:
            raise ValueError(f"Unsupported audio dimensions: {audio_data.ndim}")

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to prevent clipping.

        For multi-channel audio, normalizes globally (across all channels) to maintain
        relative channel levels.

        Args:
            audio_data: Input audio data (mono: shape (samples,), multi-channel: shape (samples, channels))

        Returns:
            Normalized audio data (same shape as input)
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            normalized_audio = audio_data / max_val
            logger.info(f"Normalized audio (max value was {max_val:.6f})")
            return normalized_audio
        return audio_data

    def get_audio_info(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """Get basic information about the audio data.

        Args:
            audio_data: Audio data array (mono: shape (samples,), multi-channel: shape (samples, channels))
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary with audio information including:
            - duration_seconds: Duration in seconds
            - sample_rate: Sample rate in Hz
            - channels: Number of channels (1 for mono)
            - channel_layout: FFmpeg channel layout string (e.g., "7.1", "5.1") or "mono"/"stereo"/"multi-channel"
            - samples: Number of samples
            - max_amplitude: Maximum amplitude across all channels
            - rms: RMS value across all channels
        """
        duration = len(audio_data) / sample_rate
        num_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]

        # Use channel layout from ffprobe if available (most accurate)
        # Otherwise determine from channel count
        if self._last_channel_layout:
            channel_layout = self._last_channel_layout
        elif num_channels == 1:
            channel_layout = "mono"
        elif num_channels == 2:
            channel_layout = "stereo"
        else:
            channel_layout = "multi-channel"

        info = {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": num_channels,
            "channel_layout": channel_layout,
            "samples": len(audio_data),
            "max_amplitude": float(np.max(np.abs(audio_data))),
            "rms": float(np.sqrt(np.mean(audio_data**2))),
        }

        return info

    def extract_channels(self, audio_data: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """Extract individual channels from multi-channel audio data.

        Args:
            audio_data: Audio data array (mono: shape (samples,), multi-channel: shape (samples, channels))

        Returns:
            List of tuples (channel_data, channel_index) for each channel
            - channel_data: 1D numpy array with shape (samples,)
            - channel_index: Zero-based channel index
        """
        if audio_data.ndim == 1:
            # Mono: return single channel
            return [(audio_data, 0)]
        elif audio_data.ndim == 2:
            # Multi-channel: extract each channel
            channels = []
            for ch_idx in range(audio_data.shape[1]):
                channel_data = audio_data[:, ch_idx]
                channels.append((channel_data, ch_idx))
            return channels
        else:
            raise ValueError(f"Unsupported audio dimensions: {audio_data.ndim}")
