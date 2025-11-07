"""
Audio processing module for Music Analyser.

This module handles audio file loading, preprocessing, and basic audio operations.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file processing and preprocessing operations."""

    def __init__(self, sample_rate: int = 44100, enable_mkv_support: bool = True) -> None:
        """Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            enable_mkv_support: Enable MKV container and TrueHD audio support
        """
        self.sample_rate = sample_rate
        self.enable_mkv_support = enable_mkv_support

    def _is_mkv_file(self, file_path: Path) -> bool:
        """Check if file is an MKV container.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has .mkv extension, False otherwise
        """
        return file_path.suffix.lower() == '.mkv'
    
    def _probe_mkv_audio_streams(self, file_path: Path) -> List[dict]:
        """Probe MKV file to get audio stream information.
        
        Uses ffprobe to extract stream metadata including codec and channel information.
        
        Args:
            file_path: Path to the MKV file
            
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
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a',  # Select audio streams only
                '-show_entries', 'stream=index,codec_name,codec_long_name,channels,channel_layout,sample_rate',
                '-of', 'json',
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            probe_data = json.loads(result.stdout)
            streams = probe_data.get('streams', [])
            
            logger.info(f"Found {len(streams)} audio stream(s) in MKV file")
            for stream in streams:
                logger.info(
                    f"  Stream {stream.get('index')}: {stream.get('codec_name')} "
                    f"({stream.get('channels')} channels, {stream.get('sample_rate')} Hz)"
                )
            
            return streams
            
        except FileNotFoundError:
            raise RuntimeError(
                "ffprobe not found. Please install ffmpeg to support MKV/TrueHD files. "
                "Download from https://ffmpeg.org/download.html"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e.stderr}")
            raise RuntimeError(f"Failed to probe MKV file: {e.stderr}")
    
    def _extract_truehd_from_mkv(
        self, 
        mkv_path: Path, 
        stream_index: int = 0,
        exclude_atmos: bool = True
    ) -> Path:
        """Extract and decode TrueHD audio from MKV container to temporary WAV file.
        
        Uses ffmpeg to extract the audio stream, decode TrueHD to PCM, and exclude
        Dolby Atmos metadata streams if present.
        
        Args:
            mkv_path: Path to the MKV file
            stream_index: Audio stream index to extract (default: 0)
            exclude_atmos: Exclude Dolby Atmos metadata streams (default: True)
            
        Returns:
            Path to temporary WAV file containing decoded PCM audio
            
        Raises:
            RuntimeError: If ffmpeg is not available or fails to extract audio
        """
        # Create temporary WAV file for decoded audio
        temp_wav = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False
        )
        temp_wav_path = Path(temp_wav.name)
        temp_wav.close()
        
        try:
            # Build ffmpeg command to extract and decode TrueHD
            # -map 0:stream_index selects the stream by global index
            # -c:a pcm_s24le decodes to 24-bit PCM (TrueHD is typically 24-bit)
            # -ar sets sample rate (will be resampled later if needed)
            # -ac preserves channel count
            cmd = [
                'ffmpeg',
                '-i', str(mkv_path),
                '-map', f'0:{stream_index}',  # Select stream by global index
                '-c:a', 'pcm_s24le',  # Decode to 24-bit PCM
                '-ar', str(self.sample_rate),  # Set target sample rate
                '-y',  # Overwrite output file
                str(temp_wav_path)
            ]
            
            # Exclude Atmos metadata if requested
            if exclude_atmos:
                # Filter out Atmos metadata streams (typically have 'atmos' in codec name)
                # This is handled by selecting specific stream index
                pass
            
            logger.info(f"Extracting TrueHD audio from MKV (stream {stream_index})...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Successfully extracted TrueHD audio to {temp_wav_path}")
            return temp_wav_path
            
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to support MKV/TrueHD files. "
                "Download from https://ffmpeg.org/download.html"
            )
        except subprocess.CalledProcessError as e:
            # Clean up temp file on error
            if temp_wav_path.exists():
                temp_wav_path.unlink()
            logger.error(f"ffmpeg failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract TrueHD audio: {e.stderr}")
    
    def load_audio(self, file_path: str | Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate.
        
        Preserves multi-channel audio. Returns shape (samples,) for mono or (samples, channels) for multi-channel.
        Supports MKV containers with TrueHD audio (requires ffmpeg).
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            - audio_data: numpy array with shape (samples,) for mono or (samples, channels) for multi-channel
            - sample_rate: Sample rate in Hz
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the audio file format is not supported
            RuntimeError: If MKV/TrueHD extraction fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Handle MKV containers with TrueHD audio
        temp_wav_path: Optional[Path] = None
        if self.enable_mkv_support and self._is_mkv_file(file_path):
            try:
                # Probe MKV to find TrueHD audio streams
                streams = self._probe_mkv_audio_streams(file_path)
                
                # Find TrueHD stream (exclude Atmos metadata)
                truehd_stream = None
                for stream in streams:
                    codec = stream.get('codec_name', '').lower()
                    if codec == 'truehd':
                        # Check if it's not Atmos metadata
                        codec_long = stream.get('codec_long_name', '').lower()
                        if 'atmos' not in codec_long:
                            truehd_stream = stream
                            break
                
                if truehd_stream:
                    # Use global stream index for mapping
                    global_stream_idx = int(truehd_stream.get('index', 0))
                    logger.info(f"Found TrueHD audio stream at global index {global_stream_idx}")
                    # Extract and decode TrueHD to temporary WAV
                    temp_wav_path = self._extract_truehd_from_mkv(file_path, stream_index=global_stream_idx)
                    # Update file_path to point to extracted WAV
                    file_path = temp_wav_path
                else:
                    logger.warning(
                        f"No TrueHD audio stream found in MKV file. "
                        f"Found streams: {[s.get('codec_name') for s in streams]}"
                    )
                    # Try to extract first audio stream anyway
                    if streams:
                        global_stream_idx = int(streams[0].get('index', 0))
                        temp_wav_path = self._extract_truehd_from_mkv(file_path, stream_index=global_stream_idx)
                        file_path = temp_wav_path
                    else:
                        raise ValueError("No audio streams found in MKV file")
            except Exception as e:
                logger.error(f"Failed to extract audio from MKV: {e}")
                raise
        
        try:
            # Load audio file preserving multi-channel audio
            # librosa.load() returns mono by default, so we use soundfile directly for multi-channel
            use_float32 = True  # Default to float32 for 50% memory reduction
            dtype = np.float32 if use_float32 else None
            
            # Try loading with soundfile first to preserve channels
            try:
                audio_data, sr = sf.read(str(file_path), dtype=dtype)
                # Resample if needed
                if sr != self.sample_rate:
                    import librosa
                    if audio_data.ndim == 1:
                        # Mono: resample directly
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                    else:
                        # Multi-channel: resample each channel
                        resampled_channels = []
                        for ch in range(audio_data.shape[1]):
                            resampled_ch = librosa.resample(
                                audio_data[:, ch], 
                                orig_sr=sr, 
                                target_sr=self.sample_rate
                            )
                            resampled_channels.append(resampled_ch)
                        audio_data = np.column_stack(resampled_channels)
                    sr = self.sample_rate
            except Exception:
                # Fallback to librosa for formats soundfile doesn't support
                audio_data, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=False, dtype=dtype)
                # librosa returns (channels, samples) when mono=False, transpose to (samples, channels)
                if audio_data.ndim == 2:
                    audio_data = audio_data.T
            
            # Ensure consistent shape: (samples,) for mono, (samples, channels) for multi-channel
            if audio_data.ndim == 1:
                logger.info(f"Loaded audio file: {file_path} (mono)")
            else:
                logger.info(f"Loaded audio file: {file_path} ({audio_data.shape[1]} channels)")
            
            logger.info(f"Sample rate: {sr} Hz, Duration: {len(audio_data)/sr:.2f}s, dtype: {audio_data.dtype}")
            
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise ValueError(f"Failed to load audio file: {e}")
        finally:
            # Clean up temporary WAV file if it was created from MKV extraction
            if temp_wav_path and temp_wav_path.exists():
                try:
                    temp_wav_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_wav_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file {temp_wav_path}: {cleanup_error}")

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
            - channel_layout: "mono", "stereo", or "multi-channel"
            - samples: Number of samples
            - max_amplitude: Maximum amplitude across all channels
            - rms: RMS value across all channels
        """
        duration = len(audio_data) / sample_rate
        num_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        
        # Determine channel layout
        if num_channels == 1:
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
