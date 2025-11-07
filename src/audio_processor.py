"""
Audio processing module for Music Analyser.

This module handles audio file loading, preprocessing, and basic audio operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file processing and preprocessing operations."""

    def __init__(self, sample_rate: int = 44100) -> None:
        """Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate

    def load_audio(self, file_path: str | Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate.
        
        Preserves multi-channel audio. Returns shape (samples,) for mono or (samples, channels) for multi-channel.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            - audio_data: numpy array with shape (samples,) for mono or (samples, channels) for multi-channel
            - sample_rate: Sample rate in Hz
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the audio file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
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
