"""
Audio processing module for Music Analyser.

This module handles audio file loading, preprocessing, and basic audio operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

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
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            ValueError: If the audio file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Load audio file with optional dtype optimization
            use_float32 = True  # Default to float32 for 50% memory reduction
            dtype = np.float32 if use_float32 else None
            audio_data, sr = librosa.load(str(file_path), sr=self.sample_rate, dtype=dtype)
            logger.info(f"Loaded audio file: {file_path}")
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
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
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
            audio_data: Audio data array
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with audio information
        """
        duration = len(audio_data) / sample_rate
        
        info = {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": 1 if audio_data.ndim == 1 else audio_data.shape[1],
            "samples": len(audio_data),
            "max_amplitude": float(np.max(np.abs(audio_data))),
            "rms": float(np.sqrt(np.mean(audio_data**2))),
        }
        
        return info
