"""
Diagnostic script to check channel mapping for 7.1 TrueHD audio.

This script analyzes the low-frequency content of each channel to identify
which channel is actually the LFE channel.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter


def analyze_channel_low_freq(
    audio_data: np.ndarray, channel_idx: int, sample_rate: int
) -> dict:
    """Analyze low-frequency content of a channel.

    Args:
        audio_data: Multi-channel audio data (samples, channels)
        channel_idx: Channel index to analyze
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary with low-frequency analysis results
    """
    channel = audio_data[:, channel_idx]

    # Calculate RMS in low-frequency octave bands using the production FFT bank.
    octave_filter = OctaveBandFilter(
        sample_rate=sample_rate,
        include_low_residual_band=False,
        include_high_residual_band=False,
    )
    octave_bank = octave_filter.create_octave_bank(channel, [16.0, 31.25, 62.5])
    rms_16 = np.sqrt(np.mean(octave_bank[:, 1] ** 2))
    rms_31 = np.sqrt(np.mean(octave_bank[:, 2] ** 2))
    rms_62 = np.sqrt(np.mean(octave_bank[:, 3] ** 2))

    # Full spectrum RMS
    rms_full = np.sqrt(np.mean(channel**2))

    # Peak level
    peak = np.max(np.abs(channel))

    return {
        "channel_idx": channel_idx,
        "rms_16hz": rms_16,
        "rms_31hz": rms_31,
        "rms_62hz": rms_62,
        "rms_full": rms_full,
        "peak": peak,
        "low_freq_ratio": (rms_16 + rms_31 + rms_62) / (rms_full + 1e-10),
    }


def main():
    """Check channel mapping for TrueHD 7.1 audio."""
    track_path = Path("Tracks/Film/A Star Is Born (2018) - Remember Us This Way.mkv")

    print("=" * 80)
    print("Channel Mapping Diagnostic")
    print("=" * 80)
    print()

    # Load audio
    processor = AudioProcessor(sample_rate=44100)
    print(f"Loading audio file: {track_path}")
    audio_data, sr = processor.load_audio(track_path)

    print(f"Audio shape: {audio_data.shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"Number of channels: {audio_data.shape[1]}")
    print()

    # Analyze each channel
    print("Analyzing low-frequency content for each channel...")
    print()

    results = []
    for ch_idx in range(audio_data.shape[1]):
        result = analyze_channel_low_freq(audio_data, ch_idx, sr)
        results.append(result)

    # Sort by low-frequency ratio (LFE should have highest)
    results.sort(key=lambda x: x["low_freq_ratio"], reverse=True)

    print("Channel Analysis Results (sorted by low-frequency content):")
    print("-" * 80)
    print(
        f"{'Ch':<4} {'RMS 16Hz':<12} {'RMS 31Hz':<12} {'RMS 62Hz':<12} {'RMS Full':<12} {'Peak':<12} {'LF Ratio':<12}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['channel_idx']:<4} "
            f"{result['rms_16hz']:<12.6f} "
            f"{result['rms_31hz']:<12.6f} "
            f"{result['rms_62hz']:<12.6f} "
            f"{result['rms_full']:<12.6f} "
            f"{result['peak']:<12.6f} "
            f"{result['low_freq_ratio']:<12.4f}"
        )

    print()
    print("=" * 80)
    print("Expected Channel Mapping (current assumption):")
    print("  Channel 0: FL (Front Left)")
    print("  Channel 1: FC (Front Center)")
    print("  Channel 2: FR (Front Right)")
    print("  Channel 3: SL (Surround Left)")
    print("  Channel 4: SR (Surround Right)")
    print("  Channel 5: SBL (Surround Back Left)")
    print("  Channel 6: SBR (Surround Back Right)")
    print("  Channel 7: LFE (Low Frequency Effects)")
    print()
    print("FFmpeg Standard 7.1 Channel Order:")
    print("  Channel 0: FL (Front Left)")
    print("  Channel 1: FR (Front Right)")
    print("  Channel 2: FC (Front Center)")
    print("  Channel 3: LFE (Low Frequency Effects)")
    print("  Channel 4: SL (Surround Left)")
    print("  Channel 5: SR (Surround Right)")
    print("  Channel 6: SBL (Surround Back Left)")
    print("  Channel 7: SBR (Surround Back Right)")
    print()
    print("The channel with the highest low-frequency ratio is likely the LFE channel.")
    print("=" * 80)


if __name__ == "__main__":
    main()
