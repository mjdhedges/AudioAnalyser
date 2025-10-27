"""
Test octave band filtering phase preservation and frequency response.

This test verifies that octave band processing:
1. Does not affect phase relationships
2. Maintains correct frequency response
3. Preserves signal integrity when bands are summed back
4. Uses the same processing as the main application

Test Strategy:
- Use swept sine wave (20Hz to 20kHz)
- Apply octave band filtering
- Analyze frequency and phase response before and after
- Sum filtered bands back together
- Compare original vs reconstructed signal
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_frequency_response(audio_data: np.ndarray, sample_rate: int, 
                               title: str = "Frequency Response"):
    """Analyze frequency and phase response of audio signal."""
    # Use Welch's method for better frequency resolution
    nperseg = min(8192, len(audio_data) // 4)
    frequencies, power_spectrum = sp_signal.welch(audio_data, fs=sample_rate, 
                                                  nperseg=nperseg, window='hann')
    
    # Convert to dB
    magnitude_db = 10 * np.log10(power_spectrum + 1e-10)
    
    # Calculate phase response using FFT with unwrapping
    # Unwrapping is essential for swept signals to show true phase progression
    fft_result = np.fft.rfft(audio_data)
    phase_rad = np.angle(fft_result)
    
    # Unwrap phase to show continuous phase progression
    # This is critical for swept sine waves and chirp signals
    phase_rad_unwrapped = np.unwrap(phase_rad)
    phase_degrees = np.degrees(phase_rad_unwrapped)
    
    # Frequency vector for phase
    phase_freq = np.fft.rfftfreq(len(audio_data), 1.0/sample_rate)
    
    logger.info(f"{title} - Max magnitude: {np.max(magnitude_db):.2f} dB")
    
    return frequencies, magnitude_db, (phase_freq, phase_degrees)


def plot_frequency_and_phase(frequencies, magnitude_db, phase_freq, phase_degrees,
                             title, output_path):
    """Plot frequency response magnitude and phase."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Magnitude plot
    ax1.semilogx(frequencies, magnitude_db, linewidth=1.5)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title(f'{title} - Magnitude Response')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([20, 20000])
    ax1.set_ylim([-80, 10])
    
    # Phase plot
    ax2.semilogx(phase_freq, phase_degrees, linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title(f'{title} - Phase Response')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([20, 20000])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_octave_bands_overlay(freq_responses, center_frequencies, output_path):
    """Plot all octave band frequency responses on one graph."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_responses)))
    
    for idx, (freq, mag_db) in enumerate(freq_responses):
        label = f"{center_frequencies[idx]:.2f} Hz" if idx < len(center_frequencies) else "Full Spectrum"
        ax.semilogx(freq, mag_db, linewidth=1.5, label=label, color=colors[idx], alpha=0.7)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Octave Band Frequency Responses - All Bands')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim([20, 20000])
    ax.set_ylim([-80, 10])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved octave band overlay to {output_path}")
    plt.close(fig)


def main():
    """Run phase and frequency response test on swept sine wave."""
    
    # Test file path
    test_file = Path("Tracks/Test Signals/SineSweep 20Hz to 20kHz.wav")
    output_dir = Path("tests/test_output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Octave Band Phase Preservation Test")
    logger.info("=" * 60)
    
    # Step 1: Load the swept sine wave
    logger.info("\nStep 1: Loading test signal...")
    audio_processor = AudioProcessor(sample_rate=44100)
    audio_data, sr = audio_processor.load_audio(test_file)
    audio_data = audio_processor.stereo_to_mono(audio_data)
    
    logger.info(f"Loaded: {len(audio_data)/sr:.2f}s at {sr} Hz, {len(audio_data)} samples")
    
    # Step 2: Analyze original signal frequency and phase response
    logger.info("\nStep 2: Analyzing original signal frequency and phase response...")
    orig_freq, orig_mag_db, (orig_phase_freq, orig_phase_deg) = analyze_frequency_response(
        audio_data, sr, "Original Signal"
    )
    
    plot_frequency_and_phase(
        orig_freq, orig_mag_db, orig_phase_freq, orig_phase_deg,
        "Original Signal",
        output_dir / "01_original_response.png"
    )
    
    # Step 3: Apply octave band filtering
    logger.info("\nStep 3: Applying octave band filtering...")
    octave_filter = OctaveBandFilter(sample_rate=sr)
    octave_bank = octave_filter.create_octave_bank(audio_data)
    
    logger.info(f"Created octave bank: {octave_bank.shape} (samples x bands)")
    logger.info(f"Bands: Full Spectrum + {len(octave_filter.OCTAVE_CENTER_FREQUENCIES)} octave bands")
    
    # Step 4: Analyze each octave band frequency response
    logger.info("\nStep 4: Analyzing each octave band frequency response...")
    octave_freq_responses = []
    center_frequencies = [0] + octave_filter.OCTAVE_CENTER_FREQUENCIES
    
    for i in range(octave_bank.shape[1]):
        band_signal = octave_bank[:, i]
        freq_label = f"{center_frequencies[i]:.2f} Hz" if i > 0 else "Full Spectrum"
        
        logger.info(f"Analyzing band {i+1}/{octave_bank.shape[1]}: {freq_label}")
        
        band_freq, band_mag_db, _ = analyze_frequency_response(
            band_signal, sr, f"Band {freq_label}"
        )
        octave_freq_responses.append((band_freq, band_mag_db))
    
    # Plot all bands on one graph
    plot_octave_bands_overlay(
        octave_freq_responses,
        center_frequencies,
        output_dir / "02_octave_bands_overlay.png"
    )
    
    # Step 5: Sum all octave bands back together
    logger.info("\nStep 5: Summing all octave bands back together...")
    reconstructed_signal = np.sum(octave_bank[:, 1:], axis=1)  # Sum all bands except full spectrum
    
    logger.info(f"Reconstructed signal: {len(reconstructed_signal)} samples")
    
    # Step 6: Analyze reconstructed signal frequency and phase response
    logger.info("\nStep 6: Analyzing reconstructed signal frequency and phase response...")
    recon_freq, recon_mag_db, (recon_phase_freq, recon_phase_deg) = analyze_frequency_response(
        reconstructed_signal, sr, "Reconstructed Signal"
    )
    
    plot_frequency_and_phase(
        recon_freq, recon_mag_db, recon_phase_freq, recon_phase_deg,
        "Reconstructed Signal (Sum of Octave Bands)",
        output_dir / "03_reconstructed_response.png"
    )
    
    # Step 7: Direct comparison - original vs reconstructed
    logger.info("\nStep 7: Creating comparison plot (Original vs Reconstructed)...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Magnitude comparison
    ax1.semilogx(orig_freq, orig_mag_db, linewidth=2, label='Original', alpha=0.7)
    ax1.semilogx(recon_freq, recon_mag_db, linewidth=1.5, label='Reconstructed', alpha=0.7, linestyle='--')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Frequency Response Comparison - Magnitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([20, 20000])
    ax1.set_ylim([-80, 10])
    
    # Phase comparison
    ax2.semilogx(orig_phase_freq, orig_phase_deg, linewidth=2, label='Original', alpha=0.7)
    ax2.semilogx(recon_phase_freq, recon_phase_deg, linewidth=1.5, label='Reconstructed', 
                 alpha=0.7, linestyle='--')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Phase Response Comparison (Unwrapped)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([20, 20000])
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_comparison.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison to {output_dir / '04_comparison.png'}")
    plt.close(fig)
    
    # Step 8: Calculate differences and metrics
    logger.info("\nStep 8: Calculating differences and metrics...")
    
    # Interpolate to common frequency axis for comparison
    common_freq = orig_freq
    recon_mag_interp = np.interp(common_freq, recon_freq, recon_mag_db)
    
    magnitude_diff = np.abs(orig_mag_db - recon_mag_interp)
    
    logger.info(f"Magnitude difference statistics:")
    logger.info(f"  Mean difference: {np.mean(magnitude_diff):.3f} dB")
    logger.info(f"  Max difference: {np.max(magnitude_diff):.3f} dB")
    logger.info(f"  Median difference: {np.median(magnitude_diff):.3f} dB")
    
    # Signal similarity (normalized cross-correlation)
    max_len = min(len(audio_data), len(reconstructed_signal))
    orig_norm = audio_data[:max_len] / np.max(np.abs(audio_data[:max_len]))
    recon_norm = reconstructed_signal[:max_len] / np.max(np.abs(reconstructed_signal[:max_len]))
    
    correlation = np.corrcoef(orig_norm, recon_norm)[0, 1]
    logger.info(f"\nSignal similarity (correlation): {correlation:.6f}")
    
    # Plot magnitude difference
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.semilogx(common_freq, magnitude_diff, linewidth=1.5, color='red', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude Difference (dB)')
    ax.set_title('Frequency Response Difference (Original - Reconstructed)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 20000])
    ax.set_ylim([0, np.max(magnitude_diff) * 1.1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_magnitude_difference.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved magnitude difference to {output_dir / '05_magnitude_difference.png'}")
    plt.close(fig)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Test signal: {test_file}")
    logger.info(f"Duration: {len(audio_data)/sr:.2f} seconds")
    logger.info(f"Sample rate: {sr} Hz")
    logger.info(f"Octave bands: {len(octave_filter.OCTAVE_CENTER_FREQUENCIES)}")
    logger.info(f"\nFrequency Response Analysis:")
    logger.info(f"  Mean magnitude difference: {np.mean(magnitude_diff):.3f} dB")
    logger.info(f"  Max magnitude difference: {np.max(magnitude_diff):.3f} dB")
    logger.info(f"\nSignal Similarity:")
    logger.info(f"  Normalized correlation: {correlation:.6f}")
    logger.info(f"\nOutput plots saved to: {output_dir}")
    logger.info("=" * 60)
    
    # Test conclusions
    if correlation > 0.95 and np.max(magnitude_diff) < 3.0:
        logger.info("\n✅ TEST PASSED: Phase and frequency response preserved well")
        logger.info("   Octave band processing maintains signal integrity")
    elif correlation > 0.85 and np.max(magnitude_diff) < 6.0:
        logger.info("\n⚠️  TEST WARNING: Some differences observed")
        logger.info("   Octave band processing has minor frequency response variations")
    else:
        logger.info("\n❌ TEST FAILED: Significant differences detected")
        logger.info("   Octave band processing may be affecting signal integrity")
    
    return correlation, magnitude_diff


if __name__ == "__main__":
    main()

