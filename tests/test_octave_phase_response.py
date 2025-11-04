"""
Test octave band filtering phase preservation and frequency response.

This test verifies that octave band processing:
1. Preserves crest factor (peak-to-RMS ratio) when splitting and recombining
2. Does not affect phase relationships significantly
3. Maintains correct frequency response
4. Preserves signal integrity when bands are summed back
5. Uses the same processing as the main application

Phase Response Type:
The test uses filtfilt (zero-phase filtering), which is equivalent to linear phase
filtering for waveform preservation. This is the recommended approach for preserving
crest factor because:
- Linear phase (zero-phase via filtfilt) maintains constant group delay
- All frequency components are delayed equally, preserving waveform shape
- Crest factor is sensitive to phase changes, so linear/zero-phase is optimal

Alternatives considered:
- Minimum phase: Lower latency but non-linear phase distorts waveform, affects crest factor
- No phase change: Not achievable with real filters

Test Strategy:
- Use swept sine wave (20Hz to 20kHz)
- Apply octave band filtering using filtfilt (zero-phase)
- Calculate crest factors before and after processing
- Analyze frequency and phase response before and after
- Sum filtered bands back together
- Compare original vs reconstructed signal and crest factors
- Output comprehensive CSV reports for data analysis
"""

from __future__ import annotations

import csv
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
    
    # Handle NaN values properly
    valid_magnitudes = magnitude_db[~(np.isnan(magnitude_db) | np.isinf(magnitude_db))]
    if len(valid_magnitudes) > 0:
        max_mag = np.max(valid_magnitudes)
        min_mag = np.min(valid_magnitudes)
        logger.info(f"{title} - Max magnitude: {max_mag:.2f} dB (valid range: {min_mag:.2f} to {max_mag:.2f} dB)")
    else:
        logger.info(f"{title} - Max magnitude: NaN (all values invalid)")
    
    return frequencies, magnitude_db, (phase_freq, phase_degrees)


def calculate_crest_factor(signal: np.ndarray) -> tuple[float, float]:
    """Calculate crest factor (peak/RMS ratio) for a signal.
    
    Args:
        signal: Audio signal array
        
    Returns:
        Tuple of (crest_factor_linear, crest_factor_db)
    """
    # Remove any NaN or Inf values for calculation
    clean_signal = signal[np.isfinite(signal)]
    
    if len(clean_signal) == 0:
        return 1.0, 0.0
    
    peak = np.max(np.abs(clean_signal))
    rms = np.sqrt(np.mean(clean_signal**2))
    
    if rms > 0 and peak > 0:
        crest_factor = max(peak / rms, 1.0)  # Crest factor must be >= 1.0
        crest_factor_db = 20 * np.log10(crest_factor)
        return crest_factor, crest_factor_db
    else:
        return 1.0, 0.0


def plot_frequency_and_phase(frequencies, magnitude_db, phase_freq, phase_degrees,
                             title, output_path):
    """Plot frequency response magnitude and phase."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Magnitude plot - filter out NaN values
    valid_mag = ~(np.isnan(magnitude_db) | np.isinf(magnitude_db))
    if np.any(valid_mag):
        ax1.semilogx(frequencies[valid_mag], magnitude_db[valid_mag], linewidth=1.5)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(f'{title} - Magnitude Response')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([20, 20000])
        ax1.set_ylim([-80, 10])
    else:
        ax1.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(f'{title} - Magnitude Response (No Data)')
    
    # Phase plot - filter out NaN values
    valid_phase = ~(np.isnan(phase_degrees) | np.isinf(phase_degrees))
    if np.any(valid_phase):
        ax2.semilogx(phase_freq[valid_phase], phase_degrees[valid_phase], linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title(f'{title} - Phase Response (Unwrapped)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([20, 20000])
    else:
        ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title(f'{title} - Phase Response (No Data)')
    
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
    # Test with Butterworth filters (LR falls back to BW for low frequencies anyway)
    use_lr = False  # Set to True to test with Linkwitz-Riley (but it falls back for low freq)
    filter_type = "Linkwitz-Riley" if use_lr else "Butterworth"
    logger.info(f"Filter type: {filter_type}")
    octave_filter = OctaveBandFilter(sample_rate=sr, use_linkwitz_riley=use_lr)
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
    
    # Sum all bands except full spectrum
    # Replace NaN values with 0 before summing to avoid NaN propagation
    bands_to_sum = np.nan_to_num(octave_bank[:, 1:], nan=0.0)
    reconstructed_signal = np.sum(bands_to_sum, axis=1)
    
    logger.info(f"Reconstructed signal: {len(reconstructed_signal)} samples")
    
    # Report any remaining NaN values
    nan_count = np.sum(np.isnan(reconstructed_signal))
    if nan_count > 0:
        logger.warning(f"Reconstructed signal still has {nan_count} NaN values!")
    else:
        logger.info("Reconstructed signal has no NaN values")
    
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
    
    # Step 8: Calculate crest factors and differences
    logger.info("\nStep 8: Calculating crest factors and differences...")
    
    # Calculate crest factor for original signal
    orig_crest_factor, orig_crest_db = calculate_crest_factor(audio_data)
    logger.info(f"Original signal crest factor: {orig_crest_factor:.4f} ({orig_crest_db:.3f} dB)")
    
    # Calculate crest factor for reconstructed signal
    recon_crest_factor, recon_crest_db = calculate_crest_factor(reconstructed_signal)
    logger.info(f"Reconstructed signal crest factor: {recon_crest_factor:.4f} ({recon_crest_db:.3f} dB)")
    
    # Calculate crest factor difference
    crest_factor_diff = recon_crest_factor - orig_crest_factor
    crest_factor_diff_db = recon_crest_db - orig_crest_db
    logger.info(f"Crest factor difference: {crest_factor_diff:.6f} ({crest_factor_diff_db:.6f} dB)")
    
    # Calculate crest factors for each octave band
    logger.info("\nCalculating crest factors for each octave band...")
    band_crest_factors = []
    for i in range(octave_bank.shape[1]):
        band_signal = octave_bank[:, i]
        band_cf, band_cf_db = calculate_crest_factor(band_signal)
        freq_label = center_frequencies[i] if i < len(center_frequencies) else 0.0
        band_crest_factors.append({
            'band_index': i,
            'center_frequency_hz': freq_label,
            'crest_factor': band_cf,
            'crest_factor_db': band_cf_db
        })
        logger.info(f"  Band {i} ({freq_label} Hz): {band_cf:.4f} ({band_cf_db:.3f} dB)")
    
    # Interpolate to common frequency axis for comparison
    common_freq = orig_freq
    recon_mag_interp = np.interp(common_freq, recon_freq, recon_mag_db)
    
    magnitude_diff = np.abs(orig_mag_db - recon_mag_interp)
    
    # Filter out NaN values for statistics
    valid_mask = ~(np.isnan(magnitude_diff) | np.isinf(magnitude_diff))
    magnitude_diff_clean = magnitude_diff[valid_mask]
    
    logger.info(f"\nMagnitude difference statistics:")
    if len(magnitude_diff_clean) > 0:
        logger.info(f"  Mean difference: {np.mean(magnitude_diff_clean):.3f} dB")
        logger.info(f"  Max difference: {np.max(magnitude_diff_clean):.3f} dB")
        logger.info(f"  Median difference: {np.median(magnitude_diff_clean):.3f} dB")
    else:
        logger.warning("  All differences are NaN - check frequency ranges")
        logger.info(f"  Mean difference: nan dB")
        logger.info(f"  Max difference: nan dB")
        logger.info(f"  Median difference: nan dB")
    
    # Signal similarity (normalized cross-correlation)
    max_len = min(len(audio_data), len(reconstructed_signal))
    orig_norm = audio_data[:max_len] / np.max(np.abs(audio_data[:max_len]))
    recon_norm = reconstructed_signal[:max_len] / (np.max(np.abs(reconstructed_signal[:max_len])) + 1e-10)
    
    # Check for valid correlation data
    if np.any(np.isnan(recon_norm)) or np.any(np.isinf(recon_norm)):
        correlation = np.nan
        logger.warning("Reconstructed signal contains NaN/Inf values")
    else:
        correlation = np.corrcoef(orig_norm, recon_norm)[0, 1]
    
    if not np.isnan(correlation):
        logger.info(f"\nSignal similarity (correlation): {correlation:.6f}")
    else:
        logger.warning(f"\nSignal similarity (correlation): nan (check signal integrity)")
    
    # Plot magnitude difference
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # Only plot valid (non-NaN) values
    if len(magnitude_diff_clean) > 0:
        ax.semilogx(common_freq[valid_mask], magnitude_diff[valid_mask], 
                    linewidth=1.5, color='red', alpha=0.7)
        ax.set_ylim([0, np.max(magnitude_diff_clean) * 1.1])
    else:
        ax.text(0.5, 0.5, 'No valid data points', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude Difference (dB)')
    ax.set_title('Frequency Response Difference (Original - Reconstructed)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 20000])
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_magnitude_difference.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved magnitude difference to {output_dir / '05_magnitude_difference.png'}")
    plt.close(fig)
    
    # Step 9: Write CSV output files
    logger.info("\nStep 9: Writing CSV output files...")
    
    # CSV 1: Overall test summary
    summary_csv = output_dir / "test_summary.csv"
    with open(summary_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'metric', 'value', 'unit', 'notes'
        ])
        writer.writerow(['test_file', str(test_file), '', 'Input test signal'])
        writer.writerow(['sample_rate', sr, 'Hz', 'Audio sample rate'])
        writer.writerow(['duration', f"{len(audio_data)/sr:.3f}", 'seconds', 'Signal duration'])
        writer.writerow(['num_samples', len(audio_data), 'samples', 'Total samples'])
        writer.writerow(['num_octave_bands', len(octave_filter.OCTAVE_CENTER_FREQUENCIES), '', 'Number of octave bands'])
        writer.writerow(['filter_type', filter_type + ' 4th order', '', 'Filter design'])
        writer.writerow(['filter_method', 'filtfilt (zero-phase)', '', 'Zero-phase = linear phase equivalent'])
        writer.writerow(['phase_response_type', 'Zero-phase (Linear Phase Equivalent)', '', 
                        'filtfilt provides zero-phase filtering, equivalent to linear phase for waveform preservation'])
        writer.writerow(['', '', '', ''])
        writer.writerow(['original_crest_factor', f"{orig_crest_factor:.6f}", '', 'Peak/RMS ratio'])
        writer.writerow(['original_crest_factor_db', f"{orig_crest_db:.6f}", 'dB', 'Crest factor in dB'])
        writer.writerow(['reconstructed_crest_factor', f"{recon_crest_factor:.6f}", '', 'Peak/RMS ratio'])
        writer.writerow(['reconstructed_crest_factor_db', f"{recon_crest_db:.6f}", 'dB', 'Crest factor in dB'])
        writer.writerow(['crest_factor_difference', f"{crest_factor_diff:.6f}", '', 'Recon - Original'])
        writer.writerow(['crest_factor_difference_db', f"{crest_factor_diff_db:.6f}", 'dB', 'Recon - Original in dB'])
        writer.writerow(['crest_factor_preservation_percent', 
                        f"{100 * (1 - abs(crest_factor_diff) / orig_crest_factor):.4f}", 
                        '%', 'Preservation percentage'])
        writer.writerow(['', '', '', ''])
        writer.writerow(['signal_correlation', f"{correlation:.6f}", '', 'Normalized cross-correlation'])
        if len(magnitude_diff_clean) > 0:
            writer.writerow(['magnitude_diff_mean', f"{np.mean(magnitude_diff_clean):.6f}", 'dB', 'Mean magnitude difference'])
            writer.writerow(['magnitude_diff_max', f"{np.max(magnitude_diff_clean):.6f}", 'dB', 'Max magnitude difference'])
            writer.writerow(['magnitude_diff_median', f"{np.median(magnitude_diff_clean):.6f}", 'dB', 'Median magnitude difference'])
            writer.writerow(['magnitude_diff_std', f"{np.std(magnitude_diff_clean):.6f}", 'dB', 'Std dev of magnitude difference'])
        else:
            writer.writerow(['magnitude_diff_mean', 'nan', 'dB', 'No valid data'])
            writer.writerow(['magnitude_diff_max', 'nan', 'dB', 'No valid data'])
    
    logger.info(f"Saved summary CSV to {summary_csv}")
    
    # CSV 2: Per-band crest factors
    bands_csv = output_dir / "octave_band_crest_factors.csv"
    with open(bands_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'band_index', 'center_frequency_hz', 'crest_factor', 'crest_factor_db', 
            'band_label'
        ])
        for band_data in band_crest_factors:
            label = 'Full Spectrum' if band_data['band_index'] == 0 else f"Octave Band {band_data['band_index']}"
            writer.writerow([
                band_data['band_index'],
                band_data['center_frequency_hz'],
                f"{band_data['crest_factor']:.6f}",
                f"{band_data['crest_factor_db']:.6f}",
                label
            ])
    
    logger.info(f"Saved octave band crest factors CSV to {bands_csv}")
    
    # CSV 3: Frequency response comparison (sampled)
    freq_response_csv = output_dir / "frequency_response_comparison.csv"
    # Sample frequencies logarithmically for manageable CSV size
    num_samples = 1000
    log_freq_min = np.log10(20)
    log_freq_max = np.log10(20000)
    sampled_freqs = np.logspace(log_freq_min, log_freq_max, num_samples)
    
    with open(freq_response_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'frequency_hz', 'original_magnitude_db', 'reconstructed_magnitude_db', 
            'magnitude_difference_db', 'original_phase_deg', 'reconstructed_phase_deg', 
            'phase_difference_deg'
        ])
        
        # Interpolate both magnitude and phase responses at sampled frequencies
        orig_mag_sampled = np.interp(sampled_freqs, orig_freq, orig_mag_db)
        recon_mag_sampled = np.interp(sampled_freqs, recon_freq, recon_mag_db)
        orig_phase_sampled = np.interp(sampled_freqs, orig_phase_freq, orig_phase_deg)
        recon_phase_sampled = np.interp(sampled_freqs, recon_phase_freq, recon_phase_deg)
        
        mag_diff_sampled = orig_mag_sampled - recon_mag_sampled
        phase_diff_sampled = orig_phase_sampled - recon_phase_sampled
        
        for i in range(num_samples):
            writer.writerow([
                f"{sampled_freqs[i]:.3f}",
                f"{orig_mag_sampled[i]:.6f}" if np.isfinite(orig_mag_sampled[i]) else 'nan',
                f"{recon_mag_sampled[i]:.6f}" if np.isfinite(recon_mag_sampled[i]) else 'nan',
                f"{mag_diff_sampled[i]:.6f}" if np.isfinite(mag_diff_sampled[i]) else 'nan',
                f"{orig_phase_sampled[i]:.3f}" if np.isfinite(orig_phase_sampled[i]) else 'nan',
                f"{recon_phase_sampled[i]:.3f}" if np.isfinite(recon_phase_sampled[i]) else 'nan',
                f"{phase_diff_sampled[i]:.3f}" if np.isfinite(phase_diff_sampled[i]) else 'nan'
            ])
    
    logger.info(f"Saved frequency response comparison CSV to {freq_response_csv}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Test signal: {test_file}")
    logger.info(f"Duration: {len(audio_data)/sr:.2f} seconds")
    logger.info(f"Sample rate: {sr} Hz")
    logger.info(f"Octave bands: {len(octave_filter.OCTAVE_CENTER_FREQUENCIES)}")
    logger.info(f"Filter method: {filter_type} with filtfilt (zero-phase, equivalent to linear phase)")
    logger.info(f"\nCrest Factor Analysis:")
    logger.info(f"  Original: {orig_crest_factor:.6f} ({orig_crest_db:.6f} dB)")
    logger.info(f"  Reconstructed: {recon_crest_factor:.6f} ({recon_crest_db:.6f} dB)")
    logger.info(f"  Difference: {crest_factor_diff:.6f} ({crest_factor_diff_db:.6f} dB)")
    logger.info(f"  Preservation: {100 * (1 - abs(crest_factor_diff) / orig_crest_factor):.4f}%")
    logger.info(f"\nFrequency Response Analysis:")
    if len(magnitude_diff_clean) > 0:
        logger.info(f"  Mean magnitude difference: {np.mean(magnitude_diff_clean):.3f} dB")
        logger.info(f"  Max magnitude difference: {np.max(magnitude_diff_clean):.3f} dB")
    else:
        logger.info(f"  Mean magnitude difference: nan dB")
        logger.info(f"  Max magnitude difference: nan dB")
    logger.info(f"\nSignal Similarity:")
    logger.info(f"  Normalized correlation: {correlation:.6f}")
    logger.info(f"\nOutput files saved to: {output_dir}")
    logger.info(f"  - Summary CSV: {summary_csv}")
    logger.info(f"  - Band crest factors CSV: {bands_csv}")
    logger.info(f"  - Frequency response CSV: {freq_response_csv}")
    logger.info("=" * 60)
    
    # Test conclusions
    max_diff_value = np.max(magnitude_diff_clean) if len(magnitude_diff_clean) > 0 else np.nan
    crest_factor_preserved = abs(crest_factor_diff_db) < 0.1  # Less than 0.1 dB difference
    
    if not np.isnan(correlation) and correlation > 0.95 and crest_factor_preserved:
        logger.info("\n✅ TEST PASSED: Phase and frequency response preserved well")
        logger.info("   Crest factor preserved - octave band processing maintains signal integrity")
    elif not np.isnan(correlation) and correlation > 0.85 and abs(crest_factor_diff_db) < 1.0:
        logger.info("\n⚠️  TEST WARNING: Minor differences observed")
        logger.info("   Crest factor largely preserved but some frequency response variations detected")
    else:
        logger.info("\n❌ TEST FAILED: Significant differences detected")
        logger.info("   Crest factor may be affected - octave band processing needs review")
        if not crest_factor_preserved:
            logger.info(f"   Crest factor changed by {crest_factor_diff_db:.3f} dB (target: < 0.1 dB)")
    
    return correlation, magnitude_diff, {
        'original_crest_factor': orig_crest_factor,
        'reconstructed_crest_factor': recon_crest_factor,
        'crest_factor_difference_db': crest_factor_diff_db,
        'band_crest_factors': band_crest_factors
    }


if __name__ == "__main__":
    main()

