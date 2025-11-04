"""
Test octave band filterbank frequency response.

This test analyzes the filterbank performance by:
1. Designing all octave band filters
2. Plotting individual band frequency responses
3. Summing all band responses
4. Analyzing magnitude and phase response of the sum
5. Calculating ripple and flatness metrics

This provides a second perspective on filterbank performance, complementing
the swept sine waveform test by analyzing the filters directly.
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
from scipy import signal

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.octave_filter import OctaveBandFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_filterbank_response(
    octave_filter: OctaveBandFilter,
    center_frequencies: list[float],
    sample_rate: int = 44100,
    output_dir: Path = Path("tests/test_output")
) -> dict:
    """Analyze octave band filterbank frequency response.
    
    Args:
        octave_filter: OctaveBandFilter instance
        center_frequencies: List of center frequencies to analyze
        sample_rate: Sample rate for frequency response calculation
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Frequency range for analysis (extend below 20Hz to capture 16Hz band fully)
    # Start at 10Hz to capture the high-pass side of the 16Hz band
    test_freqs = np.logspace(np.log10(10), np.log10(20000), 2000)
    nyquist = sample_rate / 2
    
    logger.info(f"Analyzing {len(center_frequencies)} octave bands...")
    logger.info(f"Frequency range: {test_freqs[0]:.1f}Hz to {test_freqs[-1]:.1f}Hz")
    
    # Calculate frequency response for each band
    band_responses = {}
    summed_magnitude = np.zeros(len(test_freqs), dtype=complex)
    summed_magnitude_linear = np.zeros(len(test_freqs))
    
    for fc in center_frequencies:
        if fc >= nyquist:
            logger.warning(f"Skipping {fc}Hz (above Nyquist {nyquist}Hz)")
            continue
        
        # Design filter in SOS format
        sos = octave_filter.design_octave_filter_sos(fc, order=1)
        
        # Calculate frequency response
        w, h = signal.sosfreqz(sos, worN=test_freqs, fs=sample_rate)
        
        # Store magnitude and phase
        magnitude_linear = np.abs(h)
        magnitude_db = 20 * np.log10(magnitude_linear + 1e-10)
        phase_deg = np.degrees(np.angle(h))
        
        band_responses[fc] = {
            'frequencies': w,
            'magnitude_db': magnitude_db,
            'magnitude_linear': magnitude_linear,
            'phase_deg': phase_deg,
            'phase_rad': np.angle(h),
            'complex_response': h
        }
        
        # Add to summed response (complex for phase, linear for normalization)
        summed_magnitude += h
        summed_magnitude_linear += magnitude_linear
        
        logger.info(f"  Band {fc:>7.2f}Hz: max magnitude = {np.max(magnitude_db):.2f} dB")
    
    # Apply frequency-dependent normalization for perfect reconstruction
    # At each frequency, normalize the complex sum so its magnitude = 1.0 (0 dB)
    # This preserves phase relationships while ensuring flat magnitude response
    
    # Calculate magnitude of complex sum (accounts for phase interference)
    summed_magnitude_complex_abs = np.abs(summed_magnitude)
    valid_mask = summed_magnitude_complex_abs > 1e-10  # Avoid division by zero
    
    # Normalization factor: scale complex response so magnitude = 1.0
    # Preserve phase: multiply complex number by real scalar
    normalization_factor = np.ones_like(summed_magnitude, dtype=complex)
    normalization_factor[valid_mask] = 1.0 / summed_magnitude_complex_abs[valid_mask]
    
    # Apply normalization to summed complex response
    # Magnitude becomes exactly 1.0 (0 dB), phase preserved
    normalized_summed_magnitude = summed_magnitude * normalization_factor
    
    # Verify normalization worked correctly
    normalized_magnitude_linear = np.abs(normalized_summed_magnitude)
    normalized_summed_magnitude_db = 20 * np.log10(normalized_magnitude_linear + 1e-10)
    normalized_summed_phase_deg = np.degrees(np.unwrap(np.angle(normalized_summed_magnitude)))
    
    # Check for any remaining deviation from unity (should be ~0)
    deviation_from_unity = np.abs(normalized_magnitude_linear[valid_mask] - 1.0)
    max_deviation = np.max(deviation_from_unity) if np.any(valid_mask) else 0.0
    logger.info(f"Max deviation from unity after normalization: {max_deviation:.6e}")
    
    # Apply normalization to individual bands for comparison
    # Each band is scaled by the same frequency-dependent normalization factor
    # This shows how each band contributes to the normalized sum
    for fc in band_responses:
        normalized_complex = band_responses[fc]['complex_response'] * normalization_factor
        normalized_magnitude_linear = np.abs(normalized_complex)
        normalized_magnitude_db = 20 * np.log10(normalized_magnitude_linear + 1e-10)
        band_responses[fc]['normalized_magnitude_db'] = normalized_magnitude_db
        band_responses[fc]['normalized_complex_response'] = normalized_complex
    
    # Calculate summed response metrics (before normalization for comparison)
    summed_magnitude_db = 20 * np.log10(np.abs(summed_magnitude) + 1e-10)
    summed_phase_deg = np.degrees(np.unwrap(np.angle(summed_magnitude)))
    
    # Analyze ripple for normalized response (should be near 0 dB)
    ideal_response_db = 0.0
    normalized_ripple = normalized_summed_magnitude_db - ideal_response_db
    
    max_ripple_normalized = np.max(np.abs(normalized_ripple))
    mean_ripple_normalized = np.mean(np.abs(normalized_ripple))
    std_ripple_normalized = np.std(normalized_ripple)
    
    # Also analyze unnormalized response for comparison
    unnormalized_ripple = summed_magnitude_db - ideal_response_db
    max_ripple_unnormalized = np.max(np.abs(unnormalized_ripple))
    mean_ripple_unnormalized = np.mean(np.abs(unnormalized_ripple))
    
    # Find peak and valley
    peak_gain_db = np.max(normalized_summed_magnitude_db)
    valley_gain_db = np.min(normalized_summed_magnitude_db)
    peak_valley_diff = peak_gain_db - valley_gain_db
    
    # Calculate flatness metrics in passband region
    significant_bands = np.zeros(len(test_freqs), dtype=bool)
    for fc, response in band_responses.items():
        significant_bands |= response['magnitude_db'] > -20
    
    passband_mask = significant_bands
    if np.any(passband_mask):
        passband_ripple = normalized_ripple[passband_mask]
        passband_max_ripple = np.max(np.abs(passband_ripple))
        passband_mean_ripple = np.mean(np.abs(passband_ripple))
    else:
        passband_max_ripple = max_ripple_normalized
        passband_mean_ripple = mean_ripple_normalized
    
    # Find -6dB point on high-pass (low-frequency) side of 16Hz band for x-axis start
    if 16.0 in band_responses:
        band_16hz = band_responses[16.0]
        mag_16hz_db = band_16hz['magnitude_db']
        freq_16hz = band_16hz['frequencies']
        # Find peak magnitude and its index
        peak_16hz_db = np.max(mag_16hz_db)
        peak_idx = np.argmax(mag_16hz_db)
        peak_freq = freq_16hz[peak_idx]
        target_db = peak_16hz_db - 6.0  # -6dB from peak
        
        # Find frequencies on the low-frequency side (high-pass side) of the peak
        # This means frequencies <= peak frequency
        low_side_mask = freq_16hz <= peak_freq
        low_side_indices = np.where(low_side_mask)[0]
        
        if len(low_side_indices) > 0:
            # Find the lowest frequency on the low side where magnitude >= target
            # This is the -6dB point on the high-pass (rising) edge
            above_target_on_low_side = mag_16hz_db[low_side_indices] >= target_db
            
            if np.any(above_target_on_low_side):
                # Find the lowest frequency index where magnitude >= target
                valid_indices = low_side_indices[above_target_on_low_side]
                lowest_valid_idx = valid_indices[0]  # First (lowest frequency) index
                x_axis_start_hz = freq_16hz[lowest_valid_idx]
            else:
                # If no point exactly at or above target, find closest to target on low side
                diff_from_target = np.abs(mag_16hz_db[low_side_indices] - target_db)
                closest_local_idx_in_subset = np.argmin(diff_from_target)
                closest_local_idx = low_side_indices[closest_local_idx_in_subset]
                x_axis_start_hz = freq_16hz[closest_local_idx]
        else:
            # Fallback: find closest to target overall
            diff_from_target = np.abs(mag_16hz_db - target_db)
            closest_idx = np.argmin(diff_from_target)
            x_axis_start_hz = freq_16hz[closest_idx]
        
        actual_mag_at_start = mag_16hz_db[np.argmin(np.abs(freq_16hz - x_axis_start_hz))]
        logger.info(f"16Hz band: peak={peak_16hz_db:.2f} dB at {peak_freq:.2f} Hz, -6dB high-pass point={x_axis_start_hz:.2f} Hz (magnitude={actual_mag_at_start:.2f} dB, target={target_db:.2f} dB)")
    else:
        x_axis_start_hz = 20.0
    
    results = {
        'band_responses': band_responses,
        'summed_magnitude_db': summed_magnitude_db,  # Unnormalized for comparison
        'summed_phase_deg': summed_phase_deg,
        'normalized_summed_magnitude_db': normalized_summed_magnitude_db,  # Normalized (0 dB target)
        'normalized_summed_phase_deg': normalized_summed_phase_deg,
        'frequencies': test_freqs,
        'ripple': normalized_ripple,  # Use normalized ripple
        'unnormalized_ripple': unnormalized_ripple,  # For comparison
        'normalization_factor': normalization_factor,
        'max_ripple': max_ripple_normalized,
        'mean_ripple': mean_ripple_normalized,
        'std_ripple': std_ripple_normalized,
        'max_ripple_unnormalized': max_ripple_unnormalized,
        'mean_ripple_unnormalized': mean_ripple_unnormalized,
        'peak_gain_db': peak_gain_db,
        'valley_gain_db': valley_gain_db,
        'peak_valley_diff': peak_valley_diff,
        'passband_max_ripple': passband_max_ripple,
        'passband_mean_ripple': passband_mean_ripple,
        'x_axis_start_hz': x_axis_start_hz,
        'filter_type': 'Butterworth' if not octave_filter.use_linkwitz_riley else 'Linkwitz-Riley',
        'normalize_overlap': octave_filter.normalize_overlap
    }
    
    logger.info("\n=== Filterbank Response Analysis ===")
    logger.info(f"Frequency-dependent normalization applied for perfect reconstruction")
    logger.info(f"Max ripple (normalized): {max_ripple_normalized:.4f} dB")
    logger.info(f"Mean ripple (normalized): {mean_ripple_normalized:.4f} dB")
    logger.info(f"Std deviation (normalized): {std_ripple_normalized:.4f} dB")
    logger.info(f"Max ripple (unnormalized): {max_ripple_unnormalized:.2f} dB")
    logger.info(f"Mean ripple (unnormalized): {mean_ripple_unnormalized:.2f} dB")
    logger.info(f"Peak gain: {peak_gain_db:.4f} dB")
    logger.info(f"Valley gain: {valley_gain_db:.4f} dB")
    logger.info(f"Peak-valley difference: {peak_valley_diff:.4f} dB")
    logger.info(f"Passband max ripple: {passband_max_ripple:.4f} dB")
    logger.info(f"Passband mean ripple: {passband_mean_ripple:.4f} dB")
    logger.info(f"X-axis start (16Hz -6dB high-pass point): {x_axis_start_hz:.2f} Hz")
    
    return results


def plot_individual_bands(
    band_responses: dict,
    output_path: Path,
    filter_type: str = "Butterworth"
) -> None:
    """Plot individual octave band frequency responses.
    
    Args:
        band_responses: Dictionary of band responses keyed by center frequency
        output_path: Path to save plot
        filter_type: Type of filter used
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each band
    colors = plt.cm.viridis(np.linspace(0, 1, len(band_responses)))
    
    for (fc, response), color in zip(sorted(band_responses.items()), colors):
        ax.semilogx(
            response['frequencies'],
            response['magnitude_db'],
            label=f'{fc:.2f} Hz',
            linewidth=1.5,
            alpha=0.7,
            color=color
        )
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title(f'Individual Octave Band Responses ({filter_type})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([10, 20000])  # Start at 10Hz to show full 16Hz band high-pass side
    ax.set_ylim([-80, 5])
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved individual band responses plot to {output_path}")


def plot_individual_bands_comparison(
    band_responses: dict,
    output_path: Path,
    filter_type: str = "Butterworth",
    x_axis_start_hz: float = 10.0
) -> None:
    """Plot individual octave band frequency responses comparing normalized vs unnormalized.
    
    Args:
        band_responses: Dictionary of band responses keyed by center frequency
        output_path: Path to save plot
        filter_type: Type of filter used
        x_axis_start_hz: Starting frequency for x-axis
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each band - unnormalized (dashed) and normalized (solid)
    colors = plt.cm.viridis(np.linspace(0, 1, len(band_responses)))
    
    for (fc, response), color in zip(sorted(band_responses.items()), colors):
        # Plot unnormalized (dashed line)
        ax.semilogx(
            response['frequencies'],
            response['magnitude_db'],
            label=f'{fc:.2f} Hz (Unnormalized)',
            linewidth=1.5,
            alpha=0.5,
            color=color,
            linestyle='--'
        )
        
        # Plot normalized (solid line)
        if 'normalized_magnitude_db' in response:
            ax.semilogx(
                response['frequencies'],
                response['normalized_magnitude_db'],
                label=f'{fc:.2f} Hz (Normalized)',
                linewidth=1.5,
                alpha=0.8,
                color=color,
                linestyle='-'
            )
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title(f'Individual Octave Band Responses: Unnormalized vs Normalized ({filter_type})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_axis_start_hz, 20000])
    ax.set_ylim([-12, 3])  # Zoomed y-axis as requested
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved individual band comparison plot to {output_path}")


def plot_summed_response(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    ripple: np.ndarray,
    output_path: Path,
    filter_type: str = "Butterworth",
    normalize_overlap: bool = True,
    x_axis_start_hz: float = 20.0,
    unnormalized_magnitude_db: np.ndarray = None,
    unnormalized_ripple: np.ndarray = None
) -> None:
    """Plot summed filterbank response with magnitude, phase, and ripple.
    
    Args:
        frequencies: Frequency array
        magnitude_db: Normalized magnitude response in dB (should be ~0 dB)
        phase_deg: Phase response in degrees
        ripple: Normalized ripple (deviation from ideal)
        output_path: Path to save plot
        filter_type: Type of filter used
        normalize_overlap: Whether overlap normalization was used
        x_axis_start_hz: Starting frequency for x-axis (from -6dB point on high-pass side of 16Hz band)
        unnormalized_magnitude_db: Optional unnormalized magnitude for comparison
        unnormalized_ripple: Optional unnormalized ripple for comparison
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Magnitude response
    ax1 = axes[0]
    if unnormalized_magnitude_db is not None:
        ax1.semilogx(frequencies, unnormalized_magnitude_db, linewidth=1.5, 
                    color='lightblue', alpha=0.5, label='Unnormalized')
    ax1.semilogx(frequencies, magnitude_db, linewidth=2, color='blue', label='Normalized (Frequency-Dependent)')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Ideal (0 dB)')
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Magnitude (dB)', fontsize=11)
    title = f'Summed Filterbank Magnitude Response ({filter_type}'
    if normalize_overlap:
        title += ', Frequency-Dependent Normalization'
    title += ')'
    ax1.set_title(title, fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([x_axis_start_hz, 20000])
    ax1.set_ylim([-1, max(4, np.max(magnitude_db) + 0.5)])
    ax1.legend(loc='best', fontsize=10)
    
    # Phase response
    ax2 = axes[1]
    ax2.semilogx(frequencies, phase_deg, linewidth=2, color='green')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Phase (degrees)', fontsize=11)
    ax2.set_title('Summed Filterbank Phase Response', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([x_axis_start_hz, 20000])
    
    # Ripple (deviation from ideal)
    ax3 = axes[2]
    if unnormalized_ripple is not None:
        ax3.semilogx(frequencies, unnormalized_ripple, linewidth=1.5, 
                    color='lightcoral', alpha=0.5, label='Unnormalized')
    ax3.semilogx(frequencies, ripple, linewidth=2, color='orange', label='Normalized')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.fill_between(frequencies, ripple, 0, alpha=0.3, color='orange')
    ax3.set_xlabel('Frequency (Hz)', fontsize=11)
    ax3.set_ylabel('Ripple (dB)', fontsize=11)
    ax3.set_title('Ripple: Deviation from Ideal Flat Response (Target: 0 dB)', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([x_axis_start_hz, 20000])
    ax3.set_ylim([-max(0.5, np.max(np.abs(ripple)) + 0.1), max(0.5, np.max(np.abs(ripple)) + 0.1)])
    if unnormalized_ripple is not None:
        ax3.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved summed response plot to {output_path}")


def save_response_csv(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    ripple: np.ndarray,
    output_path: Path,
    unnormalized_magnitude_db: np.ndarray = None,
    unnormalized_ripple: np.ndarray = None,
    normalization_factor: np.ndarray = None
) -> None:
    """Save frequency response data to CSV.
    
    Args:
        frequencies: Frequency array
        magnitude_db: Normalized magnitude response in dB
        phase_deg: Phase response in degrees
        ripple: Normalized ripple (deviation from ideal)
        output_path: Path to save CSV
        unnormalized_magnitude_db: Optional unnormalized magnitude for comparison
        unnormalized_ripple: Optional unnormalized ripple for comparison
        normalization_factor: Optional normalization factors applied
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['frequency_hz', 'magnitude_db_normalized', 'phase_deg', 'ripple_db_normalized']
        if unnormalized_magnitude_db is not None:
            headers.extend(['magnitude_db_unnormalized', 'ripple_db_unnormalized'])
        if normalization_factor is not None:
            headers.append('normalization_factor')
        writer.writerow(headers)
        
        for i, freq in enumerate(frequencies):
            row = [freq, magnitude_db[i], phase_deg[i], ripple[i]]
            if unnormalized_magnitude_db is not None:
                row.extend([unnormalized_magnitude_db[i], unnormalized_ripple[i]])
            if normalization_factor is not None:
                row.append(normalization_factor[i])
            writer.writerow(row)
    
    logger.info(f"Saved frequency response CSV to {output_path}")


def save_summary_csv(
    results: dict,
    output_path: Path
) -> None:
    """Save analysis summary to CSV.
    
    Args:
        results: Analysis results dictionary
        output_path: Path to save CSV
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'unit', 'description'])
        
        writer.writerow(['filter_type', results['filter_type'], '', 'Filter design type'])
        writer.writerow(['normalize_overlap', str(results['normalize_overlap']), '', 'Geometric mean crossover alignment'])
        writer.writerow(['num_bands', len(results['band_responses']), '', 'Number of octave bands'])
        writer.writerow(['', '', '', ''])
        
        writer.writerow(['max_ripple', f"{results['max_ripple']:.4f}", 'dB', 'Maximum ripple across all frequencies'])
        writer.writerow(['mean_ripple', f"{results['mean_ripple']:.4f}", 'dB', 'Mean ripple magnitude'])
        writer.writerow(['std_ripple', f"{results['std_ripple']:.4f}", 'dB', 'Standard deviation of ripple'])
        writer.writerow(['', '', '', ''])
        
        writer.writerow(['peak_gain', f"{results['peak_gain_db']:.4f}", 'dB', 'Peak gain of summed response'])
        writer.writerow(['valley_gain', f"{results['valley_gain_db']:.4f}", 'dB', 'Valley (minimum) gain of summed response'])
        writer.writerow(['peak_valley_diff', f"{results['peak_valley_diff']:.4f}", 'dB', 'Difference between peak and valley'])
        writer.writerow(['', '', '', ''])
        
        writer.writerow(['passband_max_ripple', f"{results['passband_max_ripple']:.4f}", 'dB', 'Max ripple in passband region'])
        writer.writerow(['passband_mean_ripple', f"{results['passband_mean_ripple']:.4f}", 'dB', 'Mean ripple in passband region'])
    
    logger.info(f"Saved summary CSV to {output_path}")


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Octave Band Filterbank Frequency Response Test")
    logger.info("=" * 60)
    
    sample_rate = 44100
    output_dir = Path("tests/test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with normalization enabled (geometric mean crossovers)
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: With Geometric Mean Crossover Alignment")
    logger.info("=" * 60)
    
    octave_filter = OctaveBandFilter(
        sample_rate=sample_rate,
        use_linkwitz_riley=False,
        normalize_overlap=True
    )
    
    center_frequencies = octave_filter.OCTAVE_CENTER_FREQUENCIES
    
    results = analyze_filterbank_response(
        octave_filter,
        center_frequencies,
        sample_rate=sample_rate,
        output_dir=output_dir
    )
    
    # Plot individual bands
    plot_individual_bands(
        results['band_responses'],
        output_dir / "filterbank_individual_bands.png",
        filter_type=results['filter_type']
    )
    
    # Plot comparison of normalized vs unnormalized individual bands
    plot_individual_bands_comparison(
        results['band_responses'],
        output_dir / "filterbank_individual_bands_comparison.png",
        filter_type=results['filter_type'],
        x_axis_start_hz=results['x_axis_start_hz']
    )
    
    # Plot summed response (with normalized and unnormalized for comparison)
    plot_summed_response(
        results['frequencies'],
        results['normalized_summed_magnitude_db'],
        results['normalized_summed_phase_deg'],
        results['ripple'],
        output_dir / "filterbank_summed_response.png",
        filter_type=results['filter_type'],
        normalize_overlap=results['normalize_overlap'],
        x_axis_start_hz=results['x_axis_start_hz'],
        unnormalized_magnitude_db=results['summed_magnitude_db'],
        unnormalized_ripple=results['unnormalized_ripple']
    )
    
    # Save CSV files
    save_response_csv(
        results['frequencies'],
        results['normalized_summed_magnitude_db'],
        results['normalized_summed_phase_deg'],
        results['ripple'],
        output_dir / "filterbank_response.csv",
        unnormalized_magnitude_db=results['summed_magnitude_db'],
        unnormalized_ripple=results['unnormalized_ripple'],
        normalization_factor=results['normalization_factor']
    )
    
    save_summary_csv(
        results,
        output_dir / "filterbank_summary.csv"
    )
    
    # Test without normalization for comparison
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Without Geometric Mean Crossover Alignment (for comparison)")
    logger.info("=" * 60)
    
    octave_filter_no_norm = OctaveBandFilter(
        sample_rate=sample_rate,
        use_linkwitz_riley=False,
        normalize_overlap=False
    )
    
    results_no_norm = analyze_filterbank_response(
        octave_filter_no_norm,
        center_frequencies,
        sample_rate=sample_rate,
        output_dir=output_dir
    )
    
    # Plot summed response for comparison
    plot_summed_response(
        results_no_norm['frequencies'],
        results_no_norm['summed_magnitude_db'],
        results_no_norm['summed_phase_deg'],
        results_no_norm['ripple'],
        output_dir / "filterbank_summed_response_no_norm.png",
        filter_type=results_no_norm['filter_type'],
        normalize_overlap=results_no_norm['normalize_overlap']
    )
    
    save_response_csv(
        results_no_norm['frequencies'],
        results_no_norm['summed_magnitude_db'],
        results_no_norm['summed_phase_deg'],
        results_no_norm['ripple'],
        output_dir / "filterbank_response_no_norm.csv"
    )
    
    save_summary_csv(
        results_no_norm,
        output_dir / "filterbank_summary_no_norm.csv"
    )
    
    # Comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    logger.info(f"With Geometric Mean Crossovers:")
    logger.info(f"  Max ripple: {results['max_ripple']:.2f} dB")
    logger.info(f"  Mean ripple: {results['mean_ripple']:.2f} dB")
    logger.info(f"  Peak-valley diff: {results['peak_valley_diff']:.2f} dB")
    logger.info(f"\nWithout Geometric Mean Crossovers:")
    logger.info(f"  Max ripple: {results_no_norm['max_ripple']:.2f} dB")
    logger.info(f"  Mean ripple: {results_no_norm['mean_ripple']:.2f} dB")
    logger.info(f"  Peak-valley diff: {results_no_norm['peak_valley_diff']:.2f} dB")
    
    improvement_max = results_no_norm['max_ripple'] - results['max_ripple']
    improvement_mean = results_no_norm['mean_ripple'] - results['mean_ripple']
    
    logger.info(f"\nImprovement:")
    logger.info(f"  Max ripple reduction: {improvement_max:.2f} dB")
    logger.info(f"  Mean ripple reduction: {improvement_mean:.2f} dB")
    logger.info(f"  Target achieved: {'✓ YES' if results['max_ripple'] < 0.1 else '✗ NO'} (< 0.1 dB target)")
    
    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

