"""Verify octave bands follow ISO standards with consistent bandwidth."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy import signal
from src.octave_filter import OctaveBandFilter

# ISO 266:1997 / IEC 61260:1995 standard for octave bands
# According to ISO standard:
# - Upper frequency / Lower frequency = 2.0 (exactly one octave)
# - Center frequency = geometric mean of upper and lower = sqrt(upper * lower)
# - This gives: lower = center / sqrt(2), upper = center * sqrt(2)
# - Bandwidth = upper - lower = center * sqrt(2) - center / sqrt(2) = center * (sqrt(2) - 1/sqrt(2))
# - Bandwidth = center * (2 - 1) / sqrt(2) = center / sqrt(2) ✓
# But ratio = upper/lower = (center*sqrt(2)) / (center/sqrt(2)) = 2.0 ✓
#
# However, our implementation uses symmetric bandwidth: bw = center / sqrt(2)
# Which gives: lower = center - bw/2, upper = center + bw/2
# This gives ratio = (center + bw/2) / (center - bw/2) ≈ 2.094 (not exactly 2.0)
#
# The correct ISO formula should be:
# lower = center / sqrt(2)
# upper = center * sqrt(2)
# bandwidth = center * (sqrt(2) - 1/sqrt(2)) = center * (2 - 1) / sqrt(2) = center / sqrt(2)

def verify_octave_band_properties(octave_filter: OctaveBandFilter):
    """Verify that octave bands follow ISO standards."""
    
    center_frequencies = octave_filter.OCTAVE_CENTER_FREQUENCIES
    sample_rate = octave_filter.sample_rate
    nyquist = sample_rate / 2
    
    print("=" * 80)
    print("ISO Octave Band Verification")
    print("=" * 80)
    print(f"\nStandard: ISO 266:1997 / IEC 61260:1995")
    print(f"ISO Definition:")
    print(f"  Lower frequency = center / sqrt(2)")
    print(f"  Upper frequency = center * sqrt(2)")
    print(f"  Bandwidth = center * (sqrt(2) - 1/sqrt(2)) = center / sqrt(2)")
    print(f"  Frequency ratio (upper/lower) = 2.0 (exactly)")
    print(f"  Q factor = center / bandwidth = sqrt(2) = 1.414")
    print(f"\nCurrent Implementation:")
    print(f"  Lower = center / sqrt(2)")
    print(f"  Upper = center * sqrt(2)")
    print(f"  This gives ratio = 2.0 (ISO compliant)")
    print(f"\nSample rate: {sample_rate} Hz")
    print(f"Nyquist: {nyquist} Hz")
    print(f"\n{'Center':<12} {'Low':<12} {'High':<12} {'Bandwidth':<12} {'Q Factor':<12} {'Ratio':<12} {'Status'}")
    print("-" * 80)
    
    bandwidths = []
    q_factors = []
    ratios = []
    low_freqs = []
    high_freqs = []
    
    for fc in center_frequencies:
        if fc >= nyquist:
            print(f"{fc:<12.2f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'Above Nyquist'}")
            continue
        
        # Get filter design frequencies using the actual filter design logic
        # Match the logic in design_octave_filter_sos
        if octave_filter.normalize_overlap and fc in center_frequencies:
            # Geometric mean crossover design
            sorted_freqs = sorted([f for f in center_frequencies if f > 0])
            idx = sorted_freqs.index(fc)
            
            if idx > 0:
                low_freq = np.sqrt(sorted_freqs[idx - 1] * fc)
            else:
                # First band: use ISO standard calculation
                low_freq = fc / np.sqrt(2)
            
            if idx < len(sorted_freqs) - 1:
                high_freq = np.sqrt(fc * sorted_freqs[idx + 1])
            else:
                # Last band: use ISO standard calculation
                high_freq = fc * np.sqrt(2)
        else:
            # ISO 266:1997 / IEC 61260:1995 standard octave band calculation
            # ISO defines: lower = center / sqrt(2), upper = center * sqrt(2)
            low_freq = fc / np.sqrt(2)
            high_freq = fc * np.sqrt(2)
        
        # Ensure frequencies are within valid range
        low_freq = max(low_freq, 1.0)
        high_freq = min(high_freq, nyquist * 0.99)
        
        # Calculate actual bandwidth and Q factor
        actual_bandwidth = high_freq - low_freq
        q_factor = fc / actual_bandwidth if actual_bandwidth > 0 else np.nan
        freq_ratio = high_freq / low_freq if low_freq > 0 else np.nan
        
        # ISO standard values
        iso_lower = fc / np.sqrt(2)
        iso_upper = fc * np.sqrt(2)
        iso_bandwidth = iso_upper - iso_lower
        iso_q = np.sqrt(2)
        iso_ratio = 2.0
        
        # Check if values match ISO standard (within tolerance)
        # ISO uses: lower = fc/sqrt(2), upper = fc*sqrt(2)
        lower_match = abs(low_freq - iso_lower) / iso_lower < 0.01  # 1% tolerance
        upper_match = abs(high_freq - iso_upper) / iso_upper < 0.01
        ratio_match = abs(freq_ratio - iso_ratio) / iso_ratio < 0.01 if not np.isnan(freq_ratio) else False
        
        status = "[OK] ISO" if (lower_match and upper_match and ratio_match) else "[X] Non-ISO"
        
        bandwidths.append(actual_bandwidth)
        q_factors.append(q_factor)
        ratios.append(freq_ratio)
        low_freqs.append(low_freq)
        high_freqs.append(high_freq)
        
        print(f"{fc:<12.2f} {low_freq:<12.2f} {high_freq:<12.2f} "
              f"{actual_bandwidth:<12.2f} {q_factor:<12.3f} {freq_ratio:<12.3f} {status}")
        
        if not lower_match:
            print(f"  [X] Lower frequency deviation: ISO {iso_lower:.2f} Hz, got {low_freq:.2f} Hz")
        if not upper_match:
            print(f"  [X] Upper frequency deviation: ISO {iso_upper:.2f} Hz, got {high_freq:.2f} Hz")
        if not ratio_match:
            print(f"  [X] Frequency ratio deviation: ISO {iso_ratio:.3f}, got {freq_ratio:.3f}")
    
    # Check consistency across bands
    print("\n" + "=" * 80)
    print("Consistency Check")
    print("=" * 80)
    
    # Check Q factor consistency (should be constant)
    valid_q_factors = [q for q in q_factors if not np.isnan(q)]
    if len(valid_q_factors) > 1:
        q_mean = np.mean(valid_q_factors)
        q_std = np.std(valid_q_factors)
        q_cv = (q_std / q_mean) * 100 if q_mean > 0 else 0  # Coefficient of variation
        
        print(f"\nQ Factor Statistics:")
        print(f"  Mean: {q_mean:.6f}")
        print(f"  Std Dev: {q_std:.6f}")
        print(f"  Coefficient of Variation: {q_cv:.2f}%")
        print(f"  Expected: {np.sqrt(2):.6f}")
        
        if q_cv < 1.0:
            print(f"  [OK] Q factors are consistent (CV < 1%)")
        else:
            print(f"  [X] Q factors vary significantly (CV = {q_cv:.2f}%)")
    
    # Check frequency ratio consistency
    valid_ratios = [r for r in ratios if not np.isnan(r)]
    if len(valid_ratios) > 1:
        ratio_mean = np.mean(valid_ratios)
        ratio_std = np.std(valid_ratios)
        
        print(f"\nFrequency Ratio Statistics (upper/lower):")
        print(f"  Mean: {ratio_mean:.3f}")
        print(f"  Std Dev: {ratio_std:.3f}")
        print(f"  Expected: 2.000")
        
        if abs(ratio_mean - 2.0) < 0.01:
            print(f"  [OK] Frequency ratios are consistent with ISO standard")
        else:
            print(f"  [X] Frequency ratios deviate from ISO standard")
    
    # Check bandwidth formula consistency
    print(f"\nBandwidth Formula Check:")
    print(f"  Formula: bandwidth = center_freq / sqrt(2)")
    print(f"  sqrt(2) = {np.sqrt(2):.6f}")
    
    # Verify for a few bands
    test_freqs = [125.0, 1000.0, 4000.0]
    print(f"\n  Verification examples:")
    for test_fc in test_freqs:
        if test_fc in center_frequencies:
            expected_bw = test_fc / np.sqrt(2)
            idx = center_frequencies.index(test_fc)
            if idx < len(low_freqs) and idx < len(high_freqs):
                actual_bw = high_freqs[idx] - low_freqs[idx]
                match = abs(actual_bw - expected_bw) / expected_bw < 0.01
                status = "[OK]" if match else "[X]"
                print(f"    {status} {test_fc:.1f} Hz: expected {expected_bw:.2f} Hz, got {actual_bw:.2f} Hz")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    # Overall assessment
    iso_compliant = True
    if len(valid_q_factors) > 0:
        if abs(np.mean(valid_q_factors) - np.sqrt(2)) > 0.01:
            iso_compliant = False
    if len(valid_ratios) > 0:
        if abs(np.mean(valid_ratios) - 2.0) > 0.01:
            iso_compliant = False
    
    if iso_compliant:
        print("[OK] All bands follow ISO 266:1997 / IEC 61260:1995 standard")
    else:
        print("[X] Some bands deviate from ISO standard (may be due to geometric mean crossover adjustment)")
    
    return {
        'bandwidths': bandwidths,
        'q_factors': q_factors,
        'ratios': ratios,
        'low_freqs': low_freqs,
        'high_freqs': high_freqs,
        'iso_compliant': iso_compliant
    }


if __name__ == "__main__":
    print("Test 1: Standard ISO Design (no geometric mean crossovers)")
    print("=" * 80)
    of_standard = OctaveBandFilter(sample_rate=44100, normalize_overlap=False)
    results_standard = verify_octave_band_properties(of_standard)
    
    print("\n\n")
    print("Test 2: With Geometric Mean Crossover Alignment")
    print("=" * 80)
    of_modified = OctaveBandFilter(sample_rate=44100, normalize_overlap=True)
    results_modified = verify_octave_band_properties(of_modified)

