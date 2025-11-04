"""Test the effect of normalization on summed frequency response."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy import signal
from src.octave_filter import OctaveBandFilter

sr = 44100
freqs = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
test_freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
nyquist = sr / 2

print("=== Testing Normalization Effect ===")

# Test without normalization
of_no_norm = OctaveBandFilter(sample_rate=sr, normalize_overlap=False)
summed_no_norm = np.zeros(len(test_freqs))
for fc in freqs:
    sos = of_no_norm.design_octave_filter_sos(fc, order=1)
    w, h = signal.sosfreqz(sos, worN=test_freqs, fs=sr)
    summed_no_norm += np.abs(h)

summed_db_no_norm = 20 * np.log10(summed_no_norm + 1e-10)
ripple_no_norm = summed_db_no_norm - 0
max_ripple_no_norm = np.max(np.abs(ripple_no_norm))
mean_ripple_no_norm = np.mean(np.abs(ripple_no_norm))
print(f"\nWithout normalization:")
print(f"  Max ripple: {max_ripple_no_norm:.2f} dB")
print(f"  Mean ripple: {mean_ripple_no_norm:.2f} dB")
print(f"  Peak gain: {np.max(summed_no_norm):.3f}")
print(f"  Min gain: {np.min(summed_no_norm):.3f}")

# Test with normalization
of_with_norm = OctaveBandFilter(sample_rate=sr, normalize_overlap=True)
norm_factors = of_with_norm._calculate_normalization_factors(freqs)
print(f"\nNormalization factors:")
for fc, factor in sorted(norm_factors.items()):
    print(f"  {fc:>7.2f}Hz: {factor:.4f}")

summed_with_norm = np.zeros(len(test_freqs))
for fc in freqs:
    sos = of_with_norm.design_octave_filter_sos(fc, order=1)
    w, h = signal.sosfreqz(sos, worN=test_freqs, fs=sr)
    norm_factor = norm_factors.get(fc, 1.0)
    summed_with_norm += np.abs(h) * norm_factor

summed_db_with_norm = 20 * np.log10(summed_with_norm + 1e-10)
ripple_with_norm = summed_db_with_norm - 0
max_ripple_with_norm = np.max(np.abs(ripple_with_norm))
mean_ripple_with_norm = np.mean(np.abs(ripple_with_norm))
print(f"\nWith normalization:")
print(f"  Max ripple: {max_ripple_with_norm:.2f} dB")
print(f"  Mean ripple: {mean_ripple_with_norm:.2f} dB")
print(f"  Peak gain: {np.max(summed_with_norm):.3f}")
print(f"  Min gain: {np.min(summed_with_norm):.3f}")

print(f"\nImprovement: {max_ripple_no_norm - max_ripple_with_norm:.2f} dB reduction in max ripple")


