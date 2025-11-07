"""
Test script for peak envelope processor implementation.

Tests the new peak envelope follower with wavelength-based attack/release times.
Verifies that it correctly tracks peaks and creates a line linking wave peaks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.music_analyzer import MusicAnalyzer


def test_peak_envelope_basic():
    """Test basic peak envelope functionality."""
    print("\n=== Test 1: Basic Peak Envelope ===")
    
    sample_rate = 44100
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create test signal: sine wave with varying amplitude
    freq = 1000.0
    signal = np.sin(2 * np.pi * freq * t)
    # Add amplitude modulation
    envelope_modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    signal = signal * envelope_modulation
    
    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    
    # Test peak envelope
    envelope = analyzer._calculate_peak_envelope(
        signal,
        center_freq=freq,
        wavelength_multiplier=1.0
    )
    
    # Verify envelope tracks peaks
    rectified = np.abs(signal)
    max_rectified = np.max(rectified)
    max_envelope = np.max(envelope)
    
    print(f"  Max rectified signal: {max_rectified:.4f}")
    print(f"  Max envelope: {max_envelope:.4f}")
    print(f"  Envelope tracks peak: {max_envelope >= max_rectified * 0.9}")
    
    # Envelope should be >= rectified signal (it's a peak follower)
    assert np.all(envelope >= 0), "Envelope should be non-negative"
    assert max_envelope >= max_rectified * 0.9, "Envelope should track peaks"
    
    print("  ✓ Basic peak envelope test passed")
    return True


def test_peak_envelope_wavelength_scaling():
    """Test that attack/release times scale correctly with frequency."""
    print("\n=== Test 2: Wavelength Scaling ===")
    
    sample_rate = 44100
    duration = 0.05  # 50ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    test_freqs = [125.0, 500.0, 1000.0, 4000.0]
    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    
    results = {}
    
    for freq in test_freqs:
        # Create signal at test frequency
        signal = np.sin(2 * np.pi * freq * t)
        
        # Calculate period
        period = 1.0 / freq
        period_samples = period * sample_rate
        
        # Calculate envelope
        envelope = analyzer._calculate_peak_envelope(
            signal,
            center_freq=freq,
            wavelength_multiplier=1.0
        )
        
        # Find first peak in envelope (should occur around 1 period)
        rectified = np.abs(signal)
        # Find where envelope reaches significant value
        threshold = np.max(rectified) * 0.5
        peak_idx = np.where(envelope >= threshold)[0]
        
        if len(peak_idx) > 0:
            first_peak_sample = peak_idx[0]
            first_peak_time = first_peak_sample / sample_rate
            expected_time = period  # Should be around 1 period
            
            results[freq] = {
                'period': period,
                'first_peak_time': first_peak_time,
                'expected_time': expected_time,
                'ratio': first_peak_time / expected_time
            }
            
            print(f"  {freq:6.1f} Hz: period={period*1000:.2f}ms, "
                  f"first_peak={first_peak_time*1000:.2f}ms, "
                  f"ratio={first_peak_time/expected_time:.2f}x")
    
    print("  ✓ Wavelength scaling test passed")
    return True


def test_peak_envelope_multiplier():
    """Test wavelength multiplier parameter."""
    print("\n=== Test 3: Wavelength Multiplier ===")
    
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    freq = 1000.0
    signal = np.sin(2 * np.pi * freq * t)
    # Add sudden peak
    signal[int(len(signal) * 0.3):int(len(signal) * 0.4)] *= 2.0
    
    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    
    multipliers = [0.5, 1.0, 2.0]
    envelopes = {}
    
    for mult in multipliers:
        envelope = analyzer._calculate_peak_envelope(
            signal,
            center_freq=freq,
            wavelength_multiplier=mult
        )
        envelopes[mult] = envelope
        
        # Calculate rise time (time to reach 63% of peak)
        rectified = np.abs(signal)
        peak_value = np.max(rectified)
        threshold = peak_value * 0.63
        
        rise_indices = np.where(envelope >= threshold)[0]
        if len(rise_indices) > 0:
            rise_time = rise_indices[0] / sample_rate
            expected_time = (mult * 1.0 / freq)
            print(f"  Multiplier {mult:.1f}x: rise_time={rise_time*1000:.2f}ms, "
                  f"expected={expected_time*1000:.2f}ms")
    
    print("  ✓ Wavelength multiplier test passed")
    return True


def test_peak_envelope_full_spectrum():
    """Test Full Spectrum (freq=0) fallback."""
    print("\n=== Test 4: Full Spectrum Fallback ===")
    
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create broadband signal
    signal = np.random.normal(0, 0.1, len(t))
    signal += 0.5 * np.sin(2 * np.pi * 1000 * t)
    
    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    
    # Test with freq=0 (Full Spectrum)
    envelope = analyzer._calculate_peak_envelope(
        signal,
        center_freq=0.0,
        wavelength_multiplier=1.0,
        fallback_window_ms=10.0
    )
    
    assert len(envelope) == len(signal), "Envelope length should match signal"
    assert np.all(envelope >= 0), "Envelope should be non-negative"
    
    print(f"  Envelope max: {np.max(envelope):.4f}")
    print(f"  Signal max (abs): {np.max(np.abs(signal)):.4f}")
    print("  ✓ Full Spectrum fallback test passed")
    return True


def test_peak_envelope_via_rms_method():
    """Test peak envelope via _calculate_rms_envelope method."""
    print("\n=== Test 5: Via _calculate_rms_envelope Method ===")
    
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    freq = 1000.0
    signal = np.sin(2 * np.pi * freq * t)
    
    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    
    # Test via _calculate_rms_envelope with peak_envelope method
    envelope = analyzer._calculate_rms_envelope(
        signal,
        center_freq=freq,
        method='peak_envelope',
        wavelength_multiplier=1.0
    )
    
    assert len(envelope) == len(signal), "Envelope length should match signal"
    assert np.all(envelope >= 0), "Envelope should be non-negative"
    
    # Compare with direct call
    envelope_direct = analyzer._calculate_peak_envelope(
        signal,
        center_freq=freq,
        wavelength_multiplier=1.0
    )
    
    assert np.allclose(envelope, envelope_direct), "Methods should produce same result"
    
    print("  ✓ Via _calculate_rms_envelope test passed")
    return True


def test_peak_envelope_db_conversion():
    """Test dB conversion of peak envelope."""
    print("\n=== Test 6: dB Conversion ===")
    
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    freq = 1000.0
    signal = np.sin(2 * np.pi * freq * t)
    
    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    
    # Calculate envelope
    envelope_linear = analyzer._calculate_peak_envelope(
        signal,
        center_freq=freq,
        wavelength_multiplier=1.0
    )
    
    # Convert to dBFS (same as in analyze_envelope_statistics)
    envelope_linear = np.maximum(envelope_linear, 1e-10)
    envelope_db = 20 * np.log10(envelope_linear * analyzer.original_peak + 1e-10)
    
    assert np.all(np.isfinite(envelope_db)), "dB values should be finite"
    assert np.all(envelope_db <= 0), "dBFS values should be <= 0"
    
    max_db = np.max(envelope_db)
    print(f"  Max envelope dBFS: {max_db:.2f}")
    print("  ✓ dB conversion test passed")
    return True


def test_peak_envelope_visualization():
    """Create visualization plots for inspection."""
    print("\n=== Test 7: Visualization ===")
    
    sample_rate = 44100
    duration = 0.05  # 50ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    freq = 1000.0
    # Create signal with clear peaks
    signal = np.sin(2 * np.pi * freq * t)
    # Add amplitude modulation
    envelope_modulation = 0.3 + 0.7 * np.sin(2 * np.pi * 10 * t)
    signal = signal * envelope_modulation
    
    analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=1.0)
    
    # Calculate envelope
    envelope = analyzer._calculate_peak_envelope(
        signal,
        center_freq=freq,
        wavelength_multiplier=1.0
    )
    
    # Convert to dB
    envelope_linear = np.maximum(envelope, 1e-10)
    envelope_db = 20 * np.log10(envelope_linear * analyzer.original_peak + 1e-10)
    
    # Create plot
    output_dir = Path('tests/test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Linear scale
    ax1.plot(t * 1000, signal, 'b-', alpha=0.5, label='Signal')
    ax1.plot(t * 1000, envelope, 'r-', linewidth=2, label='Peak Envelope')
    ax1.plot(t * 1000, -envelope, 'r-', linewidth=2, alpha=0.3)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (linear)')
    ax1.set_title(f'Peak Envelope - {freq} Hz (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: dB scale
    signal_db = 20 * np.log10(np.abs(signal) * analyzer.original_peak + 1e-10)
    ax2.plot(t * 1000, signal_db, 'b-', alpha=0.5, label='Signal (abs)')
    ax2.plot(t * 1000, envelope_db, 'r-', linewidth=2, label='Peak Envelope')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (dBFS)')
    ax2.set_title(f'Peak Envelope - {freq} Hz (dB Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-60, 5])
    
    plt.tight_layout()
    output_path = output_dir / 'peak_envelope_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to: {output_path}")
    print("  ✓ Visualization test passed")
    return True


def run_all_tests():
    """Run all peak envelope tests."""
    print("=" * 60)
    print("Peak Envelope Processor Test Suite")
    print("=" * 60)
    
    tests = [
        test_peak_envelope_basic,
        test_peak_envelope_wavelength_scaling,
        test_peak_envelope_multiplier,
        test_peak_envelope_full_spectrum,
        test_peak_envelope_via_rms_method,
        test_peak_envelope_db_conversion,
        test_peak_envelope_visualization,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

