"""
Test script to compare envelope detection methods.

This script tests different envelope detection approaches:
1. Rectification + Low-Pass Filter (user's suggestion)
2. Frequency-Relative RMS Window
3. Hilbert Transform Envelope
4. Current Fixed 10ms RMS Window

Saves comparison plots to tests/test_output/
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
import matplotlib.pyplot as plt
from pathlib import Path


def method1_rectification_lpf(signal: np.ndarray, sample_rate: float,
                               center_freq: float) -> np.ndarray:
    """
    Method 1: Rectification + Low-Pass Filter at 4x center frequency.
    
    User's suggested approach:
    1. Rectify signal (make all positive)
    2. Apply low-pass filter at 4x center frequency
    """
    # Rectify: take absolute value
    rectified = np.abs(signal)
    
    # Design low-pass filter at 4x center frequency
    lpf_cutoff = 4.0 * center_freq
    nyquist = sample_rate / 2.0
    
    # Ensure cutoff is below Nyquist
    lpf_cutoff = min(lpf_cutoff, nyquist * 0.95)
    
    # Design Butterworth filter (4th order for steep rolloff)
    sos = sp_signal.butter(4, lpf_cutoff, btype='low', fs=sample_rate, output='sos')
    
    # Apply filter
    envelope = sp_signal.sosfilt(sos, rectified)
    
    return envelope


def method2_frequency_relative_rms(signal: np.ndarray, sample_rate: float,
                                   center_freq: float,
                                   num_wavelengths: float = 20) -> np.ndarray:
    """
    Method 2: Frequency-relative RMS window.
    
    Calculate window size based on center frequency:
    window_time = num_wavelengths / center_freq
    """
    if center_freq > 0:
        period = 1.0 / center_freq
        window_time = num_wavelengths * period
    else:
        # Fallback for Full Spectrum
        window_time = 0.01  # 10ms
    
    window_samples = int(window_time * sample_rate)
    window_samples = max(1, min(window_samples, len(signal) // 2))
    
    # Rectangular window
    window = np.ones(window_samples) / window_samples
    
    # RMS calculation
    signal_squared = signal ** 2
    rms_squared = sp_signal.fftconvolve(signal_squared, window, mode='same')
    rms_squared = np.maximum(rms_squared, 0.0)
    envelope = np.sqrt(rms_squared)
    
    return envelope


def method3_hilbert_envelope(signal: np.ndarray, sample_rate: float,
                             center_freq: float) -> np.ndarray:
    """
    Method 3: Hilbert Transform Envelope.
    
    Uses analytical signal to extract envelope.
    """
    # Apply Hilbert transform
    analytic_signal = sp_signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    
    # Optional: smooth with low-pass filter to reduce artifacts
    nyquist = sample_rate / 2.0
    smooth_cutoff = min(center_freq * 4, nyquist * 0.95)
    sos = sp_signal.butter(2, smooth_cutoff, btype='low', fs=sample_rate, output='sos')
    envelope = sp_signal.sosfilt(sos, envelope)
    
    return envelope


def method4_current_fixed_rms(signal: np.ndarray, sample_rate: float,
                              window_ms: float = 10.0) -> np.ndarray:
    """
    Method 4: Current approach - Fixed 10ms RMS window.
    """
    window_samples = int(window_ms * sample_rate / 1000)
    window_samples = max(1, min(window_samples, len(signal) // 2))
    
    window = np.ones(window_samples) / window_samples
    
    signal_squared = signal ** 2
    rms_squared = sp_signal.fftconvolve(signal_squared, window, mode='same')
    rms_squared = np.maximum(rms_squared, 0.0)
    envelope = np.sqrt(rms_squared)
    
    return envelope


def test_envelope_methods():
    """Test all envelope methods on synthetic signals."""
    
    sample_rate = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test frequencies (octave band centers)
    test_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    
    # Create test signal: amplitude-modulated sine wave
    # Simulates a musical note with envelope
    carrier_freq = 1000.0
    modulation_freq = 2.0  # Slow envelope modulation
    
    # Generate test signal
    signal = np.sin(2 * np.pi * carrier_freq * t)
    envelope_modulation = 0.5 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)
    signal = signal * envelope_modulation
    
    # Add some noise
    noise = np.random.normal(0, 0.05, len(signal))
    signal = signal + noise
    
    # Test each center frequency
    results = {}
    
    for center_freq in test_freqs:
        print(f"\nTesting center frequency: {center_freq} Hz")
        
        # Filter signal to octave band (simplified - use bandpass)
        # In real implementation, this would use the actual octave filter
        low_cutoff = max(1, center_freq / np.sqrt(2))
        high_cutoff = min(sample_rate / 2 * 0.95, center_freq * np.sqrt(2))
        
        sos_bp = sp_signal.butter(4, [low_cutoff, high_cutoff],
                                   btype='band', fs=sample_rate, output='sos')
        band_signal = sp_signal.sosfilt(sos_bp, signal)
        
        # Apply all methods
        methods = {
            'Method 1: Rectification + LPF (4x)': method1_rectification_lpf,
            'Method 2: Freq-Relative RMS (20λ)': method2_frequency_relative_rms,
            'Method 3: Hilbert Transform': method3_hilbert_envelope,
            'Method 4: Fixed 10ms RMS': method4_current_fixed_rms,
        }
        
        envelopes = {}
        for name, method_func in methods.items():
            try:
                if 'Fixed' in name:
                    envelope = method_func(band_signal, sample_rate)
                else:
                    envelope = method_func(band_signal, sample_rate, center_freq)
                envelopes[name] = envelope
                print(f"  {name}: ✓")
            except Exception as e:
                print(f"  {name}: ✗ Error: {e}")
        
        results[center_freq] = {
            'signal': band_signal,
            'envelopes': envelopes
        }
    
    # Plot results
    output_dir = Path('tests/test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for center_freq, data in results.items():
        num_methods = len(data['envelopes'])
        fig, axes = plt.subplots(num_methods + 1, 1,
                                figsize=(14, 3 * (num_methods + 1)))
        
        # Plot original band signal
        ax = axes[0]
        ax.plot(t[:len(data['signal'])], data['signal'], 'b-', alpha=0.5, linewidth=0.5)
        ax.set_title(f'Band Signal (Center: {center_freq} Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Plot each envelope method
        for idx, (name, envelope) in enumerate(data['envelopes'].items(), 1):
            ax = axes[idx]
            ax.plot(t[:len(envelope)], envelope, 'r-', linewidth=2, label=name)
            ax.set_ylabel('Envelope')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(output_dir / f'envelope_comparison_{center_freq:.0f}Hz.png', dpi=150)
        plt.close()
        
        print(f"Saved plot: envelope_comparison_{center_freq:.0f}Hz.png")
    
    print("\n✓ Test complete! Check tests/test_output/ for comparison plots.")


if __name__ == '__main__':
    test_envelope_methods()

