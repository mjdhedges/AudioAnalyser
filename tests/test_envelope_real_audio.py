"""
Test envelope detection methods on real audio data.

This script tests the envelope methods on actual octave band data
from "Fragments of Time" to see real-world performance.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter
from src.config import config


def method1_rectification_lpf(signal: np.ndarray, sample_rate: float,
                               center_freq: float, lpf_multiplier: float = 4.0) -> np.ndarray:
    """Method 1: Rectification + Low-Pass Filter at Nx center frequency."""
    rectified = np.abs(signal)
    lpf_cutoff = lpf_multiplier * center_freq
    nyquist = sample_rate / 2.0
    lpf_cutoff = min(lpf_cutoff, nyquist * 0.95)
    sos = sp_signal.butter(4, lpf_cutoff, btype='low', fs=sample_rate, output='sos')
    envelope = sp_signal.sosfilt(sos, rectified)
    return envelope


def method2_frequency_relative_rms(signal: np.ndarray, sample_rate: float,
                                   center_freq: float,
                                   num_wavelengths: float = 20) -> np.ndarray:
    """Method 2: Frequency-relative RMS window."""
    if center_freq > 0:
        period = 1.0 / center_freq
        window_time = num_wavelengths * period
    else:
        window_time = 0.01
    
    window_samples = int(window_time * sample_rate)
    window_samples = max(1, min(window_samples, len(signal) // 2))
    
    window = np.ones(window_samples) / window_samples
    signal_squared = signal ** 2
    rms_squared = sp_signal.fftconvolve(signal_squared, window, mode='same')
    rms_squared = np.maximum(rms_squared, 0.0)
    envelope = np.sqrt(rms_squared)
    return envelope


def method4_current_fixed_rms(signal: np.ndarray, sample_rate: float,
                              window_ms: float = 10.0) -> np.ndarray:
    """Method 4: Current approach - Fixed 10ms RMS window."""
    window_samples = int(window_ms * sample_rate / 1000)
    window_samples = max(1, min(window_samples, len(signal) // 2))
    
    window = np.ones(window_samples) / window_samples
    signal_squared = signal ** 2
    rms_squared = sp_signal.fftconvolve(signal_squared, window, mode='same')
    rms_squared = np.maximum(rms_squared, 0.0)
    envelope = np.sqrt(rms_squared)
    return envelope


def test_real_audio():
    """Test envelope methods on real audio data."""
    
    audio_path = Path("Tracks/Music/Fragments of Time.wav")
    
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        print("Skipping real audio test.")
        return
    
    print(f"Loading audio: {audio_path}")
    
    # Load audio
    processor = AudioProcessor()
    audio_data, sample_rate = processor.load_audio(audio_path)
    
    # Use first 10 seconds for testing
    test_duration = 10.0
    test_samples = int(test_duration * sample_rate)
    audio_data = audio_data[:test_samples, 0] if audio_data.ndim > 1 else audio_data[:test_samples]
    
    print(f"Sample rate: {sample_rate} Hz, Duration: {test_duration}s")
    
    # Create octave filter
    center_frequencies = config.get('analysis.octave_center_frequencies', [])
    filter_order = config.get('analysis.filter_order', 4)
    
    octave_filter = OctaveBandFilter(sample_rate, filter_order)
    octave_bank = octave_filter.create_octave_bank(audio_data)
    
    print(f"Created octave bank with {octave_bank.shape[1]} bands")
    
    # Test frequencies (focus on problematic high frequencies)
    test_freqs = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    test_freq_indices = []
    
    extended_freqs = [0] + center_frequencies
    for freq in test_freqs:
        if freq in extended_freqs:
            test_freq_indices.append((extended_freqs.index(freq), freq))
    
    # Time array
    t = np.arange(len(audio_data)) / sample_rate
    
    # Test different LPF multipliers for Method 1
    lpf_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    # Methods to test
    methods = {}
    
    # Add Method 1 with different multipliers
    # Use default parameter trick to avoid lambda closure issues
    for mult in lpf_multipliers:
        def make_method(multiplier):
            return lambda s, sr, cf: method1_rectification_lpf(s, sr, cf, multiplier)
        methods[f'Method 1: Rect+LPF ({mult}x)'] = make_method(mult)
    
    # Add other methods for comparison
    methods['Method 2: Freq-Relative RMS (20λ)'] = method2_frequency_relative_rms
    methods['Method 4: Fixed 10ms RMS'] = method4_current_fixed_rms
    
    output_dir = Path('tests/test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for band_idx, center_freq in test_freq_indices:
        print(f"\nTesting band: {center_freq} Hz (index {band_idx})")
        
        # Extract band signal
        band_signal = octave_bank[:, band_idx]
        
        # Calculate envelopes
        envelopes = {}
        for name, method_func in methods.items():
            try:
                if 'Fixed' in name:
                    envelope = method_func(band_signal, sample_rate)
                elif 'Method 1' in name:
                    # Extract multiplier from name
                    import re
                    match = re.search(r'\(([\d.]+)x\)', name)
                    mult = float(match.group(1)) if match else 4.0
                    envelope = method_func(band_signal, sample_rate, center_freq, mult)
                else:
                    envelope = method_func(band_signal, sample_rate, center_freq)
                envelopes[name] = envelope
                print(f"  {name}: ✓")
            except Exception as e:
                print(f"  {name}: ✗ Error: {e}")
        
        # Plot comparison - use subplot grid for better organization
        num_methods = len(envelopes)
        # Organize: Method 1 variants in one column, others in second column
        num_lpf_methods = len([n for n in envelopes.keys() if 'Method 1' in n])
        num_other_methods = num_methods - num_lpf_methods
        
        # Create figure with subplots: signal at top, then Method 1 variants, then others
        fig = plt.figure(figsize=(18, 3 * (num_methods + 1)))
        gs = fig.add_gridspec(num_methods + 1, 1, hspace=0.3)
        
        axes = []
        
        # Signal plot
        ax = fig.add_subplot(gs[0, 0])
        axes.append(ax)
        
        # Method 1 variants (LPF tests)
        for idx, (name, envelope) in enumerate([(n, e) for n, e in envelopes.items() if 'Method 1' in n], 1):
            ax = fig.add_subplot(gs[idx, 0])
            axes.append(ax)
        
        # Other methods
        for idx, (name, envelope) in enumerate([(n, e) for n, e in envelopes.items() if 'Method 1' not in n], len([n for n in envelopes.keys() if 'Method 1' in n]) + 1):
            ax = fig.add_subplot(gs[idx, 0])
            axes.append(ax)
        
        # Plot band signal
        ax = axes[0]
        ax.plot(t, band_signal, 'b-', alpha=0.3, linewidth=0.5)
        ax.set_title(f'Octave Band Signal (Center: {center_freq} Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Plot Method 1 variants first (LPF tests)
        method1_items = [(n, e) for n, e in envelopes.items() if 'Method 1' in n]
        method1_items.sort(key=lambda x: float(re.search(r'\(([\d.]+)x\)', x[0]).group(1)) if re.search(r'\(([\d.]+)x\)', x[0]) else 999)
        
        for idx, (name, envelope) in enumerate(method1_items, 1):
            ax = axes[idx]
            # Convert to dBFS for visualization
            envelope_db = 20 * np.log10(np.abs(envelope) + 1e-10)
            envelope_db = np.clip(envelope_db, -60, 0)
            
            # Color based on multiplier (lighter for lower, darker for higher)
            multiplier = float(re.search(r'\(([\d.]+)x\)', name).group(1)) if re.search(r'\(([\d.]+)x\)', name) else 4.0
            color_intensity = multiplier / 5.0  # Normalize to 0-1
            color = plt.cm.Reds(0.3 + 0.7 * color_intensity)
            
            ax.plot(t, envelope_db, color=color, linewidth=1.5, label=name)
            ax.set_ylabel('Envelope (dBFS)')
            ax.set_title(f'{name} - {center_freq} Hz', fontsize=10)
            ax.set_ylim(-60, 0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Plot other methods
        other_items = [(n, e) for n, e in envelopes.items() if 'Method 1' not in n]
        start_idx = len(method1_items) + 1
        for idx, (name, envelope) in enumerate(other_items, start_idx):
            ax = axes[idx]
            envelope_db = 20 * np.log10(np.abs(envelope) + 1e-10)
            envelope_db = np.clip(envelope_db, -60, 0)
            
            color = 'green' if 'Method 2' in name else 'orange'
            ax.plot(t, envelope_db, color=color, linewidth=1.5, label=name)
            ax.set_ylabel('Envelope (dBFS)')
            ax.set_title(f'{name} - {center_freq} Hz', fontsize=10)
            ax.set_ylim(-60, 0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        freq_label = f"{center_freq:.0f}" if center_freq > 0 else "Full_Spectrum"
        plt.savefig(output_dir / f'real_audio_envelope_comparison_{freq_label}Hz.png', dpi=150)
        plt.close()
        
        print(f"  Saved: real_audio_envelope_comparison_{freq_label}Hz.png")
    
    print("\n✓ Real audio test complete! Check tests/test_output/ for comparison plots.")


if __name__ == '__main__':
    test_real_audio()

