"""Test low frequency band outputs."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter

# Load test signal
ap = AudioProcessor(sample_rate=44100)
audio, sr = ap.load_audio('Tracks/Test Signals/SineSweep 20Hz to 20kHz.wav')
audio = ap.stereo_to_mono(audio)

print(f'Original signal: len={len(audio)}, max={np.max(np.abs(audio)):.6f}, rms={np.sqrt(np.mean(audio**2)):.6f}')

# Test with Linkwitz-Riley (current test setting)
of_lr = OctaveBandFilter(sample_rate=sr, use_linkwitz_riley=True)
print('\n=== Linkwitz-Riley Filters ===')
for freq in [16.0, 31.25, 62.5, 125, 250]:
    try:
        b, a = of_lr.design_octave_filter(freq, order=1)
        # Check filter stability
        from scipy import signal
        poles = np.roots(a)
        max_pole = np.max(np.abs(poles))
        print(f'{freq:6.2f}Hz: Filter stability check - max pole magnitude: {max_pole:.6f}')
        
        # Try filtering a small test signal first
        test_small = audio[:1000]
        test_result_filtfilt = signal.filtfilt(b, a, test_small)
        print(f'  Small test (filtfilt): has_nan={np.any(np.isnan(test_result_filtfilt))}, has_inf={np.any(np.isinf(test_result_filtfilt))}')
        
        # Try sosfiltfilt (more stable for long signals)
        sos = signal.tf2sos(b, a)
        test_result_sos = signal.sosfiltfilt(sos, test_small)
        print(f'  Small test (sosfiltfilt): has_nan={np.any(np.isnan(test_result_sos))}, has_inf={np.any(np.isinf(test_result_sos))}')
        
        # Try full signal with sosfiltfilt
        try:
            full_result_sos = signal.sosfiltfilt(sos, audio)
            sos_has_nan = np.any(np.isnan(full_result_sos))
            sos_has_inf = np.any(np.isinf(full_result_sos))
            sos_rms = np.sqrt(np.mean(full_result_sos**2)) if not sos_has_nan else np.nan
            sos_peak = np.max(np.abs(full_result_sos)) if not sos_has_nan else np.nan
            print(f'  Full signal (sosfiltfilt): peak={sos_peak:.6e}, rms={sos_rms:.6e}, has_nan={sos_has_nan}, has_inf={sos_has_inf}')
        except Exception as e:
            print(f'  Full signal (sosfiltfilt): ERROR - {e}')
        
        # Now full signal with standard filtfilt
        band = of_lr.apply_octave_filter(audio, freq)
        rms = np.sqrt(np.mean(band**2)) if not np.any(np.isnan(band)) else np.nan
        peak = np.max(np.abs(band)) if not np.any(np.isnan(band)) else np.nan
        print(f'  Full signal: peak={peak:.6e}, rms={rms:.6e}, has_nan={np.any(np.isnan(band))}, has_inf={np.any(np.isinf(band))}, all_zero={np.all(band==0) if not np.any(np.isnan(band)) else False}')
    except Exception as e:
        print(f'{freq:6.2f}Hz: ERROR - {e}')

# Test with Butterworth
of_bw = OctaveBandFilter(sample_rate=sr, use_linkwitz_riley=False)
print('\n=== Butterworth Filters ===')
for freq in [16.0, 31.25, 62.5, 125, 250]:
    band = of_bw.apply_octave_filter(audio, freq)
    rms = np.sqrt(np.mean(band**2))
    peak = np.max(np.abs(band))
    print(f'{freq:6.2f}Hz: peak={peak:.6e}, rms={rms:.6e}, has_nan={np.any(np.isnan(band))}, all_zero={np.all(band==0)}, crest={peak/rms if rms>0 else 0:.3f}')

# Check what happens when summing
print('\n=== Summing Test (LR) ===')
bands_lr = []
for freq in [16.0, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]:
    bands_lr.append(of_lr.apply_octave_filter(audio, freq))

bands_array = np.column_stack(bands_lr)
summed = np.sum(bands_array, axis=1)
print(f'Summed signal: max={np.max(np.abs(summed)):.6f}, rms={np.sqrt(np.mean(summed**2)):.6f}')
print(f'Original:       max={np.max(np.abs(audio)):.6f}, rms={np.sqrt(np.mean(audio**2)):.6f}')
print(f'Ratio:          max={np.max(np.abs(summed))/np.max(np.abs(audio)):.3f}, rms={np.sqrt(np.mean(summed**2))/np.sqrt(np.mean(audio**2)):.3f}')

# Check low frequency content specifically
print('\n=== Low Frequency Content (0-200Hz) ===')
from scipy import signal as sp_signal
orig_freq, orig_psd = sp_signal.welch(audio, fs=sr, nperseg=8192)
summed_freq, summed_psd = sp_signal.welch(summed, fs=sr, nperseg=8192)

low_mask = orig_freq < 200
orig_low_energy = np.sum(orig_psd[low_mask])
summed_low_energy = np.sum(summed_psd[low_mask])
print(f'Original energy <200Hz: {10*np.log10(orig_low_energy+1e-10):.2f} dB')
print(f'Summed energy <200Hz:   {10*np.log10(summed_low_energy+1e-10):.2f} dB')
print(f'Energy ratio:          {summed_low_energy/(orig_low_energy+1e-10):.6f} ({10*np.log10(summed_low_energy/(orig_low_energy+1e-10)):.2f} dB)')

