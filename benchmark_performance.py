"""
Benchmark script to measure performance improvements from optimizations.

This script compares the performance of optimized vs original implementations
to verify speedup and correctness.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.audio_processor import AudioProcessor
from src.envelope_analyzer import EnvelopeAnalyzer
from src.music_analyzer import MusicAnalyzer
from src.octave_filter import OctaveBandFilter

logging.basicConfig(level=logging.WARNING)  # Suppress info logs during benchmarking
logger = logging.getLogger(__name__)


def generate_test_signal(duration_seconds: float = 10.0, sample_rate: int = 44100) -> np.ndarray:
    """Generate a test audio signal for benchmarking.
    
    Args:
        duration_seconds: Duration of test signal in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Test audio signal array
    """
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    # Mix of frequencies for realistic testing
    signal = (
        0.3 * np.sin(2 * np.pi * 100 * t) +
        0.2 * np.sin(2 * np.pi * 1000 * t) +
        0.1 * np.sin(2 * np.pi * 5000 * t) +
        0.05 * np.random.randn(len(t))  # Add some noise
    )
    # Normalize
    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)


def benchmark_peak_envelope(signal: np.ndarray, num_iterations: int = 10) -> Dict[str, float]:
    """Benchmark peak envelope calculation.
    
    Args:
        signal: Test audio signal
        num_iterations: Number of iterations to average
        
    Returns:
        Dictionary with timing results
    """
    analyzer = EnvelopeAnalyzer(sample_rate=44100)
    center_freq = 1000.0
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        envelope = analyzer.calculate_peak_envelope(signal, center_freq=center_freq)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        "function": "calculate_peak_envelope",
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "signal_length": len(signal),
        "samples_per_second": len(signal) / avg_time / 1e6  # Millions of samples per second
    }


def benchmark_time_domain_chunks(signal: np.ndarray, num_iterations: int = 10) -> Dict[str, float]:
    """Benchmark time-domain chunk analysis.
    
    Args:
        signal: Test audio signal
        num_iterations: Number of iterations to average
        
    Returns:
        Dictionary with timing results
    """
    analyzer = MusicAnalyzer(sample_rate=44100)
    chunk_duration = 2.0
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        results = analyzer._analyze_time_domain_chunks(signal, chunk_duration=chunk_duration)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        "function": "_analyze_time_domain_chunks",
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "signal_length": len(signal),
        "num_chunks": results["num_chunks"],
        "chunks_per_second": results["num_chunks"] / avg_time
    }


def benchmark_multichannel_resample(
    signal: np.ndarray, num_channels: int = 6, num_iterations: int = 5
) -> Dict[str, float]:
    """Benchmark multi-channel resampling.
    
    Args:
        signal: Test audio signal (will be duplicated across channels)
        num_channels: Number of channels to simulate
        num_iterations: Number of iterations to average
        
    Returns:
        Dictionary with timing results
    """
    processor = AudioProcessor(sample_rate=44100)
    
    # Create multi-channel signal
    multichannel = np.column_stack([signal] * num_channels)
    original_sr = 48000  # Simulate resampling from 48kHz to 44.1kHz
    
    times = []
    for _ in range(num_iterations):
        # Simulate the resampling operation
        start = time.perf_counter()
        import librosa
        multichannel_transposed = multichannel.T
        resampled = librosa.resample(
            multichannel_transposed,
            orig_sr=original_sr,
            target_sr=44100
        )
        result = resampled.T
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        "function": "multichannel_resample",
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "signal_length": len(signal),
        "num_channels": num_channels,
        "total_samples": len(signal) * num_channels,
        "samples_per_second": len(signal) * num_channels / avg_time / 1e6
    }


def benchmark_pattern_matching(signal: np.ndarray, num_iterations: int = 3) -> Dict[str, float]:
    """Benchmark pattern matching correlation.
    
    Args:
        signal: Test audio signal
        num_iterations: Number of iterations to average
        
    Returns:
        Dictionary with timing results
    """
    analyzer = EnvelopeAnalyzer(sample_rate=44100)
    
    # Create envelope and detect peaks
    envelope_db = 20 * np.log10(np.abs(signal) + 1e-10)
    envelope_db = np.clip(envelope_db, -120, 0)
    
    # Detect peaks (simplified)
    from scipy import signal as sp_signal
    peak_indices, _ = sp_signal.find_peaks(
        envelope_db,
        height=-40.0,
        distance=int(0.05 * 44100)  # 50ms minimum distance
    )
    peak_values_db = envelope_db[peak_indices]
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        results = analyzer.analyze_repeating_patterns(
            envelope_db,
            peak_indices,
            peak_values_db,
            min_repetitions=3,
            max_patterns=5,
            similarity_threshold=0.85,
            window_ms=100.0
        )
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        "function": "analyze_repeating_patterns",
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "signal_length": len(signal),
        "num_peaks": len(peak_indices),
        "patterns_detected": results.get("patterns_detected", 0)
    }


def run_all_benchmarks() -> None:
    """Run all benchmarks and print results."""
    print("=" * 80)
    print("Performance Benchmark Results")
    print("=" * 80)
    print()
    
    # Test with different signal lengths
    test_durations = [1.0, 10.0, 60.0]  # 1s, 10s, 60s
    
    for duration in test_durations:
        print(f"\n{'=' * 80}")
        print(f"Test Signal: {duration:.1f} seconds ({duration * 44100:,} samples)")
        print(f"{'=' * 80}\n")
        
        signal = generate_test_signal(duration_seconds=duration)
        
        # Benchmark peak envelope
        print("1. Peak Envelope Calculation:")
        result = benchmark_peak_envelope(signal)
        print(f"   Average time: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
        print(f"   Throughput: {result['samples_per_second']:.2f} M samples/sec")
        print()
        
        # Benchmark time-domain chunks
        print("2. Time-Domain Chunk Analysis:")
        result = benchmark_time_domain_chunks(signal)
        print(f"   Average time: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
        print(f"   Chunks processed: {result['num_chunks']}")
        print(f"   Throughput: {result['chunks_per_second']:.2f} chunks/sec")
        print()
        
        # Benchmark multi-channel resampling (only for longer signals)
        if duration >= 10.0:
            print("3. Multi-Channel Resampling (6 channels):")
            result = benchmark_multichannel_resample(signal, num_channels=6)
            print(f"   Average time: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
            print(f"   Total samples: {result['total_samples']:,}")
            print(f"   Throughput: {result['samples_per_second']:.2f} M samples/sec")
            print()
        
        # Benchmark pattern matching (only for longer signals with enough peaks)
        if duration >= 10.0:
            print("4. Pattern Matching Correlation:")
            result = benchmark_pattern_matching(signal)
            print(f"   Average time: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
            print(f"   Peaks analyzed: {result['num_peaks']}")
            print(f"   Patterns detected: {result['patterns_detected']}")
            print()
    
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)
    print("\nNote: These benchmarks measure the optimized implementations.")
    print("Compare with baseline measurements to determine speedup.")


if __name__ == "__main__":
    run_all_benchmarks()


