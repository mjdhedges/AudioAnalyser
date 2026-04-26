"""FFT power-complementary octave band filtering for Audio Analyser."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class OctaveBandFilter:
    """Create octave-band time-series using a proven FFT filter bank.

    The bank is power-complementary by construction:
    ``sum(weight_band(f) ** 2) == 1.0`` for every FFT bin. This preserves total
    signal energy when octave-band RMS values are summed as linear power.
    """

    OCTAVE_CENTER_FREQUENCIES = [
        8.0,
        16.0,
        31.25,
        62.5,
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        16000.0,
    ]

    VALID_PROCESSING_MODES = {"auto", "full_file", "block"}

    def __init__(
        self,
        sample_rate: int = 44100,
        processing_mode: str = "auto",
        block_duration_seconds: float = 30.0,
        max_memory_gb: float = 4.0,
        include_low_residual_band: bool = True,
        include_high_residual_band: bool = True,
        low_residual_center_hz: float = 4.0,
        high_residual_center_hz: float = 32000.0,
    ) -> None:
        """Initialize the FFT octave filter bank.

        Args:
            sample_rate: Sample rate of the audio signal.
            processing_mode: ``"auto"``, ``"full_file"``, or ``"block"``.
            block_duration_seconds: FFT block length for ``"block"`` mode.
            max_memory_gb: Approximate RAM budget for octave-bank processing.
            include_low_residual_band: Include a 4 Hz-and-below residual band.
            include_high_residual_band: Include an above-16 kHz residual band.
            low_residual_center_hz: Representative center for the low residual.
            high_residual_center_hz: Representative center for the high residual.

        Raises:
            ValueError: If processing mode or residual settings are invalid.
        """
        if processing_mode not in self.VALID_PROCESSING_MODES:
            raise ValueError(
                "processing_mode must be one of "
                f"{sorted(self.VALID_PROCESSING_MODES)}"
            )
        if block_duration_seconds <= 0:
            raise ValueError("block_duration_seconds must be positive")
        if max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if low_residual_center_hz <= 0:
            raise ValueError("low_residual_center_hz must be positive")
        if high_residual_center_hz <= 0:
            raise ValueError("high_residual_center_hz must be positive")

        self.sample_rate = sample_rate
        self.processing_mode = processing_mode
        self.block_duration_seconds = float(block_duration_seconds)
        self.max_memory_gb = float(max_memory_gb)
        self.max_memory_bytes = int(self.max_memory_gb * 1024**3)
        self.include_low_residual_band = include_low_residual_band
        self.include_high_residual_band = include_high_residual_band
        self.low_residual_center_hz = float(low_residual_center_hz)
        self.high_residual_center_hz = float(high_residual_center_hz)
        self._last_band_center_frequencies: list[float] = []
        self._last_effective_processing_mode = processing_mode
        self._last_output_storage = "memory"
        self._last_estimated_full_file_peak_bytes = 0
        self._last_estimated_output_bytes = 0
        self._last_block_duration_seconds = self.block_duration_seconds
        self._memmap_dir: tempfile.TemporaryDirectory[str] | None = None

    def get_band_center_frequencies(
        self,
        center_frequencies: List[float] | None = None,
    ) -> list[float]:
        """Return the analysis band centers, including configured residual bands.

        Args:
            center_frequencies: Nominal octave centers to include between residuals.

        Returns:
            Sorted representative band centers used for output columns.
        """
        if center_frequencies is None:
            center_frequencies = self.OCTAVE_CENTER_FREQUENCIES

        centers = sorted({float(freq) for freq in center_frequencies if freq > 0})

        if self.include_low_residual_band:
            centers = [self.low_residual_center_hz, *centers]
        if self.include_high_residual_band:
            centers = [*centers, self.high_residual_center_hz]

        return centers

    def create_octave_bank(
        self,
        audio_data: np.ndarray,
        center_frequencies: List[float] | None = None,
    ) -> np.ndarray:
        """Create an octave bank with the source in column 0 and bands after it.

        Args:
            audio_data: Input audio signal.
            center_frequencies: Nominal octave centers before residual bands are added.

        Returns:
            Array shaped ``(samples, 1 + bands)``.
        """
        mode = self._resolve_processing_mode(audio_data, center_frequencies)
        if mode == "block":
            return self._create_block_fft_octave_bank(audio_data, center_frequencies)
        return self._create_full_file_fft_octave_bank(audio_data, center_frequencies)

    def get_processing_metadata(self) -> dict[str, object]:
        """Return metadata describing the most recent octave-bank build."""
        return {
            "octave_filter_design": "fft_power_complementary",
            "octave_requested_processing_mode": self.processing_mode,
            "octave_effective_processing_mode": self._last_effective_processing_mode,
            "octave_output_storage": self._last_output_storage,
            "octave_max_memory_gb": self.max_memory_gb,
            "octave_fft_block_duration_seconds": self._last_block_duration_seconds,
            "octave_include_low_residual_band": self.include_low_residual_band,
            "octave_include_high_residual_band": self.include_high_residual_band,
            "octave_low_residual_center_hz": self.low_residual_center_hz,
            "octave_high_residual_center_hz": self.high_residual_center_hz,
            "octave_band_centers_hz": "|".join(
                f"{freq:g}" for freq in self._last_band_center_frequencies
            ),
            "octave_estimated_full_file_peak_gb": self._last_estimated_full_file_peak_bytes
            / 1024**3,
            "octave_estimated_output_gb": self._last_estimated_output_bytes / 1024**3,
            "octave_power_sum_rule": "sum(weight_band(f) ** 2) = 1.0",
        }

    def create_octave_bank_parallel(
        self,
        audio_data: np.ndarray,
        center_frequencies: List[float] | None = None,
        num_workers: int | None = None,
    ) -> np.ndarray:
        """Compatibility wrapper for the retired parallel IIR path.

        Args:
            audio_data: Input audio signal.
            center_frequencies: Nominal octave centers before residual bands are added.
            num_workers: Ignored. FFT processing reuses one spectrum per file/block.

        Returns:
            Array shaped ``(samples, 1 + bands)``.
        """
        if num_workers is not None:
            logger.info(
                "Ignoring num_workers; FFT octave processing is not per-band parallel"
            )
        return self.create_octave_bank(audio_data, center_frequencies)

    def get_octave_analysis(self, octave_bank: np.ndarray) -> dict:
        """Perform basic analysis on an octave bank.

        Args:
            octave_bank: Octave bank array.

        Returns:
            Dictionary with per-column peak, RMS, and dynamic-range values.
        """
        num_bands = octave_bank.shape[1]
        max_values = np.zeros(num_bands)
        max_values_db = np.zeros(num_bands)
        rms_values = np.zeros(num_bands)
        rms_values_db = np.zeros(num_bands)
        dynamic_range = np.zeros(num_bands)
        dynamic_range_db = np.zeros(num_bands)

        for band_idx in range(num_bands):
            band_signal = np.asarray(octave_bank[:, band_idx], dtype=np.float64)
            max_values[band_idx] = np.max(np.abs(band_signal))
            max_values_db[band_idx] = (
                20 * np.log10(max_values[band_idx])
                if max_values[band_idx] > 0
                else -np.inf
            )
            rms_values[band_idx] = np.sqrt(np.mean(np.square(band_signal)))
            rms_values_db[band_idx] = (
                20 * np.log10(rms_values[band_idx])
                if rms_values[band_idx] > 0
                else -np.inf
            )
            if max_values[band_idx] > 0:
                dynamic_range[band_idx] = rms_values[band_idx] / max_values[band_idx]
                dynamic_range_db[band_idx] = 20 * np.log10(dynamic_range[band_idx])
            else:
                dynamic_range_db[band_idx] = -np.inf

        center_frequencies = [0.0, *self._last_band_center_frequencies]
        if len(center_frequencies) != num_bands:
            center_frequencies = [0.0, *self.OCTAVE_CENTER_FREQUENCIES[: num_bands - 1]]

        return {
            "max_values": max_values,
            "max_values_db": max_values_db,
            "rms_values": rms_values,
            "rms_values_db": rms_values_db,
            "dynamic_range": dynamic_range,
            "dynamic_range_db": dynamic_range_db,
            "center_frequencies": center_frequencies,
        }

    def _create_full_file_fft_octave_bank(
        self,
        audio_data: np.ndarray,
        center_frequencies: List[float] | None,
    ) -> np.ndarray:
        """Create the bank from one FFT over the full input."""
        x = np.asarray(audio_data, dtype=np.float64)
        band_centers = self.get_band_center_frequencies(center_frequencies)
        self._last_band_center_frequencies = band_centers
        self._last_effective_processing_mode = "full_file"
        self._last_output_storage = "memory"

        spectrum = np.fft.rfft(x)
        fft_freqs = np.fft.rfftfreq(x.size, d=1.0 / self.sample_rate)
        weights = self._fft_power_complementary_weights(fft_freqs, band_centers)

        filtered_signals = [x]
        for weight in weights:
            filtered_signals.append(np.fft.irfft(spectrum * weight, n=x.size))

        logger.info(
            "Created FFT octave bank with %d bands using full-file mode",
            len(band_centers),
        )
        return np.column_stack(filtered_signals)

    def _create_block_fft_octave_bank(
        self,
        audio_data: np.ndarray,
        center_frequencies: List[float] | None,
    ) -> np.ndarray:
        """Create the bank by processing large non-overlapping FFT blocks."""
        x = np.asarray(audio_data, dtype=np.float64)
        band_centers = self.get_band_center_frequencies(center_frequencies)
        self._last_band_center_frequencies = band_centers
        self._last_effective_processing_mode = "block"
        block_samples = self._resolve_block_samples(len(x), len(band_centers))
        if block_samples <= 0:
            raise ValueError("block_duration_seconds is too small for the sample rate")

        output = self._allocate_octave_output(x.size, len(band_centers) + 1)
        output[:, 0] = x

        num_blocks = int(np.ceil(x.size / block_samples))
        for block_idx in range(num_blocks):
            start = block_idx * block_samples
            end = min(start + block_samples, x.size)
            block = x[start:end]
            if block.size == 0:
                continue

            spectrum = np.fft.rfft(block)
            fft_freqs = np.fft.rfftfreq(block.size, d=1.0 / self.sample_rate)
            weights = self._fft_power_complementary_weights(fft_freqs, band_centers)
            for band_idx, weight in enumerate(weights):
                output[start:end, band_idx + 1] = np.fft.irfft(
                    spectrum * weight,
                    n=block.size,
                )

        logger.info(
            "Created FFT octave bank with %d bands using %d %.1fs blocks",
            len(band_centers),
            num_blocks,
            self._last_block_duration_seconds,
        )
        return output

    def _resolve_processing_mode(
        self,
        audio_data: np.ndarray,
        center_frequencies: List[float] | None,
    ) -> str:
        """Choose full-file or block FFT processing from memory estimates."""
        sample_count = int(np.asarray(audio_data).size)
        band_count = len(self.get_band_center_frequencies(center_frequencies))
        self._last_estimated_output_bytes = self._estimate_output_bytes(
            sample_count,
            band_count,
        )
        self._last_estimated_full_file_peak_bytes = self._estimate_full_file_peak_bytes(
            sample_count,
            band_count,
        )

        if self.processing_mode == "block":
            return "block"

        if self._last_estimated_full_file_peak_bytes <= self.max_memory_bytes:
            return "full_file"

        logger.info(
            "Switching octave FFT to block mode: estimated full-file peak %.2f GB exceeds %.2f GB limit",
            self._last_estimated_full_file_peak_bytes / 1024**3,
            self.max_memory_gb,
        )
        return "block"

    @staticmethod
    def _estimate_output_bytes(sample_count: int, band_count: int) -> int:
        """Estimate bytes for the full octave-bank output array."""
        return sample_count * (band_count + 1) * np.dtype(np.float64).itemsize

    @staticmethod
    def _estimate_full_file_peak_bytes(sample_count: int, band_count: int) -> int:
        """Estimate peak RAM for full-file FFT processing."""
        fft_bins = sample_count // 2 + 1
        output = OctaveBandFilter._estimate_output_bytes(sample_count, band_count)
        weights = band_count * fft_bins * np.dtype(np.float64).itemsize
        spectrum = fft_bins * np.dtype(np.complex128).itemsize
        source = sample_count * np.dtype(np.float64).itemsize
        filtered_band = sample_count * np.dtype(np.float64).itemsize
        return output + weights + spectrum + source + filtered_band

    def _resolve_block_samples(self, sample_count: int, band_count: int) -> int:
        """Resolve a block size that keeps per-block FFT RAM under the limit."""
        requested = int(round(self.block_duration_seconds * self.sample_rate))
        if requested <= 0:
            return requested

        bytes_per_block_sample = (
            np.dtype(np.float64).itemsize * band_count / 2
            + np.dtype(np.complex128).itemsize / 2
            + np.dtype(np.float64).itemsize * 2
        )
        max_block_samples = max(
            self.sample_rate,
            int(self.max_memory_bytes / max(bytes_per_block_sample, 1)),
        )
        block_samples = min(requested, max_block_samples, sample_count)
        self._last_block_duration_seconds = block_samples / self.sample_rate

        if block_samples < requested:
            logger.info(
                "Reducing octave FFT block length from %.2fs to %.2fs to stay within %.2f GB RAM limit",
                self.block_duration_seconds,
                self._last_block_duration_seconds,
                self.max_memory_gb,
            )
        return block_samples

    def _allocate_octave_output(
        self,
        sample_count: int,
        column_count: int,
    ) -> np.ndarray:
        """Allocate octave output in memory or disk-backed storage."""
        output_bytes = sample_count * column_count * np.dtype(np.float64).itemsize
        if output_bytes <= self.max_memory_bytes:
            self._last_output_storage = "memory"
            return np.zeros((sample_count, column_count), dtype=np.float64)

        if self._memmap_dir is None:
            self._memmap_dir = tempfile.TemporaryDirectory(
                prefix="audio_analyser_octave_"
            )
        memmap_path = Path(self._memmap_dir.name) / "octave_bank.dat"
        self._last_output_storage = "disk_memmap"
        logger.info(
            "Using disk-backed octave bank: estimated output %.2f GB exceeds %.2f GB RAM limit",
            output_bytes / 1024**3,
            self.max_memory_gb,
        )
        return np.memmap(
            memmap_path,
            dtype=np.float64,
            mode="w+",
            shape=(sample_count, column_count),
        )

    @staticmethod
    def _fft_power_complementary_weights(
        fft_freqs: np.ndarray,
        band_centers: list[float],
    ) -> np.ndarray:
        """Create raised-cosine weights with a flat summed-power response."""
        center_array = np.asarray(band_centers, dtype=np.float64)
        if np.any(center_array <= 0):
            raise ValueError("band centers must be positive")
        if np.any(np.diff(center_array) <= 0):
            raise ValueError("band centers must be strictly ascending")

        weights = np.zeros((center_array.size, fft_freqs.size), dtype=np.float64)

        low_mask = fft_freqs <= center_array[0]
        high_mask = fft_freqs >= center_array[-1]
        transition_mask = ~(low_mask | high_mask)

        weights[0, low_mask] = 1.0
        weights[-1, high_mask] = 1.0

        transition_freqs = fft_freqs[transition_mask]
        if transition_freqs.size == 0:
            return weights

        transition_bins = np.flatnonzero(transition_mask)
        lower_idx = np.searchsorted(center_array, transition_freqs, side="right") - 1
        upper_idx = lower_idx + 1
        lower = center_array[lower_idx]
        upper = center_array[upper_idx]
        position = (np.log(transition_freqs) - np.log(lower)) / (
            np.log(upper) - np.log(lower)
        )
        theta = position * np.pi / 2.0

        weights[lower_idx, transition_bins] = np.cos(theta)
        weights[upper_idx, transition_bins] = np.sin(theta)

        return weights
