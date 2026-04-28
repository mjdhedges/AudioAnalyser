"""Extended tests for FFT octave filtering."""

from __future__ import annotations

import numpy as np
import pytest

from src.octave_filter import OctaveBandFilter


class TestOctaveFilterExtended:
    """Extended tests for filter-bank weights and block processing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 48000
        self.filter = OctaveBandFilter(sample_rate=self.sample_rate)

    def test_power_complementary_weights_are_flat(self) -> None:
        """Test that every FFT bin has a flat summed-power response."""
        fft_freqs = np.fft.rfftfreq(self.sample_rate * 2, d=1.0 / self.sample_rate)
        band_centers = self.filter.get_band_center_frequencies()

        weights = self.filter._fft_power_complementary_weights(fft_freqs, band_centers)
        power_sum = np.sum(np.square(weights), axis=0)

        np.testing.assert_allclose(power_sum, np.ones_like(power_sum), atol=1e-12)

    def test_per_band_weight_matches_full_weight_matrix(self) -> None:
        """Test per-band generation is numerically identical to the old matrix form."""
        fft_freqs = np.fft.rfftfreq(self.sample_rate * 2, d=1.0 / self.sample_rate)
        band_centers = self.filter.get_band_center_frequencies()

        weights = self.filter._fft_power_complementary_weights(fft_freqs, band_centers)
        per_band_weights = np.vstack(
            [
                self.filter._fft_power_complementary_weight(
                    fft_freqs,
                    band_centers,
                    band_idx,
                )
                for band_idx in range(len(band_centers))
            ]
        )

        np.testing.assert_allclose(per_band_weights, weights, atol=1e-12)

    def test_full_file_peak_estimate_uses_one_weight_vector(self) -> None:
        """Test peak estimate no longer reserves memory for all band weights."""
        sample_count = self.sample_rate * 60
        band_count = len(self.filter.get_band_center_frequencies())
        fft_bins = sample_count // 2 + 1
        old_all_weights_bytes = band_count * fft_bins * np.dtype(np.float64).itemsize
        new_one_weight_bytes = fft_bins * np.dtype(np.float64).itemsize

        old_estimate = (
            self.filter._estimate_output_bytes(sample_count, band_count)
            + old_all_weights_bytes
            + fft_bins * np.dtype(np.complex128).itemsize
            + sample_count * np.dtype(np.float64).itemsize
        )
        new_estimate = self.filter._estimate_full_file_peak_bytes(
            sample_count,
            band_count,
        )

        assert new_estimate < old_estimate
        assert new_estimate > new_one_weight_bytes

    def test_amplitude_sum_is_not_used_as_closure_criterion(self) -> None:
        """Test that overlapping bands are not simple-amplitude complementary."""
        fft_freqs = np.fft.rfftfreq(self.sample_rate * 2, d=1.0 / self.sample_rate)
        band_centers = self.filter.get_band_center_frequencies()

        weights = self.filter._fft_power_complementary_weights(fft_freqs, band_centers)
        amplitude_sum = np.sum(weights, axis=0)

        assert np.max(amplitude_sum) > 1.0

    def test_block_mode_matches_full_file_on_active_bands(self) -> None:
        """Test large-block FFT agrees closely with full-file FFT."""
        rng = np.random.default_rng(42)
        test_signal = rng.normal(0.0, 1.0, self.sample_rate * 4)
        full_filter = OctaveBandFilter(sample_rate=self.sample_rate)
        block_filter = OctaveBandFilter(
            sample_rate=self.sample_rate,
            processing_mode="block",
            block_duration_seconds=2.0,
        )

        full_bank = full_filter.create_octave_bank(test_signal)
        block_bank = block_filter.create_octave_bank(test_signal)
        full_rms = np.sqrt(np.mean(np.square(full_bank[:, 1:]), axis=0))
        block_rms = np.sqrt(np.mean(np.square(block_bank[:, 1:]), axis=0))
        active = full_rms > np.sqrt(np.mean(np.square(test_signal))) * 1e-6
        delta_db = 20.0 * np.log10(block_rms[active] / full_rms[active])

        assert full_bank.shape == block_bank.shape
        assert float(np.max(np.abs(delta_db))) < 0.1

    def test_auto_mode_switches_to_block_and_memmap_under_low_ram_limit(self) -> None:
        """Test auto mode records block/memmap metadata with a tiny estimate."""
        rng = np.random.default_rng(123)
        test_signal = rng.normal(0.0, 1.0, self.sample_rate * 2)
        octave_filter = OctaveBandFilter(
            sample_rate=self.sample_rate,
            processing_mode="auto",
            block_duration_seconds=1.0,
            max_memory_gb=0.000001,
        )

        octave_bank = octave_filter.create_octave_bank(test_signal)
        metadata = octave_filter.get_processing_metadata()

        assert octave_bank.shape[0] == len(test_signal)
        assert metadata["octave_effective_processing_mode"] == "block"
        assert metadata["octave_output_storage"] == "disk_memmap"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"block_duration_seconds": 0.0}, "block_duration_seconds"),
            ({"max_memory_gb": 0.0}, "max_memory_gb"),
            ({"low_residual_center_hz": 0.0}, "low_residual_center_hz"),
            ({"high_residual_center_hz": -1.0}, "high_residual_center_hz"),
        ],
    )
    def test_invalid_init_values_raise(
        self,
        kwargs: dict[str, float],
        message: str,
    ) -> None:
        """Test constructor validation."""
        with pytest.raises(ValueError, match=message):
            OctaveBandFilter(sample_rate=self.sample_rate, **kwargs)

    def test_invalid_band_centers_raise(self) -> None:
        """Test weight validation rejects non-ascending centers."""
        fft_freqs = np.fft.rfftfreq(1024, d=1.0 / self.sample_rate)

        with pytest.raises(ValueError, match="strictly ascending"):
            self.filter._fft_power_complementary_weights(
                fft_freqs,
                [8.0, 8.0, 16.0],
            )
