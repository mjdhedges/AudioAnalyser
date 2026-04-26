"""Tests for the FFT octave filter module."""

from __future__ import annotations

import numpy as np
import pytest

from src.octave_filter import OctaveBandFilter


def _rms(signal: np.ndarray) -> float:
    """Return linear RMS."""
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


class TestOctaveBandFilter:
    """Test cases for the FFT power-complementary octave filter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 44100
        self.filter = OctaveBandFilter(sample_rate=self.sample_rate)

    def test_init(self) -> None:
        """Test default initialization."""
        assert self.filter.sample_rate == self.sample_rate
        assert self.filter.processing_mode == "auto"
        assert len(self.filter.OCTAVE_CENTER_FREQUENCIES) == 12

    def test_residual_band_centers_are_added(self) -> None:
        """Test that configured residual bands surround nominal octave centers."""
        centers = self.filter.get_band_center_frequencies()

        assert centers[0] == 4.0
        assert centers[1:-1] == self.filter.OCTAVE_CENTER_FREQUENCIES
        assert centers[-1] == 32000.0

    def test_create_octave_bank(self) -> None:
        """Test creating an FFT octave bank."""
        duration = 1.0
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        test_signal = np.sin(2 * np.pi * 1000 * t)

        octave_bank = self.filter.create_octave_bank(test_signal)

        assert octave_bank.shape[0] == len(test_signal)
        assert octave_bank.shape[1] == 1 + len(
            self.filter.get_band_center_frequencies()
        )
        np.testing.assert_array_equal(octave_bank[:, 0], test_signal)

    def test_create_octave_bank_custom_frequencies_without_residuals(self) -> None:
        """Test creating a bank with only configured nominal centers."""
        filter_bank = OctaveBandFilter(
            sample_rate=self.sample_rate,
            include_low_residual_band=False,
            include_high_residual_band=False,
        )
        test_signal = np.random.default_rng(1234).normal(0.0, 1.0, 4096)
        custom_frequencies = [125.0, 250.0, 500.0]

        octave_bank = filter_bank.create_octave_bank(test_signal, custom_frequencies)

        assert octave_bank.shape == (len(test_signal), 1 + len(custom_frequencies))

    def test_full_file_energy_closure(self) -> None:
        """Test that band RMS values power-sum to full-band RMS."""
        rng = np.random.default_rng(20260426)
        test_signal = rng.normal(0.0, 1.0, self.sample_rate * 2)
        octave_bank = self.filter.create_octave_bank(test_signal)

        full_rms = _rms(octave_bank[:, 0])
        band_rms = np.sqrt(np.mean(np.square(octave_bank[:, 1:]), axis=0))
        band_power_sum_rms = float(np.sqrt(np.sum(np.square(band_rms))))

        np.testing.assert_allclose(band_power_sum_rms, full_rms, rtol=1e-12, atol=1e-12)

    def test_block_mode_energy_closure_and_shape(self) -> None:
        """Test that block FFT mode remains energy-closed and same-length."""
        rng = np.random.default_rng(20260426)
        test_signal = rng.normal(0.0, 1.0, self.sample_rate * 3)
        block_filter = OctaveBandFilter(
            sample_rate=self.sample_rate,
            processing_mode="block",
            block_duration_seconds=1.0,
        )

        octave_bank = block_filter.create_octave_bank(test_signal)
        full_rms = _rms(octave_bank[:, 0])
        band_rms = np.sqrt(np.mean(np.square(octave_bank[:, 1:]), axis=0))
        band_power_sum_rms = float(np.sqrt(np.sum(np.square(band_rms))))

        assert octave_bank.shape[0] == len(test_signal)
        np.testing.assert_allclose(band_power_sum_rms, full_rms, rtol=1e-12, atol=1e-12)

    def test_invalid_processing_mode(self) -> None:
        """Test invalid processing mode validation."""
        with pytest.raises(ValueError, match="processing_mode"):
            OctaveBandFilter(sample_rate=self.sample_rate, processing_mode="iir")

    def test_get_octave_analysis(self) -> None:
        """Test octave band analysis helper."""
        test_signal = np.random.default_rng(4321).normal(0.0, 1.0, 4096)
        octave_bank = self.filter.create_octave_bank(test_signal)

        analysis = self.filter.get_octave_analysis(octave_bank)

        assert "max_values" in analysis
        assert "rms_values" in analysis
        assert "dynamic_range" in analysis
        assert len(analysis["center_frequencies"]) == octave_bank.shape[1]
