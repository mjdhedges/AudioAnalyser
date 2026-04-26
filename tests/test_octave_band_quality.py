"""Quality tests for the FFT power-complementary octave band filter."""

from __future__ import annotations

import numpy as np
import pytest

from src.octave_filter import OctaveBandFilter


def _rms(signal: np.ndarray) -> float:
    """Return linear RMS."""
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


class TestOctaveBandQuality:
    """Quality tests for octave-band energy closure."""

    @pytest.fixture
    def sample_rate(self) -> int:
        """Return the proof sample rate."""
        return 48000

    @pytest.fixture
    def octave_filter(self, sample_rate: int) -> OctaveBandFilter:
        """Create an FFT octave filter."""
        return OctaveBandFilter(sample_rate=sample_rate)

    @pytest.mark.parametrize("frequency", [31.25, 1000.0, 16000.0])
    def test_single_tone_energy_closure(
        self,
        octave_filter: OctaveBandFilter,
        sample_rate: int,
        frequency: float,
    ) -> None:
        """Test exact energy closure for tones at important band centers."""
        duration = 2.0
        t = np.arange(int(sample_rate * duration)) / sample_rate
        test_signal = 0.5 * np.sin(2.0 * np.pi * frequency * t)

        octave_bank = octave_filter.create_octave_bank(test_signal)
        full_rms = _rms(octave_bank[:, 0])
        band_rms = np.sqrt(np.mean(np.square(octave_bank[:, 1:]), axis=0))
        band_power_sum_rms = float(np.sqrt(np.sum(np.square(band_rms))))

        np.testing.assert_allclose(band_power_sum_rms, full_rms, rtol=1e-12, atol=1e-12)

    def test_noise_energy_closure(
        self, octave_filter: OctaveBandFilter, sample_rate: int
    ) -> None:
        """Test exact energy closure for broadband noise."""
        rng = np.random.default_rng(20260426)
        test_signal = rng.normal(0.0, 0.25, sample_rate * 2)

        octave_bank = octave_filter.create_octave_bank(test_signal)
        full_rms = _rms(octave_bank[:, 0])
        band_rms = np.sqrt(np.mean(np.square(octave_bank[:, 1:]), axis=0))
        band_power_sum_rms = float(np.sqrt(np.sum(np.square(band_rms))))

        np.testing.assert_allclose(band_power_sum_rms, full_rms, rtol=1e-12, atol=1e-12)

    def test_band_outputs_are_valid_time_series(
        self,
        octave_filter: OctaveBandFilter,
        sample_rate: int,
    ) -> None:
        """Test each band is finite, same-length, and usable for downstream metrics."""
        rng = np.random.default_rng(7)
        test_signal = rng.normal(0.0, 0.25, sample_rate)

        octave_bank = octave_filter.create_octave_bank(test_signal)
        bands = octave_bank[:, 1:]
        peak = np.max(np.abs(bands), axis=0)
        rms = np.sqrt(np.mean(np.square(bands), axis=0))
        crest_factor = np.divide(peak, rms, out=np.ones_like(peak), where=rms > 0)

        assert octave_bank.shape[0] == len(test_signal)
        assert np.all(np.isfinite(octave_bank))
        assert np.all(crest_factor >= 1.0)
