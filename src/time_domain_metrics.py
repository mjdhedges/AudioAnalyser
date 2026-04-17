"""Time-domain metrics calculators (strategy-based).

This module isolates time-domain crest factor computation so it can be selected
via config without tangling the rest of the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

import numpy as np

from src.signal_metrics import (
    compute_peak_hold_envelope,
    compute_slow_rms_envelope,
    sampled_max_abs,
)


@dataclass(frozen=True)
class TimeDomainCrestFactorResult:
    """Container for time-domain crest factor outputs."""

    mode: Literal["slow", "fixed_chunk"]
    time_points: np.ndarray
    crest_factors: np.ndarray
    crest_factors_db: np.ndarray
    peak_levels: np.ndarray
    rms_levels: np.ndarray
    peak_levels_dbfs: np.ndarray
    rms_levels_dbfs: np.ndarray
    chunk_duration: float
    time_step_seconds: float
    num_chunks: int
    total_duration: float
    # Metadata to make exports/plots unambiguous
    rms_method: str
    peak_method: str


class TimeDomainCrestFactorCalculator(Protocol):
    def compute(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int,
        original_peak: float,
        config: Mapping[str, object],
    ) -> TimeDomainCrestFactorResult: ...


def _dbfs(levels: np.ndarray, original_peak: float, floor_dbfs: float) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.float64)
    out = np.full(levels.shape, -np.inf, dtype=np.float64)
    mask = levels > 0
    if np.any(mask):
        out[mask] = 20 * np.log10(levels[mask] * float(original_peak))
    return np.where(np.isfinite(out), out, float(floor_dbfs))


class SlowTimeDomainCalculator:
    """IEC-style SLOW time weighting: peak-hold + slow RMS, sampled at fixed step."""

    def __init__(self, peak_hold_tau_seconds: float) -> None:
        self.peak_hold_tau_seconds = float(peak_hold_tau_seconds)

    def compute(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int,
        original_peak: float,
        config: Mapping[str, object],
    ) -> TimeDomainCrestFactorResult:
        total_samples = int(audio_data.size)
        total_duration = total_samples / float(sample_rate) if sample_rate > 0 else 0.0

        window_seconds = float(config.get("time_domain_slow_window_seconds", 1.0) or 1.0)
        step_seconds = float(config.get("time_domain_slow_step_seconds", window_seconds) or window_seconds)
        rms_tau_seconds = float(config.get("time_domain_slow_rms_tau_seconds", 1.0) or 1.0)

        window_samples = max(int(window_seconds * sample_rate), 1)
        step_samples = max(int(step_seconds * sample_rate), 1)

        if total_samples < window_samples:
            empty = np.array([], dtype=np.float64)
            return TimeDomainCrestFactorResult(
                mode="slow",
                time_points=empty,
                crest_factors=empty,
                crest_factors_db=empty,
                peak_levels=empty,
                rms_levels=empty,
                peak_levels_dbfs=empty,
                rms_levels_dbfs=empty,
                chunk_duration=window_seconds,
                time_step_seconds=step_seconds,
                num_chunks=0,
                total_duration=total_duration,
                rms_method=f"slow_rms_tau={rms_tau_seconds}",
                peak_method=f"peak_hold_tau={self.peak_hold_tau_seconds}",
            )

        num_windows = (total_samples - window_samples) // step_samples + 1
        start_indices = np.arange(num_windows, dtype=np.int64) * step_samples
        end_indices = start_indices + window_samples - 1

        peak_env = compute_peak_hold_envelope(
            audio_data,
            sample_rate,
            tau=self.peak_hold_tau_seconds,
        )
        rms_env = compute_slow_rms_envelope(audio_data, sample_rate, tau=rms_tau_seconds)

        end_indices = np.clip(end_indices, 0, peak_env.size - 1)
        peak_levels = peak_env[end_indices]
        end_indices = np.clip(end_indices, 0, rms_env.size - 1)
        rms_levels = rms_env[end_indices]

        crest_factors = np.divide(
            peak_levels,
            rms_levels,
            out=np.ones_like(peak_levels),
            where=(rms_levels > 0),
        )
        crest_factors = np.maximum(crest_factors, 1.0)
        crest_factors_db = 20 * np.log10(crest_factors)
        crest_factors_db = np.where(np.isfinite(crest_factors_db), crest_factors_db, 0.0)

        peak_levels_dbfs = _dbfs(peak_levels, original_peak, -120.0)
        rms_levels_dbfs = _dbfs(rms_levels, original_peak, -120.0)
        time_points = (end_indices + 1) / float(sample_rate)

        return TimeDomainCrestFactorResult(
            mode="slow",
            time_points=np.asarray(time_points, dtype=np.float64),
            crest_factors=np.asarray(crest_factors, dtype=np.float64),
            crest_factors_db=np.asarray(crest_factors_db, dtype=np.float64),
            peak_levels=np.asarray(peak_levels, dtype=np.float64),
            rms_levels=np.asarray(rms_levels, dtype=np.float64),
            peak_levels_dbfs=np.asarray(peak_levels_dbfs, dtype=np.float64),
            rms_levels_dbfs=np.asarray(rms_levels_dbfs, dtype=np.float64),
            chunk_duration=window_seconds,
            time_step_seconds=step_seconds,
            num_chunks=int(num_windows),
            total_duration=float(total_duration),
            rms_method=f"slow_rms_tau={rms_tau_seconds}",
            peak_method=f"peak_hold_tau={self.peak_hold_tau_seconds}",
        )


class FixedChunkTimeDomainCalculator:
    """Fixed window peak+RMS measured per chunk."""

    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = float(window_seconds)

    def compute(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int,
        original_peak: float,
        config: Mapping[str, object],
    ) -> TimeDomainCrestFactorResult:
        total_samples = int(audio_data.size)
        total_duration = total_samples / float(sample_rate) if sample_rate > 0 else 0.0

        window_samples = max(int(self.window_seconds * sample_rate), 1)
        step_samples = window_samples

        if total_samples < window_samples:
            empty = np.array([], dtype=np.float64)
            return TimeDomainCrestFactorResult(
                mode="fixed_chunk",
                time_points=empty,
                crest_factors=empty,
                crest_factors_db=empty,
                peak_levels=empty,
                rms_levels=empty,
                peak_levels_dbfs=empty,
                rms_levels_dbfs=empty,
                chunk_duration=self.window_seconds,
                time_step_seconds=self.window_seconds,
                num_chunks=0,
                total_duration=total_duration,
                rms_method="window_rms",
                peak_method="window_peak",
            )

        num_windows = (total_samples - window_samples) // step_samples + 1
        # Peak per chunk
        peak_levels = sampled_max_abs(audio_data, window_samples, step_samples)
        peak_levels = peak_levels[:num_windows]

        # RMS per chunk via cumulative sum of squares
        x2 = np.square(audio_data.astype(np.float64, copy=False))
        csum = np.concatenate(([0.0], np.cumsum(x2)))
        starts = np.arange(num_windows, dtype=np.int64) * step_samples
        ends = starts + window_samples
        sum_sq = csum[ends] - csum[starts]
        mean_sq = sum_sq / float(window_samples)
        rms_levels = np.sqrt(np.clip(mean_sq, 0.0, None))

        crest_factors = np.divide(
            peak_levels,
            rms_levels,
            out=np.ones_like(peak_levels),
            where=(rms_levels > 0),
        )
        crest_factors = np.maximum(crest_factors, 1.0)
        crest_factors_db = 20 * np.log10(crest_factors)
        crest_factors_db = np.where(np.isfinite(crest_factors_db), crest_factors_db, 0.0)

        peak_levels_dbfs = _dbfs(peak_levels, original_peak, -120.0)
        rms_levels_dbfs = _dbfs(rms_levels, original_peak, -120.0)
        end_indices = starts + window_samples - 1
        time_points = (end_indices + 1) / float(sample_rate)

        return TimeDomainCrestFactorResult(
            mode="fixed_chunk",
            time_points=np.asarray(time_points, dtype=np.float64),
            crest_factors=np.asarray(crest_factors, dtype=np.float64),
            crest_factors_db=np.asarray(crest_factors_db, dtype=np.float64),
            peak_levels=np.asarray(peak_levels, dtype=np.float64),
            rms_levels=np.asarray(rms_levels, dtype=np.float64),
            peak_levels_dbfs=np.asarray(peak_levels_dbfs, dtype=np.float64),
            rms_levels_dbfs=np.asarray(rms_levels_dbfs, dtype=np.float64),
            chunk_duration=self.window_seconds,
            time_step_seconds=self.window_seconds,
            num_chunks=int(num_windows),
            total_duration=float(total_duration),
            rms_method="window_rms",
            peak_method="window_peak",
        )

