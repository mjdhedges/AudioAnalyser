"""Shared plotting utilities."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt


def add_calibrated_spl_axis(
    ax: plt.Axes, y_limits: Tuple[float, float], is_lfe: bool = False
) -> None:
    """Add calibrated SPL axis aligned with a dBFS axis.

    Args:
        ax: Matplotlib axis to attach the SPL axis to.
        y_limits: Tuple representing the current dBFS axis limits (min, max).
        is_lfe: Whether the plot represents the LFE channel (adds +10 dB).
    """
    base_offset = 115.0 if is_lfe else 105.0
    spl_limits = (y_limits[0] + base_offset, y_limits[1] + base_offset)

    ax_spl = ax.twinx()
    ax_spl.set_ylim(spl_limits)
    ax_spl.set_ylabel("Calibrated SPL at RSP (dB)", color="g")
    ax_spl.tick_params(axis="y", labelcolor="g")
    ax_spl.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax_spl.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax_spl.grid(True, alpha=0.2, which="major", linestyle="--", color="g")
    ax_spl.grid(True, alpha=0.1, which="minor", linestyle=":", color="g")

