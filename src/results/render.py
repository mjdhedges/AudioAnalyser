"""Render plots from result bundles without loading source audio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting_utils import add_calibrated_spl_axis
from src.results.reader import ChannelResult, ResultBundle

logger = logging.getLogger(__name__)

_OCTAVE_TIME_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

_DEFAULT_OCTAVE_TICKS_HZ: list[tuple[float, str]] = [
    (8.0, "8"),
    (16.0, "16"),
    (31.25, "31.25"),
    (62.5, "62.5"),
    (125.0, "125"),
    (250.0, "250"),
    (500.0, "500"),
    (1000.0, "1k"),
    (2000.0, "2k"),
    (4000.0, "4k"),
    (8000.0, "8k"),
    (16000.0, "16k"),
]


def render_bundle_spectrum_plots(
    bundle: ResultBundle,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render octave spectrum and crest-factor plots for all channels."""
    output_paths: list[Path] = []
    for channel in bundle.channels():
        channel_output_dir = output_dir / channel.channel_id
        metadata = channel.read_json("metadata")
        plotting_config = channel.read_json("plotting_config")
        title_suffix = _title_suffix(bundle, channel, metadata)
        output_paths.append(
            render_channel_octave_spectrum(
                channel=channel,
                output_path=channel_output_dir / "octave_spectrum.png",
                plotting_config=plotting_config,
                title_suffix=title_suffix,
                dpi=dpi,
            )
        )
        output_paths.append(
            render_channel_crest_factor_spectrum(
                channel=channel,
                output_path=channel_output_dir / "crest_factor.png",
                plotting_config=plotting_config,
                title_suffix=title_suffix,
                dpi=dpi,
            )
        )
    return output_paths


def render_channel_octave_spectrum(
    *,
    channel: ChannelResult,
    output_path: Path,
    plotting_config: dict,
    title_suffix: str = "",
    dpi: int = 300,
) -> Path:
    """Render octave peak/RMS spectrum from bundle table data."""
    octave_data = _octave_band_rows(channel)
    if octave_data.empty:
        raise ValueError(f"No octave band data available for {channel.channel_id}")

    full_spectrum = _full_spectrum_row(channel)
    plot_freqs = octave_data["frequency_numeric"].to_numpy(dtype=float)
    plot_max_db = _finite_series(
        octave_data["max_amplitude_db"].to_numpy(dtype=float),
        -60.0,
    )
    plot_rms_db = _finite_series(octave_data["rms_db"].to_numpy(dtype=float), -60.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogx(plot_freqs, plot_max_db, "b-o", label="Max Peak (dBFS)", linewidth=2)
    ax.semilogx(plot_freqs, plot_rms_db, "r-s", label="RMS (dBFS)", linewidth=2)

    if full_spectrum is not None:
        track_peak_db = _safe_float(full_spectrum.get("max_amplitude_db"), -60.0)
        track_rms_db = _safe_float(full_spectrum.get("rms_db"), -60.0)
        ax.axhline(
            y=track_peak_db,
            color="blue",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Track Peak ({track_peak_db:.1f} dBFS)",
        )
        ax.axhline(
            y=track_rms_db,
            color="red",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Track RMS ({track_rms_db:.1f} dBFS)",
        )

    _add_extreme_chunk_overlay(ax, channel)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dBFS)")
    ax.set_title(f"Octave Band Analysis - Peak and RMS Levels (dBFS){title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _apply_frequency_axis(ax, plot_freqs, plotting_config, "octave_spectrum_xlim")
    ax.set_ylim(plotting_config.get("octave_spectrum_ylim", [-60, 3]))
    _set_octave_ticks(ax, plot_freqs)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered octave spectrum plot: %s", output_path)
    return output_path


def render_channel_crest_factor_spectrum(
    *,
    channel: ChannelResult,
    output_path: Path,
    plotting_config: dict,
    title_suffix: str = "",
    dpi: int = 300,
) -> Path:
    """Render octave crest-factor spectrum from bundle table data."""
    octave_data = _octave_band_rows(channel)
    if octave_data.empty:
        raise ValueError(f"No octave band data available for {channel.channel_id}")

    plot_freqs = octave_data["frequency_numeric"].to_numpy(dtype=float)
    crest_db = _finite_series(
        octave_data["crest_factor_db"].to_numpy(dtype=float),
        0.0,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogx(
        plot_freqs,
        crest_db,
        "g-o",
        label="Crest Factor (dB)",
        linewidth=2,
    )

    extreme_chunks = channel.read_table("extreme_chunks_octave_analysis")
    if not extreme_chunks.empty:
        for chunk_type, color, label_prefix in (
            ("min_crest", "g", "Min Crest Chunk"),
            ("max_crest", "m", "Max Crest Chunk"),
        ):
            chunk_rows = extreme_chunks[extreme_chunks["chunk_type"] == chunk_type]
            if chunk_rows.empty:
                continue
            chunk_rows = chunk_rows.sort_values("frequency_hz", kind="stable")
            ax.semilogx(
                chunk_rows["frequency_hz"],
                _finite_series(
                    chunk_rows["chunk_crest_factor_db"].to_numpy(dtype=float),
                    0.0,
                ),
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label=f"{label_prefix} ({_safe_float(chunk_rows['crest_factor_db'].iloc[0], 0.0):.1f} dB)",
            )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Crest Factor (dB)")
    ax.set_title(f"Octave Band Crest Factor Analysis{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _apply_frequency_axis(ax, plot_freqs, plotting_config, "crest_factor_xlim")
    ax.set_ylim(
        [
            float(plotting_config.get("crest_factor_ylim_min", 0)),
            float(plotting_config.get("crest_factor_ylim_max", 30)),
        ]
    )
    _set_octave_ticks(ax, plot_freqs)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered crest factor spectrum plot: %s", output_path)
    return output_path


def render_bundle_histograms(
    bundle: ResultBundle,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render linear and log histogram plots for all channels in a bundle.

    Args:
        bundle: Loaded analysis result bundle.
        output_dir: Destination directory for rendered plot files.
        dpi: Output DPI.

    Returns:
        Paths to generated plot files.
    """
    output_paths: list[Path] = []
    for channel in bundle.channels():
        channel_output_dir = output_dir / channel.channel_id
        metadata = channel.read_json("metadata")
        title_suffix = _title_suffix(bundle, channel, metadata)
        output_paths.append(
            render_channel_histogram(
                channel=channel,
                artifact_name="histogram_linear",
                output_path=channel_output_dir / "histograms.png",
                title=f"Amplitude Distribution by Octave Band{title_suffix}",
                x_label="Amplitude",
                x_limits=(-1.0, 1.0),
                dpi=dpi,
            )
        )
        plotting_config = channel.read_json("plotting_config")
        noise_floor_db = float(plotting_config.get("log_histogram_noise_floor_db", -60))
        max_db = float(plotting_config.get("log_histogram_max_db", 0))
        output_paths.append(
            render_channel_histogram(
                channel=channel,
                artifact_name="histogram_log_db",
                output_path=channel_output_dir / "histograms_log_db.png",
                title=(
                    "Amplitude Distribution by Octave Band "
                    f"(Log dBFS Scale){title_suffix}"
                ),
                x_label="Amplitude (dBFS)",
                x_limits=(noise_floor_db, max_db),
                dpi=dpi,
            )
        )
    return output_paths


def render_bundle_time_plots(
    bundle: ResultBundle,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render time-domain plots for all channels in a bundle."""
    output_paths: list[Path] = []
    for channel in bundle.channels():
        channel_output_dir = output_dir / channel.channel_id
        metadata = channel.read_json("metadata")
        title_suffix = _title_suffix(bundle, channel, metadata)
        output_paths.append(
            render_channel_crest_factor_time(
                channel=channel,
                output_path=channel_output_dir / "crest_factor_time.png",
                title_suffix=title_suffix,
                dpi=dpi,
            )
        )
        output_paths.append(
            render_channel_octave_crest_factor_time(
                channel=channel,
                output_path=channel_output_dir / "octave_crest_factor_time.png",
                title_suffix=title_suffix,
                dpi=dpi,
            )
        )
    return output_paths


def render_bundle_envelope_plots(
    bundle: ResultBundle,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render pattern and independent envelope plots for all channels."""
    output_paths: list[Path] = []
    for channel in bundle.channels():
        channel_output_dir = output_dir / channel.channel_id
        metadata = channel.read_json("metadata")
        envelope_config = channel.read_json("envelope_config")
        try:
            envelope_data = channel.read_json("envelope_plot_data")
        except (FileNotFoundError, KeyError):
            continue

        title_suffix = _title_suffix(bundle, channel, metadata)
        output_paths.extend(
            render_channel_pattern_envelopes(
                envelope_data=envelope_data,
                output_dir=channel_output_dir / "pattern_envelopes",
                envelope_config=envelope_config,
                title_suffix=title_suffix,
                channel_name=str(metadata.get("channel_name") or channel.channel_name),
                dpi=dpi,
            )
        )
        output_paths.extend(
            render_channel_independent_envelopes(
                envelope_data=envelope_data,
                output_dir=channel_output_dir / "independent_envelopes",
                envelope_config=envelope_config,
                title_suffix=title_suffix,
                channel_name=str(metadata.get("channel_name") or channel.channel_name),
                dpi=dpi,
            )
        )
    return output_paths


def render_bundle_group_outputs(
    bundle: ResultBundle,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render group-level plots and manifest from bundle channel tables."""
    output_paths: list[Path] = []
    groups = _group_bundle_channels(bundle)
    for group_name, channels in groups.items():
        group_dir = output_dir / group_name
        output_paths.append(
            _render_group_crest_factor_time(
                bundle=bundle,
                group_name=group_name,
                channels=channels,
                output_path=group_dir / "crest_factor_time.png",
                dpi=dpi,
            )
        )
        output_paths.append(
            _render_group_octave_spectrum(
                bundle=bundle,
                group_name=group_name,
                channels=channels,
                output_path=group_dir / "octave_spectrum.png",
                dpi=dpi,
            )
        )
    output_paths.append(_write_worst_channels_manifest(bundle, output_dir))
    decay_path = _render_peak_decay_groups(
        bundle, groups, output_dir / "peak_decay_groups.png", dpi
    )
    if decay_path is not None:
        output_paths.append(decay_path)
    return output_paths


def render_channel_pattern_envelopes(
    *,
    envelope_data: dict,
    output_dir: Path,
    envelope_config: dict,
    title_suffix: str = "",
    channel_name: str = "",
    dpi: int = 300,
) -> list[Path]:
    """Render repeating-pattern envelope plots from stored envelope windows."""
    output_paths: list[Path] = []
    num_envelopes = int(envelope_config.get("envelope_plots_num_pattern_envelopes", 10))
    ylim = _envelope_ylim(envelope_config)
    is_lfe_channel = "LFE" in channel_name.upper()

    for frequency_label, band_data in envelope_data.items():
        pattern_analysis = band_data.get("pattern_analysis", {})
        pattern_envelopes = []
        patterns_detected = int(pattern_analysis.get("patterns_detected", 0) or 0)
        for pattern_num in range(1, patterns_detected + 1):
            pattern = pattern_analysis.get(f"pattern_{pattern_num}", {})
            windows = pattern.get("envelope_windows", []) or []
            time_windows = pattern.get("time_windows_ms", []) or []
            peak_times = pattern.get("peak_times_seconds", []) or []
            for idx, envelope_window in enumerate(windows):
                if envelope_window is None or idx >= len(time_windows):
                    continue
                time_window = time_windows[idx]
                if time_window is None:
                    continue
                values = np.asarray(envelope_window, dtype=float)
                times = np.asarray(time_window, dtype=float)
                if not values.size or not times.size:
                    continue
                peak_value_db = float(np.nanmax(values))
                peak_time_seconds = (
                    _safe_float(peak_times[idx], 0.0) if idx < len(peak_times) else 0.0
                )
                pattern_envelopes.append(
                    {
                        "pattern_num": pattern_num,
                        "time_ms": times,
                        "envelope": values,
                        "peak_value_db": peak_value_db,
                        "peak_time_seconds": peak_time_seconds,
                    }
                )

        if not pattern_envelopes:
            continue
        pattern_envelopes.sort(key=lambda item: item["peak_value_db"], reverse=True)
        top_envelopes = pattern_envelopes[:num_envelopes]
        output_path = (
            output_dir
            / f"pattern_envelopes_{_safe_frequency_filename(frequency_label)}.png"
        )
        _render_envelope_window_plot(
            output_path=output_path,
            envelopes=top_envelopes,
            title=f"Top {len(top_envelopes)} Pattern Envelopes - {frequency_label}{title_suffix}",
            ylim=ylim,
            is_lfe_channel=is_lfe_channel,
            dpi=dpi,
            label_prefix="Pattern",
        )
        output_paths.append(output_path)
    return output_paths


def render_channel_independent_envelopes(
    *,
    envelope_data: dict,
    output_dir: Path,
    envelope_config: dict,
    title_suffix: str = "",
    channel_name: str = "",
    dpi: int = 300,
) -> list[Path]:
    """Render independent worst-case envelope plots from stored windows."""
    output_paths: list[Path] = []
    num_envelopes = int(
        envelope_config.get("envelope_plots_num_independent_envelopes", 3)
    )
    ylim = _envelope_ylim(envelope_config)
    is_lfe_channel = "LFE" in channel_name.upper()

    for frequency_label, band_data in envelope_data.items():
        worst_cases = band_data.get("worst_case_envelopes", []) or []
        envelopes = []
        for envelope in worst_cases[:num_envelopes]:
            envelope_window = envelope.get("envelope_window")
            time_window = envelope.get("time_window_ms")
            if envelope_window is None or time_window is None:
                continue
            values = np.asarray(envelope_window, dtype=float)
            times = np.asarray(time_window, dtype=float)
            if not values.size or not times.size:
                continue
            envelopes.append(
                {
                    "rank": envelope.get("rank"),
                    "time_ms": times,
                    "envelope": values,
                    "peak_value_db": _safe_float(
                        envelope.get("peak_value_db"),
                        float(np.nanmax(values)),
                    ),
                    "peak_time_seconds": _safe_float(
                        envelope.get("peak_time_seconds"),
                        0.0,
                    ),
                    "decay_times": envelope.get("decay_times", {}),
                }
            )

        if not envelopes:
            continue
        output_path = (
            output_dir
            / f"independent_envelopes_{_safe_frequency_filename(frequency_label)}.png"
        )
        _render_envelope_window_plot(
            output_path=output_path,
            envelopes=envelopes,
            title=f"Top {len(envelopes)} Independent Envelopes - {frequency_label}{title_suffix}",
            ylim=ylim,
            is_lfe_channel=is_lfe_channel,
            dpi=dpi,
            label_prefix="Rank",
            show_decay=True,
        )
        output_paths.append(output_path)
    return output_paths


def render_channel_crest_factor_time(
    *,
    channel: ChannelResult,
    output_path: Path,
    title_suffix: str = "",
    dpi: int = 300,
) -> Path:
    """Render full-spectrum crest factor over time from bundle table data."""
    time_data = channel.read_table("time_domain_analysis")
    if time_data.empty:
        raise ValueError(f"No time-domain data available for {channel.channel_id}")

    time_points = time_data["time_seconds"].to_numpy(dtype=float)
    crest_factors_db = _finite_series(
        time_data["crest_factor_db"].to_numpy(dtype=float),
        fallback=-60.0,
    )
    peak_levels_dbfs = _finite_series(
        time_data["peak_level_dbfs"].to_numpy(dtype=float),
        fallback=-120.0,
    )
    rms_levels_dbfs = _finite_series(
        time_data["rms_level_dbfs"].to_numpy(dtype=float),
        fallback=-120.0,
    )
    peak_levels = time_data.get("peak_level", pd.Series(dtype=float)).to_numpy(
        dtype=float
    )
    rms_levels = time_data.get("rms_level", pd.Series(dtype=float)).to_numpy(
        dtype=float
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    channel_label = _channel_label(channel.channel_name)
    color = "#1f77b4"

    ax1.plot(
        time_points,
        crest_factors_db,
        color=color,
        linewidth=2,
        alpha=0.8,
        label=channel_label,
    )
    ax1.set_xlim(time_points.min(), time_points.max())
    ax1.set_ylim([0, 30])
    ax1.set_ylabel("Crest Factor (dB)")
    ax1.set_title(f"Crest Factor vs Time{title_suffix}")
    ax1.grid(True, alpha=0.3, which="major")
    ax1.grid(True, alpha=0.15, which="minor")
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.legend(loc="best", fontsize=10)

    ax2.plot(
        time_points,
        peak_levels_dbfs,
        color=color,
        linewidth=2,
        alpha=0.8,
        linestyle="-",
        label=f"{channel_label} Peak",
    )
    ax2.plot(
        time_points,
        rms_levels_dbfs,
        color=color,
        linewidth=2,
        alpha=0.8,
        linestyle="--",
        label=f"{channel_label} RMS",
    )
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Level (dBFS)")
    ax2.set_title(f"Peak and RMS Levels vs Time{title_suffix}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9, ncol=2)
    level_ylim = (-40, 3)
    ax2.set_ylim(level_ylim)
    add_calibrated_spl_axis(
        ax2, level_ylim, is_lfe="LFE" in channel.channel_name.upper()
    )

    finite_cf = crest_factors_db[np.isfinite(crest_factors_db)]
    avg_crest_db = _average_crest_db(peak_levels, rms_levels)
    max_crest_db = float(np.max(finite_cf)) if finite_cf.size else 0.0
    min_crest_db = float(np.min(finite_cf)) if finite_cf.size else 0.0
    ax1.text(
        0.02,
        0.95,
        (
            f"Ave. Crest Factor: {avg_crest_db:.1f} dB | "
            f"Max Crest Factor: {max_crest_db:.1f} dB | "
            f"Min Crest Factor: {min_crest_db:.1f} dB"
        ),
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered crest factor time plot: %s", output_path)
    return output_path


def render_channel_octave_crest_factor_time(
    *,
    channel: ChannelResult,
    output_path: Path,
    title_suffix: str = "",
    dpi: int = 300,
) -> Path:
    """Render octave-band crest factor over time from bundle table data."""
    octave_time = channel.read_table("octave_time_metrics")
    if octave_time.empty:
        raise ValueError(f"No octave time data available for {channel.channel_id}")

    fig, ax = plt.subplots(figsize=(14, 8))
    ordered = octave_time.sort_values(["frequency_hz", "time_seconds"], kind="stable")
    for idx, (frequency_hz, rows) in enumerate(
        ordered.groupby("frequency_hz", sort=True)
    ):
        ax.plot(
            rows["time_seconds"],
            _finite_series(rows["crest_factor_db"].to_numpy(dtype=float), 0.0),
            color=_OCTAVE_TIME_COLORS[idx % len(_OCTAVE_TIME_COLORS)],
            linewidth=1.5,
            label=_format_frequency_label(float(frequency_hz)),
            alpha=0.8,
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Crest Factor (dB)")
    ax.set_title(f"Octave Band Crest Factor vs Time{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim([0, 30])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered octave crest factor time plot: %s", output_path)
    return output_path


def render_channel_histogram(
    *,
    channel: ChannelResult,
    artifact_name: str,
    output_path: Path,
    title: str,
    x_label: str,
    x_limits: Optional[tuple[float, float]] = None,
    dpi: int = 300,
) -> Path:
    """Render one histogram grid from pre-binned bundle data."""
    histogram_data = channel.read_table(artifact_name)
    frequency_groups = _histogram_groups(histogram_data)
    num_bands = max(len(frequency_groups), 1)
    num_cols = 4
    num_rows = int(np.ceil(num_bands / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
    axes = np.asarray(axes).flatten()

    for idx, (label, rows) in enumerate(frequency_groups):
        ax = axes[idx]
        if rows.empty:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title(f"{label} Hz (No Data)")
        else:
            widths = rows["bin_right"].to_numpy() - rows["bin_left"].to_numpy()
            ax.bar(
                rows["bin_center"],
                rows["bin_density"],
                width=widths,
                align="center",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.2,
            )
            ax.set_title(_format_frequency_title(label, artifact_name))
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    for idx in range(len(frequency_groups), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered histogram plot: %s", output_path)
    return output_path


def _histogram_groups(dataframe: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if dataframe.empty:
        return []
    ordered = dataframe.copy()
    ordered["_sort_key"] = ordered["frequency_hz"].fillna(np.inf)
    ordered = ordered.sort_values(["_sort_key", "bin_left"], kind="stable")
    return [
        (str(label), rows.drop(columns=["_sort_key"]))
        for label, rows in ordered.groupby("frequency_label", sort=False)
    ]


def _render_envelope_window_plot(
    *,
    output_path: Path,
    envelopes: list[dict],
    title: str,
    ylim: tuple[float, float],
    is_lfe_channel: bool,
    dpi: int,
    label_prefix: str,
    show_decay: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x_bounds = []

    for idx, envelope in enumerate(envelopes):
        color = colors[idx % len(colors)]
        time_ms = np.asarray(envelope["time_ms"], dtype=float)
        values = np.asarray(envelope["envelope"], dtype=float)
        peak_value_db = _safe_float(
            envelope.get("peak_value_db"), float(np.nanmax(values))
        )
        peak_time_seconds = _safe_float(envelope.get("peak_time_seconds"), 0.0)
        identifier = envelope.get("pattern_num", envelope.get("rank", idx + 1))
        label = (
            f"{label_prefix} {identifier} - Peak: {peak_value_db:.1f} dBFS "
            f"@ {peak_time_seconds:.1f}s"
        )

        ax.plot(time_ms, values, color=color, linewidth=2, label=label, alpha=0.8)
        ax.plot(
            0,
            peak_value_db,
            "o",
            color=color,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )
        x_bounds.extend([float(np.nanmin(time_ms)), float(np.nanmax(time_ms))])

        if show_decay:
            decay_times = envelope.get("decay_times", {}) or {}
            for decay_db in (-3, -6, -9, -12):
                decay_key = f"decay_{abs(decay_db)}db_ms"
                decay_time_ms = decay_times.get(decay_key)
                if decay_time_ms is None:
                    continue
                decay_level = peak_value_db + decay_db
                ax.plot(
                    _safe_float(decay_time_ms, 0.0),
                    decay_level,
                    "x",
                    color=color,
                    markersize=6,
                    alpha=0.7,
                )

    ax.set_xlabel("Time (ms relative to peak)")
    ax.set_ylabel("RMS Level (dBFS)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(ylim)
    add_calibrated_spl_axis(ax, ylim, is_lfe=is_lfe_channel)
    if x_bounds:
        ax.set_xlim([min(x_bounds), max(x_bounds)])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered envelope plot: %s", output_path)


def _render_group_crest_factor_time(
    *,
    bundle: ResultBundle,
    group_name: str,
    channels: list[ChannelResult],
    output_path: Path,
    dpi: int,
) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = _channel_colors()
    all_peak_levels = []
    all_rms_levels = []
    all_crest_factors = []

    for idx, channel in enumerate(channels):
        data = channel.read_table("time_domain_analysis")
        if data.empty:
            continue
        color = colors[idx % len(colors)]
        label = _channel_label(channel.channel_name)
        time_seconds = data["time_seconds"].to_numpy(dtype=float)
        crest_db = _finite_series(data["crest_factor_db"].to_numpy(dtype=float), 0.0)
        peak_dbfs = _finite_series(
            data["peak_level_dbfs"].to_numpy(dtype=float), -120.0
        )
        rms_dbfs = _finite_series(data["rms_level_dbfs"].to_numpy(dtype=float), -120.0)
        peak_linear = np.power(10.0, peak_dbfs / 20.0)
        rms_linear = np.power(10.0, rms_dbfs / 20.0)

        all_crest_factors.extend(crest_db[np.isfinite(crest_db)].tolist())
        all_peak_levels.extend(peak_linear[peak_linear > 0].tolist())
        all_rms_levels.extend(rms_linear[rms_linear > 0].tolist())

        ax1.plot(
            time_seconds, crest_db, color=color, linewidth=1.5, alpha=0.8, label=label
        )
        ax2.plot(
            time_seconds,
            peak_dbfs,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=f"{label} Peak",
        )
        ax2.plot(
            time_seconds,
            rms_dbfs,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"{label} RMS",
        )

    ax1.set_ylabel("Crest Factor (dB)")
    ax1.set_title(f"Crest Factor Over Time - {group_name} - {_track_title(bundle)}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    if all_crest_factors:
        ax1.set_ylim([0, max(30, float(np.nanmax(all_crest_factors)) * 1.1)])

    level_ylim = (-60, 3)
    if all_peak_levels and all_rms_levels:
        highest_peak = 20 * np.log10(max(all_peak_levels))
        lowest_rms = 20 * np.log10(max(min(all_rms_levels), 1e-10))
        level_ylim = (min(-60, lowest_rms - 3), max(3, highest_peak + 1))
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Level (dBFS)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize="small")
    ax2.set_ylim(level_ylim)
    add_calibrated_spl_axis(ax2, level_ylim, is_lfe=(group_name == "LFE"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered group crest factor time plot: %s", output_path)
    return output_path


def _render_group_octave_spectrum(
    *,
    bundle: ResultBundle,
    group_name: str,
    channels: list[ChannelResult],
    output_path: Path,
    dpi: int,
) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = _channel_colors()
    all_freqs = []

    for idx, channel in enumerate(channels):
        octave_data = _octave_band_rows(channel)
        if octave_data.empty:
            continue
        color = colors[idx % len(colors)]
        label = _channel_label(channel.channel_name)
        freqs = octave_data["frequency_numeric"].to_numpy(dtype=float)
        crest_db = _finite_series(
            octave_data["crest_factor_db"].to_numpy(dtype=float), 0.0
        )
        peak_db = _finite_series(
            octave_data["max_amplitude_db"].to_numpy(dtype=float), -60.0
        )
        rms_db = _finite_series(octave_data["rms_db"].to_numpy(dtype=float), -60.0)
        all_freqs.extend(freqs.tolist())

        ax1.semilogx(
            freqs,
            np.maximum(crest_db, 0.0),
            color=color,
            marker="o",
            linewidth=2,
            alpha=0.8,
            label=label,
        )
        ax2.semilogx(
            freqs,
            peak_db,
            color=color,
            marker="o",
            linewidth=2,
            alpha=0.8,
            label=f"{label} Peak",
        )
        ax2.semilogx(
            freqs,
            rms_db,
            color=color,
            marker="s",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"{label} RMS",
        )

    ax1.set_ylabel("Crest Factor (dB)")
    ax1.set_title(f"Octave Spectrum by Channel - {group_name} - {_track_title(bundle)}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_ylim([0, 30])

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Level (dBFS)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize="small")
    ax2.set_ylim([-60, 3])
    if all_freqs:
        freqs_array = np.asarray(all_freqs, dtype=float)
        _set_octave_ticks(ax2, freqs_array)
        ax2.set_xlim([max(3.0, float(np.nanmin(freqs_array)) / 1.25), 40000.0])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered group octave spectrum plot: %s", output_path)
    return output_path


def _write_worst_channels_manifest(bundle: ResultBundle, output_dir: Path) -> Path:
    rows = []
    candidates: dict[str, tuple[ChannelResult, float, str]] = {}
    for channel in bundle.channels():
        group = _worst_channel_group(channel.channel_name)
        if group is None:
            continue
        score, metric_used = _channel_worst_score(channel)
        previous = candidates.get(group)
        if previous is None or score > previous[1]:
            candidates[group] = (channel, score, metric_used)

    for group, (channel, score, metric_used) in candidates.items():
        rows.append(
            {
                "group": group,
                "folder": channel.channel_name,
                "score": score,
                "metric_used": metric_used,
            }
        )
    manifest_path = output_dir / "worst_channels_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["group", "folder", "score", "metric_used"]).to_csv(
        manifest_path,
        index=False,
    )
    logger.info("Rendered worst-channel manifest: %s", manifest_path)
    return manifest_path


def _render_peak_decay_groups(
    bundle: ResultBundle,
    groups: dict[str, list[ChannelResult]],
    output_path: Path,
    dpi: int,
) -> Optional[Path]:
    group_rows: dict[str, list[dict]] = {}
    group_peak_levels: dict[str, list[float]] = {}
    for group_name, channels in groups.items():
        rows = []
        peaks = []
        for channel in channels:
            sustained = channel.read_table("sustained_peaks_summary")
            if sustained.empty:
                continue
            full_spectrum = sustained[sustained["frequency_hz"] == "Full Spectrum"]
            if full_spectrum.empty:
                continue
            rows.append(full_spectrum.iloc[0].to_dict())
            advanced = channel.read_table("advanced_statistics")
            if not advanced.empty:
                values = dict(zip(advanced["parameter"], advanced["value"]))
                peaks.append(_safe_float(values.get("max_true_peak_dbfs"), 0.0))
        if rows:
            group_rows[group_name] = rows
            group_peak_levels[group_name] = peaks

    if not group_rows:
        return None

    max_time = 0.0
    for rows in group_rows.values():
        times_ms, _ = _aggregate_decay_times(rows)
        if times_ms.size and np.any(times_ms > 0):
            max_time = max(max_time, float(np.nanmax(times_ms)))
    x_max = max(200, int(max_time * 1.2)) if max_time > 0 else 200
    t_axis = np.linspace(0, x_max, 400)
    colors = {
        "Screen": "#1f77b4",
        "Surround+Height": "#2ca02c",
        "LFE": "#d62728",
        "All Channels": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for group_name, rows in group_rows.items():
        times_ms, levels_db = _aggregate_decay_times(rows)
        valid_mask = times_ms > 0
        if not np.any(valid_mask):
            continue
        times_valid = times_ms[valid_mask]
        levels_valid = levels_db[valid_mask]
        tau, amplitude = _fit_decay_curve(times_valid, levels_valid)
        curve_db = -amplitude * np.log1p(t_axis / max(tau, 1e-3))
        peaks = group_peak_levels.get(group_name, [])
        peak_dbfs = max(peaks) if peaks else 0.0
        peak_label = f"Peak: {peak_dbfs:.1f} dBFS" if peak_dbfs > -60 else "Peak: N/A"
        ax.plot(
            t_axis,
            curve_db,
            label=f"{group_name} (tau~{tau:.1f} ms, {peak_label})",
            color=colors.get(group_name),
            linewidth=2,
        )
        ax.scatter(
            times_valid,
            levels_valid,
            color=colors.get(group_name),
            marker="o",
            s=50,
            zorder=5,
        )

    ax.set_xlabel("Time after peak (ms)")
    ax.set_ylabel("Relative level (dB)")
    ax.set_title(
        f"Worst Case (P95) Peak Decay by Channel Group - {_track_title(bundle)}"
    )
    ax.set_ylim([-20, 0])
    ax.set_yticks(np.arange(-21, 1, 3))
    ax.set_yticks(np.arange(-20, 1, 1), minor=True)
    ax.grid(True, alpha=0.3, which="major", linewidth=1.0)
    ax.grid(True, alpha=0.15, which="minor", linewidth=0.5)
    ax.legend()
    ax.set_xlim([0, x_max])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rendered peak decay group plot: %s", output_path)
    return output_path


def _aggregate_decay_times(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    times = []
    for threshold in (3, 6, 9, 12):
        p95_key = f"t{threshold}_ms_p95"
        p90_key = f"t{threshold}_ms_p90"
        values = [_safe_float(row.get(p95_key), 0.0) for row in rows]
        values = [value for value in values if value > 0]
        if not values:
            times.append(0.0)
            continue
        aggregated = max(values)
        if aggregated >= 4950.0:
            fallback_values = [_safe_float(row.get(p90_key), 0.0) for row in rows]
            fallback_values = [value for value in fallback_values if value > 0]
            if fallback_values:
                aggregated = max(fallback_values)
        times.append(aggregated)
    return np.asarray(times, dtype=float), -np.asarray([3, 6, 9, 12], dtype=float)


def _fit_decay_curve(
    times_ms: np.ndarray, levels_db: np.ndarray
) -> tuple[float, float]:
    if len(times_ms) < 2 or np.any(times_ms <= 0) or np.any(levels_db >= 0):
        return 50.0, 12.0
    mid_idx = len(times_ms) // 2
    tau = float(times_ms[mid_idx])
    amplitude = abs(float(levels_db[mid_idx])) / 0.693
    return max(tau, 1.0), max(amplitude, 1.0)


def _group_bundle_channels(bundle: ResultBundle) -> dict[str, list[ChannelResult]]:
    grouped = {"Screen": [], "Surround+Height": [], "LFE": [], "All Channels": []}
    for channel in bundle.channels():
        name = channel.channel_name
        if name in {"Channel 1 FL", "Channel 2 FR", "Channel 3 FC"}:
            grouped["Screen"].append(channel)
        elif "LFE" in name:
            grouped["LFE"].append(channel)
        elif any(
            name.startswith(prefix)
            for prefix in (
                "Channel 5",
                "Channel 6",
                "Channel 7",
                "Channel 8",
                "Channel 9",
                "Channel 10",
                "Channel 11",
                "Channel 12",
            )
        ):
            grouped["Surround+Height"].append(channel)
        else:
            grouped["All Channels"].append(channel)
    return {name: channels for name, channels in grouped.items() if channels}


def _channel_worst_score(channel: ChannelResult) -> tuple[float, str]:
    sustained = channel.read_table("sustained_peaks_summary")
    if not sustained.empty:
        full_spectrum = sustained[sustained["frequency_hz"] == "Full Spectrum"]
        if not full_spectrum.empty:
            row = full_spectrum.iloc[0]
            for key in (
                "t6_ms_p95",
                "t9_ms_p95",
                "t3_ms_p95",
                "t12_ms_p95",
                "hold_ms_p95",
            ):
                if key in row:
                    return _safe_float(row[key], 0.0), f"sustained:{key}"
    advanced = channel.read_table("advanced_statistics")
    if not advanced.empty:
        values = dict(zip(advanced["parameter"], advanced["value"]))
        for key in ("peak_saturation_percent", "clip_events_rate_per_sec"):
            if key in values:
                return _safe_float(values[key], 0.0), f"advanced:{key}"
    return 0.0, "none"


def _worst_channel_group(channel_name: str) -> Optional[str]:
    if channel_name in {"Channel 1 FL", "Channel 2 FR", "Channel 3 FC"}:
        return "screen"
    if "LFE" in channel_name:
        return "lfe"
    if channel_name.startswith(("Channel 5", "Channel 6", "Channel 7", "Channel 8")):
        return "surround"
    return None


def _channel_colors() -> list[str]:
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]


def _envelope_ylim(envelope_config: dict) -> tuple[float, float]:
    return (
        float(envelope_config.get("envelope_plots_ylim_min", -30)),
        float(envelope_config.get("envelope_plots_ylim_max", 0)),
    )


def _safe_frequency_filename(frequency_label: str) -> str:
    return str(frequency_label).replace(" ", "_").replace(".", "_")


def _octave_band_rows(channel: ChannelResult) -> pd.DataFrame:
    octave_data = channel.read_table("octave_band_analysis")
    if octave_data.empty:
        return octave_data
    rows = octave_data[octave_data["frequency_hz"] != "Full Spectrum"].copy()
    rows["frequency_numeric"] = pd.to_numeric(rows["frequency_hz"], errors="coerce")
    rows = rows[np.isfinite(rows["frequency_numeric"])]
    return rows.sort_values("frequency_numeric", kind="stable")


def _full_spectrum_row(channel: ChannelResult) -> Optional[pd.Series]:
    octave_data = channel.read_table("octave_band_analysis")
    if octave_data.empty:
        return None
    full_rows = octave_data[octave_data["frequency_hz"] == "Full Spectrum"]
    if full_rows.empty:
        return None
    return full_rows.iloc[0]


def _add_extreme_chunk_overlay(
    ax,
    channel: ChannelResult,
) -> None:
    extreme_chunks = channel.read_table("extreme_chunks_octave_analysis")
    if extreme_chunks.empty:
        return

    overlays = [
        ("min_crest", "g", "Min Crest"),
        ("max_crest", "m", "Max Crest"),
    ]
    for chunk_type, color, label_prefix in overlays:
        chunk_rows = extreme_chunks[extreme_chunks["chunk_type"] == chunk_type]
        if chunk_rows.empty:
            continue
        chunk_rows = chunk_rows.sort_values("frequency_hz", kind="stable")
        time_seconds = _safe_float(chunk_rows["time_seconds"].iloc[0], 0.0)
        crest_db = _safe_float(chunk_rows["crest_factor_db"].iloc[0], 0.0)
        ax.semilogx(
            chunk_rows["frequency_hz"],
            _finite_series(chunk_rows["rms_db"].to_numpy(dtype=float), -60.0),
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label=f"{label_prefix} RMS ({crest_db:.1f} dB @ {time_seconds:.0f}s)",
        )
        ax.semilogx(
            chunk_rows["frequency_hz"],
            _finite_series(
                chunk_rows["max_amplitude_db"].to_numpy(dtype=float),
                -60.0,
            ),
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
            label=f"{label_prefix} Peak (@ {time_seconds:.0f}s)",
        )


def _apply_frequency_axis(
    ax,
    plot_freqs: np.ndarray,
    plotting_config: dict,
    xlim_key: str,
) -> None:
    xlim = plotting_config.get(xlim_key, [7, 20000])
    try:
        x_left_cfg = float(xlim[0])
        x_right = float(xlim[1])
    except (TypeError, ValueError, IndexError):
        x_left_cfg, x_right = 7.0, 20000.0
    x_left_data = float(np.min(plot_freqs)) / 1.25 if plot_freqs.size else x_left_cfg
    ax.set_xlim([min(x_left_cfg, x_left_data), x_right])


def _set_octave_ticks(ax, plot_freqs: np.ndarray) -> None:
    tick_x, tick_labels = _octave_semilog_ticks(plot_freqs)
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels)


def _octave_semilog_ticks(plot_freqs: np.ndarray) -> tuple[list[float], list[str]]:
    if plot_freqs.size == 0:
        return [f for f, _ in _DEFAULT_OCTAVE_TICKS_HZ], [
            label for _, label in _DEFAULT_OCTAVE_TICKS_HZ
        ]
    finite_freqs = plot_freqs[np.isfinite(plot_freqs)]
    if finite_freqs.size == 0:
        return [f for f, _ in _DEFAULT_OCTAVE_TICKS_HZ], [
            label for _, label in _DEFAULT_OCTAVE_TICKS_HZ
        ]

    min_f = float(np.min(finite_freqs))
    ticks: list[float] = []
    labels: list[str] = []
    for frequency_hz, label in _DEFAULT_OCTAVE_TICKS_HZ:
        if frequency_hz >= min_f * 0.85:
            ticks.append(frequency_hz)
            labels.append(label)
    for frequency in finite_freqs:
        freq = float(frequency)
        if (
            not ticks
            or min(abs(freq - tick) / max(tick, 1e-9) for tick in ticks) > 0.04
        ):
            ticks.append(freq)
            labels.append(
                f"{freq:g}" if freq >= 100 else f"{freq:.3g}".rstrip("0").rstrip(".")
            )
    order = sorted(range(len(ticks)), key=lambda i: ticks[i])
    return [ticks[i] for i in order], [labels[i] for i in order]


def _format_frequency_title(label: str, artifact_name: str) -> str:
    suffix = " (Log dB)" if artifact_name == "histogram_log_db" else ""
    if label == "Full Spectrum":
        return f"{label}{suffix}"
    return f"{label} Hz{suffix}"


def _format_frequency_label(frequency_hz: float) -> str:
    if frequency_hz >= 1000:
        return (
            f"{frequency_hz / 1000:.0f}k Hz"
            if frequency_hz % 1000 == 0
            else f"{frequency_hz / 1000:.1f}k Hz"
        )
    return (
        f"{frequency_hz:.0f} Hz"
        if frequency_hz == int(frequency_hz)
        else f"{frequency_hz:.1f} Hz"
    )


def _channel_label(channel_name: str) -> str:
    label = channel_name.replace("Channel ", "").replace("Channel", "").strip()
    return label or channel_name or "Channel"


def _finite_series(values: np.ndarray, fallback: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, fallback)


def _safe_float(value, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if np.isfinite(parsed) else fallback


def _average_crest_db(peak_levels: np.ndarray, rms_levels: np.ndarray) -> float:
    valid_rms = rms_levels[np.isfinite(rms_levels) & (rms_levels > 0)]
    valid_peaks = peak_levels[np.isfinite(peak_levels) & (peak_levels > 0)]
    if not valid_peaks.size or not valid_rms.size:
        return 0.0
    highest_peak = float(np.max(valid_peaks))
    avg_rms = float(np.mean(valid_rms))
    if highest_peak <= 0 or avg_rms <= 0:
        return 0.0
    return float(20 * np.log10(highest_peak / avg_rms))


def _title_suffix(
    bundle: ResultBundle,
    channel: ChannelResult,
    metadata: dict,
) -> str:
    track_name = Path(str(bundle.track.get("track_name") or "")).stem
    channel_name = metadata.get("channel_name") or channel.channel_name
    parts = [part for part in (track_name, channel_name) if part]
    return f" - {' - '.join(parts)}" if parts else ""


def _track_title(bundle: ResultBundle) -> str:
    return Path(str(bundle.track.get("track_name") or bundle.path.stem)).stem
