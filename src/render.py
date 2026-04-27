"""Render plots and reports from Audio Analyser result bundles."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from src.config import Config
from src.results import find_result_bundles, generate_bundle_report, load_result_bundle
from src.results.render import (
    render_bundle_envelope_plots,
    render_bundle_group_outputs,
    render_bundle_histograms,
    render_bundle_spectrum_plots,
    render_bundle_time_plots,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--results",
    "-r",
    "results_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a .aaresults bundle or a directory containing bundles.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory where rendered plots will be written.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("config.toml"),
    show_default=True,
    help="Configuration file used for render settings.",
)
@click.option(
    "--dpi",
    type=int,
    default=None,
    help="Override plot output DPI. Defaults to plotting.render_dpi from config.",
)
@click.option(
    "--spectrum-plots/--no-spectrum-plots",
    default=True,
    show_default=True,
    help="Render octave spectrum and crest-factor spectrum plots from bundle data.",
)
@click.option(
    "--histograms/--no-histograms",
    default=True,
    show_default=True,
    help="Render linear and log histogram plots from bundle data.",
)
@click.option(
    "--time-plots/--no-time-plots",
    default=True,
    show_default=True,
    help="Render crest-factor time plots from bundle data.",
)
@click.option(
    "--envelope-plots/--no-envelope-plots",
    default=True,
    show_default=True,
    help="Render pattern and independent envelope plots from bundle data.",
)
@click.option(
    "--group-plots/--no-group-plots",
    default=True,
    show_default=True,
    help="Render group plots and worst-channel manifest from bundle data.",
)
@click.option(
    "--reports/--no-reports",
    default=False,
    show_default=True,
    help="Generate Markdown reports from bundle data and rendered plots.",
)
def main(
    results_path: Path,
    output_dir: Path,
    config_path: Path,
    dpi: int,
    spectrum_plots: bool,
    histograms: bool,
    time_plots: bool,
    envelope_plots: bool,
    group_plots: bool,
    reports: bool,
) -> None:
    """Render plots from processed result bundles without loading source audio."""
    config = Config(config_path)
    render_dpi = _resolve_render_dpi(config, dpi)
    bundles = find_result_bundles(results_path)
    if not bundles:
        logger.error("No .aaresults bundles found under %s", results_path)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for bundle_path in bundles:
        bundle = load_result_bundle(bundle_path)
        bundle_output_dir = output_dir / bundle_path.stem
        if spectrum_plots:
            generated.extend(
                render_bundle_spectrum_plots(
                    bundle=bundle,
                    output_dir=bundle_output_dir,
                    dpi=render_dpi,
                )
            )
        if histograms:
            generated.extend(
                render_bundle_histograms(
                    bundle=bundle,
                    output_dir=bundle_output_dir,
                    dpi=render_dpi,
                )
            )
        if time_plots:
            generated.extend(
                render_bundle_time_plots(
                    bundle=bundle,
                    output_dir=bundle_output_dir,
                    dpi=render_dpi,
                )
            )
        if envelope_plots:
            generated.extend(
                render_bundle_envelope_plots(
                    bundle=bundle,
                    output_dir=bundle_output_dir,
                    dpi=render_dpi,
                )
            )
        if group_plots:
            generated.extend(
                render_bundle_group_outputs(
                    bundle=bundle,
                    output_dir=bundle_output_dir,
                    dpi=render_dpi,
                )
            )
        if reports:
            generated.append(
                generate_bundle_report(
                    bundle=bundle,
                    rendered_output_dir=bundle_output_dir,
                )
            )

    logger.info("Generated %d output file(s).", len(generated))


def _resolve_render_dpi(config: Config, override_dpi: Optional[int]) -> int:
    """Resolve runtime render DPI from CLI override or configuration."""
    if override_dpi is not None:
        return int(override_dpi)
    for key_path in (
        "plotting.render_dpi",
        "plotting.batch_dpi",
        "plotting.dpi",
    ):
        value = config.get(key_path)
        if value is not None:
            return int(value)
    return 150


if __name__ == "__main__":
    main()
