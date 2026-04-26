"""Render plots and reports from Audio Analyser result bundles."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src.results import find_result_bundles, load_result_bundle
from src.results.render import (
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
    "--dpi", type=int, default=300, show_default=True, help="Plot output DPI."
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
def main(
    results_path: Path,
    output_dir: Path,
    dpi: int,
    spectrum_plots: bool,
    histograms: bool,
    time_plots: bool,
) -> None:
    """Render plots from processed result bundles without loading source audio."""
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
                    dpi=dpi,
                )
            )
        if histograms:
            generated.extend(
                render_bundle_histograms(
                    bundle=bundle,
                    output_dir=bundle_output_dir,
                    dpi=dpi,
                )
            )
        if time_plots:
            generated.extend(
                render_bundle_time_plots(
                    bundle=bundle,
                    output_dir=bundle_output_dir,
                    dpi=dpi,
                )
            )

    logger.info("Rendered %d plot file(s).", len(generated))


if __name__ == "__main__":
    main()
