"""Generate reports for all tracks in the analysis folder."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.report_generator import generate_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Generate reports for all tracks."""
    analysis_dir = Path("analysis")
    reports_dir = Path("reports")

    if not analysis_dir.exists():
        logger.error(f"Analysis directory not found: {analysis_dir}")
        sys.exit(1)

    reports_dir.mkdir(exist_ok=True)

    # Get all track directories
    track_dirs = [
        d for d in analysis_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    if not track_dirs:
        logger.warning("No track directories found in analysis folder")
        return

    logger.info(f"Found {len(track_dirs)} track(s) to process")

    for track_dir in sorted(track_dirs):
        track_name = track_dir.name
        # Output path points to the folder, main report will be analysis.md inside
        output_path = reports_dir / track_name / "analysis.md"

        try:
            logger.info(f"Processing: {track_name}")
            generate_report(track_dir, output_path)
            logger.info(f"✓ Generated: {output_path}")
        except Exception as e:
            logger.error(f"✗ Failed to generate report for {track_name}: {e}", exc_info=True)

    logger.info("Report generation complete")


if __name__ == "__main__":
    main()

