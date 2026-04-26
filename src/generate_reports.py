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

    def is_track_dir(d: Path) -> bool:
        """Heuristically determine if a directory is a track analysis folder."""
        if not d.is_dir() or d.name.startswith("."):
            return False

        # Exclude per-channel and per-group output folders.
        if d.name.startswith("Channel "):
            return False
        if d.name in {"Screen", "LFE", "Surround+Height", "Mono", "All Channels"}:
            return False

        # Mono/stereo tracks may have results directly in the folder.
        if (d / "analysis_results.csv").exists():
            return True

        # Multi-channel tracks typically have channel/group subfolders with results.
        # Only treat as a track folder if the immediate children look like channel/group
        # analysis folders (to avoid picking up category folders like analysis/Film).
        for child in d.iterdir():
            if not child.is_dir():
                continue
            if not (child / "analysis_results.csv").exists():
                continue
            if child.name.startswith("Channel ") or child.name in {
                "Screen",
                "LFE",
                "Surround+Height",
                "Mono",
                "All Channels",
            }:
                return True
        return False

    # Find track directories recursively (analysis may be organized by category).
    # We only treat a folder as a track if it contains analysis outputs (CSV).
    track_dirs = [d for d in analysis_dir.rglob("*") if is_track_dir(d)]

    if not track_dirs:
        logger.warning("No track directories found in analysis folder")
        return

    logger.info(f"Found {len(track_dirs)} track(s) to process")

    for track_dir in sorted(track_dirs, key=lambda p: str(p).lower()):
        track_name = track_dir.name
        # Output path points to the folder, main report will be analysis.md inside
        rel_track_dir = track_dir.relative_to(analysis_dir)
        output_path = reports_dir / rel_track_dir / "analysis.md"

        try:
            logger.info(f"Processing: {track_name}")
            generate_report(track_dir, output_path)
            logger.info(f"✓ Generated: {output_path}")
        except Exception as e:
            logger.error(f"✗ Failed to generate report for {track_name}: {e}", exc_info=True)

    logger.info("Report generation complete")


if __name__ == "__main__":
    main()

