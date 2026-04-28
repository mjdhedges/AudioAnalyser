"""Versioned analysis result bundles for Audio Analyser."""

from src.results.bundle import write_channel_result_bundle
from src.results.reader import ResultBundle, find_result_bundles, load_result_bundle


def generate_bundle_report(*args, **kwargs):
    """Generate a Markdown report for a result bundle."""
    from src.report_generator import generate_bundle_report as _generate_bundle_report

    return _generate_bundle_report(*args, **kwargs)

__all__ = [
    "ResultBundle",
    "find_result_bundles",
    "generate_bundle_report",
    "load_result_bundle",
    "write_channel_result_bundle",
]
