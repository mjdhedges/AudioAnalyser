"""About dialog content for the desktop GUI."""

from __future__ import annotations

from src.version_info import get_about_version_text

APP_NAME = "Audio Analyser"
AUTHOR = "Michael Hedges"
GITHUB_URL = "https://github.com/mjdhedges/AudioAnalyser"
LICENSE = "GPL-3.0"
CITATION = (
    "Michael Hedges, Audio Analyser, GitHub repository, "
    "https://github.com/mjdhedges/AudioAnalyser"
)


def about_html() -> str:
    """Return rich text for the About dialog."""
    return f"""
    <h2>{APP_NAME}</h2>
    <p>Offline octave-band, crest-factor, envelope, and report generation for
    music, film, and test-signal audio.</p>
    <p><b>Author:</b> {AUTHOR}</p>
    <p><b>Build:</b> {get_about_version_text()}</p>
    <p><b>License:</b> {LICENSE}</p>
    <p><b>GitHub:</b>
    <a href="{GITHUB_URL}">{GITHUB_URL}</a></p>
    <p><b>Citation:</b><br>{CITATION}</p>
    """.strip()
