"""GUI asset helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

ICON_FILENAME = "audioanalyser_icon.jpeg"


def app_icon_path() -> Path:
    """Return the best-known path to the application icon."""
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        bundled_path = Path(bundled_root) / ICON_FILENAME
        if bundled_path.exists():
            return bundled_path

    return Path(__file__).resolve().parents[2] / ICON_FILENAME


def app_icon() -> QIcon:
    """Load the Audio Analyser application icon."""
    return QIcon(str(app_icon_path()))
