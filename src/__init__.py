"""
Audio Analyser - A Python project for audio analysis.
"""

from __future__ import annotations

__author__ = "Michael Hedges"
__email__ = ""


def __getattr__(name: str):
    """Lazy ``__version__`` to avoid import cycles with ``version_info``."""
    if name == "__version__":
        from src.version_info import get_release_version

        return get_release_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
