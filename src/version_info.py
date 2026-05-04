"""Release version and git identity for reports, manifests, and the GUI.

Embedded builds load generated ``src/_version_build.py`` (created by
``packaging/write_build_info.py`` before PyInstaller). Development checkouts
fall back to ``importlib.metadata`` and optional ``git describe``.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PACKAGE_NAME = "audio-analyser"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _try_generated_module() -> Optional[Any]:
    try:
        from src import _version_build as generated  # type: ignore[import-not-found]
    except ImportError:
        return None
    return generated


def _metadata_version() -> str:
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0-dev"


def _git_line(args: list[str]) -> Optional[str]:
    if getattr(sys, "frozen", False):
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        return out or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def get_application_dict() -> Dict[str, Any]:
    """Return analyzer identity fields stored in bundle manifests and reports."""
    gen = _try_generated_module()
    if gen is not None:
        return {
            "name": _PACKAGE_NAME,
            "version": str(getattr(gen, "VERSION", "")),
            "git_commit": str(getattr(gen, "GIT_COMMIT_FULL", "")),
            "git_describe": str(getattr(gen, "GIT_DESCRIBE", "")),
            "build_date": str(getattr(gen, "BUILD_DATE_UTC", "")),
            "working_tree_dirty": bool(getattr(gen, "WORKING_TREE_DIRTY", False)),
        }

    describe = _git_line(["describe", "--always", "--tags", "--dirty"])
    commit = _git_line(["rev-parse", "HEAD"]) or ""
    version = _metadata_version()
    dirty = bool(describe and describe.endswith("-dirty"))
    return {
        "name": _PACKAGE_NAME,
        "version": version,
        "git_commit": commit,
        "git_describe": describe or version,
        "build_date": "",
        "working_tree_dirty": dirty,
    }


def resolve_application_for_report(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer embedded manifest ``application`` when the bundle recorded it."""
    embedded = manifest.get("application")
    if isinstance(embedded, dict) and embedded:
        return {**get_application_dict(), **embedded}
    return get_application_dict()


def format_analyzer_markdown_lines(
    application: Optional[Dict[str, Any]] = None,
) -> list[str]:
    """Return Markdown bullets for the analyzer / build (no trailing blank line)."""
    app = application or get_application_dict()
    ver = app.get("version") or "unknown"
    commit_full = str(app.get("git_commit") or "")
    short = commit_full[:12] if len(commit_full) >= 12 else (commit_full or "unknown")
    describe = str(app.get("git_describe") or "").strip()
    build_date = str(app.get("build_date") or "").strip()
    dirty = app.get("working_tree_dirty")

    lines = [
        f"- **Analyzer:** `{app.get('name', _PACKAGE_NAME)}` **version** `{ver}`",
        f"- **Git commit:** `{short}`" + (f" (`{describe}`)" if describe else ""),
    ]
    if build_date:
        lines.append(f"- **Build time (UTC):** `{build_date}`")
    if dirty:
        lines.append("- **Working tree:** dirty (uncommitted changes at build time)")
    return lines


def get_release_version() -> str:
    """PEP 440-ish release string for ``src.__version__``."""
    return str(get_application_dict().get("version") or _metadata_version())


def get_about_version_text() -> str:
    """Plain-text summary for dialogs."""
    app = get_application_dict()
    ver = app.get("version") or "unknown"
    commit = str(app.get("git_commit") or "")
    short = commit[:12] if len(commit) >= 12 else (commit or "")
    parts = [f"Version {ver}"]
    if short:
        parts.append(f"commit {short}")
    if app.get("working_tree_dirty"):
        parts.append("(dirty)")
    return " · ".join(parts)
