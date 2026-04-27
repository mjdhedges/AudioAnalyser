"""Command builders used by the desktop GUI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AnalysisCommandOptions:
    """User-selected analysis options from the GUI."""

    input_path: Path
    project_dir: Path
    batch_workers: int
    max_memory_gb: float


@dataclass(frozen=True)
class RenderCommandOptions:
    """User-selected render options from the GUI."""

    results_dir: Path
    output_dir: Path
    reports: bool = True


def analysis_output_dir(project_dir: Path) -> Path:
    """Return the analysis output folder inside a GUI project folder."""
    return project_dir / "analysis"


def render_output_dir(project_dir: Path) -> Path:
    """Return the rendered output folder inside a GUI project folder."""
    return project_dir / "rendered"


def build_analysis_command(
    options: AnalysisCommandOptions,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Build the analysis subprocess command.

    Args:
        options: Analysis runtime options selected in the GUI.
        python_executable: Python interpreter to run. Defaults to the current
            interpreter so development and virtual-environment runs behave
            consistently.

    Returns:
        Command list suitable for ``QProcess.start(program, arguments)``.

    Raises:
        ValueError: If worker or memory values are invalid.
    """
    if options.batch_workers < 1:
        raise ValueError("batch_workers must be >= 1")
    if options.max_memory_gb <= 0:
        raise ValueError("max_memory_gb must be > 0")

    executable = python_executable or sys.executable
    return [
        executable,
        "-m",
        "src.main",
        "--input",
        str(options.input_path),
        "--output-dir",
        str(analysis_output_dir(options.project_dir)),
        "--batch-workers",
        str(options.batch_workers),
        "--max-memory-gb",
        f"{options.max_memory_gb:g}",
    ]


def build_render_command(
    options: RenderCommandOptions,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Build the render subprocess command.

    Args:
        options: Render options selected in the GUI.
        python_executable: Python interpreter to run.

    Returns:
        Command list suitable for ``QProcess.start(program, arguments)``.
    """
    executable = python_executable or sys.executable
    command = [
        executable,
        "-m",
        "src.render",
        "--results",
        str(options.results_dir),
        "--output-dir",
        str(options.output_dir),
    ]
    if options.reports:
        command.append("--reports")
    return command


def resolve_render_results_path(input_path: Path, analysis_output_dir: Path) -> Path:
    """Resolve the results path that should be rendered after analysis.

    Single-file analysis writes one bundle directly under ``analysis_output_dir``.
    Folder analysis may produce multiple bundles, so the renderer should receive
    the output directory.

    Args:
        input_path: File or folder selected by the user.
        analysis_output_dir: Analysis output directory selected by the user.

    Returns:
        Bundle path for single-file input, otherwise the analysis output folder.
    """
    if input_path.is_file() or (input_path.suffix and not input_path.is_dir()):
        return analysis_output_dir / f"{input_path.stem}.aaresults"
    return analysis_output_dir
