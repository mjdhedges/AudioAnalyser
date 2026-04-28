"""Command builders used by the desktop GUI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.gui.cli import ANALYSIS_CLI_ARG, RENDER_CLI_ARG

FROZEN_CLI_EXE = "AudioAnalyserCli.exe"


@dataclass(frozen=True)
class AnalysisCommandOptions:
    """User-selected analysis options from the GUI."""

    input_path: Path
    project_dir: Path
    batch_workers: int
    max_memory_gb: float
    progress_json: bool = True


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
    command = _module_command(
        executable=executable,
        module="src.main",
        frozen_arg=ANALYSIS_CLI_ARG,
    )
    config_path = _frozen_config_path(executable)
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    command.extend(
        [
            "--input",
            str(options.input_path),
            "--output-dir",
            str(analysis_output_dir(options.project_dir)),
            "--batch-workers",
            str(options.batch_workers),
            "--max-memory-gb",
            f"{options.max_memory_gb:g}",
        ]
    )
    if options.progress_json:
        command.append("--progress-json")
    return command


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
    command = _module_command(
        executable=executable,
        module="src.render",
        frozen_arg=RENDER_CLI_ARG,
    )
    config_path = _frozen_config_path(executable)
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    command.extend(
        [
            "--results",
            str(options.results_dir),
            "--output-dir",
            str(options.output_dir),
        ]
    )
    if options.reports:
        command.append("--reports")
    return command


def _module_command(executable: str, module: str, frozen_arg: str) -> List[str]:
    """Return a subprocess command prefix for development or frozen builds."""
    if getattr(sys, "frozen", False):
        executable_path = Path(executable)
        cli_executable = executable_path.with_name(FROZEN_CLI_EXE)
        if cli_executable.exists():
            return [str(cli_executable), frozen_arg]
        return [executable, frozen_arg]
    return [executable, "-m", module]


def _frozen_config_path(executable: str) -> Optional[Path]:
    """Return the packaged config path when running as a frozen application."""
    if not getattr(sys, "frozen", False):
        return None

    candidates = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root) / "config.toml")
    candidates.append(Path(executable).resolve().parent / "config.toml")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
