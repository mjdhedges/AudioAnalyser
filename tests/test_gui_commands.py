"""Tests for GUI subprocess command construction."""

from pathlib import Path

import pytest

from src.gui.commands import (
    AnalysisCommandOptions,
    RenderCommandOptions,
    analysis_output_dir,
    build_analysis_command,
    build_render_command,
    render_output_dir,
    resolve_render_results_path,
)


def test_build_analysis_command_includes_gui_options() -> None:
    """Analysis command should expose GUI-selected runtime controls."""
    command = build_analysis_command(
        AnalysisCommandOptions(
            input_path=Path("tracks/Music"),
            project_dir=Path("Project"),
            batch_workers=3,
            max_memory_gb=6.5,
        ),
        python_executable="python",
    )

    assert command == [
        "python",
        "-m",
        "src.main",
        "--input",
        "tracks\\Music" if "\\" in str(Path("tracks/Music")) else "tracks/Music",
        "--output-dir",
        str(Path("Project") / "analysis"),
        "--batch-workers",
        "3",
        "--max-memory-gb",
        "6.5",
    ]


def test_build_analysis_command_rejects_invalid_options() -> None:
    """Invalid GUI values should fail before a subprocess is launched."""
    with pytest.raises(ValueError, match="batch_workers"):
        build_analysis_command(
            AnalysisCommandOptions(
                input_path=Path("track.wav"),
                project_dir=Path("Project"),
                batch_workers=0,
                max_memory_gb=4.0,
            )
        )

    with pytest.raises(ValueError, match="max_memory_gb"):
        build_analysis_command(
            AnalysisCommandOptions(
                input_path=Path("track.wav"),
                project_dir=Path("Project"),
                batch_workers=1,
                max_memory_gb=0.0,
            )
        )


def test_build_render_command_adds_reports_flag() -> None:
    """Render command should optionally request Markdown reports."""
    command = build_render_command(
        RenderCommandOptions(
            results_dir=Path("analysis"),
            output_dir=Path("rendered"),
            reports=True,
        ),
        python_executable="python",
    )

    assert command == [
        "python",
        "-m",
        "src.render",
        "--results",
        "analysis",
        "--output-dir",
        "rendered",
        "--reports",
    ]


def test_resolve_render_results_path_uses_single_file_bundle(tmp_path) -> None:
    """Single-file GUI runs should render only the just-created bundle."""
    input_path = tmp_path / "Sinewave 30sec.wav"
    input_path.write_text("placeholder", encoding="utf-8")

    assert resolve_render_results_path(input_path, Path("analysis")) == (
        Path("analysis") / "Sinewave 30sec.aaresults"
    )


def test_resolve_render_results_path_keeps_folder_batch_output(tmp_path) -> None:
    """Folder GUI runs should render the analysis output directory."""
    input_dir = tmp_path / "Tracks"
    input_dir.mkdir()

    assert resolve_render_results_path(input_dir, Path("analysis")) == Path("analysis")


def test_project_folder_resolves_standard_output_dirs() -> None:
    """GUI project folders should contain analysis and rendered outputs."""
    project_dir = Path("Project")

    assert analysis_output_dir(project_dir) == project_dir / "analysis"
    assert render_output_dir(project_dir) == project_dir / "rendered"
