"""Tests for GUI subprocess command construction."""

from pathlib import Path

import pytest

from src.gui.commands import (
    AnalysisCommandOptions,
    FROZEN_CLI_EXE,
    RenderCommandOptions,
    analysis_output_dir,
    build_analysis_command,
    build_render_command,
    render_output_dir,
    resolve_render_results_path,
)
from src.gui.cli import ANALYSIS_CLI_ARG, RENDER_CLI_ARG


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
        "--progress-json",
    ]


def test_build_analysis_command_can_render_each_completed_bundle() -> None:
    """GUI analysis can request in-process rendering for total progress tracking."""
    command = build_analysis_command(
        AnalysisCommandOptions(
            input_path=Path("tracks/Music"),
            project_dir=Path("Project"),
            batch_workers=2,
            max_memory_gb=4.0,
            render_after_analysis=True,
            render_output_dir=Path("Project") / "rendered",
            render_reports=False,
        ),
        python_executable="python",
    )

    assert "--render-output-dir" in command
    assert str(Path("Project") / "rendered") in command
    assert "--no-render-reports" in command


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


def test_build_analysis_command_uses_internal_cli_when_frozen(monkeypatch) -> None:
    """Packaged GUI should launch analysis through the same executable."""
    monkeypatch.setattr("sys.frozen", True, raising=False)

    command = build_analysis_command(
        AnalysisCommandOptions(
            input_path=Path("track.wav"),
            project_dir=Path("Project"),
            batch_workers=1,
            max_memory_gb=4.0,
        ),
        python_executable="AudioAnalyser.exe",
    )

    assert command[:2] == ["AudioAnalyser.exe", ANALYSIS_CLI_ARG]


def test_build_analysis_command_uses_cli_companion_when_available(
    monkeypatch, tmp_path
) -> None:
    """Frozen GUI should prefer the console companion executable."""
    monkeypatch.setattr("sys.frozen", True, raising=False)
    gui_executable = tmp_path / "AudioAnalyser.exe"
    cli_executable = tmp_path / FROZEN_CLI_EXE
    gui_executable.touch()
    cli_executable.touch()

    command = build_analysis_command(
        AnalysisCommandOptions(
            input_path=Path("track.wav"),
            project_dir=Path("Project"),
            batch_workers=1,
            max_memory_gb=4.0,
        ),
        python_executable=str(gui_executable),
    )

    assert command[:2] == [str(cli_executable), ANALYSIS_CLI_ARG]


def test_build_analysis_command_passes_packaged_config(monkeypatch, tmp_path) -> None:
    """Frozen analysis runs should explicitly pass the packaged config."""
    monkeypatch.setattr("sys.frozen", True, raising=False)
    gui_executable = tmp_path / "AudioAnalyser.exe"
    config_path = tmp_path / "config.toml"
    gui_executable.touch()
    config_path.write_text("[analysis]\n", encoding="utf-8")

    command = build_analysis_command(
        AnalysisCommandOptions(
            input_path=Path("track.wav"),
            project_dir=Path("Project"),
            batch_workers=1,
            max_memory_gb=4.0,
        ),
        python_executable=str(gui_executable),
    )

    assert command[2:4] == ["--config", str(config_path)]


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


def test_build_render_command_can_skip_reports() -> None:
    """Render command should allow plot-only rendering."""
    command = build_render_command(
        RenderCommandOptions(
            results_dir=Path("analysis"),
            output_dir=Path("rendered"),
            reports=False,
        ),
        python_executable="python",
    )

    assert "--reports" not in command


def test_build_render_command_uses_internal_cli_when_frozen(monkeypatch) -> None:
    """Packaged GUI should launch rendering through the same executable."""
    monkeypatch.setattr("sys.frozen", True, raising=False)

    command = build_render_command(
        RenderCommandOptions(
            results_dir=Path("analysis"),
            output_dir=Path("rendered"),
        ),
        python_executable="AudioAnalyser.exe",
    )

    assert command[:2] == ["AudioAnalyser.exe", RENDER_CLI_ARG]


def test_build_render_command_passes_packaged_config(monkeypatch, tmp_path) -> None:
    """Frozen render runs should explicitly pass the packaged config."""
    monkeypatch.setattr("sys.frozen", True, raising=False)
    gui_executable = tmp_path / "AudioAnalyser.exe"
    config_path = tmp_path / "config.toml"
    gui_executable.touch()
    config_path.write_text("[plotting]\n", encoding="utf-8")

    command = build_render_command(
        RenderCommandOptions(
            results_dir=Path("analysis"),
            output_dir=Path("rendered"),
        ),
        python_executable=str(gui_executable),
    )

    assert command[2:4] == ["--config", str(config_path)]


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
