"""Tests for analyzer version metadata helpers."""

from src.version_info import (
    format_analyzer_markdown_lines,
    get_about_version_text,
    get_application_dict,
)


def test_get_application_dict_has_expected_keys() -> None:
    app = get_application_dict()
    assert app["name"] == "audio-analyser"
    for key in ("version", "git_commit", "git_describe", "working_tree_dirty"):
        assert key in app


def test_format_analyzer_markdown_lines_non_empty() -> None:
    lines = format_analyzer_markdown_lines()
    assert any("Analyzer" in line for line in lines)
    assert any("Git commit" in line for line in lines)


def test_get_about_version_text_includes_version_word() -> None:
    text = get_about_version_text()
    assert "Version" in text
