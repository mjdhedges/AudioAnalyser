"""Tests for GUI About dialog content."""

from src.gui.about import AUTHOR, GITHUB_URL, LICENSE, about_html


def test_about_html_includes_project_metadata() -> None:
    """About text should include project, author, license, and citation details."""
    html = about_html()

    assert GITHUB_URL in html
    assert AUTHOR in html
    assert LICENSE in html
    assert "Citation" in html
