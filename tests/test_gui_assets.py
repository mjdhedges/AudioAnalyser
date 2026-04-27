"""Tests for GUI asset resolution."""

from src.gui.assets import ICON_FILENAME, app_icon_path


def test_app_icon_path_points_to_existing_icon() -> None:
    """The GUI should be able to resolve its application icon."""
    path = app_icon_path()

    assert path.name == ICON_FILENAME
    assert path.exists()
