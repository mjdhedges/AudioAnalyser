"""Desktop GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.gui.assets import app_icon
from src.gui.main_window import MainWindow


def main() -> int:
    """Run the Audio Analyser desktop GUI."""
    app = QApplication(sys.argv)
    icon = app_icon()
    app.setWindowIcon(icon)
    window = MainWindow(icon=icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
