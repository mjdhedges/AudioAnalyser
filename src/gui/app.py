"""Desktop GUI entry point."""

from __future__ import annotations

import sys
from multiprocessing import freeze_support

from PySide6.QtWidgets import QApplication

from src.gui.assets import app_icon
from src.gui.cli import ANALYSIS_CLI_ARG, RENDER_CLI_ARG
from src.gui.main_window import MainWindow


def main() -> int:
    """Run the Audio Analyser desktop GUI."""
    freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == ANALYSIS_CLI_ARG:
        from src.main import main as analysis_main

        return analysis_main.main(
            args=sys.argv[2:],
            prog_name="audio-analyser-analysis",
            standalone_mode=True,
        )
    if len(sys.argv) > 1 and sys.argv[1] == RENDER_CLI_ARG:
        from src.render import main as render_main

        return render_main.main(
            args=sys.argv[2:],
            prog_name="audio-analyser-render",
            standalone_mode=True,
        )

    app = QApplication(sys.argv)
    icon = app_icon()
    app.setWindowIcon(icon)
    window = MainWindow(icon=icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
