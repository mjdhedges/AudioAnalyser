"""Console entry point used by frozen GUI builds."""

from __future__ import annotations

import sys

from src.gui.cli import ANALYSIS_CLI_ARG, RENDER_CLI_ARG


def main() -> int:
    """Dispatch packaged subprocess calls to the analysis or render CLI."""
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

    message = f"Expected {ANALYSIS_CLI_ARG} or {RENDER_CLI_ARG}."
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
