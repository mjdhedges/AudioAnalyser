"""PDF export helpers for Markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import markdown
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication, QPageSize, QPdfWriter, QTextDocument

_PDF_QT_APP: Optional[QGuiApplication] = None


def markdown_report_to_pdf(
    markdown_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Convert a Markdown report into a PDF file.

    Args:
        markdown_path: Path to the Markdown report.
        output_path: Optional PDF output path. Defaults to the same stem with a
            `.pdf` suffix.

    Returns:
        Path to the generated PDF file.
    """
    pdf_path = output_path or markdown_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_qgui_application()

    markdown_text = markdown_path.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        markdown_text,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )
    html = _report_html_document(body_html)

    writer = QPdfWriter(str(pdf_path))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setResolution(96)
    writer.setTitle(markdown_path.stem)

    document = QTextDocument()
    document.setBaseUrl(QUrl.fromLocalFile(str(markdown_path.parent.resolve()) + "/"))
    document.setHtml(html)
    document.print_(writer)

    return pdf_path


def _ensure_qgui_application() -> QGuiApplication:
    """Return an existing Qt app or create one for headless PDF rendering."""
    global _PDF_QT_APP
    app = QGuiApplication.instance()
    if app is None:
        _PDF_QT_APP = QGuiApplication(["audio-analyser-pdf"])
        app = _PDF_QT_APP
    return app


def _report_html_document(body_html: str) -> str:
    """Wrap report body HTML with print-friendly styling."""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      font-size: 10pt;
      line-height: 1.35;
      color: #202124;
    }}
    h1, h2, h3 {{
      page-break-after: avoid;
    }}
    h1 {{
      font-size: 20pt;
    }}
    h2 {{
      font-size: 15pt;
      margin-top: 18pt;
    }}
    h3 {{
      font-size: 12pt;
      margin-top: 14pt;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 8pt 0;
    }}
    th, td {{
      border: 1px solid #d0d7de;
      padding: 4pt;
      vertical-align: top;
    }}
    th {{
      background: #f6f8fa;
      font-weight: bold;
    }}
    code {{
      font-family: Consolas, monospace;
      font-size: 9pt;
    }}
    img {{
      max-width: 100%;
      height: auto;
      margin: 6pt 0 12pt 0;
    }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""
