"""Tests for report generation helper prose."""

from __future__ import annotations

from PySide6.QtGui import QColor, QImage

from src.report_generator import (
    REPORT_IMAGE_MAX_WIDTH_PX,
    _copy_report_image,
    _frequency_report_label,
    _octave_processing_sentence,
)


def test_octave_processing_sentence_reports_effective_auto_block_mode() -> None:
    """Report prose should state how configurable octave processing ran."""
    metadata = {
        "octave_filter_design": "fft_power_complementary",
        "octave_requested_processing_mode": "auto",
        "octave_effective_processing_mode": "block",
        "octave_output_storage": "disk_memmap",
        "octave_max_memory_gb": "4.0",
        "octave_fft_block_duration_seconds": "30.0",
    }

    sentence = _octave_processing_sentence(metadata)

    assert "FFT power-complementary" in sentence
    assert "`block` mode" in sentence
    assert "requested `auto`" in sentence
    assert "30-second FFT blocks" in sentence
    assert "`disk_memmap` octave-bank storage" in sentence
    assert "4.0 GB per-track memory estimate" in sentence


def test_octave_processing_sentence_handles_missing_metadata() -> None:
    """Report prose should be explicit when metadata is unavailable."""
    sentence = _octave_processing_sentence({})

    assert "metadata was not found" in sentence


def test_frequency_report_label_preserves_fractional_octave_centers() -> None:
    """Report headings should not round fractional octave labels."""
    assert _frequency_report_label(31.25) == "31.25 Hz"
    assert _frequency_report_label(62.5) == "62.5 Hz"
    assert _frequency_report_label(1000.0) == "1000 Hz"


def test_copy_report_image_downsamples_for_pdf_embedding(tmp_path) -> None:
    """Report image copies should be compact enough for PDF embedding."""
    source_path = tmp_path / "source.png"
    image = QImage(2400, 1200, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    assert image.save(str(source_path), "PNG")
    stale_png_path = tmp_path / "report" / "images" / "wide_plot.png"
    stale_png_path.parent.mkdir(parents=True)
    assert image.save(str(stale_png_path), "PNG")

    rel_path = _copy_report_image(source_path, tmp_path / "report", "wide_plot.png")
    copied_path = tmp_path / "report" / rel_path
    copied_image = QImage(str(copied_path))

    assert copied_path.suffix == ".jpg"
    assert copied_image.width() == REPORT_IMAGE_MAX_WIDTH_PX
    assert not stale_png_path.exists()
