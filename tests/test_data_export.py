"""
Tests for the data export module.
"""

import pytest
import numpy as np
import tempfile
import pandas as pd
from pathlib import Path

from src.data_export import DataExporter
from src.music_analyzer import MusicAnalyzer


class TestDataExporter:
    """Test cases for DataExporter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_exporter = DataExporter(sample_rate=44100, original_peak=1.0)

    def test_init(self):
        """Test DataExporter initialization."""
        assert self.data_exporter.sample_rate == 44100
        assert self.data_exporter.original_peak == 1.0

    def test_export_analysis_results(self):
        """Test exporting analysis results to CSV."""
        # Create test analysis results
        analyzer = MusicAnalyzer(sample_rate=44100)
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [125, 250]  # 0 is added automatically
        results = analyzer.analyze_octave_bands(octave_bank, center_frequencies)

        # Export to temporary file
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            self.data_exporter.export_analysis_results(results, tmp_path)

            # Check that file was created and has content
            assert Path(tmp_path).exists()

            # Read and verify CSV content
            df = pd.read_csv(tmp_path)
            assert len(df) > 0
            assert "frequency_hz" in df.columns
            assert "max_amplitude" in df.columns
            assert "rms" in df.columns

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_calculate_advanced_statistics(self):
        """Test calculating advanced statistics."""
        # Create test data
        sample_rate = 44100
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration))

        # Create test analysis results
        analyzer = MusicAnalyzer(sample_rate=sample_rate)
        octave_bank = np.random.randn(len(audio_data), 3)
        center_frequencies = [125, 250]
        analysis_results = analyzer.analyze_octave_bands(
            octave_bank, center_frequencies
        )

        # Create test time analysis with correct keys
        time_points = np.linspace(0, duration, 100)
        time_analysis = {
            "time_points": time_points,
            "crest_factors_db": np.random.rand(100) * 20 - 10,  # dB values
            "peak_levels_dbfs": np.random.rand(100) * 20 - 20,  # dBFS values
            "rms_levels_dbfs": np.random.rand(100) * 20 - 30,  # dBFS values
        }

        # Calculate advanced statistics
        stats = self.data_exporter.calculate_advanced_statistics(
            audio_data, analysis_results, time_analysis
        )

        # Should return dictionary with statistics
        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_export_comprehensive_results(self):
        """Test exporting comprehensive results."""
        # Create test data
        sample_rate = 44100
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration))

        # Create test analysis results
        analyzer = MusicAnalyzer(sample_rate=sample_rate)
        octave_bank = np.random.randn(len(audio_data), 3)
        center_frequencies = [125, 250]
        analysis_results = analyzer.analyze_octave_bands(
            octave_bank, center_frequencies
        )

        # Create test time analysis with correct keys (including both linear and dB versions)
        time_points = np.linspace(0, duration, 100)
        crest_factors = np.random.rand(100) * 5 + 1  # Linear crest factors
        crest_factors_db = 20 * np.log10(crest_factors)  # dB values
        peak_levels = np.random.rand(100) * 0.5 + 0.5  # Linear peak levels
        peak_levels_dbfs = 20 * np.log10(peak_levels)  # dBFS values
        rms_levels = np.random.rand(100) * 0.3 + 0.1  # Linear RMS levels
        rms_levels_dbfs = 20 * np.log10(rms_levels)  # dBFS values

        time_analysis = {
            "time_points": time_points,
            "crest_factors": crest_factors,  # Linear values
            "crest_factors_db": crest_factors_db,  # dB values
            "peak_levels": peak_levels,  # Linear values
            "peak_levels_dbfs": peak_levels_dbfs,  # dBFS values
            "rms_levels": rms_levels,  # Linear values
            "rms_levels_dbfs": rms_levels_dbfs,  # dBFS values
            "chunk_duration": duration / len(time_points),  # Duration per chunk
            "num_chunks": len(time_points),  # Number of chunks
        }

        # Create test metadata
        track_metadata = {
            "track_name": "test_track.wav",
            "track_path": "/path/to/test_track.wav",
            "content_type": "Music",
            "channel_index": 0,
            "channel_name": "Channel 1 Left",
            "total_channels": 2,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "samples": len(audio_data),
            "channels": 1,
            "original_peak": 1.0,
            "original_peak_dbfs": 0.0,
            "analysis_date": pd.Timestamp.now().isoformat(),
            "octave_filter_design": "fft_power_complementary",
            "octave_requested_processing_mode": "auto",
            "octave_effective_processing_mode": "block",
            "octave_output_storage": "disk_memmap",
            "octave_max_memory_gb": 4.0,
            "octave_fft_block_duration_seconds": 30.0,
            "octave_include_low_residual_band": True,
            "octave_include_high_residual_band": True,
            "octave_power_sum_rule": "sum(weight_band(f) ** 2) = 1.0",
        }

        # Export to temporary file
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            self.data_exporter.export_comprehensive_results(
                analysis_results,
                time_analysis,
                track_metadata,
                tmp_path,
                audio_data=audio_data,
            )

            # Check that file was created
            assert Path(tmp_path).exists()

            # Read and verify CSV content (it's a multi-section CSV, not standard format)
            with open(tmp_path, "r") as f:
                content = f.read()
                assert len(content) > 0
                assert "TRACK_METADATA" in content
                assert "OCTAVE_BAND_ANALYSIS" in content
                assert "octave_filter_design,fft_power_complementary" in content
                assert "octave_effective_processing_mode,block" in content
                assert "octave_output_storage,disk_memmap" in content
                assert "octave_max_memory_gb,4.0" in content

        finally:
            Path(tmp_path).unlink(missing_ok=True)
