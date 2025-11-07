"""
Tests for the visualization module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.visualization import PlotGenerator
from src.music_analyzer import MusicAnalyzer


class TestPlotGenerator:
    """Test cases for PlotGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plot_generator = PlotGenerator(sample_rate=44100, original_peak=1.0, dpi=300)

    def test_init(self):
        """Test PlotGenerator initialization."""
        assert self.plot_generator.sample_rate == 44100
        assert self.plot_generator.original_peak == 1.0
        assert self.plot_generator.dpi == 300

    def test_create_octave_spectrum_plot(self):
        """Test creating octave spectrum plot."""
        # Create test analysis results
        analyzer = MusicAnalyzer(sample_rate=44100)
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [125, 250]  # 0 is added automatically
        results = analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Test plot creation
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            self.plot_generator.create_octave_spectrum_plot(results, tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_histogram_plots(self):
        """Test creating histogram plots."""
        # Create test analysis results
        analyzer = MusicAnalyzer(sample_rate=44100)
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [125, 250]  # 0 is added automatically
        results = analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Test histogram creation
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.plot_generator.create_histogram_plots(
                results, tmp_dir, octave_bank=octave_bank
            )
            
            # Check that histogram file was created
            hist_file = Path(tmp_dir) / "histograms.png"
            assert hist_file.exists()

    def test_create_histogram_plots_log_db(self):
        """Test creating log dB histogram plots."""
        # Create test analysis results
        analyzer = MusicAnalyzer(sample_rate=44100)
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [125, 250]  # 0 is added automatically
        results = analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Test histogram creation with config
        from src.config import config
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.plot_generator.create_histogram_plots_log_db(
                results, tmp_dir, config=config.get_plotting_config(), octave_bank=octave_bank
            )
            
            # Check that histogram file was created
            hist_file = Path(tmp_dir) / "histograms_log_db.png"
            assert hist_file.exists()

    def test_create_crest_factor_time_plot(self):
        """Test creating crest factor time plot."""
        # Create test time analysis with correct keys
        time_points = np.linspace(0, 10, 100)
        time_analysis = {
            "time_points": time_points,
            "crest_factors_db": np.random.rand(100) * 20 - 10,  # dB values
            "peak_levels_dbfs": np.random.rand(100) * 20 - 20,  # dBFS values
            "rms_levels_dbfs": np.random.rand(100) * 20 - 30  # dBFS values
        }
        
        # Test plot creation
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            self.plot_generator.create_crest_factor_time_plot(time_analysis, tmp_path)
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_octave_crest_factor_time_plot(self):
        """Test creating octave crest factor time plot."""
        # Create test data
        octave_bank = np.random.randn(100, 3)
        time_points = np.linspace(0, 10, 100)
        time_analysis = {
            "time_points": time_points,
            "crest_factors_db": np.random.rand(100) * 20 - 10,  # dB values
            "peak_levels_dbfs": np.random.rand(100) * 20 - 20,  # dBFS values
            "rms_levels_dbfs": np.random.rand(100) * 20 - 30  # dBFS values
        }
        center_frequencies = [125, 250]
        
        # Test plot creation
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            self.plot_generator.create_octave_crest_factor_time_plot(
                octave_bank, time_analysis, center_frequencies, tmp_path
            )
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_crest_factor_plot(self):
        """Test creating crest factor plot."""
        # Create test analysis results
        analyzer = MusicAnalyzer(sample_rate=44100)
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [125, 250]  # 0 is added automatically
        results = analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Create test time analysis with correct keys
        time_points = np.linspace(0, 10, 100)
        time_analysis = {
            "time_points": time_points,
            "crest_factors_db": np.random.rand(100) * 20 - 10,  # dB values
            "peak_levels_dbfs": np.random.rand(100) * 20 - 20,  # dBFS values
            "rms_levels_dbfs": np.random.rand(100) * 20 - 30  # dBFS values
        }
        
        # Test plot creation
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            self.plot_generator.create_crest_factor_plot(
                results, tmp_path, time_analysis=time_analysis
            )
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

