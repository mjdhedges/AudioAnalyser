"""
Tests for the music analyzer module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.music_analyzer import MusicAnalyzer


class TestMusicAnalyzer:
    """Test cases for MusicAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MusicAnalyzer(sample_rate=44100)

    def test_init(self):
        """Test MusicAnalyzer initialization."""
        assert self.analyzer.sample_rate == 44100

    def test_analyze_octave_bands(self):
        """Test octave band analysis."""
        # Create test octave bank
        num_samples = 1000
        num_bands = 5
        octave_bank = np.random.randn(num_samples, num_bands)
        center_frequencies = [0, 125, 250, 500, 1000]
        
        # Perform analysis
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Check results structure
        assert "band_data" in results
        assert "statistics" in results
        assert "center_frequencies" in results
        
        # Check that all bands are analyzed
        assert len(results["band_data"]) == num_bands
        assert len(results["statistics"]) == num_bands
        
        # Check statistics for one band
        band_stats = results["statistics"]["Full Spectrum"]
        assert "max_amplitude" in band_stats
        assert "max_amplitude_db" in band_stats
        assert "rms" in band_stats
        assert "rms_db" in band_stats
        assert "dynamic_range" in band_stats
        assert "dynamic_range_db" in band_stats
        assert "mean" in band_stats
        assert "std" in band_stats
        assert "percentiles" in band_stats

    def test_calculate_band_statistics(self):
        """Test band statistics calculation."""
        # Create test signal
        test_signal = np.array([0.1, -0.2, 0.3, -0.1, 0.4])
        
        # Calculate statistics
        stats = self.analyzer._calculate_band_statistics(test_signal)
        
        # Check calculated values
        assert stats["max_amplitude"] == 0.4
        assert stats["max_amplitude_db"] == 20 * np.log10(0.4)
        
        # Check RMS calculation
        expected_rms = np.sqrt(np.mean(test_signal**2))
        assert abs(stats["rms"] - expected_rms) < 1e-10
        
        # Check percentiles
        assert "p10" in stats["percentiles"]
        assert "p50" in stats["percentiles"]
        assert "p99" in stats["percentiles"]

    def test_export_analysis_results(self):
        """Test exporting analysis results to CSV."""
        # Create test analysis results
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [0, 125, 250]
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            try:
                self.analyzer.export_analysis_results(results, tmp_file.name)
                
                # Check that file was created and has content
                assert Path(tmp_file.name).exists()
                with open(tmp_file.name, 'r') as f:
                    content = f.read()
                    assert 'frequency_hz' in content
                    assert 'max_amplitude' in content
                    assert 'rms' in content
                    
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_create_octave_spectrum_plot(self):
        """Test creating octave spectrum plot."""
        # Create test analysis results
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [0, 125, 250]
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Test plot creation (without showing)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                # This should not raise an exception
                self.analyzer.create_octave_spectrum_plot(results, tmp_file.name)
                assert Path(tmp_file.name).exists()
                
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)

    def test_create_histogram_plots(self):
        """Test creating histogram plots."""
        # Create test analysis results
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [0, 125, 250]
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Test histogram creation (without showing)
        with tempfile.TemporaryDirectory() as tmp_dir:
            # This should not raise an exception
            self.analyzer.create_histogram_plots(results, tmp_dir)
            
            # Check that histogram file was created
            hist_file = Path(tmp_dir) / "histograms.png"
            assert hist_file.exists()
