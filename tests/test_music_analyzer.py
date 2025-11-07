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
        center_frequencies = [125, 250, 500, 1000]  # Note: 0 is added automatically for full spectrum
        
        # Perform analysis
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Check results structure
        # Note: band_data was removed to save memory - only statistics are returned
        assert "statistics" in results
        assert "center_frequencies" in results
        
        # Check that all bands are analyzed (including full spectrum)
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
        center_frequencies = [125, 250]  # 0 is added automatically
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Export to temporary file
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            self.analyzer.export_analysis_results(results, tmp_path)
            
            # Check that file was created and has content
            assert Path(tmp_path).exists()
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert 'frequency_hz' in content or 'Frequency' in content
                assert 'max_amplitude' in content or 'Max Amplitude' in content
                assert 'rms' in content or 'RMS' in content
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_octave_spectrum_plot(self):
        """Test creating octave spectrum plot."""
        # Create test analysis results
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [125, 250]  # 0 is added automatically
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Test plot creation (without showing)
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # This should not raise an exception
            self.analyzer.create_octave_spectrum_plot(results, tmp_path)
            assert Path(tmp_path).exists()
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_create_histogram_plots(self):
        """Test creating histogram plots."""
        # Create test analysis results
        octave_bank = np.random.randn(100, 3)
        center_frequencies = [125, 250]  # 0 is added automatically
        results = self.analyzer.analyze_octave_bands(octave_bank, center_frequencies)
        
        # Test histogram creation (without showing)
        # Note: create_histogram_plots requires octave_bank parameter now
        with tempfile.TemporaryDirectory() as tmp_dir:
            # This should not raise an exception
            # Pass octave_bank as well since band_data was removed
            self.analyzer.create_histogram_plots(results, tmp_dir, octave_bank=octave_bank)
            
            # Check that histogram file was created
            hist_file = Path(tmp_dir) / "histograms.png"
            assert hist_file.exists()
