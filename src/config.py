"""
Configuration management module for Audio Analyser.

This module handles loading and managing configuration parameters from TOML files
with support for command-line overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import toml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for Audio Analyser."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default config.toml
        """
        self.config_path = config_path or Path("config.toml")
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from TOML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = toml.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(
                    f"Configuration file {self.config_path} not found, using defaults"
                )
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "analysis": {
                "chunk_duration_seconds": 2.0,
                "sample_rate": 44100,
                "tracks_dir": "Tracks",
                "output_dir": "analysis",
                "peak_hold_tau_seconds": 2.0,
                "time_domain_crest_factor_mode": "slow",
                "time_domain_slow_window_seconds": 1.0,
                "time_domain_slow_step_seconds": 1.0,
                "time_domain_slow_rms_tau_seconds": 1.0,
                "octave_center_frequencies": [
                    8.0,
                    16.0,
                    31.25,
                    62.5,
                    125.0,
                    250.0,
                    500.0,
                    1000.0,
                    2000.0,
                    4000.0,
                    8000.0,
                    16000.0,
                ],
                "octave_filter_mode": "auto",
                "octave_fft_block_duration_seconds": 30.0,
                "octave_max_memory_gb": 4.0,
                "octave_include_low_residual_band": True,
                "octave_include_high_residual_band": True,
                "octave_low_residual_center_hz": 4.0,
                "octave_high_residual_center_hz": 32000.0,
            },
            "plotting": {
                "octave_spectrum_figsize": [12, 8],
                "crest_factor_figsize": [12, 8],
                "histogram_figsize": [15, 10],
                "crest_factor_time_figsize": [14, 10],
                "octave_crest_factor_time_figsize": [14, 12],
                "octave_spectrum_xlim": [3, 40000],
                "octave_spectrum_ylim": [-60, 3],
                "crest_factor_xlim": [3, 40000],
                "crest_factor_ylim_min": 0,
                "crest_factor_ylim_max": 30,
                "octave_crest_factor_time_ylim": [0, 40],
                "histogram_bins": 51,
                "histogram_range": [-1, 1],
                "log_histogram_bins": 51,
                "log_histogram_noise_floor_db": -60,
                "log_histogram_max_db": 0,
                "log_histogram_max_bin_size_db": 3.0,
                "dpi": 300,
                "render_dpi": 150,
                "batch_dpi": 150,
                "high_quality_dpi": 300,
                "bbox_inches": "tight",
            },
            "advanced_stats": {
                "hot_peaks_threshold_db": -1.0,
                "clip_events_threshold_db": -0.1,
                "peak_saturation_threshold_db": -3.0,
                "transient_threshold_db": 3.0,
                "peak_distribution_ranges_min": [-0.1, -1.0, -3.0, -6.0, -12.0, -60.0],
                "peak_distribution_ranges_max": [0.0, -0.1, -1.0, -3.0, -6.0, -12.0],
                "peak_distribution_ranges_labels": [
                    "clipping",
                    "hot",
                    "loud",
                    "moderate",
                    "quiet",
                    "very_quiet",
                ],
                "bass_frequencies": [8.0, 16.0, 31.25, 62.5, 125.0],
                "mid_frequencies": [250.0, 500.0, 1000.0, 2000.0],
                "treble_frequencies": [4000.0, 8000.0, 16000.0],
            },
            "file_handling": {
                "supported_formats": ["wav", "flac", "mp3", "aiff", "m4a"],
                "create_subdirectories": True,
                "use_track_name_for_output": True,
            },
            "export": {
                "generate_analysis_bundle": True,
                "generate_legacy_csv": False,
                "include_track_metadata": True,
                "include_advanced_statistics": True,
                "include_octave_band_analysis": True,
                "include_time_domain_analysis": True,
                "include_time_domain_summary": True,
                "include_histogram_data": True,
                "include_extreme_chunks_analysis": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "include_debug_info": False,
            },
            "performance": {
                "max_chunk_size_mb": 100,
                "enable_memory_optimization": True,
                "skip_octave_cf_time": False,
                "export_octave_cf_time_data": False,
            },
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'analysis.chunk_duration_seconds')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_optional_positive_int(self, key_path: str) -> Optional[int]:
        """Parse a positive worker/thread count, or None if unset / empty / non-positive (auto-detect)."""
        raw = self.get(key_path, None)
        if raw is None:
            return None
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped == "":
                return None
            try:
                raw = int(stripped)
            except ValueError:
                logger.warning("Ignoring invalid integer for %s: %r", key_path, raw)
                return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid integer for %s: %r", key_path, raw)
            return None
        if value <= 0:
            return None
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split(".")
        config = self._config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the final value
        config[keys[-1]] = value

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration section."""
        return self.get("analysis", {})

    def get_plotting_config(self) -> Dict[str, Any]:
        """Get plotting configuration section."""
        return self.get("plotting", {})

    def get_advanced_stats_config(self) -> Dict[str, Any]:
        """Get advanced statistics configuration section."""
        return self.get("advanced_stats", {})

    def get_file_handling_config(self) -> Dict[str, Any]:
        """Get file handling configuration section."""
        return self.get("file_handling", {})

    def get_export_config(self) -> Dict[str, Any]:
        """Get export configuration section."""
        return self.get("export", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.get("logging", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration section."""
        return self.get("performance", {})

    def get_octave_center_frequencies(self) -> List[float]:
        """Return active 1/1-octave band center frequencies (Hz) for analysis.

        Uses ``analysis.octave_center_frequencies`` from the loaded TOML when
        non-empty; otherwise falls back to ``OctaveBandFilter.OCTAVE_CENTER_FREQUENCIES``
        (includes an 8 Hz band below the IEC 16 Hz series for sub-bass / LFE work).

        Returns:
            Positive center frequencies in ascending order (as configured).
        """
        from src.octave_filter import OctaveBandFilter

        raw = self.get("analysis.octave_center_frequencies")
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            # Sorted ascending and de-duplicated: FFT band weights require a stable
            # center-frequency order and labels must match column order.
            return sorted({float(x) for x in raw})
        return list(OctaveBandFilter.OCTAVE_CENTER_FREQUENCIES)

    def override_from_args(self, **kwargs) -> None:
        """Override configuration values from command line arguments.

        Args:
            **kwargs: Command line arguments to override configuration
        """
        # Map command line argument names to configuration paths
        arg_mapping = {
            "chunk_duration": "analysis.chunk_duration_seconds",
            "sample_rate": "analysis.sample_rate",
            "dpi": "plotting.dpi",
            "log_level": "logging.level",
            "test_start_time": "analysis.test_start_time",
            "test_duration": "analysis.test_duration",
            "peak_hold_tau": "analysis.peak_hold_tau_seconds",
        }

        for arg_name, config_path in arg_mapping.items():
            if arg_name in kwargs and kwargs[arg_name] is not None:
                self.set(config_path, kwargs[arg_name])
                logger.info(f"Overrode {config_path} with {kwargs[arg_name]}")

    def save_config(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration. If None, uses original config path
        """
        save_path = path or self.config_path
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                toml.dump(self._config, f)
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Global configuration instance
config = Config()
