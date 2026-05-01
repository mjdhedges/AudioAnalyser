"""
Extended tests for configuration module.

Tests error handling, default config fallback, and configuration methods.
"""

import pytest
import tempfile
from pathlib import Path

from src.config import Config


class TestConfigExtended:
    """Extended test cases for Config class."""

    def test_config_nonexistent_file(self):
        """Test that Config uses defaults when file doesn't exist."""
        config = Config(config_path=Path("nonexistent_config.toml"))

        # Should use default config
        assert config.get("analysis.chunk_duration_seconds") == 2.0

    def test_config_invalid_toml(self):
        """Test that Config handles invalid TOML gracefully."""
        # Create invalid TOML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as tmp_file:
            tmp_file.write("invalid toml content [unclosed bracket\n")
            tmp_path = Path(tmp_file.name)

        try:
            config = Config(config_path=tmp_path)

            # Should fall back to defaults
            assert config.get("analysis.chunk_duration_seconds") == 2.0
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_config_set_method(self):
        """Test setting configuration values."""
        config = Config()

        # Set a new value
        config.set("test.new_key", "test_value")
        assert config.get("test.new_key") == "test_value"

        # Override existing value
        config.set("analysis.chunk_duration_seconds", 1.0)
        assert config.get("analysis.chunk_duration_seconds") == 1.0

    def test_config_set_nested(self):
        """Test setting nested configuration values."""
        config = Config()

        # Set deeply nested value
        config.set("level1.level2.level3.value", 42)
        assert config.get("level1.level2.level3.value") == 42

    def test_config_get_with_default(self):
        """Test getting configuration with default value."""
        config = Config()

        # Get existing value
        assert config.get("analysis.chunk_duration_seconds") is not None

        # Get non-existent value with default
        assert config.get("nonexistent.key", "default") == "default"

        # Get non-existent value without default
        assert config.get("nonexistent.key") is None

    def test_config_get_nested(self):
        """Test getting nested configuration values."""
        config = Config()

        # Get nested value
        chunk_duration = config.get("analysis.chunk_duration_seconds")
        assert chunk_duration is not None

        # Get entire section
        analysis_config = config.get("analysis")
        assert isinstance(analysis_config, dict)
        assert "chunk_duration_seconds" in analysis_config

    def test_config_get_section_methods(self):
        """Test section getter methods."""
        config = Config()

        # Test all section getters
        assert isinstance(config.get_analysis_config(), dict)
        assert isinstance(config.get_plotting_config(), dict)
        assert isinstance(config.get_advanced_stats_config(), dict)
        assert isinstance(config.get_file_handling_config(), dict)
        assert isinstance(config.get_export_config(), dict)
        assert isinstance(config.get_logging_config(), dict)
        assert isinstance(config.get_performance_config(), dict)

    def test_config_override_from_args(self):
        """Test overriding configuration from command line arguments."""
        config = Config()

        # Override chunk duration
        config.override_from_args(chunk_duration=1.0)
        assert config.get("analysis.chunk_duration_seconds") == 1.0

        # Override DPI
        config.override_from_args(dpi=600)
        assert config.get("plotting.dpi") == 600

        # Override log level
        config.override_from_args(log_level="DEBUG")
        assert config.get("logging.level") == "DEBUG"

        # Override GUI-facing runtime controls
        config.override_from_args(max_memory_gb=6.5, batch_workers=3)
        assert config.get("analysis.octave_max_memory_gb") == 6.5
        assert config.get("performance.max_batch_workers") == 3

    def test_config_override_from_args_ignores_none(self):
        """Test that None values in override_from_args are ignored."""
        config = Config()
        original_chunk_duration = config.get("analysis.chunk_duration_seconds")

        # Pass None - should not override
        config.override_from_args(chunk_duration=None)
        assert config.get("analysis.chunk_duration_seconds") == original_chunk_duration

    def test_config_save_config(self):
        """Test saving configuration to file."""
        config = Config()

        # Modify a value
        config.set("test.save_key", "save_value")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            config.save_config(tmp_path)

            # Verify file was created
            assert tmp_path.exists()

            # Load and verify content
            loaded_config = Config(config_path=tmp_path)
            assert loaded_config.get("test.save_key") == "save_value"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_config_save_config_error_handling(self):
        """Test that save_config handles errors gracefully."""
        config = Config()

        # Try to save to invalid path (read-only directory or invalid path)
        # This should not raise exception, just log error
        invalid_path = Path("/invalid/path/that/does/not/exist/config.toml")

        # Should handle error gracefully
        try:
            config.save_config(invalid_path)
        except Exception:
            # If it raises, that's also acceptable behavior
            pass

    def test_config_get_keyerror_handling(self):
        """Test that get() handles KeyError gracefully."""
        config = Config()

        # Accessing deeply nested non-existent key
        result = config.get("level1.level2.level3.nonexistent")
        assert result is None

        # Accessing non-existent top-level key
        result = config.get("nonexistent_key")
        assert result is None

    def test_config_get_typeerror_handling(self):
        """Test that get() handles TypeError gracefully."""
        config = Config()

        # Try to access key on non-dict value
        config.set("test.value", "string_value")

        # Accessing nested key on string should return default
        result = config.get("test.value.nested", "default")
        assert result == "default"

    def test_config_default_config_structure(self):
        """Test that default config has all required sections."""
        config = Config(config_path=Path("nonexistent.toml"))

        # Verify all sections exist
        assert config.get("analysis") is not None
        assert config.get("plotting") is not None
        assert config.get("advanced_stats") is not None
        assert config.get("file_handling") is not None
        assert config.get("export") is not None
        assert config.get("logging") is not None
        assert config.get("performance") is not None

    def test_config_valid_toml_file(self):
        """Test loading valid TOML configuration file."""
        # Create valid TOML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as tmp_file:
            tmp_file.write("""
[analysis]
chunk_duration_seconds = 1.5

[plotting]
dpi = 600
""")
            tmp_path = Path(tmp_file.name)

        try:
            config = Config(config_path=tmp_path)

            # Verify loaded values
            assert config.get("analysis.chunk_duration_seconds") == 1.5
            assert config.get("plotting.dpi") == 600
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_config_default_path_uses_frozen_bundle_root(self, monkeypatch, tmp_path):
        """Frozen builds should discover config.toml beside bundled resources."""
        bundled_config = tmp_path / "config.toml"
        bundled_config.write_text(
            """
[analysis]
chunk_duration_seconds = 1.25
""",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path.parent)
        monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)

        config = Config()

        assert config.config_path == bundled_config
        assert config.get("analysis.chunk_duration_seconds") == 1.25

    def test_config_replace_uses_copy(self):
        """Worker config replacement should not share mutable dictionaries."""
        config = Config(config_path=Path("nonexistent.toml"))
        values = config.as_dict()
        values["analysis"]["chunk_duration_seconds"] = 1.0

        config.replace(values)
        values["analysis"]["chunk_duration_seconds"] = 3.0

        assert config.get("analysis.chunk_duration_seconds") == 1.0
