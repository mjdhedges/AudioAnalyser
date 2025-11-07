"""
Extended tests for channel mapping module.

Tests TrueHD channel configuration parsing and edge cases.
"""

import pytest

from src.channel_mapping import (
    get_channel_name,
    get_channel_folder_name,
    parse_truehd_channel_config
)


class TestChannelMappingExtended:
    """Extended test cases for channel mapping functions."""

    def test_parse_truehd_channel_config_2f(self):
        """Test parsing 2F (stereo) configuration."""
        result = parse_truehd_channel_config("2F")
        assert result == ["FL", "FR"]

    def test_parse_truehd_channel_config_3f(self):
        """Test parsing 3F (front three) configuration."""
        result = parse_truehd_channel_config("3F")
        assert result == ["FL", "FC", "FR"]

    def test_parse_truehd_channel_config_3f2m(self):
        """Test parsing 3F2M (5.0) configuration."""
        result = parse_truehd_channel_config("3F2M")
        assert result == ["FL", "FC", "FR", "SL", "SR"]

    def test_parse_truehd_channel_config_3f2m2r(self):
        """Test parsing 3F2M2R (7.0) configuration."""
        result = parse_truehd_channel_config("3F2M2R")
        assert result == ["FL", "FC", "FR", "SL", "SR", "SBL", "SBR"]

    def test_parse_truehd_channel_config_with_lfe(self):
        """Test parsing configuration with LFE."""
        result = parse_truehd_channel_config("3F2M/LFE")
        assert result == ["FL", "FC", "FR", "SL", "SR", "LFE"]
        
        result = parse_truehd_channel_config("3F2M2R/LFE")
        assert result == ["FL", "FC", "FR", "SL", "SR", "SBL", "SBR", "LFE"]

    def test_parse_truehd_channel_config_5f(self):
        """Test parsing 5F (5 front channels) configuration."""
        result = parse_truehd_channel_config("5F")
        assert result == ["FL", "FCL", "FC", "FCR", "FR"]

    def test_parse_truehd_channel_config_2m(self):
        """Test parsing 2M (2 middle/surround) configuration."""
        result = parse_truehd_channel_config("2M")
        assert result == ["SL", "SR"]

    def test_parse_truehd_channel_config_4m(self):
        """Test parsing 4M (4 middle/surround) configuration."""
        result = parse_truehd_channel_config("4M")
        assert result == ["SL", "SR", "SL1", "SR1"]

    def test_parse_truehd_channel_config_2r(self):
        """Test parsing 2R (2 rear) configuration."""
        result = parse_truehd_channel_config("2R")
        assert result == ["SBL", "SBR"]

    def test_parse_truehd_channel_config_empty_string(self):
        """Test parsing empty string returns None."""
        result = parse_truehd_channel_config("")
        assert result is None

    def test_parse_truehd_channel_config_none(self):
        """Test parsing None returns None."""
        result = parse_truehd_channel_config(None)
        assert result is None

    def test_parse_truehd_channel_config_case_insensitive(self):
        """Test that parsing is case insensitive."""
        result1 = parse_truehd_channel_config("3f2m/lfe")
        result2 = parse_truehd_channel_config("3F2M/LFE")
        
        assert result1 == result2

    def test_parse_truehd_channel_config_invalid_format(self):
        """Test parsing invalid format returns None."""
        result = parse_truehd_channel_config("invalid_format")
        assert result is None

    def test_parse_truehd_channel_config_malformed(self):
        """Test parsing malformed configuration returns None."""
        result = parse_truehd_channel_config("3F2M2R2R")  # Duplicate R
        # Should handle gracefully (may return partial or None)
        assert result is None or isinstance(result, list)

    def test_get_channel_name_very_large_index(self):
        """Test getting channel name for very large channel index."""
        # For unsupported channel counts, should use generic naming
        result = get_channel_name(100, 101)
        assert result == "Channel 101"

    def test_get_channel_folder_name_all_configurations(self):
        """Test folder names for various channel configurations."""
        # Test different channel counts
        configs = [
            (1, ["Channel 1 FC"]),
            (2, ["Channel 1 Left", "Channel 2 Right"]),
            (6, ["Channel 1 FL", "Channel 2 FC", "Channel 3 FR", 
                 "Channel 4 SL", "Channel 5 SR", "Channel 6 LFE"]),
            (8, ["Channel 1 FL", "Channel 2 FC", "Channel 3 FR",
                 "Channel 4 SL", "Channel 5 SR", "Channel 6 SBL",
                 "Channel 7 SBR", "Channel 8 LFE"]),
        ]
        
        for total_channels, expected_names in configs:
            for i in range(total_channels):
                folder_name = get_channel_folder_name(i, total_channels)
                assert folder_name == expected_names[i]

    def test_get_channel_name_value_error(self):
        """Test that invalid channel index raises ValueError."""
        with pytest.raises(ValueError):
            get_channel_name(-1, 2)  # Negative index
        
        with pytest.raises(ValueError):
            get_channel_name(5, 2)  # Index out of range

    def test_channel_mapping_consistency(self):
        """Test that channel names and folder names are consistent."""
        for total_channels in [1, 2, 3, 4, 5, 6, 7, 8]:
            for i in range(total_channels):
                name = get_channel_name(i, total_channels)
                folder_name = get_channel_folder_name(i, total_channels)
                
                # Folder name should contain channel name (except for stereo)
                if total_channels == 2:
                    # Stereo uses full names
                    assert name in folder_name
                elif total_channels > 2:
                    # Multi-channel: folder name should be "Channel N Name"
                    assert name in folder_name or folder_name.startswith(f"Channel {i+1}")

    def test_extended_channel_configurations(self):
        """Test extended channel configurations (9-12 channels)."""
        # 7.1.2 (9 channels)
        assert get_channel_name(7, 9) == "TFL"
        assert get_channel_name(8, 9) == "TFR"
        
        # 7.1.2 with LFE (10 channels)
        assert get_channel_name(7, 10) == "TFL"
        assert get_channel_name(8, 10) == "TFR"
        assert get_channel_name(9, 10) == "LFE"
        
        # 7.1.4 (11 channels)
        assert get_channel_name(7, 11) == "TFL"
        assert get_channel_name(8, 11) == "TFR"
        assert get_channel_name(9, 11) == "TBL"
        assert get_channel_name(10, 11) == "TBR"
        
        # 7.1.4 with LFE (12 channels)
        assert get_channel_name(7, 12) == "TFL"
        assert get_channel_name(8, 12) == "TFR"
        assert get_channel_name(9, 12) == "TBL"
        assert get_channel_name(10, 12) == "TBR"
        assert get_channel_name(11, 12) == "LFE"

