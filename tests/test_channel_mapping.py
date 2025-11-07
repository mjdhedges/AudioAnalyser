"""
Tests for the channel mapping module.

Tests RP22 standard channel naming and folder name generation.
"""

import pytest

from src.channel_mapping import (
    get_channel_name,
    get_channel_folder_name
)
from src import channel_mapping


class TestChannelMapping:
    """Test cases for channel mapping functions."""

    def test_stereo_channel_names(self):
        """Test stereo channel naming."""
        # Channel 0 (Left) - returns full name
        assert get_channel_name(0, 2) == "Channel 1 Left"
        assert get_channel_folder_name(0, 2) == "Channel 1 Left"
        
        # Channel 1 (Right) - returns full name
        assert get_channel_name(1, 2) == "Channel 2 Right"
        assert get_channel_folder_name(1, 2) == "Channel 2 Right"

    def test_mono_channel_name(self):
        """Test mono channel naming."""
        # Mono uses FC (Front Center) in RP22 standard
        assert get_channel_name(0, 1) == "FC"
        assert get_channel_folder_name(0, 1) == "Channel 1 FC"

    def test_51_surround_channel_names(self):
        """Test 5.1 surround channel naming (RP22 standard)."""
        # 5.1 surround: FL, FC, FR, SL, SR, LFE
        expected_names = ["FL", "FC", "FR", "SL", "SR", "LFE"]
        
        for i, expected_name in enumerate(expected_names):
            assert get_channel_name(i, 6) == expected_name
            assert get_channel_folder_name(i, 6) == f"Channel {i+1} {expected_name}"

    def test_71_surround_channel_names(self):
        """Test 7.1 surround channel naming (RP22 standard)."""
        # 7.1 surround: FL, FC, FR, SL, SR, SBL, SBR, LFE
        expected_names = ["FL", "FC", "FR", "SL", "SR", "SBL", "SBR", "LFE"]
        
        for i, expected_name in enumerate(expected_names):
            assert get_channel_name(i, 8) == expected_name
            assert get_channel_folder_name(i, 8) == f"Channel {i+1} {expected_name}"

    def test_extended_channel_names(self):
        """Test extended RP22 channel names for larger channel counts."""
        # Test 7.1.2 configuration (10 channels): FL, FC, FR, SL, SR, SBL, SBR, TFL, TFR, LFE
        assert get_channel_name(7, 10) == "TFL"  # Top Front Left
        assert get_channel_name(8, 10) == "TFR"  # Top Front Right
        
        # Test 7.1.4 configuration (12 channels): FL, FC, FR, SL, SR, SBL, SBR, TFL, TFR, TBL, TBR, LFE
        # Channel 10 is TBL, channel 11 is TBR
        assert get_channel_name(9, 12) == "TBL"  # Top Back Left (index 9)
        assert get_channel_name(10, 12) == "TBR"  # Top Back Right (index 10)

    def test_standard_channel_maps(self):
        """Test that STANDARD_CHANNEL_MAPS dictionary is properly defined."""
        # Access the constant via module
        channel_maps = getattr(channel_mapping, 'STANDARD_CHANNEL_MAPS', None)
        assert channel_maps is not None
        assert isinstance(channel_maps, dict)
        assert len(channel_maps) > 0
        
        # Check some standard configurations exist
        assert 1 in channel_maps  # Mono
        assert 2 in channel_maps  # Stereo
        assert 6 in channel_maps  # 5.1
        assert 8 in channel_maps  # 7.1
        
        # Check 5.1 mapping
        assert channel_maps[6] == ["FL", "FC", "FR", "SL", "SR", "LFE"]

    def test_generic_channel_names(self):
        """Test generic channel naming for unsupported channel counts."""
        # For very high channel counts, should fall back to generic naming
        assert get_channel_name(0, 50) == "Channel 1"
        assert get_channel_name(49, 50) == "Channel 50"
        
        # Folder names append the channel name
        assert get_channel_folder_name(0, 50) == "Channel 1 Channel 1"
        assert get_channel_folder_name(49, 50) == "Channel 50 Channel 50"

    def test_channel_folder_name_format(self):
        """Test that folder names are properly formatted."""
        # Should always start with "Channel N"
        folder_name = get_channel_folder_name(5, 8)
        assert folder_name.startswith("Channel 6")  # Index 5 = Channel 6
        
        # Should include channel name for known channels
        folder_name = get_channel_folder_name(0, 2)
        assert "Left" in folder_name

    def test_edge_cases(self):
        """Test edge cases for channel mapping."""
        # Single channel uses FC
        assert get_channel_name(0, 1) == "FC"
        
        # Very large channel count uses generic naming
        assert get_channel_name(31, 32) == "Channel 32"
        
        # Channel index out of bounds should raise ValueError
        with pytest.raises(ValueError):
            get_channel_name(100, 8)
