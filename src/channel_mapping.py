"""
Channel mapping module for multi-channel audio analysis.

Implements RP22 standard channel naming conventions for cinema and surround sound formats.
"""

from __future__ import annotations

from typing import Dict, Optional

# RP22 Standard Channel Names
# Reference: SMPTE RP22 standard for channel assignment

# Primary front speakers
FL = "FL"  # Front Left
FC = "FC"  # Front Center
FR = "FR"  # Front Right

# Surround speakers
SL = "SL"  # Surround Left
SR = "SR"  # Surround Right

# Surround back speakers
SBL = "SBL"  # Surround Back Left
SBR = "SBR"  # Surround Back Right

# Front wide speakers
FWL = "FWL"  # Front Wide Left
FWR = "FWR"  # Front Wide Right

# Additional surround variants
SL1 = "SL1"  # Surround Left 1
SR1 = "SR1"  # Surround Right 1
SL2 = "SL2"  # Surround Left 2
SR2 = "SR2"  # Surround Right 2

# Additional front speakers for large screens
FCL = "FCL"  # Front Center Left
FCR = "FCR"  # Front Center Right

# Surround center variants
SC = "SC"  # Surround Center
SCL = "SCL"  # Surround Center Left
SCR = "SCR"  # Surround Center Right

# Top speakers (commonly used with x.1.2, x.1.6, and 3.1.2 AV receivers)
TFL = "TFL"  # Top Front Left
TFR = "TFR"  # Top Front Right
TBL = "TBL"  # Top Back Left
TBR = "TBR"  # Top Back Right
TML = "TML"  # Top Middle Left
TMR = "TMR"  # Top Middle Right
TMC = "TMC"  # Top Middle Center ("Voice of God")

# Height speakers
HFC = "HFC"  # Height Front Center
HFR = "HFR"  # Height Front Right
HBL = "HBL"  # Height Back Left
HBR = "HBR"  # Height Back Right

# Low Frequency Effects
LFE = "LFE"  # Low Frequency Effects (subwoofer)

# Standard channel mapping configurations
# Maps channel count to list of channel names in order
STANDARD_CHANNEL_MAPS: Dict[int, list[str]] = {
    1: [FC],  # Mono
    2: ["Channel 1 Left", "Channel 2 Right"],  # Stereo (custom naming)
    3: [FL, FC, FR],  # 3.0
    4: [FL, FR, SL, SR],  # Quad
    5: [FL, FC, FR, SL, SR],  # 5.0
    6: [FL, FC, FR, SL, SR, LFE],  # 5.1
    7: [FL, FC, FR, SL, SR, SBL, SBR],  # 7.0
    8: [FL, FC, FR, SL, SR, SBL, SBR, LFE],  # 7.1
    # Extended configurations
    9: [FL, FC, FR, SL, SR, SBL, SBR, TFL, TFR],  # 7.1.2
    10: [FL, FC, FR, SL, SR, SBL, SBR, TFL, TFR, LFE],  # 7.1.2 with LFE
    11: [FL, FC, FR, SL, SR, SBL, SBR, TFL, TFR, TBL, TBR],  # 7.1.4
    12: [FL, FC, FR, SL, SR, SBL, SBR, TFL, TFR, TBL, TBR, LFE],  # 7.1.4 with LFE
}


def get_channel_name(channel_index: int, total_channels: int) -> str:
    """Get RP22 standard channel name for a given channel index.
    
    For stereo (2 channels), returns "Channel 1 Left" and "Channel 2 Right".
    For multi-channel, returns RP22 standard names based on channel count.
    
    Args:
        channel_index: Zero-based channel index
        total_channels: Total number of channels in the audio
        
    Returns:
        Channel name string (e.g., "FL", "Channel 1 Left", "LFE")
        
    Raises:
        ValueError: If channel_index is out of range or total_channels is invalid
    """
    if channel_index < 0 or channel_index >= total_channels:
        raise ValueError(
            f"Channel index {channel_index} out of range for {total_channels} channels"
        )
    
    # Special handling for stereo (2 channels)
    if total_channels == 2:
        if channel_index == 0:
            return "Channel 1 Left"
        elif channel_index == 1:
            return "Channel 2 Right"
    
    # Use standard channel map if available
    if total_channels in STANDARD_CHANNEL_MAPS:
        channel_names = STANDARD_CHANNEL_MAPS[total_channels]
        if channel_index < len(channel_names):
            return channel_names[channel_index]
    
    # Fallback: generate generic name
    return f"Channel {channel_index + 1}"


def get_channel_folder_name(channel_index: int, total_channels: int) -> str:
    """Get folder name for channel output directory.
    
    For stereo: "Channel 1 Left", "Channel 2 Right"
    For multi-channel: Uses RP22 standard names (FL, FC, FR, etc.)
    
    Args:
        channel_index: Zero-based channel index
        total_channels: Total number of channels
        
    Returns:
        Folder name string suitable for directory creation
    """
    channel_name = get_channel_name(channel_index, total_channels)
    
    # For stereo, use the full name as-is
    if total_channels == 2:
        return channel_name
    
    # For multi-channel, format as "Channel N Name" for clarity
    return f"Channel {channel_index + 1} {channel_name}"


def parse_truehd_channel_config(channel_config_str: str) -> Optional[list[str]]:
    """Parse TrueHD channel configuration string to channel names.
    
    Example: "3F2M2R/LFE" means:
    - 3F: 3 Front channels (FL, FC, FR)
    - 2M: 2 Middle/Surround channels (SL, SR)
    - 2R: 2 Rear channels (SBL, SBR)
    - LFE: Low Frequency Effects
    
    Args:
        channel_config_str: Channel configuration string from TrueHD metadata
                           (e.g., "3F2M2R/LFE", "2F/LFE")
        
    Returns:
        List of channel names in order, or None if parsing fails
    """
    if not channel_config_str:
        return None
    
    channel_names = []
    
    try:
        # Split by / to separate LFE
        parts = channel_config_str.upper().split("/")
        main_config = parts[0]
        has_lfe = len(parts) > 1 and "LFE" in parts[1]
        
        # Parse main configuration
        # Format: NF, NM, NR where N is count
        import re
        
        # Extract front channels (F)
        front_match = re.search(r"(\d+)F", main_config)
        if front_match:
            front_count = int(front_match.group(1))
            if front_count == 1:
                channel_names.append(FC)
            elif front_count == 2:
                channel_names.extend([FL, FR])
            elif front_count == 3:
                channel_names.extend([FL, FC, FR])
            elif front_count == 5:
                channel_names.extend([FL, FCL, FC, FCR, FR])
        
        # Extract middle/surround channels (M)
        middle_match = re.search(r"(\d+)M", main_config)
        if middle_match:
            middle_count = int(middle_match.group(1))
            if middle_count == 2:
                channel_names.extend([SL, SR])
            elif middle_count == 4:
                channel_names.extend([SL, SR, SL1, SR1])
        
        # Extract rear channels (R)
        rear_match = re.search(r"(\d+)R", main_config)
        if rear_match:
            rear_count = int(rear_match.group(1))
            if rear_count == 2:
                channel_names.extend([SBL, SBR])
            elif rear_count == 4:
                channel_names.extend([SBL, SBR, SBL, SBR])  # May need adjustment
        
        # Add LFE if present
        if has_lfe:
            channel_names.append(LFE)
        
        return channel_names if channel_names else None
        
    except Exception:
        return None

