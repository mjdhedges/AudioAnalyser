#!/usr/bin/env python3
"""
Simple test script to verify the configuration system works
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

try:
    print("Testing Music Analyser configuration system...")
    
    # Test importing config
    from config import config
    print("✅ Config imported successfully")
    
    # Test loading directory defaults
    tracks_dir = config.get('analysis.tracks_dir', 'Tracks')
    output_dir = config.get('analysis.output_dir', 'analysis')
    print(f"✅ Tracks directory: {tracks_dir}")
    print(f"✅ Output directory: {output_dir}")
    
    # Test if tracks directory exists
    tracks_path = Path(tracks_dir)
    if tracks_path.exists():
        print(f"✅ Tracks directory exists: {tracks_path}")
        audio_files = list(tracks_path.rglob("*.wav")) + list(tracks_path.rglob("*.flac")) + list(tracks_path.rglob("*.mp3"))
        print(f"✅ Found {len(audio_files)} audio files")
    else:
        print(f"❌ Tracks directory not found: {tracks_path}")
    
    # Test other config values
    chunk_duration = config.get('analysis.chunk_duration_seconds', 2.0)
    sample_rate = config.get('analysis.sample_rate', 44100)
    print(f"✅ Chunk duration: {chunk_duration}s")
    print(f"✅ Sample rate: {sample_rate}Hz")
    
    print("✅ All configuration tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Try installing dependencies: pip install toml")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

