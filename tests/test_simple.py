"""Simple test to verify imports and basic functionality."""
import sys
from pathlib import Path

print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported")
    
    import scipy.signal
    print("✓ scipy imported")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.audio_processor import AudioProcessor
    print("✓ AudioProcessor imported")
    
    from src.octave_filter import OctaveBandFilter
    print("✓ OctaveBandFilter imported")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

