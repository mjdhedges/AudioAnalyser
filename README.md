# Music Analyser

A Python implementation of octave band music analysis, based on the MATLAB `musicanalyser.m` script. This tool performs comprehensive frequency analysis on audio files using octave band filtering, statistical analysis, and visualization.

## Features

- **Audio Processing**: Load and preprocess various audio formats (WAV, FLAC, MP3, etc.)
- **Octave Band Filtering**: Apply octave band filters at standard frequencies (31.25Hz to 16kHz)
- **Statistical Analysis**: Calculate max values, RMS, dynamic range, and percentiles
- **Visualization**: Generate octave spectrum plots and amplitude distribution histograms
- **Data Export**: Export analysis results to CSV format
- **Command Line Interface**: Easy-to-use CLI with configurable options

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd music-analyser
```

2. Create and activate a virtual environment:
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
python -m venv venv
.\venv\Scripts\activate.bat

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# Basic usage
python -m src.main --input "song.flac"

# With custom output directory
python -m src.main --input "song.flac" --output-dir results

# Disable plots (faster processing)
python -m src.main --input "song.flac" --no-plot

# Custom sample rate
python -m src.main --input "song.flac" --sample-rate 48000

# Get help
python -m src.main --help
```

### Command Line Options

- `--input, -i`: Input audio file path (required)
- `--output-dir, -o`: Output directory for results (default: 'output')
- `--sample-rate, -sr`: Sample rate for processing (default: 44100)
- `--plot/--no-plot`: Generate plots (default: True)
- `--export-csv/--no-export-csv`: Export results to CSV (default: True)

### Programmatic Usage

```python
from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter
from src.music_analyzer import MusicAnalyzer

# Initialize components
audio_processor = AudioProcessor(sample_rate=44100)
octave_filter = OctaveBandFilter(sample_rate=44100)
analyzer = MusicAnalyzer(sample_rate=44100)

# Load and process audio
audio_data, sr = audio_processor.load_audio("song.flac")
audio_data = audio_processor.stereo_to_mono(audio_data)

# Create octave bank
octave_bank = octave_filter.create_octave_bank(audio_data)

# Perform analysis
results = analyzer.analyze_octave_bands(
    octave_bank, 
    octave_filter.OCTAVE_CENTER_FREQUENCIES
)

# Generate plots and export data
analyzer.create_octave_spectrum_plot(results, "spectrum.png")
analyzer.export_analysis_results(results, "results.csv")
```

## Analysis Output

The tool generates several outputs:

### 1. Octave Spectrum Plot
- Logarithmic frequency axis (20Hz - 20kHz)
- Peak and RMS levels for each octave band
- Similar to MATLAB's `semilogx` plot

### 2. Amplitude Distribution Histograms
- Histogram for each octave band
- Shows amplitude distribution patterns
- Useful for understanding signal characteristics

### 3. CSV Export
- Comprehensive statistical data
- Max amplitude, RMS, dynamic range
- Percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th)
- Frequency-specific analysis results

## Technical Details

### Octave Band Frequencies
The tool analyzes audio using standard octave band center frequencies:
- 31.25 Hz, 62.5 Hz, 125 Hz, 250 Hz, 500 Hz
- 1000 Hz, 2000 Hz, 4000 Hz, 8000 Hz, 16000 Hz

### Filter Design
- Uses Butterworth bandpass filters (4th order)
- Bandwidth calculated as center_frequency / √2 for octave bands
- Zero-phase filtering using `scipy.signal.filtfilt`

### Statistical Measures
- **Max Amplitude**: Peak signal level
- **RMS**: Root Mean Square (average power)
- **Dynamic Range**: Ratio of RMS to peak
- **Percentiles**: Distribution analysis

## Development Commands

- Run tests: `pytest`
- Format code: `black src tests`
- Lint code: `flake8 src tests`
- Type checking: `mypy src`
- Sort imports: `isort src tests`

## Project Structure

```
music-analyser/
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py             # CLI entry point
│   ├── audio_processor.py   # Audio loading and preprocessing
│   ├── octave_filter.py    # Octave band filtering
│   └── music_analyzer.py   # Analysis and visualization
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_audio_processor.py
│   ├── test_octave_filter.py
│   └── test_music_analyzer.py
├── docs/                   # Documentation
│   └── musicanalyser.m     # Original MATLAB script
├── .cursor/                # Cursor IDE rules
│   └── rules/
├── venv/                   # Virtual environment (not in git)
├── .gitignore              # Git ignore rules
├── .cursorrules            # Cursor IDE rules
├── pyproject.toml          # Project configuration
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md              # This file
```

## Comparison with MATLAB Version

This Python implementation replicates the functionality of the original MATLAB `musicanalyser.m` script:

| MATLAB Function | Python Equivalent |
|----------------|-------------------|
| `audioread()` | `librosa.load()` |
| `octdsgn()` | `OctaveBandFilter.design_octave_filter()` |
| `filter()` | `scipy.signal.filtfilt()` |
| `poctave()` | Custom octave band analysis |
| `semilogx()` | `matplotlib.pyplot.semilogx()` |
| `histogram()` | `matplotlib.pyplot.hist()` |

## Dependencies

### Core Libraries
- **librosa**: Audio processing and analysis
- **numpy**: Numerical computations
- **scipy**: Signal processing and filtering
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation and CSV export
- **soundfile**: Audio file I/O
- **click**: Command-line interface

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **isort**: Import sorting

## License

MIT License - see LICENSE file for details.
