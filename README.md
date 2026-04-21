# Music Analyser

A Python implementation of octave band music analysis, based on the MATLAB `musicanalyser.m` script. This tool performs comprehensive frequency analysis on audio files using octave band filtering, statistical analysis, and visualization.

## Features

- **Audio Processing**: Load and preprocess various audio formats (WAV, FLAC, MP3, MKV, etc.)
- **Multi-Channel Support**: Process stereo and surround audio with separate analysis per channel
- **MKV/TrueHD Support**: Extract and decode Dolby TrueHD audio from MKV containers (requires ffmpeg)
- **RP22 Channel Naming**: Automatic channel identification using RP22 standard (FL, FC, FR, SL, SR, etc.)
- **Octave Band Filtering**: Apply octave band filters at standard frequencies (16Hz to 16kHz) including cinema/LFE analysis
- **Advanced Statistics**: Comprehensive analysis including clipping detection, dynamic range, spectral characteristics
- **Visualization**: Generate octave spectrum plots, crest factor analysis, and amplitude distribution histograms
- **Data Export**: Export comprehensive analysis results to CSV format with content type tagging
- **Configuration System**: TOML-based configuration with command-line overrides
- **Content Classification**: Automatic tagging of Music/Film/Test Signal content based on folder structure
- **Command Line Interface**: Professional CLI with extensive customization options

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- **ffmpeg** (required for MKV/TrueHD support): Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

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

## Configuration

The Music Analyser uses a TOML-based configuration system (`config.toml`) that allows you to customize all analysis parameters. You can override any setting via command-line arguments.

### Configuration File (`config.toml`)

The configuration file is organized into sections:

```toml
[analysis]
chunk_duration_seconds = 2.0        # Time-domain analysis chunk size
sample_rate = 44100                 # Audio processing sample rate
tracks_dir = "Tracks"               # Default tracks directory
output_dir = "analysis"             # Default output directory
octave_center_frequencies = [16.0, 31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]

[plotting]
octave_spectrum_figsize = [12, 8]   # Figure dimensions
octave_spectrum_xlim = [15, 20000]  # X-axis limits (Hz)
octave_spectrum_ylim = [-60, 3]     # Y-axis limits (dBFS)
dpi = 300                           # Plot resolution

[advanced_stats]
hot_peaks_threshold_db = -1.0       # Near-clipping threshold
clip_events_threshold_db = -0.1     # Actual clipping threshold
peak_saturation_threshold_db = -3.0  # Heavily compressed threshold
transient_threshold_db = 3.0        # Significant level change threshold

[export]
include_track_metadata = true       # Include track information
include_advanced_statistics = true  # Include advanced metrics
include_histogram_data = true       # Include amplitude distributions
```

### Command Line Overrides

You can override any configuration parameter via command-line arguments:

```bash
# Override chunk duration for finer time resolution
python -m src.main --input "track.wav" --chunk-duration 1.0

# Override plot DPI for high-resolution output
python -m src.main --input "track.wav" --dpi 600

# Override logging level for debugging
python -m src.main --input "track.wav" --log-level DEBUG

# Use custom configuration file
python -m src.main --input "track.wav" --config "custom_config.toml"
```

## Usage

### Command Line Interface

```bash
# Batch processing - analyze all tracks in Tracks directory
python -m src.main

# Analyze tracks in custom directory
python -m src.main --tracks-dir "MyMusic" --output-dir "results"

# Single file analysis (file path)
python -m src.main --input "song.flac"

# Batch processing from a directory (directory path; recursive)
python -m src.main --input "MyMusic" --output-dir "results"

# Batch processing with all artifacts (analysis + manifest + decay plot)
python -m src.main --tracks-dir "Tracks" --output-dir "analysis_output"

# Post-processing only (regenerate manifests/plots for existing results)
python -m src.main --tracks-dir "Tracks" --output-dir "analysis_output" --post-only

# Custom sample rate for batch processing
python -m src.main --sample-rate 48000

# Get help
python -m src.main --help
```

### Command Line Options

#### Basic Options
- `--input, -i`: Input path. If it's a file: single-file analysis. If it's a directory: batch analysis (recursive).
- `--tracks-dir, -t`: Directory containing tracks to analyze (default: from config)
- `--output-dir, -o`: Output directory for results (default: from config). In batch mode, subfolders under the input folder are recreated under this directory (each track is written to a path that mirrors its location relative to the batch input folder).
- `--batch/--single`: Deprecated. Mode is inferred from `--input` (file=single, dir=batch).
- `--export-csv/--no-export-csv`: Export results to CSV (default: True)
- `--generate-manifest/--no-generate-manifest`: Generate worst-channel manifest (default: True)
- `--generate-decay-plot/--no-generate-decay-plot`: Generate combined decay plot (default: True)
- `--post-only`: Skip analysis and only run post-processing (manifests/plots) on existing results

#### Configuration Overrides
- `--sample-rate, -sr`: Sample rate for processing (overrides config)
- `--chunk-duration, -cd`: Duration of analysis chunks in seconds (overrides config)
- `--dpi, -d`: DPI for plot output (overrides config)
- `--log-level, -l`: Logging level: DEBUG, INFO, WARNING, ERROR (overrides config)
- `--config, -c`: Path to custom configuration file (default: config.toml)

### Content Type Classification

The analyzer automatically tags tracks based on their folder structure:

```
Tracks/
├── Music/           # Tagged as "Music"
│   ├── song1.flac
│   └── song2.wav
├── Film/            # Tagged as "Film"
│   ├── movie1.wav
│   └── movie2.flac
└── Test Signals/    # Tagged as "Test Signal"
    ├── pink_noise.wav
    └── sine_sweep.wav
```

This enables easy filtering of analysis results by content type in the CSV export.

### Multi-Channel Audio Analysis

The analyzer supports multi-channel audio processing with automatic channel identification:

#### Stereo Files
- **Channel 1 Left**: Left channel analysis in `Channel 1 Left/` folder
- **Channel 2 Right**: Right channel analysis in `Channel 2 Right/` folder

#### Multi-Channel Files (RP22 Standard)
Multi-channel audio uses RP22 standard channel naming:

**Primary Channels:**
- **FL** - Front Left
- **FC** - Front Center
- **FR** - Front Right
- **SL** - Surround Left
- **SR** - Surround Right
- **SBL** - Surround Back Left
- **SBR** - Surround Back Right
- **LFE** - Low Frequency Effects

**Extended Channels:**
- **FWL/FWR** - Front Wide Left/Right
- **SL1/SR1, SL2/SR2** - Surround Left/Right 1 & 2
- **FCL/FCR** - Front Center Left/Right
- **SC** - Surround Center
- **SCL/SCR** - Surround Center Left/Right
- **TFL/TFR** - Top Front Left/Right
- **TBL/TBR** - Top Back Left/Right
- **TML/TMR** - Top Middle Left/Right
- **TMC** - Top Middle Center
- **HFC/HFR/HBL/HBR** - Height Front/Back Center/Left/Right

Each channel is analyzed separately and results are stored in channel-specific folders (e.g., `Channel 1 FL/`, `Channel 2 FC/`).

#### MKV Container and TrueHD Support

The analyzer can extract and decode Dolby TrueHD audio from MKV video containers:

**Requirements:**
- ffmpeg must be installed and available in PATH
- Enable MKV support in `config.toml`: `[mkv_support] enable = true`

**Features:**
- Automatic TrueHD stream detection
- Excludes Dolby Atmos metadata streams (configurable)
- Decodes TrueHD to PCM for analysis
- Preserves multi-channel channel order
- Temporary file cleanup after extraction

**Configuration:**
```toml
[mkv_support]
enable = true              # Enable MKV/TrueHD support
exclude_atmos = true       # Exclude Dolby Atmos metadata streams
```

**Usage:**
```bash
# Analyze MKV file with TrueHD audio
python -m src.main --input "movie.mkv"
```

### Programmatic Usage

```python
from src.audio_processor import AudioProcessor
from src.octave_filter import OctaveBandFilter
from src.music_analyzer import MusicAnalyzer

# Initialize components
audio_processor = AudioProcessor(sample_rate=44100)
octave_filter = OctaveBandFilter(sample_rate=44100)
analyzer = MusicAnalyzer(sample_rate=44100)

# Load and process audio (preserves multi-channel)
audio_data, sr = audio_processor.load_audio("song.flac")

# For multi-channel audio, extract individual channels
channels = audio_processor.extract_channels(audio_data)
for channel_data, channel_idx in channels:
    # Process each channel separately
    pass

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

The tool generates comprehensive analysis outputs with professional-grade metrics and visualizations.

### Directory Structure

For a **single file**, outputs go under ``analysis/<track stem>/``. For **batch** analysis of a folder, each file is written under ``analysis/<relative path without extension>/``, mirroring subfolders (e.g. ``tracks/Music/a.flac`` → ``analysis/Music/a/``).

```
analysis/
├── Track Name 1/
│   ├── octave_spectrum.png              # Octave spectrum plot (16Hz-16kHz)
│   ├── crest_factor.png                 # Crest factor analysis
│   ├── crest_factor_time.png            # Crest factor vs time
│   ├── octave_crest_factor_time.png     # Octave band crest factors vs time
│   ├── histograms.png                   # Linear amplitude distributions
│   ├── histograms_log_db.png            # Log dBFS amplitude distributions
│   └── analysis_results.csv             # Comprehensive data export
├── Track Name 2/
│   └── ...
```

### Visualizations

#### 1. Octave Spectrum Plot (`octave_spectrum.png`)
- **Frequency Range**: 16Hz to 16kHz (cinema-ready including LFE)
- **Metrics**: Peak and RMS levels for each octave band
- **Reference Lines**: Track peak/RMS, min/max crest factor chunks
- **Format**: Logarithmic frequency axis, dBFS amplitude scale

#### 2. Crest Factor Analysis (`crest_factor.png`)
- **Metrics**: Peak-to-RMS ratio for each octave band
- **Reference**: Track average crest factor
- **Extreme Analysis**: Min/max crest factor chunk overlays
- **Format**: Logarithmic frequency axis, dB crest factor scale

#### 3. Time-Domain Analysis (`crest_factor_time.png`)
- **Metrics**: Crest factor, peak, and RMS levels over time
- **Resolution**: Configurable chunk duration (default: 2 seconds)
- **Statistics**: Average, maximum, minimum crest factors
- **Format**: Time series plots with dual y-axes

#### 4. Octave Band Time Analysis (`octave_crest_factor_time.png`)
- **Metrics**: Crest factor for each octave band over time
- **Resolution**: Same time chunks as main analysis
- **Visualization**: Multi-line plot with frequency-specific colors

Note: This plot is relatively expensive to generate. You can skip it with
`--skip-octave-cf-time` (it is not currently persisted in `analysis_results.csv`,
so generating it later requires recomputation).

If you want to keep the underlying time-series without generating the PNG,
use `--export-octave-cf-time-data` (writes `octave_crest_factor_time.csv`).
- **Format**: Time series with fixed 0-40dB crest factor scale

#### 5. Amplitude Distributions (`histograms.png` & `histograms_log_db.png`)
- **Linear Histograms**: Amplitude distribution (-1 to +1 range)
- **Log dBFS Histograms**: Amplitude distribution in dBFS scale
- **Bands**: Separate histogram for each octave band
- **Statistics**: Bin counts and densities for statistical analysis

### CSV Data Export (`analysis_results.csv`)

The CSV export contains comprehensive analysis data organized in sections:

#### 1. Track Metadata
- Track name, path, content type (Music/Film/Test Signal)
- Duration, sample rate, channels, original peak level
- Analysis timestamp

#### 2. Advanced Statistics (27 metrics)
- **Clipping Analysis**: Hot peaks, clip events, peak saturation rates
- **Frequency Dynamics**: Crest factor variance, bass/treble ratios
- **Temporal Analysis**: Dynamic consistency, transient density
- **Spectral Analysis**: Spectral centroid, energy distribution
- **Peak Distribution**: Sample distribution across dBFS ranges

#### 3. Octave Band Analysis
- **11 Frequency Bands**: 16Hz to 16kHz (including cinema LFE)
- **Metrics per Band**: Max amplitude, RMS, dynamic range, crest factor
- **Statistics**: Mean, std, percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th)

#### 4. Time-Domain Analysis
- **Chunk Data**: Crest factor, peak, RMS for each time chunk
- **Summary Statistics**: Average, max, min crest factors over time

#### 5. Histogram Data
- **Amplitude Distributions**: Bin centers, counts, densities for each frequency band
- **Statistical Analysis**: Complete amplitude distribution data

#### 6. Extreme Chunks Analysis
- **Min/Max Crest Factor Chunks**: Detailed analysis of most/least dynamic sections
- **Per-Band Metrics**: RMS, peak, crest factor for each octave band in extreme chunks

## Technical Details

### Octave Band Frequencies
The tool analyzes audio using standard octave band center frequencies including cinema/LFE analysis:
- **16 Hz** (LFE/Cinema), 31.25 Hz, 62.5 Hz, 125 Hz, 250 Hz, 500 Hz
- 1000 Hz, 2000 Hz, 4000 Hz, 8000 Hz, 16000 Hz

### Filter Design
- Uses Butterworth bandpass filters (4th order for standard frequencies)
- **2nd order filters** for very low frequencies (< 1% of Nyquist) for numerical stability
- Bandwidth calculated as center_frequency / √2 for octave bands
- Zero-phase filtering using `scipy.signal.filtfilt`

### Advanced Statistics
- **Clipping Detection**: Hot peaks (>-1dBFS), clip events (>-0.1dBFS), peak saturation (>-3dBFS)
- **Dynamic Analysis**: Crest factor variance, temporal consistency, transient density
- **Spectral Analysis**: Spectral centroid, bass/mid/treble energy ratios
- **Peak Distribution**: Sample distribution across 6 dBFS ranges

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Comprehensive system architecture and design details
- **[FINAL_PERFORMANCE_SUMMARY.md](FINAL_PERFORMANCE_SUMMARY.md)**: Complete performance optimization history
- **[PHASE3_OPTIMIZATION_SUMMARY.md](PHASE3_OPTIMIZATION_SUMMARY.md)**: Latest optimizations (batch processing & caching)

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
│   ├── config.py           # Configuration management
│   ├── audio_processor.py  # Audio loading and preprocessing
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
├── config.toml             # Configuration file
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
