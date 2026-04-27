# Audio Analyser - Architecture Documentation

## Overview

The Audio Analyser is a Python application that performs comprehensive octave band frequency analysis on audio files. This document explains the system architecture, component design, and data flow.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Entry Point                          │
│                         (src/main.py)                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     Batch Processing                      │  │
│  │   ┌───────────┐  ┌───────────┐  ┌───────────┐           │  │
│  │   │ Worker 1  │  │ Worker 2  │  │ Worker 3  │  ...      │  │
│  │   └───────────┘  └───────────┘  └───────────┘           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Processing Pipeline                      │
│                   (analyze_single_track)                         │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐         │
│  │   Audio     │──▶│   Octave    │──▶│  Analysis &  │         │
│  │  Processor  │   │   Filter    │   │ Visualization│         │
│  └─────────────┘   └─────────────┘   └──────────────┘         │
│         │                  │                    │               │
│         ▼                  ▼                    ▼               │
│     ┌──────┐         ┌────────┐           ┌────────┐          │
│     │ Mono │         │ Cached │           │  CSV   │          │
│     │Conv. │         │ Filters│           │ Export │          │
│     └──────┘         └────────┘           └────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Configuration Layer                        │
│                      (src/config.py)                              │
│  • TOML-based configuration                                       │
│  • Command-line argument overrides                                │
│  • Performance settings (caching, parallelization)                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Audio Processing Layer                       │
│                   (src/audio_processor.py)                        │
│  • Audio file loading (librosa)                                   │
│  • Format conversion and preprocessing                            │
│  • Stereo-to-mono conversion                                      │
│  • Audio normalization                                            │
│  • Float32 optimization for memory                                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Octave Band Filtering                        │
│                   (src/octave_filter.py)                          │
│  • FFT power-complementary octave bank                            │
│  • Full-file or large-block FFT processing                        │
│  • Low/high residual bands for energy outside nominal octaves     │
│  • Inverse FFT time-series reconstruction per band                │
│  • Disk-based caching (NPY format)                                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Analysis & Visualization                       │
│                  (src/music_analyzer.py)                          │
│  • Octave band statistical analysis                               │
│  • Time-domain chunk analysis                                     │
│  • Extreme chunk identification                                   │
│  • Advanced statistics (clipping, dynamics, etc.)                │
│  • Plot generation (Matplotlib with Agg backend)                  │
│  • CSV data export                                                │
└──────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Main Entry Point (`src/main.py`)

**Responsibilities:**
- Command-line interface using Click
- Configuration management
- Batch vs single file processing
- Result caching coordination
- Progress tracking

**Key Functions:**
- `main()`: CLI entry point with argument parsing
- `analyze_single_track()`: Core analysis orchestration
- `get_config_hash()`: Config hash for cache invalidation
- `check_result_cache()`: Cache validation
- `save_result_cache()`: Cache metadata storage

**Processing Modes:**
- **Single File**: Analyze one track into a `.aaresults` bundle
- **Batch (Sequential)**: Process tracks one at a time
- **Batch (Parallel)**: Process multiple tracks simultaneously

### 2. Configuration (`src/config.py`)

**Responsibilities:**
- Load TOML configuration file
- Provide configuration access API
- Support command-line overrides
- Default value management

**Configuration Sections:**
- `[analysis]`: Core analysis parameters (sample rate, chunk duration, etc.)
- `[plotting]`: Render settings (DPI, figure sizes, axes)
- `[advanced_stats]`: Thresholds for clipping detection, etc.
- `[export]`: Bundle output and legacy CSV compatibility options
- `[performance]`: Optimization settings (caching, parallel processing)
- `[multi_channel]`: Multi-channel processing settings (enable/disable)
- `[mkv_support]`: MKV container and TrueHD audio extraction settings

### 3. Audio Processor (`src/audio_processor.py`)

**Responsibilities:**
- Load audio files (librosa/soundfile with float32 optimization)
- Preserve multi-channel audio (no automatic mono conversion)
- Extract individual channels from multi-channel audio
- Normalize audio data (global normalization for multi-channel)
- Extract audio metadata (channel count, layout)
- MKV container detection and TrueHD audio extraction

**Key Features:**
- Float32 by default (50% memory reduction)
- Handles multiple audio formats (WAV, FLAC, MP3, MKV, etc.)
- Preserves original peak for dBFS calculations
- Multi-channel support: preserves channel structure (samples × channels)
- MKV/TrueHD support: extracts and decodes Dolby TrueHD from MKV containers

**Multi-Channel Processing:**
- `load_audio()`: Preserves multi-channel shape (samples,) for mono or (samples, channels) for multi-channel
- `extract_channels()`: Returns list of (channel_data, channel_index) tuples
- `get_audio_info()`: Returns channel count and layout (mono/stereo/multi-channel)
- `normalize_audio()`: Global normalization across all channels to maintain relative levels

**MKV/TrueHD Support:**
- `_is_mkv_file()`: Detects MKV containers by file extension
- `_probe_mkv_audio_streams()`: Uses ffprobe to identify audio streams and codecs
- `_extract_truehd_from_mkv()`: Uses ffmpeg to extract and decode TrueHD to PCM
- Automatic TrueHD stream detection (excludes Dolby Atmos metadata)
- Temporary file management with automatic cleanup
- Requires ffmpeg/ffprobe in PATH

### 4. Octave Band Filter (`src/octave_filter.py`)

**Responsibilities:**
- Create an FFT power-complementary octave bank
- Preserve total RMS energy through flat summed power
- Return same-length filtered time-series signals for each band
- Support auto, full-file FFT, and large-block FFT processing
- Use temporary disk-backed octave storage when the time-series bank would
  exceed the configured RAM budget

**Key Classes:**
- `OctaveBandFilter`: Main FFT octave-bank class

**Filter Specifications:**
- Nominal centers: 8, 16, 31.25, 62.5, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz
- Optional residual band at 4 Hz and below
- Optional high residual band above the 16 kHz octave region up to Nyquist
- Power complementarity: `sum(weight_band(f) ** 2) = 1.0`

**Caching Strategy:**
- Cache octave banks to disk (NPY format)
- Memory-mapped loading for efficiency
- Automatic invalidation on source file change

### 5. Channel Mapping (`src/channel_mapping.py`)

**Responsibilities:**
- Map channel indices to RP22 standard channel names
- Generate human-readable channel labels for metadata and legacy output
- Support stereo and multi-channel layouts

**Key Functions:**
- `get_channel_name()`: Returns RP22 channel name (FL, FC, FR, etc.) or generic "Channel N"
- `get_channel_folder_name()`: Returns formatted legacy folder/display name (e.g., "Channel 1 FL")

**Channel Naming:**
- **Stereo**: Channel 1 Left, Channel 2 Right
- **Multi-Channel**: RP22 standard names (FL, FC, FR, SL, SR, SBL, SBR, LFE, etc.)
- Supports up to 32 channels with extended RP22 names (TFL, TFR, TBL, TBR, etc.)

### 6. Audio Analyzer (`src/music_analyzer.py`)

**Responsibilities:**
- Core octave band statistical analysis
- Time-domain analysis (chunking)
- Orchestration of analysis pipeline
- Delegation to specialized modules via composition

**Key Classes:**
- `MusicAnalyzer`: Main analysis class using composition pattern

**Composition:**
- `EnvelopeAnalyzer`: Handles envelope processing (delegated from `envelope_analyzer.py`)
- `DataExporter`: Calculates advanced statistics and optional legacy CSV exports

**Analysis Types:**
1. **Octave Band Analysis**: HS RMS, peak, dynamic range, crest factor per band
2. **Time-Domain Analysis**: Crest factor, peak, RMS over time chunks
3. **Extreme Chunks**: Identification of min/max crest factor sections

**Note:** The original methods are kept as wrappers for backward compatibility, delegating to the composed objects.

### 7. Track Processor (`src/track_processor.py`)

**Responsibilities:**
- Single channel processing pipeline
- Orchestration of analysis and bundle export for one channel
- Optional legacy channel-specific CSV output management

**Key Classes:**
- `TrackProcessor`: Handles complete processing pipeline for a single channel

**Processing Flow:**
1. Normalize channel audio
2. Create octave bank
3. Perform comprehensive analysis
4. Analyze envelope statistics
5. Write/update the track `.aaresults` bundle
6. Optionally write legacy `analysis_results.csv`

### 8. Bundle Rendering (`src/render.py`, `src/results/render.py`)

**Responsibilities:**
- Read `.aaresults` bundles
- Generate channel plots, envelope plots, group plots, worst-channel manifests, and reports
- Keep graph/report generation out of the audio analysis pass

**Key Classes:**
- `ResultBundle`: Reader-facing bundle abstraction
- `ChannelResult`: Per-channel bundle abstraction

**Visualizations:**
1. Octave spectrum plot
2. Crest factor plot
3. Time-domain crest factor plot
4. Octave band crest factor time plot
5. Linear amplitude histograms
6. Log dBFS amplitude histograms
7. Pattern envelope plots
8. Independent envelope plots
9. Group crest-factor, octave-spectrum, peak-decay, and worst-channel outputs

## Data Flow

### Analysis Pipeline

```
1. Audio File Input (WAV, FLAC, MP3, MKV, etc.)
   ↓
2. Detect File Type
   ├─► MKV? → Extract TrueHD audio (ffmpeg)
   └─► Other → Direct load
   ↓
3. Load Audio (float32, preserves multi-channel)
   ↓
4. Detect Channel Count & Layout
   ├─► Mono → Process directly
   ├─► Stereo → Extract channels (Left, Right)
   └─► Multi-Channel → Extract channels (RP22 naming)
   ↓
5. For Each Channel:
   ├─► Normalize (global normalization)
   ├─► Create Octave Bank (11 filtered bands + full spectrum)
   │   ├─► Check disk cache
   │   ├─► Parallel filtering (if enabled)
   │   └─► Save to cache
   ├─► Perform Comprehensive Analysis
   │   ├─► Octave band statistics
   │   ├─► Time-domain chunking
   │   └─► Extreme chunk analysis
   ├─► Analyze envelope and sustained-peak data
   ├─► Write per-channel CSV/JSON artifacts
   └─► Update .aaresults manifest
   ↓
6. Save Cache Metadata in manifest.json
   ↓
7. Render Pass:
   ├─► Read .aaresults bundle
   ├─► Generate graphs and group outputs
   └─► Generate analysis.md report
```

### Caching Strategy

```
First Run:
  Audio File → Create Octave Bank → Cache to Disk → Analysis → Results
                              ↓
                             Cache
                              ↓
Re-run (File Unchanged):
  Audio File → Load Octave Bank from Cache → Results (instant)
  
Re-run (File Changed):
  Audio File (modified) → Invalidate Cache → Create New → Cache → Results
```

## Performance Optimizations

### Phase 0: Critical Fixes
- Eliminated redundant octave bank creation
- Reduced filter operations from 1,851 to 39 per track (97.9% reduction)

### Phase 1: Vectorization
- Vectorized all time-domain calculations
- Eliminated nested Python loops
- 10-20x speedup for plot generation

### Phase 2: Advanced Optimizations
- Parallel octave bank processing (4-8x speedup)
- Float32 memory optimization (50% reduction)
- Smart disk caching (near-instant re-analysis)

### Phase 3: Batch & Caching
- Result caching (100-400x re-analysis speedup)
- Parallel batch processing (3-4x speedup)
- Matplotlib Agg backend (20-30% faster)

**Total Speedup:** 10,000-40,000x compared to original implementation

## Configuration & Customization

### Key Configuration Options

```toml
[analysis]
octave_filter_mode = "auto"           # Auto full-file/block FFT selection
octave_max_memory_gb = 4.0            # Octave processing RAM budget
octave_fft_block_duration_seconds = 30.0

[performance]
enable_parallel_batch = true        # Parallel batch processing
enable_result_cache = true          # Cache analysis results
use_float32 = true                  # Memory optimization

[plotting]
dpi = 300                           # High quality plots
batch_dpi = 150                     # Faster batch plots
render_dpi = 150                    # Bundle render pass DPI

[export]
generate_analysis_bundle = true     # Default analysis artifact
generate_legacy_csv = false         # Optional compatibility export
```

### Cache Directories

```
analysis/
└── Music/
    └── Track Name.aaresults/
        ├── manifest.json   # Bundle index plus cache metadata
        └── channels/
            ├── channel_01/
            └── channel_02/

rendered/
└── Track Name/
    ├── octave_spectrum.png
    ├── peak_decay_groups.png
    ├── worst_channels_manifest.csv
    └── analysis.md
```

## Error Handling & Reliability

### Cache Invalidation
- Source file modification time check
- Configuration hash comparison
- Automatic regeneration on changes

### Fallback Mechanisms
- Sequential processing if parallel fails
- Cache regeneration on corruption
- Graceful error handling with logging

### Memory Management
- Explicit cleanup with `del` statements
- `plt.close('all')` for matplotlib figures
- `gc.collect()` for forced garbage collection
- Float32 optimization for large arrays

## Extensibility

### Adding New Analyzers
1. Add method to `MusicAnalyzer` class
2. Integrate into `analyze_comprehensive()`
3. Persist the derived data in the `.aaresults` bundle
4. Update renderer/report consumers if the new data should be visualized

### Adding New Visualizations
1. Add or extend a renderer in `src.results.render`
2. Read only from `ResultBundle` artifacts
3. Add focused tests for bundle replay and CLI rendering

### Adding New Optimizations
1. Identify bottleneck (use profiling)
2. Implement optimization (vectorization, caching, parallelization)
3. Add configuration options
4. Test performance impact
5. Document in optimization summary

## Testing

### Test Coverage
- Unit tests for each component
- Integration tests for full pipeline
- Performance benchmarks
- Cache validation tests

### Running Tests
```bash
pytest tests/              # Run all tests
pytest tests/ -v          # Verbose output
pytest tests/ --cov       # Coverage report
```

## Future Enhancements

### Planned
- Incremental result updates
- Cache size management
- Distributed processing

### Potential
- Real-time analysis
- Web interface
- Database storage
- Machine learning analysis

## References

- **README.md**: User guide and quick start
- **FINAL_PERFORMANCE_SUMMARY.md**: Performance optimization details
- **PHASE3_OPTIMIZATION_SUMMARY.md**: Latest optimization batch
- **config.toml**: Default configuration values
- **docs/musicanalyser.m**: Original MATLAB implementation

