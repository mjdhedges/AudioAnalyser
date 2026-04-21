"""
Main entry point for the Music Analyser application.

This module provides the command-line interface for the music analysis tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from src.audio_processor import AudioProcessor
from src.music_analyzer import MusicAnalyzer
from src.octave_filter import OctaveBandFilter
from src.config import config, Config
from src.channel_mapping import get_channel_name, get_channel_folder_name
from src.track_processor import TrackProcessor
from src.post.worst_channels import select_worst_channels
from src.post.group_decay_plot import generate_group_decay_plot
from src.post.group_crest_factor_time import generate_group_crest_factor_time_plot
from src.post.group_octave_spectrum import generate_group_octave_spectrum_plot
from src.post.lfe_octave_time import generate_lfe_octave_time_plot, generate_lfe_full_channel_plot
from src.post.channel_deep_dive import generate_channel_deep_dive_plot


def determine_content_type(track_path: Path) -> str:
    """Determine content type based on folder structure.
    
    Args:
        track_path: Path to the audio file
        
    Returns:
        Content type string: 'Music', 'Film', or 'Test Signal'
    """
    # Get the parent directory name
    parent_dir = track_path.parent.name
    
    # Map folder names to content types
    content_type_mapping = {
        'Music': 'Music',
        'Film': 'Film', 
        'Test Signals': 'Test Signal'
    }
    
    # Return mapped content type or default to 'Unknown'
    return content_type_mapping.get(parent_dir, 'Unknown')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_cache_path(track_path: Path, cache_dir: Path) -> Path:
    """Generate cache file path for a track.
    
    Args:
        track_path: Path to the audio track
        cache_dir: Base cache directory
        
    Returns:
        Path to the cache file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use track name and hash for uniqueness
    track_hash = hash(str(track_path))
    cache_file = cache_dir / f"{track_path.stem}_{abs(track_hash)}.npy"
    return cache_file


def resolve_track_output_dir(
    output_dir: Path,
    track_path: Path,
    batch_tracks_root: Optional[Path],
) -> Path:
    """Resolve the folder where one track's analysis outputs are stored.

    In batch mode, mirror the directory layout under ``batch_tracks_root`` so that
    ``tracks/Music/a.flac`` becomes ``output_dir/Music/a`` (extension stripped).
    Single-file / non-batch uses a single folder named after the file stem.
    """
    if batch_tracks_root is not None:
        try:
            rel = track_path.resolve().relative_to(batch_tracks_root.resolve())
            return (output_dir / rel).with_suffix("")
        except ValueError:
            logger.warning(
                "Track path is not under batch tracks directory; using flat output folder: %s",
                track_path,
            )
    return output_dir / track_path.stem


def _is_channel_output_folder_name(name: str) -> bool:
    """Heuristic: channel outputs live under folders like ``Channel 1 Left`` or ``FC``."""
    u = name.upper()
    if u.startswith("CHANNEL "):
        return True
    if u in ("FC",):
        return True
    return False


def find_track_output_dirs(base: Path) -> list[Path]:
    """Find per-track analysis folders under ``base`` (flat or nested).

    Primary: directories that contain ``.cache_meta.json`` from a completed run.
    Fallback (e.g. caching disabled): infer from ``analysis_results.csv`` paths.
    """
    if not base.exists():
        return []
    roots = {p.parent for p in base.rglob(".cache_meta.json")}
    if roots:
        return sorted(roots)
    inferred: set[Path] = set()
    for csv_path in base.rglob("analysis_results.csv"):
        parent = csv_path.parent
        if _is_channel_output_folder_name(parent.name):
            inferred.add(parent.parent)
        else:
            inferred.add(parent)
    return sorted(inferred)


def get_config_hash() -> str:
    """Generate a hash of relevant configuration parameters.
    
    Returns:
        Hexadecimal hash string of config
    """
    # Get relevant config sections that affect analysis results
    relevant_config = {
        'sample_rate': config.get('analysis.sample_rate', 44100),
        'chunk_duration': config.get('analysis.chunk_duration_seconds', 2.0),
        'octave_frequencies': config.get('analysis.octave_center_frequencies', []),
        'filter_order': config.get('analysis.filter_order', 4),
        'use_linkwitz_riley': config.get('analysis.use_linkwitz_riley', True),
        'use_cascade_complementary': config.get('analysis.use_cascade_complementary', True),
        'normalize_overlap': config.get('analysis.normalize_overlap', False),
        'sustained_peaks_search_window_seconds': config.get('envelope_analysis.sustained_peaks_search_window_seconds', 5.0),
        'dpi': config.get('plotting.dpi', 300),
    }
    
    # Convert to JSON string and hash
    config_str = json.dumps(relevant_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def check_result_cache(
    track_path: Path,
    track_output_dir: Path,
    config_hash: str,
    use_cache: bool,
    require_octave_crest_factor_time: bool = True,
    require_octave_crest_factor_time_csv: bool = False,
) -> bool:
    """Check if cached analysis results are still valid.
    
    Args:
        track_path: Path to the audio track
        track_output_dir: Directory where this track's artifacts are stored
        config_hash: Hash of current configuration
        use_cache: Whether to use caching
        
    Returns:
        True if cache is valid and exists, False otherwise
    """
    if not use_cache:
        return False
    
    # Check if all required output files exist
    required_files = [
        'octave_spectrum.png',
        'crest_factor.png',
        'histograms.png',
        'histograms_log_db.png',
        'crest_factor_time.png',
        'analysis_results.csv'
    ]
    if require_octave_crest_factor_time:
        required_files.append('octave_crest_factor_time.png')
    if require_octave_crest_factor_time_csv:
        required_files.append('octave_crest_factor_time.csv')
    
    for filename in required_files:
        file_path = track_output_dir / filename
        if not file_path.exists():
            return False
    
    # Check if cache metadata exists
    cache_meta_path = track_output_dir / '.cache_meta.json'
    if not cache_meta_path.exists():
        return False
    
    # Load cache metadata
    try:
        with open(cache_meta_path, 'r') as f:
            cache_meta = json.load(f)
        
        # Check if source file has been modified
        if cache_meta.get('source_mtime', 0) != track_path.stat().st_mtime:
            return False
        
        # Check if config has changed
        if cache_meta.get('config_hash') != config_hash:
            return False
        
        # Check if all cached files are newer than source
        for filename in required_files:
            file_path = track_output_dir / filename
            if file_path.stat().st_mtime <= track_path.stat().st_mtime:
                return False
        
        return True
    except Exception as e:
        logger.warning(f"Error checking cache metadata: {e}")
        return False


def save_result_cache(track_path: Path, track_output_dir: Path, config_hash: str) -> None:
    """Save cache metadata for analysis results.
    
    Args:
        track_path: Path to the audio track
        track_output_dir: Directory where this track's artifacts are stored
        config_hash: Hash of current configuration
    """
    track_output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_meta = {
        'source_path': str(track_path),
        'source_mtime': track_path.stat().st_mtime,
        'config_hash': config_hash,
        'cache_date': pd.Timestamp.now().isoformat()
    }
    
    cache_meta_path = track_output_dir / '.cache_meta.json'
    try:
        with open(cache_meta_path, 'w') as f:
            json.dump(cache_meta, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache metadata: {e}")

def analyze_single_track(
    track_path: Path,
    output_dir: Path,
    sample_rate: int,
    chunk_duration: float,
    use_cache: bool = True,
    channel_filters: Optional[tuple[str, ...]] = None,
    batch_tracks_root: Optional[Path] = None,
    skip_octave_crest_factor_time: bool = False,
    export_octave_crest_factor_time_data: bool = False,
) -> tuple[bool, float]:
    """Analyze a single audio track, processing each channel separately.
    
    For mono files, processes single channel. For stereo/multi-channel files,
    processes each channel separately and stores results in channel-specific subfolders.
    
    Args:
        track_path: Path to the audio file
        output_dir: Base output directory
        batch_tracks_root: If set (batch mode), mirror input paths under ``output_dir``
        sample_rate: Sample rate for processing
        chunk_duration: Duration of analysis chunks in seconds
        use_cache: Whether to use result caching
        
    Returns:
        Tuple of (success, elapsed_seconds).
    """
    def _tokenize(values: list[str]) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            if not value:
                continue
            norm = value.lower().strip()
            if norm:
                tokens.add(norm)
                tokens.add(''.join(ch for ch in norm if ch.isalnum()))
        return tokens
    
    started = time.perf_counter()
    try:
        # Create track-specific output directory (mirror batch folder layout when applicable)
        track_name = track_path.stem  # Get filename without extension
        track_output_dir = resolve_track_output_dir(output_dir, track_path, batch_tracks_root)
        track_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analyzing: {track_path.name}")
        logger.info(f"Output directory: {track_output_dir}")
        
        # Check if result cache is valid
        enable_result_cache = use_cache and config.get('performance.enable_result_cache', True)
        config_hash = get_config_hash()
        
        if enable_result_cache and check_result_cache(
            track_path,
            track_output_dir,
            config_hash,
            use_cache,
            require_octave_crest_factor_time=(not skip_octave_crest_factor_time),
            require_octave_crest_factor_time_csv=export_octave_crest_factor_time_data,
        ):
            logger.info(f"Result cache valid - skipping analysis for {track_path.name}")
            return True, time.perf_counter() - started
        
        # Initialize components
        enable_mkv_support = config.get('mkv_support.enable', True)
        audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            enable_mkv_support=enable_mkv_support
        )
        
        # Use cascade complementary Linkwitz-Riley filters for optimal crest factor preservation
        use_linkwitz_riley = config.get('analysis.use_linkwitz_riley', True)
        use_cascade = config.get('analysis.use_cascade_complementary', True)
        normalize_overlap = config.get('analysis.normalize_overlap', False)
        
        octave_filter = OctaveBandFilter(
            sample_rate=sample_rate,
            use_linkwitz_riley=use_linkwitz_riley,
            use_cascade=use_cascade,
            normalize_overlap=normalize_overlap
        )
        
        if use_cascade:
            logger.info("Using cascade complementary filter approach for optimal crest factor preservation")
        
        # Load audio file (preserves multi-channel)
        logger.info("Loading audio file...")
        # Get test section parameters (for testing long files)
        test_start_time = config.get('analysis.test_start_time', None)
        test_duration = config.get('analysis.test_duration', None)
        if test_start_time is not None or test_duration is not None:
            logger.info(
                f"Test mode: Analyzing section from {test_start_time or 0:.2f}s "
                f"for {test_duration or 'full'} seconds"
            )
        audio_data, sr = audio_processor.load_audio(
            track_path, 
            start_time=test_start_time, 
            duration=test_duration
        )
        
        # Get audio info to determine channel count
        audio_info = audio_processor.get_audio_info(audio_data, sr)
        num_channels = audio_info["channels"]
        channel_layout = audio_info["channel_layout"]
        
        logger.info(f"Audio info: {channel_layout}, {num_channels} channel(s), "
                   f"Duration={audio_info['duration_seconds']:.2f}s")
        
        # Store original peak level before normalization (across all channels)
        original_peak = np.max(np.abs(audio_data))
        
        # Extract channels
        channels = audio_processor.extract_channels(audio_data)
        
        # Determine content type
        content_type = determine_content_type(track_path)
        
        # Initialize track processor
        track_processor = TrackProcessor(sample_rate=sample_rate, original_peak=original_peak)
        
        # Prepare channel filters (if any)
        filter_specs = []
        if channel_filters:
            for raw in channel_filters:
                norm = raw.lower().strip()
                forms = {norm, ''.join(ch for ch in norm if ch.isalnum())}
                filter_specs.append({"raw": raw, "forms": forms})
        
        matched_filters: set[str] = set()
        
        # Process each channel separately
        success_count = 0
        processed_total = 0
        for channel_data, channel_index in channels:
            # Get channel name (simple name for CSV) and folder name (for directory)
            # Use FFmpeg channel layout if available for correct mapping
            channel_name = get_channel_name(channel_index, num_channels, channel_layout)
            channel_folder_name = get_channel_folder_name(channel_index, num_channels, channel_layout)
            
            # Create channel-specific output directory
            if num_channels == 1:
                # Mono: use track directory directly (backward compatibility)
                channel_output_dir = track_output_dir
            else:
                # Multi-channel: create channel subfolder
                channel_output_dir = track_output_dir / channel_folder_name
                channel_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Apply channel filters if provided
            if filter_specs:
                candidate_tokens = _tokenize([
                    channel_name,
                    channel_folder_name,
                    f"channel {channel_index + 1}",
                    f"channel{channel_index + 1}",
                    f"ch{channel_index + 1}",
                    str(channel_index + 1),
                ])
                
                matched = False
                for spec in filter_specs:
                    if spec["forms"] & candidate_tokens:
                        matched = True
                        matched_filters.add(spec["raw"])
                if not matched:
                    logger.info(
                        f"Skipping channel {channel_name} ({channel_folder_name}) due to channel filter"
                    )
                    continue
            
            processed_total += 1
            
            # Process this channel
            success = track_processor.process_channel(
                channel_data=channel_data,
                channel_index=channel_index,
                channel_name=channel_name,
                channel_folder_name=channel_folder_name,
                total_channels=num_channels,
                track_path=track_path,
                track_output_dir=track_output_dir,
                channel_output_dir=channel_output_dir,
                audio_processor=audio_processor,
                octave_filter=octave_filter,
                chunk_duration=chunk_duration,
                config=config,
                content_type=content_type,
                original_peak=original_peak,
                skip_octave_crest_factor_time=skip_octave_crest_factor_time,
                export_octave_crest_factor_time_data=export_octave_crest_factor_time_data,
            )
            
            if success:
                success_count += 1
        
        if filter_specs:
            unmatched = {spec["raw"] for spec in filter_specs if spec["raw"] not in matched_filters}
            if unmatched:
                logger.warning(
                    f"Channel filters not matched for track {track_name}: {', '.join(sorted(unmatched))}"
                )
        
        # Memory cleanup
        del audio_data
        
        if processed_total == 0:
            logger.warning("No channels were processed (filters omitted all channels).")
            return False, time.perf_counter() - started
        
        if success_count == processed_total:
            logger.info(
                f"Analysis complete for {track_path.name} "
                f"({success_count} channel(s) processed)"
            )
            
            # Save cache metadata
            if enable_result_cache:
                save_result_cache(track_path, track_output_dir, config_hash)
            
            return True, time.perf_counter() - started
        else:
            logger.warning(
                f"Analysis partially complete: {success_count}/{processed_total} "
                f"channel(s) processed successfully"
            )
            return False, time.perf_counter() - started
        
    except Exception as e:
        logger.error(f"Error analyzing {track_path.name}: {e}")
        return False, time.perf_counter() - started


@click.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True, path_type=Path),
    help='Input path: audio file (single) OR directory (batch recurse)'
)
@click.option(
    '--tracks-dir', '-t',
    type=click.Path(exists=True, path_type=Path),
    help='Directory containing tracks to analyze (default: from config)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for results (default: from config)'
)
@click.option(
    '--sample-rate', '-sr',
    type=int,
    help='Sample rate for audio processing (overrides config)'
)
@click.option(
    '--chunk-duration', '-cd',
    type=float,
    help='Duration of analysis chunks in seconds (overrides config)'
)
@click.option(
    '--dpi', '-d',
    type=int,
    help='DPI for plot output (overrides config)'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    help='Logging level (overrides config)'
)
@click.option(
    '--config', '-c',
    'config_path',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file (default: config.toml)'
)
@click.option(
    '--batch/--single',
    default=True,
    help='DEPRECATED: Mode is inferred from --input (file=single, dir=batch).'
)
@click.option(
    '--export-csv/--no-export-csv',
    default=True,
    help='Export results to CSV'
)
@click.option(
    '--generate-manifest/--no-generate-manifest',
    default=True,
    help='Generate worst-channel manifest per track after analysis'
)
@click.option(
    '--generate-decay-plot/--no-generate-decay-plot',
    default=True,
    help='Generate combined group decay plot per track after analysis'
)
@click.option(
    '--run-group-plots/--no-run-group-plots',
    default=True,
    help='Generate group crest-factor and octave spectrum plots'
)
@click.option(
    '--run-lfe-deep-dive/--no-run-lfe-deep-dive',
    default=True,
    help='Generate LFE deep dive plots'
)
@click.option(
    '--run-screen-deep-dive/--no-run-screen-deep-dive',
    default=True,
    help='Generate Screen deep dive plots'
)
@click.option(
    '--run-surround-deep-dive/--no-run-surround-deep-dive',
    default=True,
    help='Generate Surround+Height deep dive plots'
)
@click.option(
    '--post-only',
    is_flag=True,
    default=False,
    help='Skip main analysis and only run post-processing on existing results'
)
@click.option(
    '--test-start-time',
    type=float,
    default=None,
    help='Start time in seconds for test mode (analyze only a section of the file)'
)
@click.option(
    '--test-duration',
    type=float,
    default=None,
    help='Duration in seconds for test mode (limit analysis length)'
)
@click.option(
    '--channel',
    'channel_filters',
    multiple=True,
    help='Only process specified channel names (e.g., FL, "Channel 4 LFE").'
)
@click.option(
    '--skip-post/--run-post',
    default=False,
    help='Skip post-processing (group plots, deep dives, manifests).'
)
@click.option(
    '--skip-octave-cf-time/--run-octave-cf-time',
    default=None,
    help='Skip generating octave_crest_factor_time.png (saves significant time).'
)
@click.option(
    '--export-octave-cf-time-data/--no-export-octave-cf-time-data',
    default=None,
    help='Export octave_crest_factor_time.csv (time-series) for later processing.'
)
@click.option(
    '--peak-hold-tau',
    type=float,
    help='Peak-hold time constant in seconds (overrides config).'
)
def main(input: Optional[Path], tracks_dir: Optional[Path], output_dir: Optional[Path], 
         sample_rate: Optional[int], chunk_duration: Optional[float],
         dpi: Optional[int], log_level: Optional[str], config_path: Optional[Path],
         batch: bool, export_csv: bool,
         generate_manifest: bool, generate_decay_plot: bool,
         run_group_plots: bool, run_lfe_deep_dive: bool,
         run_screen_deep_dive: bool, run_surround_deep_dive: bool,
         post_only: bool,
         test_start_time: Optional[float], test_duration: Optional[float],
         channel_filters: tuple[str, ...], skip_post: bool,
         skip_octave_cf_time: bool,
         export_octave_cf_time_data: bool,
         peak_hold_tau: Optional[float]) -> None:
    """Music Analyser - Analyze audio files using octave band filtering.
    
    This tool performs comprehensive octave band analysis on audio files,
    similar to the MATLAB musicanalyser.m script. It generates frequency
    analysis, statistical measures, and visualizations.
    
    Examples:
        # Analyze all tracks in Tracks directory
        python -m src.main
        
        # Analyze a single file
        python -m src.main --input "song.flac" --single
        
        # Analyze tracks in custom directory
        python -m src.main --tracks-dir "MyMusic" --output-dir "results"
    """
    try:
        # Load configuration
        global config
        if config_path:
            config = Config(config_path)
        
        # Override configuration with command line arguments
        config.override_from_args(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            dpi=dpi,
            log_level=log_level,
            test_start_time=test_start_time,
            test_duration=test_duration,
            peak_hold_tau=peak_hold_tau,
        )
        
        if skip_post and post_only:
            logger.error("--skip-post cannot be combined with --post-only mode.")
            sys.exit(1)
        
        # Update logging level if specified
        if log_level:
            logging.getLogger().setLevel(getattr(logging, log_level))
        
        # Get configuration values
        sample_rate = config.get('analysis.sample_rate', 44100)
        chunk_duration = config.get('analysis.chunk_duration_seconds', 2.0)

        if chunk_duration <= 0:
            logger.error(f"Invalid chunk duration: {chunk_duration}. Must be > 0.")
            sys.exit(1)
        
        # Use config defaults if CLI options not provided
        if tracks_dir is None:
            tracks_dir = Path(config.get('analysis.tracks_dir', 'Tracks'))
        if output_dir is None:
            output_dir = Path(config.get('analysis.output_dir', 'analysis'))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Music Analyser")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Sample rate: {sample_rate} Hz")
        logger.info(f"Chunk duration: {chunk_duration} seconds")

        # Config-driven defaults for optional expensive artifacts.
        if skip_octave_cf_time is None:
            skip_octave_cf_time = bool(config.get("performance.skip_octave_cf_time", False))
        if export_octave_cf_time_data is None:
            export_octave_cf_time_data = bool(config.get("performance.export_octave_cf_time_data", False))

        # Helper: run worst-channel manifest via internal function
        def _run_select_worst_channels(track_dir: Path) -> bool:
            try:
                select_worst_channels(track_dir)
                logger.info(f"Worst-channel manifest generated for {track_dir.name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to generate worst-channel manifest for {track_dir.name}: {e}")
                return False

        # Helper: run combined decay plot via internal function
        def _run_group_decay_plot(track_dir: Path) -> bool:
            try:
                output_img = track_dir / "peak_decay_groups.png"
                generate_group_decay_plot(track_dir, output_img)
                logger.info(f"Group decay plot generated for {track_dir.name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to generate decay plot for {track_dir.name}: {e}")
                return False

        # Helper: run group crest factor time plots via internal function
        def _run_group_crest_factor_time_plot(track_dir: Path) -> bool:
            try:
                generate_group_crest_factor_time_plot(track_dir, track_dir)
                logger.info(f"Group crest factor time plots generated for {track_dir.name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to generate group crest factor time plots for {track_dir.name}: {e}")
                return False

        # Helper: run group octave spectrum plots via internal function
        def _run_group_octave_spectrum_plot(track_dir: Path) -> bool:
            try:
                generate_group_octave_spectrum_plot(track_dir, track_dir)
                logger.info(f"Group octave spectrum plots generated for {track_dir.name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to generate group octave spectrum plots for {track_dir.name}: {e}")
                return False
        
        def _run_lfe_full_channel_plot(track_dir: Path) -> bool:
            try:
                output_path = track_dir / "LFE" / "lfe_full_channel.png"
                result = generate_lfe_full_channel_plot(track_dir, output_path)
                if result:
                    logger.info(f"LFE full channel plot generated for {track_dir.name}")
                    return True
                return False
            except Exception as e:
                logger.warning(f"Failed to generate LFE full channel plot for {track_dir.name}: {e}")
                return False
        
        def _run_lfe_octave_time_plot(track_dir: Path, original_track_path: Optional[Path] = None) -> bool:
            try:
                output_path = track_dir / "LFE" / "lfe_octave_time.png"
                result = generate_lfe_octave_time_plot(track_dir, output_path, original_track_path)
                if result:
                    logger.info(f"LFE octave band time plot generated for {track_dir.name}")
                    return True
                return False
            except Exception as e:
                logger.warning(f"Failed to generate LFE octave band time plot for {track_dir.name}: {e}")
                return False
        
        def _run_screen_deep_dive_plot(track_dir: Path, original_track_path: Optional[Path] = None) -> bool:
            try:
                output_dir = track_dir / "Screen"
                result = generate_channel_deep_dive_plot(track_dir, "Screen", output_dir, original_track_path)
                if result:
                    logger.info(f"Screen deep dive plots generated for {track_dir.name}")
                    return True
                return False
            except Exception as e:
                logger.warning(f"Failed to generate Screen deep dive plots for {track_dir.name}: {e}")
                return False
        
        def _run_surround_deep_dive_plot(track_dir: Path, original_track_path: Optional[Path] = None) -> bool:
            try:
                output_dir = track_dir / "Surround+Height"
                result = generate_channel_deep_dive_plot(track_dir, "Surround+Height", output_dir, original_track_path)
                if result:
                    logger.info(f"Surround+Height deep dive plots generated for {track_dir.name}")
                    return True
                return False
            except Exception as e:
                logger.warning(f"Failed to generate Surround+Height deep dive plots for {track_dir.name}: {e}")
                return False

        # Post-only mode: skip analysis and just run post-processing
        if post_only:
            logger.info("Post-only mode: running post-processing on existing results")
            for td in find_track_output_dirs(output_dir):
                if generate_manifest:
                    _run_select_worst_channels(td)
                if generate_decay_plot:
                    _run_group_decay_plot(td)
                if run_group_plots:
                    _run_group_crest_factor_time_plot(td)
                    _run_group_octave_spectrum_plot(td)
                if run_lfe_deep_dive:
                    _run_lfe_full_channel_plot(td)
                    _run_lfe_octave_time_plot(td)
                if run_screen_deep_dive:
                    _run_screen_deep_dive_plot(td)
                if run_surround_deep_dive:
                    _run_surround_deep_dive_plot(td)
            logger.info("Post-processing complete.")
            return
        
        # ------------------------------------------------------------------
        # Mode selection (simple path-based behavior)
        # ------------------------------------------------------------------
        # If --input is provided:
        #   - file  => single-file processing
        #   - dir   => batch processing over that directory (recursive)
        # Otherwise:
        #   - fall back to --tracks-dir (or config default) in batch mode.
        if input is not None:
            if input.is_dir():
                tracks_dir = input
                input = None
                batch = True
            else:
                batch = False

        if batch:
            # Batch processing - analyze all tracks in directory
            logger.info(f"Batch processing tracks from: {tracks_dir}")
            batch_started = time.perf_counter()
            
            # Find all audio files
            audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.mkv'}
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(tracks_dir.glob(f"**/*{ext}"))
                audio_files.extend(tracks_dir.glob(f"**/*{ext.upper()}"))
            
            # On Windows, paths are case-insensitive: *.flac and *.FLAC globs both match the same
            # files, doubling the list and scheduling duplicate ProcessPool jobs for one track.
            raw_count = len(audio_files)
            unique_by_resolve: dict[Path, Path] = {}
            for p in audio_files:
                key = p.resolve()
                if key not in unique_by_resolve:
                    unique_by_resolve[key] = p
            audio_files = sorted(
                unique_by_resolve.values(),
                key=lambda x: str(x).casefold(),
            )
            if raw_count > len(audio_files):
                logger.info(
                    "Deduplicated track list: %s paths → %s unique files "
                    "(case-insensitive extension globs on this OS can list the same file twice)",
                    raw_count,
                    len(audio_files),
                )
            
            if not audio_files:
                logger.error(f"No audio files found in {tracks_dir}")
                logger.info(f"Supported formats: {', '.join(audio_extensions)}")
                sys.exit(1)
            
            logger.info(f"Found {len(audio_files)} audio files to analyze")
            
            # Process tracks (parallel or sequential based on config)
            successful_analyses = 0
            failed_analyses = 0
            total_tracks = len(audio_files)
            per_track_timings: list[tuple[Path, float, bool]] = []
            
            enable_parallel_batch = config.get('performance.enable_parallel_batch', False)
            
            if enable_parallel_batch:
                # PARALLEL PROCESSING: Process multiple tracks simultaneously
                from concurrent.futures import ProcessPoolExecutor, as_completed
                import multiprocessing
                
                max_workers = config.get('performance.max_batch_workers', None)
                available_cores = multiprocessing.cpu_count()
                
                if max_workers is None:
                    max_workers = 2  # Default: 2 workers to keep memory reasonable
                
                logger.info(f"Using parallel batch processing with {max_workers} workers")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_track = {
                        executor.submit(
                            analyze_single_track,
                            track_path,
                            output_dir,
                            sample_rate,
                            chunk_duration,
                            channel_filters=channel_filters,
                            batch_tracks_root=tracks_dir,
                            skip_octave_crest_factor_time=skip_octave_cf_time,
                            export_octave_crest_factor_time_data=export_octave_cf_time_data,
                        ): (idx, track_path, total_tracks)
                        for idx, track_path in enumerate(sorted(audio_files), 1)
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_track):
                        idx, track_path, total_tracks = future_to_track[future]
                        try:
                            success, elapsed_s = future.result(timeout=600)  # 10 min timeout per track
                            per_track_timings.append((track_path, elapsed_s, success))
                            if success:
                                successful_analyses += 1
                                logger.info(f"[{idx}/{total_tracks}] ✓ {track_path.name}")
                            else:
                                failed_analyses += 1
                                logger.error(f"[{idx}/{total_tracks}] ✗ {track_path.name}")
                        except Exception as e:
                            failed_analyses += 1
                            logger.error(f"[{idx}/{total_tracks}] ✗ {track_path.name}: {e}")
                            per_track_timings.append((track_path, float("nan"), False))
            else:
                # SEQUENTIAL PROCESSING: Process tracks one at a time
                for idx, track_path in enumerate(sorted(audio_files), 1):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Processing track {idx}/{total_tracks} ({100*idx/total_tracks:.1f}%) - {track_path.name}")
                    
                    success, elapsed_s = analyze_single_track(
                        track_path,
                        output_dir,
                        sample_rate,
                        chunk_duration,
                        channel_filters=channel_filters,
                        batch_tracks_root=tracks_dir,
                        skip_octave_crest_factor_time=skip_octave_cf_time,
                        export_octave_crest_factor_time_data=export_octave_cf_time_data,
                    )
                    per_track_timings.append((track_path, elapsed_s, success))
                    if success:
                        successful_analyses += 1
                    else:
                        failed_analyses += 1
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch analysis complete!")
            logger.info(f"Successfully analyzed: {successful_analyses} tracks")
            logger.info(f"Failed analyses: {failed_analyses} tracks")
            logger.info(f"Results saved to: {output_dir}")
            batch_elapsed_s = time.perf_counter() - batch_started
            logger.info(f"Total batch wall time: {batch_elapsed_s:.2f}s")
            if per_track_timings:
                # Slowest-first; NaNs go last.
                per_track_timings_sorted = sorted(
                    per_track_timings,
                    key=lambda x: (x[1] == x[1], x[1]),  # NaN sorts after numbers
                    reverse=True,
                )
                logger.info("Per-track wall time (slowest first):")
                for p, elapsed_s, success in per_track_timings_sorted:
                    status = "OK" if success else "FAIL"
                    if elapsed_s == elapsed_s:
                        logger.info(f"  {elapsed_s:8.2f}s  [{status}]  {p.name}")
                    else:
                        logger.info(f"       n/a  [{status}]  {p.name}")
            # Post-process generated track folders (if enabled)
            if not skip_post:
                logger.info("Starting post-processing for generated tracks...")
                for td in find_track_output_dirs(output_dir):
                    if generate_manifest:
                        _run_select_worst_channels(td)
                    if generate_decay_plot:
                        _run_group_decay_plot(td)
                    if run_group_plots:
                        _run_group_crest_factor_time_plot(td)
                        _run_group_octave_spectrum_plot(td)
                    if run_lfe_deep_dive:
                        _run_lfe_full_channel_plot(td)
                        _run_lfe_octave_time_plot(td)
                    if run_screen_deep_dive:
                        _run_screen_deep_dive_plot(td)
                    if run_surround_deep_dive:
                        _run_surround_deep_dive_plot(td)
                logger.info("Post-processing complete.")
            elif skip_post:
                logger.info("Skipping post-processing (--skip-post).")
            
        else:
            # Single file processing
            if not input:
                logger.error("--input must be a file path for single-file analysis")
                sys.exit(1)
            
            logger.info(f"Single file analysis: {input}")
            
            success, elapsed_s = analyze_single_track(
                input,
                output_dir,
                sample_rate,
                chunk_duration,
                channel_filters=channel_filters,
                batch_tracks_root=None,
                skip_octave_crest_factor_time=skip_octave_cf_time,
                export_octave_crest_factor_time_data=export_octave_cf_time_data,
            )
            if success:
                logger.info("Analysis complete!")
                logger.info(f"Track wall time: {elapsed_s:.2f}s")
                logger.info(f"Results saved to: {output_dir}")
                # Post-process this track folder (if enabled)
                if not skip_post:
                    track_dir = resolve_track_output_dir(output_dir, input, None)
                    if track_dir.exists():
                        if generate_manifest:
                            _run_select_worst_channels(track_dir)
                        if generate_decay_plot:
                            _run_group_decay_plot(track_dir)
                        if run_group_plots:
                            _run_group_crest_factor_time_plot(track_dir)
                            _run_group_octave_spectrum_plot(track_dir)
                        if run_lfe_deep_dive:
                            _run_lfe_full_channel_plot(track_dir)
                            _run_lfe_octave_time_plot(track_dir, original_track_path=input)
                        if run_screen_deep_dive:
                            _run_screen_deep_dive_plot(track_dir, original_track_path=input)
                        if run_surround_deep_dive:
                            _run_surround_deep_dive_plot(track_dir, original_track_path=input)
                    else:
                        logger.warning(
                            f"Expected track directory not found for post-processing: {track_dir}"
                        )
                elif skip_post:
                    logger.info("Skipping post-processing (--skip-post).")
            else:
                logger.error("Analysis failed!")
                logger.info(f"Track wall time: {elapsed_s:.2f}s")
                sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
