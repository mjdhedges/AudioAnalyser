"""
Main entry point for the Music Analyser application.

This module provides the command-line interface for the music analysis tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from src.audio_processor import AudioProcessor
from src.music_analyzer import MusicAnalyzer
from src.octave_filter import OctaveBandFilter
from src.config import config, Config


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
        'dpi': config.get('plotting.dpi', 300),
    }
    
    # Convert to JSON string and hash
    config_str = json.dumps(relevant_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def check_result_cache(track_path: Path, output_dir: Path, 
                      config_hash: str, use_cache: bool) -> bool:
    """Check if cached analysis results are still valid.
    
    Args:
        track_path: Path to the audio track
        output_dir: Output directory for results
        config_hash: Hash of current configuration
        use_cache: Whether to use caching
        
    Returns:
        True if cache is valid and exists, False otherwise
    """
    if not use_cache:
        return False
    
    track_name = track_path.stem
    track_output_dir = output_dir / track_name
    
    # Check if all required output files exist
    required_files = [
        'octave_spectrum.png',
        'crest_factor.png',
        'histograms.png',
        'histograms_log_db.png',
        'crest_factor_time.png',
        'octave_crest_factor_time.png',
        'analysis_results.csv'
    ]
    
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


def save_result_cache(track_path: Path, output_dir: Path, config_hash: str) -> None:
    """Save cache metadata for analysis results.
    
    Args:
        track_path: Path to the audio track
        output_dir: Output directory for results
        config_hash: Hash of current configuration
    """
    track_name = track_path.stem
    track_output_dir = output_dir / track_name
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


def analyze_single_track(track_path: Path, output_dir: Path, sample_rate: int, chunk_duration: float, 
                         use_cache: bool = True) -> bool:
    """Analyze a single audio track.
    
    Args:
        track_path: Path to the audio file
        output_dir: Base output directory
        sample_rate: Sample rate for processing
        chunk_duration: Duration of analysis chunks in seconds
        use_cache: Whether to use result caching
        
    Returns:
        True if analysis was successful, False otherwise
    """
    try:
        # Create track-specific output directory
        track_name = track_path.stem  # Get filename without extension
        track_output_dir = output_dir / track_name
        track_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analyzing: {track_path.name}")
        logger.info(f"Output directory: {track_output_dir}")
        
        # Check if result cache is valid
        enable_result_cache = use_cache and config.get('performance.enable_result_cache', True)
        config_hash = get_config_hash()
        
        if enable_result_cache and check_result_cache(track_path, output_dir, config_hash, use_cache):
            logger.info(f"Result cache valid - skipping analysis for {track_path.name}")
            return True
        
        # Initialize components
        audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Use cascade complementary Linkwitz-Riley filters for optimal crest factor preservation
        # This approach ensures non-overlapping bands, linear phase, and better phase coherence
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
        
        # Load and preprocess audio (now uses float32 by default for 50% memory savings)
        logger.info("Loading audio file...")
        audio_data, sr = audio_processor.load_audio(track_path)
        
        # Convert to mono if stereo
        audio_data = audio_processor.stereo_to_mono(audio_data)
        
        # Store original peak level before normalization for dBFS calculation
        original_peak = np.max(np.abs(audio_data))
        
        # Initialize analyzer with original peak for dBFS calculations
        # Use lower DPI for batch processing if enabled
        use_batch_dpi = config.get('performance.enable_parallel_batch', False) or \
                       config.get('performance.enable_result_cache', False)
        plot_dpi = config.get('plotting.batch_dpi', 150) if use_batch_dpi else \
                   config.get('plotting.dpi', 300)
        
        analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=original_peak, dpi=plot_dpi)
        
        # Normalize audio
        audio_data = audio_processor.normalize_audio(audio_data)
        
        # Get audio info
        audio_info = audio_processor.get_audio_info(audio_data, sr)
        logger.info(f"Audio info: Duration={audio_info['duration_seconds']:.2f}s, "
                   f"RMS={audio_info['rms']:.4f}, Max={audio_info['max_amplitude']:.4f}")
        
        # Try to load cached octave bank or create new one
        # Note: Cache is invalidated when filter settings change (use_cascade, use_linkwitz_riley)
        # This ensures cached banks match current filter configuration
        enable_cache = use_cache and config.get('performance.enable_octave_cache', True)
        cache_dir = Path(config.get('performance.cache_dir', 'cache/octave_banks'))
        
        if enable_cache:
            # Include filter settings in cache key to invalidate when settings change
            filter_config_hash = f"lr_{use_linkwitz_riley}_cascade_{use_cascade}_norm_{normalize_overlap}"
            cache_file_base = get_cache_path(track_path, cache_dir)
            cache_file = cache_file_base.parent / f"{cache_file_base.stem}_{filter_config_hash}{cache_file_base.suffix}"
            
            # Check if cache exists and is valid
            if cache_file.exists():
                # Check if source file hasn't been modified since caching
                if cache_file.stat().st_mtime > track_path.stat().st_mtime:
                    try:
                        logger.info(f"Loading cached octave bank: {cache_file.name}")
                        octave_bank = np.load(cache_file, mmap_mode='r')  # Memory-mapped for efficiency
                        logger.info(f"Loaded octave bank from cache ({octave_bank.shape[1]} bands)")
                    except Exception as e:
                        logger.warning(f"Failed to load cache: {e}, creating new octave bank")
                        octave_bank = None
                else:
                    logger.info("Cache expired (source file modified), creating new octave bank")
                    octave_bank = None
            else:
                logger.debug(f"Cache not found for filter config {filter_config_hash}, creating new octave bank")
                octave_bank = None
        else:
            octave_bank = None
        
        # Create octave bank if not loaded from cache
        if octave_bank is None:
            import multiprocessing
            parallel_enabled = config.get('performance.enable_parallel_processing', False)
            
            # Note: Cascade complementary approach is sequential by design and cannot be parallelized
            # Parallel processing only applies to parallel filter bank approach
            if parallel_enabled and not use_cascade:
                logger.info("Creating octave bank with parallel processing...")
                num_workers = config.get('performance.max_workers', None)
                octave_bank = octave_filter.create_octave_bank_parallel(audio_data, num_workers=num_workers)
            else:
                if parallel_enabled and use_cascade:
                    logger.info("Parallel processing disabled for cascade complementary approach (sequential by design)")
                logger.info("Creating octave bank...")
                octave_bank = octave_filter.create_octave_bank(audio_data)
            
            # Save to cache (cache_file already includes filter config hash)
            if enable_cache:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_file, octave_bank)
                    logger.info(f"Cached octave bank to: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to cache octave bank: {e}")
        
        # Perform comprehensive analysis (efficient - runs octave analysis only once)
        logger.info("Performing comprehensive analysis...")
        comprehensive_results = analyzer.analyze_comprehensive(
            audio_data, 
            octave_bank, 
            octave_filter.OCTAVE_CENTER_FREQUENCIES,
            chunk_duration=chunk_duration
        )
        
        # Extract results for backward compatibility
        analysis_results = comprehensive_results["main_analysis"]
        time_analysis = comprehensive_results["time_analysis"]
        chunk_octave_analysis = comprehensive_results["chunk_octave_analysis"]
        
        # Generate plots
        logger.info("Generating plots...")
        analyzer.create_octave_spectrum_plot(
            analysis_results,
            output_path=str(track_output_dir / "octave_spectrum.png"),
            time_analysis=time_analysis,
            chunk_octave_analysis=chunk_octave_analysis
        )
        analyzer.create_crest_factor_plot(
            analysis_results,
            output_path=str(track_output_dir / "crest_factor.png"),
            time_analysis=time_analysis,
            chunk_octave_analysis=chunk_octave_analysis
        )
        analyzer.create_histogram_plots(
            analysis_results,
            output_dir=str(track_output_dir),
            octave_bank=octave_bank
        )
        analyzer.create_histogram_plots_log_db(
            analysis_results,
            output_dir=str(track_output_dir),
            config=config.get_plotting_config(),
            octave_bank=octave_bank
        )
        analyzer.create_crest_factor_time_plot(
            time_analysis,
            output_path=str(track_output_dir / "crest_factor_time.png")
        )
        analyzer.create_octave_crest_factor_time_plot(
            octave_bank,
            time_analysis,
            octave_filter.OCTAVE_CENTER_FREQUENCIES,
            output_path=str(track_output_dir / "octave_crest_factor_time.png")
        )
        
        # Envelope statistics analysis (worst-case and pattern detection)
        logger.info("Performing envelope statistics analysis...")
        envelope_stats = analyzer.analyze_envelope_statistics(
            octave_bank,
            octave_filter.OCTAVE_CENTER_FREQUENCIES,
            config=config.get('envelope_analysis', {})
        )
        
        # Create envelope visualization plots
        logger.info("Creating envelope visualization plots...")
        analyzer.create_pattern_envelope_plots(
            envelope_stats,
            octave_filter.OCTAVE_CENTER_FREQUENCIES,
            output_dir=str(track_output_dir),
            config=config.get('envelope_analysis', {})
        )
        analyzer.create_independent_envelope_plots(
            envelope_stats,
            octave_filter.OCTAVE_CENTER_FREQUENCIES,
            output_dir=str(track_output_dir),
            config=config.get('envelope_analysis', {})
        )
        
        # MEMORY CLEANUP: Close all matplotlib figures and free memory immediately
        import matplotlib.pyplot as plt
        import gc
        plt.close('all')  # Close all figures
        gc.collect()  # Force garbage collection
        
        # Export results to CSV
        logger.info("Exporting results to CSV...")
        
        # Determine content type based on folder structure
        content_type = determine_content_type(track_path)
        
        # Prepare comprehensive export data
        track_metadata = {
            "track_name": track_path.name,
            "track_path": str(track_path),
            "content_type": content_type,
            "duration_seconds": audio_info["duration_seconds"],
            "sample_rate": sr,
            "samples": len(audio_data),
            "channels": audio_info.get("channels", 1),
            "original_peak": original_peak,
            "original_peak_dbfs": 20 * np.log10(original_peak),
            "analysis_date": pd.Timestamp.now().isoformat()
        }
        
        analyzer.export_comprehensive_results(
            analysis_results,
            time_analysis,
            track_metadata,
            str(track_output_dir / "analysis_results.csv"),
            chunk_octave_analysis=chunk_octave_analysis,
            audio_data=audio_data,
            envelope_statistics=envelope_stats
        )
        
        # Memory cleanup - explicitly delete large arrays to reduce memory footprint
        del audio_data, octave_bank, comprehensive_results
        logger.info(f"Analysis complete for {track_path.name}")
        
        # Save cache metadata
        if enable_result_cache:
            save_result_cache(track_path, output_dir, config_hash)
        
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing {track_path.name}: {e}")
        return False


@click.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True, path_type=Path),
    help='Input audio file path (for single file analysis)'
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
    help='Process all tracks in directory (batch) or single file'
)
@click.option(
    '--export-csv/--no-export-csv',
    default=True,
    help='Export results to CSV'
)
def main(input: Optional[Path], tracks_dir: Optional[Path], output_dir: Optional[Path], 
         sample_rate: Optional[int], chunk_duration: Optional[float],
         dpi: Optional[int], log_level: Optional[str], config_path: Optional[Path],
         batch: bool, export_csv: bool) -> None:
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
            log_level=log_level
        )
        
        # Update logging level if specified
        if log_level:
            logging.getLogger().setLevel(getattr(logging, log_level))
        
        # Get configuration values
        sample_rate = config.get('analysis.sample_rate', 44100)
        chunk_duration = config.get('analysis.chunk_duration_seconds', 2.0)
        
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
        
        if batch:
            # Batch processing - analyze all tracks in directory
            logger.info(f"Batch processing tracks from: {tracks_dir}")
            
            # Find all audio files
            audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg'}
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(tracks_dir.glob(f"**/*{ext}"))
                audio_files.extend(tracks_dir.glob(f"**/*{ext.upper()}"))
            
            if not audio_files:
                logger.error(f"No audio files found in {tracks_dir}")
                logger.info(f"Supported formats: {', '.join(audio_extensions)}")
                sys.exit(1)
            
            logger.info(f"Found {len(audio_files)} audio files to analyze")
            
            # Process tracks (parallel or sequential based on config)
            successful_analyses = 0
            failed_analyses = 0
            total_tracks = len(audio_files)
            
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
                            analyze_single_track, track_path, output_dir, sample_rate, chunk_duration
                        ): (idx, track_path, total_tracks)
                        for idx, track_path in enumerate(sorted(audio_files), 1)
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_track):
                        idx, track_path, total_tracks = future_to_track[future]
                        try:
                            result = future.result(timeout=600)  # 10 min timeout per track
                            if result:
                                successful_analyses += 1
                                logger.info(f"[{idx}/{total_tracks}] ✓ {track_path.name}")
                            else:
                                failed_analyses += 1
                                logger.error(f"[{idx}/{total_tracks}] ✗ {track_path.name}")
                        except Exception as e:
                            failed_analyses += 1
                            logger.error(f"[{idx}/{total_tracks}] ✗ {track_path.name}: {e}")
            else:
                # SEQUENTIAL PROCESSING: Process tracks one at a time
                for idx, track_path in enumerate(sorted(audio_files), 1):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Processing track {idx}/{total_tracks} ({100*idx/total_tracks:.1f}%) - {track_path.name}")
                    
                    if analyze_single_track(track_path, output_dir, sample_rate, chunk_duration):
                        successful_analyses += 1
                    else:
                        failed_analyses += 1
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch analysis complete!")
            logger.info(f"Successfully analyzed: {successful_analyses} tracks")
            logger.info(f"Failed analyses: {failed_analyses} tracks")
            logger.info(f"Results saved to: {output_dir}")
            
        else:
            # Single file processing
            if not input:
                logger.error("--input is required when using --single mode")
                sys.exit(1)
            
            logger.info(f"Single file analysis: {input}")
            
            if analyze_single_track(input, output_dir, sample_rate, chunk_duration):
                logger.info("Analysis complete!")
                logger.info(f"Results saved to: {output_dir}")
            else:
                logger.error("Analysis failed!")
                sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
