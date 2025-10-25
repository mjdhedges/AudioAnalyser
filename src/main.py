"""
Main entry point for the Music Analyser application.

This module provides the command-line interface for the music analysis tool.
"""

from __future__ import annotations

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


def analyze_single_track(track_path: Path, output_dir: Path, sample_rate: int, chunk_duration: float) -> bool:
    """Analyze a single audio track.
    
    Args:
        track_path: Path to the audio file
        output_dir: Base output directory
        sample_rate: Sample rate for processing
        
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
        
        # Initialize components
        audio_processor = AudioProcessor(sample_rate=sample_rate)
        octave_filter = OctaveBandFilter(sample_rate=sample_rate)
        
        # Load and preprocess audio
        logger.info("Loading audio file...")
        audio_data, sr = audio_processor.load_audio(track_path)
        
        # Convert to mono if stereo
        audio_data = audio_processor.stereo_to_mono(audio_data)
        
        # Store original peak level before normalization for dBFS calculation
        original_peak = np.max(np.abs(audio_data))
        
        # Initialize analyzer with original peak for dBFS calculations
        analyzer = MusicAnalyzer(sample_rate=sample_rate, original_peak=original_peak)
        
        # Normalize audio
        audio_data = audio_processor.normalize_audio(audio_data)
        
        # Get audio info
        audio_info = audio_processor.get_audio_info(audio_data, sr)
        logger.info(f"Audio info: Duration={audio_info['duration_seconds']:.2f}s, "
                   f"RMS={audio_info['rms']:.4f}, Max={audio_info['max_amplitude']:.4f}")
        
        # Create octave bank
        logger.info("Creating octave bank...")
        octave_bank = octave_filter.create_octave_bank(audio_data)
        
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
            output_dir=str(track_output_dir)
        )
        analyzer.create_histogram_plots_log_db(
            analysis_results,
            output_dir=str(track_output_dir)
        )
        analyzer.create_crest_factor_time_plot(
            time_analysis,
            output_path=str(track_output_dir / "crest_factor_time.png")
        )
        analyzer.create_octave_crest_factor_time_plot(
            audio_data,
            time_analysis,
            output_path=str(track_output_dir / "octave_crest_factor_time.png")
        )
        
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
            audio_data=audio_data
        )
        
        logger.info(f"Analysis complete for {track_path.name}")
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
    default='Tracks',
    help='Directory containing tracks to analyze (default: Tracks)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='analysis',
    help='Output directory for results (default: analysis)'
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
def main(input: Optional[Path], tracks_dir: Path, output_dir: Path, 
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
                audio_files.extend(tracks_dir.glob(f"*{ext}"))
                audio_files.extend(tracks_dir.glob(f"*{ext.upper()}"))
            
            if not audio_files:
                logger.error(f"No audio files found in {tracks_dir}")
                logger.info(f"Supported formats: {', '.join(audio_extensions)}")
                sys.exit(1)
            
            logger.info(f"Found {len(audio_files)} audio files to analyze")
            
            # Process each track
            successful_analyses = 0
            failed_analyses = 0
            
            for track_path in sorted(audio_files):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing track {successful_analyses + failed_analyses + 1}/{len(audio_files)}")
                
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
