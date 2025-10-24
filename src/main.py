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

from src.audio_processor import AudioProcessor
from src.music_analyzer import MusicAnalyzer
from src.octave_filter import OctaveBandFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Input audio file path'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='output',
    help='Output directory for results and plots'
)
@click.option(
    '--sample-rate', '-sr',
    type=int,
    default=44100,
    help='Sample rate for audio processing'
)
@click.option(
    '--plot/--no-plot',
    default=True,
    help='Generate plots'
)
@click.option(
    '--export-csv/--no-export-csv',
    default=True,
    help='Export results to CSV'
)
def main(input: Path, output_dir: Path, sample_rate: int, 
         plot: bool, export_csv: bool) -> None:
    """Music Analyser - Analyze audio files using octave band filtering.
    
    This tool performs comprehensive octave band analysis on audio files,
    similar to the MATLAB musicanalyser.m script. It generates frequency
    analysis, statistical measures, and visualizations.
    
    Example:
        python -m src.main --input "song.flac" --output-dir results
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Music Analyser")
        logger.info(f"Input file: {input}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Sample rate: {sample_rate} Hz")
        
        # Initialize components
        audio_processor = AudioProcessor(sample_rate=sample_rate)
        octave_filter = OctaveBandFilter(sample_rate=sample_rate)
        analyzer = MusicAnalyzer(sample_rate=sample_rate)
        
        # Load and preprocess audio
        logger.info("Loading audio file...")
        audio_data, sr = audio_processor.load_audio(input)
        
        # Convert to mono if stereo
        audio_data = audio_processor.stereo_to_mono(audio_data)
        
        # Normalize audio
        audio_data = audio_processor.normalize_audio(audio_data)
        
        # Get audio info
        audio_info = audio_processor.get_audio_info(audio_data, sr)
        logger.info(f"Audio info: {audio_info}")
        
        # Create octave bank
        logger.info("Creating octave bank...")
        octave_bank = octave_filter.create_octave_bank(audio_data)
        
        # Perform analysis
        logger.info("Performing octave band analysis...")
        analysis_results = analyzer.analyze_octave_bands(
            octave_bank, 
            octave_filter.OCTAVE_CENTER_FREQUENCIES
        )
        
        # Generate outputs
        if plot:
            logger.info("Generating plots...")
            analyzer.create_octave_spectrum_plot(
                analysis_results,
                output_path=str(output_dir / "octave_spectrum.png")
            )
            analyzer.create_histogram_plots(
                analysis_results,
                output_dir=str(output_dir)
            )
        
        if export_csv:
            logger.info("Exporting results to CSV...")
            analyzer.export_analysis_results(
                analysis_results,
                str(output_dir / "analysis_results.csv")
            )
        
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
