"""
Track processing module for Audio Analyser.

This module handles the processing pipeline for individual audio channels.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio_processor import AudioProcessor
from src.config import Config
from src.data_export import DataExporter
from src.envelope_analyzer import EnvelopeAnalyzer
from src.music_analyzer import MusicAnalyzer
from src.octave_filter import OctaveBandFilter
from src.results import write_channel_result_bundle

logger = logging.getLogger(__name__)


class TrackProcessor:
    """Handles track processing operations."""

    def __init__(self, sample_rate: int = 44100, original_peak: float = 1.0) -> None:
        """Initialize the track processor.

        Args:
            sample_rate: Sample rate for audio processing
            original_peak: Original peak level before normalization (for dBFS calculation)
        """
        self.sample_rate = sample_rate
        self.original_peak = original_peak

    def process_channel(
        self,
        channel_data: np.ndarray,
        channel_index: int,
        channel_name: str,
        channel_folder_name: str,
        total_channels: int,
        track_path: Path,
        track_output_dir: Path,
        channel_output_dir: Path,
        audio_processor: AudioProcessor,
        octave_filter: OctaveBandFilter,
        chunk_duration: float,
        config: Config,
        content_type: str,
        original_peak: float,
        skip_octave_crest_factor_time: bool = False,
        export_octave_crest_factor_time_data: bool = False,
    ) -> bool:
        """Process a single channel through the full analysis pipeline.

        Args:
            channel_data: Single channel audio data (1D array)
            channel_index: Zero-based channel index
            channel_name: Channel name for CSV (e.g., "FL", "Channel 1 Left")
            channel_folder_name: Channel folder name for directory (e.g., "Channel 1 Left", "Channel 1 FL")
            total_channels: Total number of channels
            track_path: Path to the audio file
            track_output_dir: Base track output directory
            channel_output_dir: Channel-specific output directory
            audio_processor: AudioProcessor instance
            octave_filter: OctaveBandFilter instance
            chunk_duration: Duration of analysis chunks in seconds
            config: Configuration object
            content_type: Content type (Music/Film/Test Signal)
            original_peak: Original peak level for dBFS calculation

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            logger.info(
                f"Processing channel {channel_index + 1}/{total_channels}: {channel_name}"
            )

            # Initialize components with original peak for dBFS calculations
            use_batch_dpi = config.get(
                "performance.enable_parallel_batch", False
            ) or config.get("performance.enable_result_cache", False)
            plot_dpi = (
                config.get("plotting.batch_dpi", 150)
                if use_batch_dpi
                else config.get("plotting.dpi", 300)
            )

            peak_hold_tau = config.get("analysis.peak_hold_tau_seconds", 1.0)
            time_domain_mode = config.get(
                "analysis.time_domain_crest_factor_mode", "slow"
            )
            analyzer = MusicAnalyzer(
                sample_rate=self.sample_rate,
                original_peak=original_peak,
                dpi=plot_dpi,
                peak_hold_tau=peak_hold_tau,
                time_domain_crest_factor_mode=time_domain_mode,
                analysis_config=config.get("analysis", {}),
            )
            envelope_analyzer = EnvelopeAnalyzer(
                sample_rate=self.sample_rate, original_peak=original_peak
            )
            data_exporter = DataExporter(
                sample_rate=self.sample_rate, original_peak=original_peak
            )

            # Normalize channel audio
            channel_data = audio_processor.normalize_audio(channel_data)

            # Get audio info for this channel
            audio_info = audio_processor.get_audio_info(channel_data, self.sample_rate)
            logger.info(
                f"Channel {channel_name} info: Duration={audio_info['duration_seconds']:.2f}s, "
                f"RMS={audio_info['rms']:.4f}, Max={audio_info['max_amplitude']:.4f}"
            )

            configured_center_frequencies = config.get_octave_center_frequencies()
            center_frequencies = octave_filter.get_band_center_frequencies(
                configured_center_frequencies
            )

            logger.info(f"Creating FFT octave bank for channel {channel_name}...")
            octave_bank = octave_filter.create_octave_bank(
                channel_data,
                configured_center_frequencies,
            )
            octave_processing_metadata = octave_filter.get_processing_metadata()

            # Perform comprehensive analysis
            logger.info(
                f"Performing comprehensive analysis for channel {channel_name}..."
            )
            comprehensive_results = analyzer.analyze_comprehensive(
                channel_data,
                octave_bank,
                center_frequencies,
                chunk_duration=chunk_duration,
            )

            # Extract results
            analysis_results = comprehensive_results["main_analysis"]
            time_analysis = comprehensive_results["time_analysis"]
            chunk_octave_analysis = comprehensive_results["chunk_octave_analysis"]
            # Keep lightweight views of the already computed bank so plots/export do not
            # rebuild band mappings or accidentally fall back to full-spectrum data.
            analysis_results["band_data"] = {
                "Full Spectrum": octave_bank[:, 0],
                **{
                    f"{freq:.3f}": octave_bank[:, idx + 1]
                    for idx, freq in enumerate(center_frequencies)
                },
            }

            # Envelope statistics analysis
            logger.info(
                f"Performing envelope statistics analysis for channel {channel_name}..."
            )
            envelope_stats = envelope_analyzer.analyze_envelope_statistics(
                octave_bank,
                center_frequencies,
                config=config.get("envelope_analysis", {}),
            )

            logger.info(
                "Analysis data ready for channel %s; graph rendering is handled by src.render.",
                channel_name,
            )

            # Memory cleanup
            gc.collect()

            # Prepare channel-specific metadata
            track_metadata = {
                "track_name": track_path.name,
                "track_path": str(track_path),
                "content_type": content_type,
                "channel_index": channel_index,
                "channel_name": channel_name,
                "total_channels": total_channels,
                "duration_seconds": audio_info["duration_seconds"],
                "sample_rate": self.sample_rate,
                "samples": len(channel_data),
                "channels": 1,  # Single channel being processed
                "original_peak": original_peak,
                "original_peak_dbfs": 20 * np.log10(original_peak),
                "analysis_date": pd.Timestamp.now().isoformat(),
                **octave_processing_metadata,
            }

            logger.info(
                f"Calculating advanced statistics for channel {channel_name}..."
            )
            advanced_statistics = data_exporter.calculate_advanced_statistics(
                channel_data,
                analysis_results,
                time_analysis,
            )

            if config.get("export.generate_legacy_csv", False):
                logger.info(f"Exporting legacy CSV for channel {channel_name}...")
                channel_output_dir.mkdir(parents=True, exist_ok=True)
                data_exporter.export_comprehensive_results(
                    analysis_results,
                    time_analysis,
                    track_metadata,
                    str(channel_output_dir / "analysis_results.csv"),
                    chunk_octave_analysis=chunk_octave_analysis,
                    audio_data=channel_data,
                    envelope_statistics=envelope_stats,
                )

            if config.get("export.generate_analysis_bundle", True):
                write_channel_result_bundle(
                    track_output_dir=track_output_dir,
                    track_metadata=track_metadata,
                    analysis_results=analysis_results,
                    time_analysis=time_analysis,
                    chunk_octave_analysis=chunk_octave_analysis,
                    envelope_statistics=envelope_stats,
                    octave_bank=octave_bank,
                    center_frequencies=center_frequencies,
                    channel_data=channel_data,
                    plotting_config=config.get_plotting_config(),
                    envelope_config=config.get("envelope_analysis", {}),
                    analysis_config=config.get("analysis", {}),
                    advanced_statistics=advanced_statistics,
                )

            # Memory cleanup
            del channel_data, octave_bank, comprehensive_results

            logger.info(f"Analysis complete for channel {channel_name}")
            return True

        except Exception as e:
            logger.error(f"Error processing channel {channel_name}: {e}")
            return False
