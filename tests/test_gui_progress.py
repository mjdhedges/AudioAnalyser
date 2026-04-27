"""Tests for GUI progress state tracking."""

from src.gui.progress import ProgressTracker


def test_progress_tracker_tracks_file_lifecycle() -> None:
    """Progress events should update file state and summary counts."""
    tracker = ProgressTracker()

    tracker.handle_event({"event": "analysis_started", "total_tracks": 2})
    first = tracker.handle_event(
        {
            "event": "file_queued",
            "index": 1,
            "total": 2,
            "path": "a.wav",
            "name": "a.wav",
        }
    )
    assert first is not None
    assert first.status == "Waiting"

    first = tracker.handle_event(
        {
            "event": "file_started",
            "index": 1,
            "total": 2,
            "path": "a.wav",
            "name": "a.wav",
        }
    )
    assert first is not None
    assert first.status == "Running"

    first = tracker.handle_event(
        {
            "event": "file_finished",
            "index": 1,
            "total": 2,
            "path": "a.wav",
            "name": "a.wav",
            "success": True,
            "elapsed_seconds": 1.25,
        }
    )
    assert first is not None
    assert first.status == "Finished"
    assert first.elapsed_seconds == 1.25
    assert tracker.completed_files == 1
    assert tracker.successful_files == 1
    assert tracker.failed_files == 0


def test_progress_tracker_does_not_double_count_finished_file() -> None:
    """Repeated finish events should not inflate completed counts."""
    tracker = ProgressTracker()
    tracker.handle_event({"event": "analysis_started", "total_tracks": 1})

    event = {
        "event": "file_finished",
        "index": 1,
        "total": 1,
        "path": "a.wav",
        "name": "a.wav",
        "success": False,
    }
    tracker.handle_event(event)
    tracker.handle_event(event)

    assert tracker.completed_files == 1
    assert tracker.successful_files == 0
    assert tracker.failed_files == 1
