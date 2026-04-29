"""Progress state tracking for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class FileProgress:
    """Progress state for one input file."""

    index: int
    total: int
    path: str
    name: str
    status: str
    stage: str = ""
    elapsed_seconds: Optional[float] = None
    analysis_seconds: Optional[float] = None
    render_seconds: Optional[float] = None
    error: Optional[str] = None


class ProgressTracker:
    """Track file-level progress events emitted by ``src.main``."""

    def __init__(self) -> None:
        """Initialize an empty progress tracker."""
        self.total_files = 0
        self.completed_files = 0
        self.total_steps = 0
        self.completed_steps = 0
        self.successful_files = 0
        self.failed_files = 0
        self.render_enabled = False
        self.files: Dict[str, FileProgress] = {}

    def reset(self) -> None:
        """Clear all tracked progress."""
        self.total_files = 0
        self.completed_files = 0
        self.total_steps = 0
        self.completed_steps = 0
        self.successful_files = 0
        self.failed_files = 0
        self.render_enabled = False
        self.files.clear()

    def handle_event(self, event: Mapping[str, Any]) -> Optional[FileProgress]:
        """Apply a progress event.

        Args:
            event: Parsed progress event from ``src.main --progress-json``.

        Returns:
            Updated file progress for file events, otherwise ``None``.
        """
        event_name = str(event.get("event", ""))
        if event_name == "analysis_started":
            self.total_files = int(event.get("total_tracks") or 0)
            self.render_enabled = bool(event.get("render_enabled"))
            self.total_steps = int(
                event.get("total_steps")
                or self.total_files * (2 if self.render_enabled else 1)
            )
            self.completed_files = 0
            self.completed_steps = 0
            self.successful_files = 0
            self.failed_files = 0
            self.files.clear()
            return None
        if event_name in {"file_queued", "file_started", "file_submitted"}:
            status = {
                "file_queued": "Waiting",
                "file_started": "Analyzing",
                "file_submitted": "Submitted",
            }[event_name]
            return self._upsert_file(event, status=status, stage="Analysis")
        if event_name == "file_finished":
            success = bool(event.get("success"))
            status = "Analysis done" if success and self.render_enabled else "Finished"
            if not success:
                status = "Failed"
            existing = self.files.get(str(event.get("path", "")))
            was_analysis_complete = existing is not None and (
                existing.analysis_seconds is not None
                or existing.status in {"Analysis done", "Finished", "Failed"}
            )
            file_progress = self._upsert_file(event, status=status, stage="Analysis")
            file_progress.analysis_seconds = _optional_float(
                event.get("elapsed_seconds")
            )
            file_progress.elapsed_seconds = file_progress.analysis_seconds
            file_progress.error = (
                str(event.get("error")) if event.get("error") is not None else None
            )
            if not was_analysis_complete:
                self.completed_steps += 1
                if self.render_enabled and not success:
                    self.completed_steps += 1
            if (not self.render_enabled or not success) and not was_analysis_complete:
                self.completed_files += 1
                if success:
                    self.successful_files += 1
                else:
                    self.failed_files += 1
            return file_progress
        if event_name == "render_started":
            return self._upsert_file(event, status="Rendering", stage="Render")
        if event_name == "render_finished":
            success = bool(event.get("success"))
            status = "Finished" if success else "Render failed"
            existing = self.files.get(str(event.get("path", "")))
            was_render_complete = (
                existing is not None and existing.render_seconds is not None
            )
            file_progress = self._upsert_file(event, status=status, stage="Render")
            file_progress.render_seconds = _optional_float(event.get("elapsed_seconds"))
            analysis_seconds = file_progress.analysis_seconds or 0.0
            file_progress.elapsed_seconds = analysis_seconds + (
                file_progress.render_seconds or 0.0
            )
            file_progress.error = (
                str(event.get("error")) if event.get("error") is not None else None
            )
            if not was_render_complete:
                self.completed_steps += 1
                self.completed_files += 1
                if success:
                    self.successful_files += 1
                else:
                    self.failed_files += 1
            return file_progress
        if event_name == "analysis_finished":
            self.successful_files = int(
                event.get("successful") or self.successful_files
            )
            self.failed_files = int(event.get("failed") or self.failed_files)
            self.completed_files = self.successful_files + self.failed_files
            self.completed_steps = int(
                event.get("completed_steps") or self.completed_steps
            )
            return None
        return None

    def _upsert_file(
        self,
        event: Mapping[str, Any],
        status: str,
        stage: str,
    ) -> FileProgress:
        path = str(event.get("path", ""))
        file_progress = self.files.get(path)
        if file_progress is None:
            file_progress = FileProgress(
                index=int(event.get("index") or 0),
                total=int(event.get("total") or self.total_files),
                path=path,
                name=str(event.get("name") or path),
                status=status,
                stage=stage,
            )
            self.files[path] = file_progress
        else:
            file_progress.status = status
            file_progress.stage = stage
        return file_progress


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
