"""Reader API for Audio Analyser result bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import pandas as pd


@dataclass(frozen=True)
class ChannelResult:
    """One channel's data inside an analysis result bundle."""

    channel_id: str
    channel_index: int
    channel_name: str
    path: Path
    artifacts: Dict[str, str]

    def read_json(self, artifact_name: str) -> Dict[str, Any]:
        """Read a JSON artifact for this channel."""
        path = self.artifact_path(artifact_name)
        return json.loads(path.read_text(encoding="utf-8"))

    def read_table(self, artifact_name: str) -> pd.DataFrame:
        """Read a tabular CSV artifact for this channel."""
        return pd.read_csv(self.artifact_path(artifact_name))

    def artifact_path(self, artifact_name: str) -> Path:
        """Resolve a named artifact path."""
        relative_path = self.artifacts.get(artifact_name)
        candidate = self.path / f"{artifact_name}.json"
        if relative_path is None and candidate.exists():
            return candidate
        if relative_path is None:
            raise KeyError(f"Unknown channel artifact: {artifact_name}")
        return self.path / relative_path


class ResultBundle:
    """Portable per-track analysis result bundle."""

    def __init__(self, path: Path) -> None:
        """Load a result bundle.

        Args:
            path: Path to a `.aaresults` bundle directory.

        Raises:
            FileNotFoundError: If `manifest.json` is missing.
        """
        self.path = path
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
        self.manifest: Dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    @property
    def track(self) -> Dict[str, Any]:
        """Track-level metadata from the bundle manifest."""
        return self.manifest.get("track", {})

    @property
    def schema_version(self) -> int:
        """Bundle schema version."""
        return int(self.manifest.get("schema_version", 0))

    def channels(self) -> list[ChannelResult]:
        """Return all channel records in manifest order."""
        return [self._channel_from_manifest(entry) for entry in self._channel_entries()]

    def get_channel(self, channel_id: str) -> ChannelResult:
        """Return one channel by ID."""
        for channel in self.channels():
            if channel.channel_id == channel_id:
                return channel
        raise KeyError(f"Channel not found in bundle: {channel_id}")

    def _channel_entries(self) -> Iterable[Dict[str, Any]]:
        return self.manifest.get("channels", [])

    def _channel_from_manifest(self, entry: Dict[str, Any]) -> ChannelResult:
        return ChannelResult(
            channel_id=str(entry["channel_id"]),
            channel_index=int(entry.get("channel_index", 0)),
            channel_name=str(entry.get("channel_name", "")),
            path=self.path / str(entry["relative_path"]),
            artifacts=dict(entry.get("artifacts", {})),
        )


def load_result_bundle(path: Union[Path, str]) -> ResultBundle:
    """Load a `.aaresults` directory."""
    return ResultBundle(Path(path))


def find_result_bundles(path: Union[Path, str]) -> list[Path]:
    """Find result bundles at `path` or recursively below it."""
    root = Path(path)
    if root.suffix == ".aaresults" and root.is_dir():
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.aaresults") if p.is_dir())
