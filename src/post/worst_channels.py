from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def _parse_sustained_summary(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {}
    text = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = text.index("[SUSTAINED_PEAKS_SUMMARY]") + 1
    except ValueError:
        return {}
    if start >= len(text):
        return {}
    header = text[start].split(",")
    key_to_idx = {k: i for i, k in enumerate(header)}
    # Find Full Spectrum row
    for line in text[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",")
        if parts[0] == "Full Spectrum":
            out: Dict[str, float] = {}
            for k, idx in key_to_idx.items():
                if k == "frequency_hz":
                    continue
                try:
                    out[k] = float(parts[idx])
                except Exception:
                    pass
            return out
    return {}


def _parse_advanced_stats(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {}
    text = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = text.index("[ADVANCED_STATISTICS]") + 1
    except ValueError:
        return {}
    if start >= len(text):
        return {}
    out: Dict[str, float] = {}
    for line in text[start + 1 :]:
        if not line or line.startswith("["):
            break
        parts = line.split(",")
        if len(parts) < 2:
            continue
        key = parts[0]
        val = parts[1]
        try:
            out[key] = float(val)
        except Exception:
            pass
    return out


def _score_channel(csv_path: Path, metric: str) -> Tuple[float, str]:
    """Returns (score, source_metric_used)."""
    sust = _parse_sustained_summary(csv_path)
    adv = _parse_advanced_stats(csv_path)
    if metric in ("t3_ms_p95", "t6_ms_p95", "t9_ms_p95", "t12_ms_p95", "hold_ms_p95"):
        key = metric
        if key in sust:
            return float(sust[key]), f"sustained:{key}"
        for fallback in ("t9_ms_p95", "t6_ms_p95", "t3_ms_p95", "t12_ms_p95", "hold_ms_p95"):
            if fallback in sust:
                return float(sust[fallback]), f"sustained:{fallback}"
    if "peak_saturation_percent" in adv:
        return float(adv["peak_saturation_percent"]), "advanced:peak_saturation_percent"
    if "clip_events_rate_per_sec" in adv:
        return float(adv["clip_events_rate_per_sec"]), "advanced:clip_events_rate_per_sec"
    return 0.0, "none"


def select_worst_channels(track_dir: Path, metric: str = "t6_ms_p95") -> Path:
    """Create worst-channels manifest for a track directory.
    
    Args:
        track_dir: Path containing Channel X subfolders
        metric: Metric to rank channels (default: t6_ms_p95)
    
    Returns:
        Path to the written manifest CSV.
    """
    screen = {"Channel 1 FL", "Channel 2 FR", "Channel 3 FC"}
    lfe = {"Channel 4 LFE"}
    surround_prefixes = ("Channel 5", "Channel 6", "Channel 7", "Channel 8")

    candidates: Dict[str, Tuple[Path, float, str]] = {}
    for sub in track_dir.iterdir():
        if not sub.is_dir():
            continue
        csv_path = sub / "analysis_results.csv"
        if not csv_path.exists():
            continue
        folder_name = sub.name
        if folder_name in screen:
            group = "screen"
        elif folder_name in lfe:
            group = "lfe"
        elif folder_name.startswith(surround_prefixes):
            group = "surround"
        else:
            continue
        score, used = _score_channel(csv_path, metric)
        prev = candidates.get(group)
        if prev is None or score > prev[1]:
            candidates[group] = (sub, score, used)

    manifest_lines = ["group,folder,score,metric_used"]
    for group, (folder, score, used) in candidates.items():
        manifest_lines.append(f"{group},{folder.name},{score},{used}")

    manifest_path = track_dir / "worst_channels_manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    return manifest_path



