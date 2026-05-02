"""Prove mono, stereo, and multichannel grouping rules.

This proof builds synthetic analysis-output folders and runs the production
grouping helpers against them. It checks that report/post-processing grouping
does not drop, duplicate, or mislabel channels across mono, stereo, cinema,
surround, and height layouts.
"""

from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.channel_mapping import get_channel_folder_name, get_channel_name
from src.post.group_crest_factor_time import _group_channels as group_time_channels
from src.post.group_octave_spectrum import _group_channels as group_octave_channels
from src.report_generator import _determine_channel_groups
from src.results.render import _classify_channel_name


PROOF_DIR = Path(__file__).resolve().parent
SYNTHETIC_ROOT = PROOF_DIR / "synthetic_tracks"


@dataclass(frozen=True)
class LayoutCase:
    """Expected grouping behavior for one synthetic track layout."""

    name: str
    total_channels: int
    channel_layout: str | None
    expected_post_groups: dict[str, tuple[str, ...]]
    expected_report_groups: dict[str, tuple[str, ...]]
    root_mono: bool = False


def analysis_csv_text() -> str:
    """Return a minimal analysis_results.csv parsed by grouping helpers."""
    return "\n".join(
        [
            "[TIME_DOMAIN_ANALYSIS]",
            "time_seconds,crest_factor_db,peak_level_dbfs,rms_level_dbfs,peak_level,rms_level",
            "1.0,6.0,-3.0,-9.0,0.7,0.35",
            "",
            "[OCTAVE_BAND_ANALYSIS]",
            "frequency_hz,max_amplitude_db,rms_db,crest_factor_db",
            "Full Spectrum,-3.0,-9.0,6.0",
            "31.25,-20.0,-26.0,6.0",
            "62.5,-18.0,-24.0,6.0",
            "",
        ]
    )


def write_analysis_csv(path: Path) -> None:
    """Write minimal analysis CSV contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(analysis_csv_text(), encoding="utf-8")


def layout_folders(total_channels: int, channel_layout: str | None) -> list[str]:
    """Return production folder names for a channel layout."""
    return [
        get_channel_folder_name(index, total_channels, channel_layout)
        for index in range(total_channels)
    ]


def make_cases() -> list[LayoutCase]:
    """Return grouping proof cases."""
    stereo_folders = tuple(layout_folders(2, "stereo"))
    five_one = tuple(layout_folders(6, "5.1"))
    five_one_two = tuple(layout_folders(8, "5.1.2"))
    seven_one_four = tuple(layout_folders(12, "7.1.4"))

    return [
        LayoutCase(
            name="mono_root",
            total_channels=1,
            channel_layout="mono",
            expected_post_groups={},
            expected_report_groups={"Mono": ("",)},
            root_mono=True,
        ),
        LayoutCase(
            name="stereo",
            total_channels=2,
            channel_layout="stereo",
            expected_post_groups={"All Channels": stereo_folders},
            expected_report_groups={"All Channels": stereo_folders},
        ),
        LayoutCase(
            name="ffmpeg_5_1",
            total_channels=6,
            channel_layout="5.1",
            expected_post_groups={
                "Screen": five_one[0:3],
                "LFE": (five_one[3],),
                "Surround+Height": five_one[4:6],
            },
            expected_report_groups={
                "Screen": five_one[0:3],
                "LFE": (five_one[3],),
                "Surround+Height": five_one[4:6],
            },
        ),
        LayoutCase(
            name="ffmpeg_5_1_2",
            total_channels=8,
            channel_layout="5.1.2",
            expected_post_groups={
                "Screen": five_one_two[0:3],
                "LFE": (five_one_two[3],),
                "Surround+Height": five_one_two[4:8],
            },
            expected_report_groups={
                "Screen": five_one_two[0:3],
                "LFE": (five_one_two[3],),
                "Surround+Height": five_one_two[4:8],
            },
        ),
        LayoutCase(
            name="ffmpeg_7_1_4",
            total_channels=12,
            channel_layout="7.1.4",
            expected_post_groups={
                "Screen": seven_one_four[0:3],
                "LFE": (seven_one_four[3],),
                "Surround+Height": seven_one_four[4:12],
            },
            expected_report_groups={
                "Screen": seven_one_four[0:3],
                "LFE": (seven_one_four[3],),
                "Surround+Height": seven_one_four[4:12],
            },
        ),
    ]


def build_track(case: LayoutCase) -> Path:
    """Create a synthetic track directory for one layout case."""
    track_dir = SYNTHETIC_ROOT / case.name
    if track_dir.exists():
        shutil.rmtree(track_dir)
    track_dir.mkdir(parents=True)

    if case.root_mono:
        write_analysis_csv(track_dir / "analysis_results.csv")
        return track_dir

    for folder in layout_folders(case.total_channels, case.channel_layout):
        write_analysis_csv(track_dir / folder / "analysis_results.csv")
    return track_dir


def normalize_grouped_paths(grouped: dict[str, object]) -> dict[str, tuple[str, ...]]:
    """Normalize grouping helper outputs to group -> folder-name tuple."""
    normalized: dict[str, tuple[str, ...]] = {}
    for group, values in grouped.items():
        folders = []
        for value in values:  # type: ignore[union-attr]
            if isinstance(value, tuple):
                folders.append(str(value[0]))
            else:
                folders.append(str(value))
        normalized[group] = tuple(folders)
    return normalized


def flat_folders(groups: dict[str, tuple[str, ...]]) -> list[str]:
    """Flatten grouped folder names."""
    return [folder for folders in groups.values() for folder in folders]


def compare_groups(
    observed: dict[str, tuple[str, ...]],
    expected: dict[str, tuple[str, ...]],
) -> tuple[bool, str]:
    """Compare observed and expected groups and explain failures."""
    normalized_observed = {
        group: tuple(sorted(folders)) for group, folders in observed.items()
    }
    normalized_expected = {
        group: tuple(sorted(folders)) for group, folders in expected.items()
    }
    if normalized_observed != normalized_expected:
        return False, f"expected={expected}; observed={observed}"

    flattened = flat_folders(observed)
    duplicates = sorted({folder for folder in flattened if flattened.count(folder) > 1})
    if duplicates:
        return False, f"duplicated={duplicates}"
    return True, ""


def analyze_layout_case(case: LayoutCase) -> list[dict[str, object]]:
    """Run production grouping helpers for one synthetic layout."""
    track_dir = build_track(case)
    rows: list[dict[str, object]] = []

    checks = [
        (
            "group_crest_factor_time",
            normalize_grouped_paths(group_time_channels(track_dir)),
            case.expected_post_groups,
        ),
        (
            "group_octave_spectrum",
            normalize_grouped_paths(group_octave_channels(track_dir)),
            case.expected_post_groups,
        ),
        (
            "report_generator",
            normalize_grouped_paths(_determine_channel_groups(track_dir)),
            case.expected_report_groups,
        ),
    ]

    for helper, observed, expected in checks:
        passed, detail = compare_groups(observed, expected)
        rows.append(
            {
                "case": case.name,
                "helper": helper,
                "expected_groups": repr(expected),
                "observed_groups": repr(observed),
                "expected_channel_count": len(flat_folders(expected)),
                "observed_channel_count": len(flat_folders(observed)),
                "pass": passed,
                "detail": detail,
            }
        )
    return rows


def analyze_channel_name_classification() -> list[dict[str, object]]:
    """Check renderer channel-name classification for labels used in reports."""
    cases = [
        ("FL", "screen"),
        ("FR", "screen"),
        ("FC", "screen"),
        ("Channel 1 FL", "screen"),
        ("Channel 4 LFE", "lfe"),
        ("Low Frequency Effects", "lfe"),
        ("SL", "surround"),
        ("SR", "surround"),
        ("SBL", "surround"),
        ("SBR", "surround"),
        ("TFL", "surround"),
        ("TFR", "surround"),
        ("TBL", "surround"),
        ("TBR", "surround"),
        ("Channel 1 Left", None),
        ("Channel 2 Right", None),
        ("Channel 1 FC", "screen"),
    ]
    rows = []
    for channel_name, expected in cases:
        observed = _classify_channel_name(channel_name)
        rows.append(
            {
                "channel_name": channel_name,
                "expected_group": expected or "",
                "observed_group": observed or "",
                "pass": observed == expected,
            }
        )
    return rows


def analyze_channel_mapping() -> list[dict[str, object]]:
    """Check production channel name/folder generation for representative layouts."""
    rows = []
    for total_channels, layout in (
        (1, "mono"),
        (2, "stereo"),
        (6, "5.1"),
        (8, "5.1.2"),
        (12, "7.1.4"),
    ):
        for index in range(total_channels):
            rows.append(
                {
                    "layout": layout,
                    "channel_index": index,
                    "channel_number": index + 1,
                    "channel_name": get_channel_name(index, total_channels, layout),
                    "folder_name": get_channel_folder_name(
                        index,
                        total_channels,
                        layout,
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_group_counts(rows: list[dict[str, object]]) -> None:
    """Plot observed channel counts by case and helper."""
    helpers = list(dict.fromkeys(str(row["helper"]) for row in rows))
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    x = list(range(len(cases)))

    for helper_idx, helper in enumerate(helpers):
        values = [
            int(
                next(
                    row
                    for row in rows
                    if row["case"] == case and row["helper"] == helper
                )["observed_channel_count"]
            )
            for case in cases
        ]
        offsets = [pos + (helper_idx - 1) * width for pos in x]
        ax.bar(offsets, values, width, label=helper)

    ax.set_xticks(x)
    ax.set_xticklabels([case.replace("_", "\n") for case in cases])
    ax.set_ylabel("Grouped channel count")
    ax.set_title("Channel Grouping Counts by Production Helper")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROOF_DIR / "grouped_channel_counts.png", dpi=160)
    plt.close(fig)


def write_summary(
    grouping_rows: list[dict[str, object]],
    classification_rows: list[dict[str, object]],
) -> None:
    """Write a proof README."""
    grouping_pass = all(bool(row["pass"]) for row in grouping_rows)
    classification_pass = all(bool(row["pass"]) for row in classification_rows)
    passed = grouping_pass and classification_pass
    helper_rows = "\n".join(
        "| "
        f"{row['case']} | "
        f"{row['helper']} | "
        f"{row['observed_channel_count']} | "
        f"{row['pass']} |"
        for row in grouping_rows
    )
    classification_failures = [
        row for row in classification_rows if not bool(row["pass"])
    ]
    failure_text = (
        "None"
        if not classification_failures
        else ", ".join(str(row["channel_name"]) for row in classification_failures)
    )

    summary = f"""# Channel Grouping Proof

Status: **{"PASS" if passed else "FAIL"}**

This proof validates mono, stereo, and multichannel grouping rules used by
report and post-processing outputs. It creates synthetic analysis-output folders
with valid minimal CSV sections, then runs the production grouping helpers.

## Scope

- Channel naming: `src.channel_mapping`
- Time-series group plots: `src.post.group_crest_factor_time._group_channels`
- Octave-spectrum group plots: `src.post.group_octave_spectrum._group_channels`
- Markdown report grouping: `src.report_generator._determine_channel_groups`
- Bundle renderer classification: `src.results.render._classify_channel_name`

## Results

- Filesystem grouping helpers: **{"PASS" if grouping_pass else "FAIL"}**
- Channel-name classification: **{"PASS" if classification_pass else "FAIL"}**
- Classification failures: {failure_text}

| Case | Helper | Observed channel count | Pass |
| --- | --- | ---: | --- |
{helper_rows}

## Interpretation

Mono report grouping is represented by a root-level `analysis_results.csv` and
maps to the `Mono` report group. Stereo channel folders are intentionally not
cinema-screen channels; they group under `All Channels`.

For cinema layouts, `FL`, `FR`, and `FC` group as `Screen`; `LFE` groups as
`LFE`; surround, back, and top/height folders group as `Surround+Height` in the
post-processing group plots.

The markdown report fallback classifies channel folders by channel-name tokens,
not by loose substring matching. This matters for height channels such as
`TFL/TFR`: they contain `FL/FR` as substrings but must group with
`Surround+Height`, not `Screen`.

## Outputs

- `grouping_results.csv`
- `channel_classification.csv`
- `channel_mapping_results.csv`
- `grouped_channel_counts.png`
- `synthetic_tracks/`
"""
    (PROOF_DIR / "README.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the proof."""
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC_ROOT.exists():
        shutil.rmtree(SYNTHETIC_ROOT)
    SYNTHETIC_ROOT.mkdir(parents=True)

    grouping_rows: list[dict[str, object]] = []
    for case in make_cases():
        grouping_rows.extend(analyze_layout_case(case))

    classification_rows = analyze_channel_name_classification()
    mapping_rows = analyze_channel_mapping()

    write_csv(PROOF_DIR / "grouping_results.csv", grouping_rows)
    write_csv(PROOF_DIR / "channel_classification.csv", classification_rows)
    write_csv(PROOF_DIR / "channel_mapping_results.csv", mapping_rows)
    plot_group_counts(grouping_rows)
    write_summary(grouping_rows, classification_rows)

    passed = all(bool(row["pass"]) for row in grouping_rows) and all(
        bool(row["pass"]) for row in classification_rows
    )
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Wrote proof outputs to: {PROOF_DIR}")


if __name__ == "__main__":
    main()
