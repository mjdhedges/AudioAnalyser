# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Audio Analyser GUI and subprocess runner."""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# SPECPATH is the directory containing this spec file (``packaging/``).
# We want the repository root (parent of ``packaging/``).
ROOT = Path(SPECPATH).parent
ICON = ROOT / "audioanalyser_icon.png"
WIN_VER_FILE = ROOT / "packaging" / "windows_file_version_info.txt"
_EXE_VER_KW = {"version": str(WIN_VER_FILE)} if WIN_VER_FILE.is_file() else {}

datas = [
    (str(ROOT / "audioanalyser_icon.png"), "."),
    (str(ROOT / "config.toml"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    # Bundle FFmpeg runtime binaries and minimal license/source-info files.
    # Do not bundle full corresponding-source archives inside the app payload.
    (str(ROOT / "vendor" / "ffmpeg" / "bin"), "vendor/ffmpeg/bin"),
    (str(ROOT / "vendor" / "ffmpeg" / "LICENSE"), "vendor/ffmpeg"),
    (str(ROOT / "vendor" / "ffmpeg" / "README.txt"), "vendor/ffmpeg"),
    (
        str(ROOT / "vendor" / "ffmpeg" / "CORRESPONDING_SOURCE.md"),
        "vendor/ffmpeg",
    ),
]
datas += collect_data_files("librosa")
datas += collect_data_files("matplotlib")
datas += collect_data_files("scipy")

hiddenimports = []
hiddenimports += ["PySide6.QtCore", "PySide6.QtGui"]
hiddenimports += collect_submodules("numba")
hiddenimports += collect_submodules("soundfile")


gui_analysis = Analysis(
    [str(ROOT / "src" / "gui" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
cli_analysis = Analysis(
    [str(ROOT / "src" / "gui" / "cli_runner.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

gui_pyz = PYZ(gui_analysis.pure)
cli_pyz = PYZ(cli_analysis.pure)

gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name="AudioAnalyser",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    **_EXE_VER_KW,
)
cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="AudioAnalyserCli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    **_EXE_VER_KW,
)
coll = COLLECT(
    gui_exe,
    cli_exe,
    gui_analysis.binaries,
    gui_analysis.datas,
    cli_analysis.binaries,
    cli_analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AudioAnalyser",
)
