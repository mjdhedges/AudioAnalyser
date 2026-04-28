# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Audio Analyser GUI and subprocess runner."""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent.parent
ICON = ROOT / "audioanalyser_icon.jpeg"

datas = [
    (str(ROOT / "audioanalyser_icon.jpeg"), "."),
    (str(ROOT / "config.toml"), "."),
    (str(ROOT / "vendor" / "ffmpeg"), "vendor/ffmpeg"),
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
