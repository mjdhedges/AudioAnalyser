# Windows GUI Packaging

Audio Analyser can be packaged as a Windows desktop application with
PyInstaller. The packaged GUI includes the Python runtime and Python package
dependencies in the `dist/AudioAnalyser/` folder.

## Build

From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
.\packaging\build_windows_gui.ps1 -Clean
```

The executable is written to:

```text
dist/AudioAnalyser/AudioAnalyser.exe
```

The distribution also includes `AudioAnalyserCli.exe`, a console companion used
internally by the GUI to run analysis and rendering while preserving progress
output.

## Runtime Assets

The PyInstaller spec includes:

- `audioanalyser_icon.png`
- `config.toml`
- `vendor/ffmpeg/bin/ffmpeg.exe`
- `vendor/ffmpeg/bin/ffprobe.exe`
- FFmpeg license/readme files
- package data needed by `librosa`, `matplotlib`, and `scipy`
- PDF report support via PySide6 and Python-Markdown

The GUI launches analysis and rendering subprocesses through the packaged
`AudioAnalyserCli.exe` companion, so a separate Python installation is not
required for normal use.

## FFmpeg

Packaged Windows builds include FFmpeg so MKV/TrueHD processing works without a
separate FFmpeg install. The bundled files live under `vendor/ffmpeg/` in source
and are copied into the packaged `_internal/vendor/ffmpeg/` runtime folder.

Source/development runs can also use a system FFmpeg install on `PATH`. If
`ffmpeg.exe` or `ffprobe.exe` cannot be found, MKV analysis fails with a clear
log message explaining that FFmpeg must be installed and added to `PATH`.

The FFmpeg binaries are distributed with their upstream license/readme files.

