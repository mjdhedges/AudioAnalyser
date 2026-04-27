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

- `audioanalyser_icon.jpeg`
- `config.toml`
- package data needed by `librosa`, `matplotlib`, and `scipy`

The GUI launches analysis and rendering subprocesses through the packaged
`AudioAnalyserCli.exe` companion, so a separate Python installation is not
required for normal use.

## ffmpeg

MKV/TrueHD processing still requires the external ffmpeg tools package. The
current packaged GUI expects both `ffmpeg.exe` and `ffprobe.exe` to be available
on `PATH`.

Install ffmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html),
add the ffmpeg `bin` folder to `PATH`, then restart Audio Analyser. If those
tools are missing, MKV analysis fails with a clear log message explaining that
ffmpeg must be installed and added to `PATH`.

Bundling ffmpeg can be added later by copying `ffmpeg.exe` and `ffprobe.exe`
into a tracked `vendor/ffmpeg/` folder and adding them to the PyInstaller
`binaries` list.

