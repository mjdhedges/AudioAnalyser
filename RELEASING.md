# Releasing Audio Analyser

Step-by-step checklist for a **versioned Windows GUI build**. Version strings embedded in the executable come from **Git** (`git describe`) via `packaging/write_build_info.py`, which runs automatically inside `packaging/build_windows_gui.ps1`.

For packaging details (assets, FFmpeg, CLI companion), see [`docs/windows_gui_packaging.md`](docs/windows_gui_packaging.md).

## Prerequisites

- Repository clone with `.git` present (needed for commit/tag metadata).
- Virtual environment with dependencies installed (`requirements.txt`, `requirements-dev.txt`).
- PowerShell at the repository root.

## Release steps

1. **Activate the virtual environment**

   ```powershell
   cd "D:\Software Projects\AudioAnalyser"
   .\venv\Scripts\Activate.ps1
   ```

2. **Commit everything intended for this release**

   Use a clean working tree if you do not want `-dirty` in build metadata:

   ```powershell
   git status
   git add -A
   git commit -m "chore: prepare release v0.3.8"
   ```

3. **Align `pyproject.toml` (recommended)**

   Set `[project].version` to the same semantic version as the tag so editable installs and metadata stay consistent:

   ```toml
   version = "0.3.8"
   ```

   Commit this change if you modified it.

4. **Create an annotated or lightweight tag**

   Use the form `vMAJOR.MINOR.PATCH` so release tooling can parse it:

   ```powershell
   git tag v0.3.8
   ```

5. **Push commits and the tag**

   ```powershell
   git push origin master
   git push origin v0.3.8
   ```

   Replace `master` with your release branch name if different.

6. **Build the Windows distribution**

   ```powershell
   .\packaging\build_windows_gui.ps1 -Clean
   ```

   Outputs:

   - `dist/AudioAnalyser/AudioAnalyser.exe`
   - `dist/AudioAnalyser/AudioAnalyserCli.exe`

7. **Verify version metadata**

   - Right-click `AudioAnalyser.exe` → **Properties** → **Details** (file/product version and related fields).
   - Optional: open **About** in the GUI and confirm the build line.

## Notes

- **Tag format:** Prefer `v0.3.8`, not `0.3.8`, so `write_build_info.py` matches the usual `git describe` tag pattern.
- **Dirty builds:** Uncommitted changes produce `-dirty` in `git describe` and in embedded metadata; avoid that for official drops.
- **`pyproject.toml` vs Git:** The frozen EXE uses Git-derived data from `write_build_info.py`. Keeping `pyproject.toml` in sync avoids confusion for `pip install -e .` and documentation.
