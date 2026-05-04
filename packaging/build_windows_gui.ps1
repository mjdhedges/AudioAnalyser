param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
}

if ($Clean) {
    Remove-Item -Recurse -Force ".\build", ".\dist" -ErrorAction SilentlyContinue
}

python .\packaging\write_build_info.py
if ($LASTEXITCODE -ne 0) {
    throw "write_build_info.py failed with exit code $LASTEXITCODE"
}

python -m PyInstaller ".\packaging\audio-analyser-gui.spec" --noconfirm
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Built GUI distribution at: $repoRoot\dist\AudioAnalyser"

