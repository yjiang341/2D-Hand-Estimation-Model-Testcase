Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -Path $projectRoot

Write-Host "Installing/Updating build dependency..."
python -m pip install --upgrade pyinstaller

Write-Host "Building GUI executable..."
python -m PyInstaller --noconfirm --clean --distpath build_tools/dist --workpath build_tools/build build_tools/HandPoseAudioBridge.spec

Write-Host "Build complete. Output folder: $projectRoot/build_tools/dist/HandPoseAudioBridge"
