param(
    [string]$EntryScript = "GUI_Panel/gui_main.py",
    [string]$AppName = "HandPoseEstimator",
    [ValidateSet("onedir", "onefile")]
    [string]$BundleMode = "onedir",
    [string]$DistDir = "dist",
    [string]$BuildDir = "build",
    [switch]$Console
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Python {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

function Ensure-PyInstaller([string]$pythonExe) {
    $hasPyInstaller = $true
    try {
        & $pythonExe -m PyInstaller --version *> $null
        if ($LASTEXITCODE -ne 0) {
            $hasPyInstaller = $false
        }
    } catch {
        $hasPyInstaller = $false
    }

    if (-not $hasPyInstaller) {
        Write-Host "[build] PyInstaller not found. Installing..."
        & $pythonExe -m pip install pyinstaller
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install PyInstaller."
        }
    }
}

$entryPath = Join-Path $PSScriptRoot $EntryScript
if (-not (Test-Path $entryPath)) {
    throw "Entry script not found: $EntryScript`nTip: pass -EntryScript with an existing file, e.g. bridge_main.py or live_main.py"
}

$pythonExe = Resolve-Python
Write-Host "[build] Python: $pythonExe"
Write-Host "[build] Entry : $EntryScript"
Write-Host "[build] Name  : $AppName"
Write-Host "[build] Mode  : $BundleMode"

Ensure-PyInstaller -pythonExe $pythonExe

$distPath = Join-Path $PSScriptRoot $DistDir
$buildPath = Join-Path $PSScriptRoot $BuildDir
$hooksPath = Join-Path $PSScriptRoot "packaging_hooks"
$venvRoot = Join-Path $PSScriptRoot ".venv"
$mpTasksCPath = Join-Path $venvRoot "Lib\site-packages\mediapipe\tasks\c"
$mpTasksCDll = Join-Path $mpTasksCPath "libmediapipe.dll"
$modelAssetPath = Join-Path $PSScriptRoot "Models\hand_landmarker.task"
New-Item -ItemType Directory -Force -Path $distPath | Out-Null
New-Item -ItemType Directory -Force -Path $buildPath | Out-Null

$pyiArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--name", $AppName,
    "--distpath", $distPath,
    "--workpath", $buildPath,
    "--specpath", $buildPath,
    "--additional-hooks-dir", $hooksPath,
    "--hidden-import", "mediapipe.tasks.c",
    "--exclude-module", "mediapipe.tasks.python.test",
    "--exclude-module", "mediapipe.tasks.python.benchmark"
)

if ($BundleMode -eq "onefile") {
    $pyiArgs += "--onefile"
}

if (-not $Console) {
    $pyiArgs += "--windowed"
}

# Fallback for MediaPipe task runtime artifacts in frozen apps.
if (Test-Path $mpTasksCPath) {
    $pyiArgs += @("--add-data", "${mpTasksCPath};mediapipe/tasks/c")
}

if (Test-Path $mpTasksCDll) {
    $pyiArgs += @("--add-binary", "${mpTasksCDll};mediapipe/tasks/c")
}

if (Test-Path $modelAssetPath) {
    $pyiArgs += @("--add-data", "${modelAssetPath};Models")
}

$pyiArgs += $entryPath

Write-Host "[build] Running PyInstaller..."
& $pythonExe @pyiArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

if ($BundleMode -eq "onefile") {
    $exePath = Join-Path $distPath ("{0}.exe" -f $AppName)
} else {
    $exePath = Join-Path (Join-Path $distPath $AppName) ("{0}.exe" -f $AppName)
}

if (-not (Test-Path $exePath)) {
    throw "Build completed but executable not found at expected path: $exePath"
}

Write-Host "[build] Success"
Write-Host "[build] Executable: $exePath"
