param(
  [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$PythonCandidates = @()
$PythonCandidates += Join-Path $env:LOCALAPPDATA "Programs\\Python\\Python311\\python.exe"
$PythonCandidates += Join-Path $env:LOCALAPPDATA "Programs\\Python\\Python310\\python.exe"
$PythonCandidates += Join-Path $env:LOCALAPPDATA "Programs\\Python\\Python312\\python.exe"
$PythonCandidates += Join-Path $env:LOCALAPPDATA "Programs\\Python\\Python313\\python.exe"

try {
  $PythonCandidates += (Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA "Programs\\Python") -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName "python.exe" })
} catch { }

function Resolve-PythonExe {
  foreach ($cand in $PythonCandidates) {
    if ($cand -and (Test-Path $cand)) { return $cand }
  }

  if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
      $pyList = py -0p 2>$null
      if ($pyList -and ($pyList -notmatch "No installed Pythons found")) {
        return "py"
      }
    } catch { }
  }

  throw ("Python not found. Install Python 3.10+ (and ensure a real python.exe exists; " +
         "the Microsoft Store stub at WindowsApps\\python.exe won't work).")
}

$VenvMarker = ".venv_path"
$VenvDir = ".venv"
if (Test-Path $VenvMarker) {
  $markerValue = (Get-Content -Raw $VenvMarker).Trim()
  if ($markerValue) { $VenvDir = $markerValue }
}
$VenvDirPath = Join-Path $PSScriptRoot $VenvDir

if ($Rebuild -and (Test-Path $VenvDirPath)) {
  Write-Host "Rebuilding virtualenv ($VenvDir)..."
  try {
    Remove-Item -Recurse -Force $VenvDirPath
  } catch {
    $fallback = ".venv_keras3"
    if (Test-Path $fallback) {
      $fallback = ".venv_keras3_{0}" -f (Get-Date -Format "yyyyMMddHHmmss")
    }
    Write-Host "WARN: Could not delete $VenvDir (files locked). Creating $fallback instead."
    $VenvDir = $fallback
    $VenvDirPath = Join-Path $PSScriptRoot $VenvDir
    Set-Content -NoNewline -Encoding UTF8 $VenvMarker $VenvDir
  }
}

if (-not (Test-Path (Join-Path $VenvDirPath "Scripts\\python.exe"))) {
  $BasePython = Resolve-PythonExe
  if ($BasePython -eq "py") {
    py -3 -m venv $VenvDirPath
  } else {
    & $BasePython -m venv $VenvDirPath
  }
}

$Py = Join-Path $VenvDirPath "Scripts\\python.exe"

& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }
& $Py -m pip install --upgrade -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed (exit $LASTEXITCODE). If this is a network error, re-run with network access enabled." }

Write-Host "Installed:"
& $Py -c "import tensorflow as tf, keras; print('  tensorflow', tf.__version__); print('  keras', keras.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed (exit $LASTEXITCODE)." }

Write-Host "OK: Environment ready. Run: .\\run.ps1"
