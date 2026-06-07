$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$VenvMarker = ".venv_path"
$VenvDir = ".venv"
if (Test-Path $VenvMarker) {
  $markerValue = (Get-Content -Raw $VenvMarker).Trim()
  if ($markerValue) { $VenvDir = $markerValue }
}
$VenvDirPath = Join-Path $PSScriptRoot $VenvDir
$Py = Join-Path $VenvDirPath "Scripts\\python.exe"

if (-not (Test-Path $Py)) {
  Write-Host "Virtualenv not found; running setup..."
  & .\setup.ps1
}

try {
  $envCheck = @'
import sys
from pathlib import Path

try:
    import h5py
except Exception:
    h5py = None

try:
    import keras
except Exception:
    keras = None

p = Path("best_model.h5")
if not p.exists() or h5py is None or keras is None:
    sys.exit(0)

with h5py.File(p, "r") as f:
    v = f.attrs.get("keras_version")
    if v is None:
        sys.exit(0)
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")

saved_major = str(v).split(".", 1)[0]
runtime_major = str(getattr(keras, "__version__", "")).split(".", 1)[0]
if saved_major and runtime_major and saved_major != runtime_major:
    print(
        f"ERROR: Keras version mismatch. best_model.h5 was saved with Keras {v}, "
        f"but this environment has Keras {keras.__version__}."
    )
    print("Fix: run: .\\setup.ps1 -Rebuild")
    sys.exit(2)
'@
  $envCheck | & $Py -
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} catch {
  # If env check fails for any reason, let app.py surface a useful error.
}

& $Py app.py
