# Music Genre App (Flask)

## Prerequisites
- Install Python (recommended: 3.10 or 3.11).
- Have `best_model.h5` and `genres.pkl` available (by default next to `app.py`).
- This app expects a TensorFlow/Keras version compatible with the model file (the bundled `best_model.h5` was saved with Keras 3).

## Setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or run:
```powershell
.\setup.ps1
```

If you get a Keras version mismatch error (the bundled model was saved with Keras 3), rebuild the env:
```powershell
.\setup.ps1 -Rebuild
```

## Run
```powershell
.\.venv\Scripts\python.exe app.py
```
Open `http://127.0.0.1:5000/`.

## Pages
- `/` Home
- `/classify` Upload + predict
- `/about` About
- `/contact` Contact

Or run:
```powershell
.\run.ps1
```

If typing `python` opens the Microsoft Store, use `py -3.11` or `.\.venv\Scripts\python.exe` instead.

## VS Code interpreter
This repo includes `.vscode/settings.json` to default the interpreter to `.venv`.

## Paths (optional)
- `MODEL_PATH` (default: `best_model.h5`)
- `GENRES_PATH` (default: `genres.pkl`)
- `UPLOAD_DIR` (default: `uploads`)
- `HOST` / `PORT`
