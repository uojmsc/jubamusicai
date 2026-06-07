@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
if exist ".venv_path" (
  for /f "usebackq delims=" %%A in (".venv_path") do set "VENV_DIR=%%A"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Virtualenv not found. Run setup.ps1 first.
  exit /b 1
)

"%VENV_DIR%\Scripts\python.exe" app.py
