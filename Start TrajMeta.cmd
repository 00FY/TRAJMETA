@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Set up the project virtual environment first. See README.md.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8765 in your browser after the server starts.
".venv\Scripts\python.exe" webapp\server.py
pause
