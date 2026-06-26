@echo off
REM Start the Private Doc backend (requires Ollama running: "ollama serve").
cd /d "%~dp0"
echo Starting Private Doc backend on http://127.0.0.1:5000 ...
python server.py
pause
