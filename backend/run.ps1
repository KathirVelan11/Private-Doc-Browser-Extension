# Start the Private Doc backend. Requires Ollama running (ollama serve).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
Write-Host "Starting Private Doc backend on http://127.0.0.1:5000 ..." -ForegroundColor Cyan
python server.py
