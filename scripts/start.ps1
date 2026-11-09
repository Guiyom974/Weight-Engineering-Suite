# One-command start for Windows PowerShell: creates .venv, installs the suite, creates the demo workspace if
# needed, and opens the guided Studio. Re-running it reuses everything already set up.
# Usage: scripts\start.ps1 [-Workspace demo]
# If script execution is disabled on your machine, run:  powershell -ExecutionPolicy Bypass -File scripts\start.ps1
param([string]$Workspace = "demo")
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    Write-Host "Creating the virtual environment (.venv)..."
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { python -m venv .venv }
}
$python = Join-Path ".venv" "Scripts\python.exe"
$neuroforge = Join-Path ".venv" "Scripts\neuroforge.exe"

if (-not (Test-Path $neuroforge)) {
    Write-Host "Installing the suite (first run downloads PyTorch, a few minutes)..."
    & $python -m pip install --upgrade pip
    & $python -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

if (-not (Test-Path (Join-Path $Workspace "neuroforge.yaml"))) {
    Write-Host "Creating the demo workspace in $Workspace (toy model with a planted defect)..."
    & $neuroforge init -w $Workspace --demo
    if ($LASTEXITCODE -ne 0) { throw "neuroforge init failed" }
}

& $neuroforge doctor -w $Workspace
& $neuroforge studio -w $Workspace
