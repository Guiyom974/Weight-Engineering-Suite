#!/usr/bin/env bash
# One-command start for macOS and Linux: creates .venv, installs the suite, creates the demo workspace if needed,
# and opens the guided Studio. Re-running it reuses everything already set up.
# Usage: scripts/start.sh [workspace-dir]   (default: demo)
set -euo pipefail

cd "$(dirname "$0")/.."
WORKSPACE="${1:-demo}"
PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "Creating the virtual environment (.venv)..."
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! command -v neuroforge >/dev/null 2>&1; then
  echo "Installing the suite (first run downloads PyTorch, a few minutes)..."
  python -m pip install --upgrade pip
  python -m pip install -e .
fi

if [ ! -f "$WORKSPACE/neuroforge.yaml" ]; then
  echo "Creating the demo workspace in $WORKSPACE (toy model with a planted defect)..."
  neuroforge init -w "$WORKSPACE" --demo
fi

neuroforge doctor -w "$WORKSPACE" || true
exec neuroforge studio -w "$WORKSPACE"
