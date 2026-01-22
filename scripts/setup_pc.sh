#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements.txt}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script to keep the venv active:"
  echo "  source scripts/setup_pc.sh"
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r "$REQUIREMENTS_FILE"

echo "Venv active: $VENV_DIR"
