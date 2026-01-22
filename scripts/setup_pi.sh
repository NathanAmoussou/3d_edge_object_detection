#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements.pi.txt}"
RUN_APT="${RUN_APT:-1}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script to keep the venv active:"
  echo "  source scripts/setup_pi.sh"
  exit 1
fi

if [[ "$RUN_APT" == "1" ]]; then
  sudo apt update
  sudo apt install -y git python3-venv python3-pip
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install --prefer-binary -r "$REQUIREMENTS_FILE"

echo "Venv active: $VENV_DIR"
