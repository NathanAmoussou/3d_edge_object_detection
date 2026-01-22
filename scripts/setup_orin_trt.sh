#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements.orin_trt.txt}"
RUN_APT="${RUN_APT:-1}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-0}"
TRT_BIN_DIR="${TRT_BIN_DIR:-/usr/src/tensorrt/bin}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script to keep the venv active:"
  echo "  source scripts/setup_orin_trt.sh"
  exit 1
fi

if [[ "$RUN_APT" == "1" ]]; then
  sudo apt update
  sudo apt install -y \
    git \
    python3.8-venv \
    python3-pip \
    python3-libnvinfer \
    python3-pycuda \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvonnxparsers8 \
    libnvparsers8

  if [[ "$INSTALL_BUILD_DEPS" == "1" ]]; then
    sudo apt install -y \
      build-essential \
      pkg-config \
      python3.8-dev \
      libboost-python-dev \
      libboost-thread-dev || true
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN"
  return 1
fi

"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel

if [[ -z "${TORCH_WHL:-}" ]]; then
  echo "Set TORCH_WHL to the Jetson torch wheel URL or local path."
  echo "Example: TORCH_WHL=/path/to/torch.whl source scripts/setup_orin_trt.sh"
  return 1
fi

python -m pip install "$TORCH_WHL"
if [[ -n "${TORCHVISION_WHL:-}" ]]; then
  python -m pip install "$TORCHVISION_WHL"
fi

python -m pip install --prefer-binary -r "$REQUIREMENTS_FILE"

if [[ -d "$TRT_BIN_DIR" ]]; then
  export PATH="$TRT_BIN_DIR:$PATH"
fi
export CUDA_MODULE_LOADING=LAZY

if ! command -v trtexec >/dev/null 2>&1; then
  echo "Warning: trtexec not found in PATH. Expected under $TRT_BIN_DIR."
fi

echo "Venv active: $VENV_DIR"
