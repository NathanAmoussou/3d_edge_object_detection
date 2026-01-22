#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements.orin_trt.txt}"
RUN_APT="${RUN_APT:-1}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-0}"
TRT_BIN_DIR="${TRT_BIN_DIR:-/usr/src/tensorrt/bin}"
WHEELS_DIR="${WHEELS_DIR:-$ROOT_DIR/wheels}"
DEFAULT_TORCH_WHL="https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
TORCH_WHL="${TORCH_WHL:-$DEFAULT_TORCH_WHL}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script to keep the venv active:"
  echo "  source scripts/setup_orin_trt.sh"
  exit 1
fi

if [[ "$RUN_APT" == "1" ]]; then
  sudo apt update
  sudo apt install -y \
    git \
    wget \
    python3.8 \
    python3.8-venv \
    python3-pip \
    python3-libnvinfer \
    python3-pycuda \
    python3-libnvinfer-dev \
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

mkdir -p "$WHEELS_DIR"

resolve_wheel() {
  local src="$1"
  if [[ "$src" =~ ^https?:// ]]; then
    local fname
    fname="$(basename "${src%%\?*}")"
    local dst="$WHEELS_DIR/$fname"
    echo "Downloading wheel: $src -> $dst"
    wget -q --show-progress -O "$dst" "$src"
    echo "$dst"
  else
    echo "$src"
  fi
}

python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

TORCH_WHL_LOCAL="$(resolve_wheel "$TORCH_WHL")"
python -m pip install --no-cache-dir --force-reinstall "$TORCH_WHL_LOCAL"

if [[ -n "${TORCHVISION_WHL:-}" ]]; then
  TORCHVISION_WHL_LOCAL="$(resolve_wheel "$TORCHVISION_WHL")"
  python -m pip install --no-cache-dir --force-reinstall "$TORCHVISION_WHL_LOCAL"
fi

python -m pip install --prefer-binary -r "$REQUIREMENTS_FILE"

if [[ -d "$TRT_BIN_DIR" ]]; then
  export PATH="$TRT_BIN_DIR:$PATH"
fi
export CUDA_MODULE_LOADING=LAZY

if ! command -v trtexec >/dev/null 2>&1; then
  echo "Error: trtexec not found in PATH. Expected under $TRT_BIN_DIR."
  return 1
fi

python - <<'PY'
import sys

print("python:", sys.version.split()[0])

try:
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
except Exception as exc:
    raise SystemExit(f"tensorrt import failed: {exc}")

try:
    import pycuda.driver as cuda
    cuda.init()
    print("cuda devices:", cuda.Device.count())
except Exception as exc:
    raise SystemExit(f"pycuda init failed: {exc}")

try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() == False (CUDA build required)")
except Exception as exc:
    raise SystemExit(f"torch import/validation failed: {exc}")
PY

echo "Venv active: $VENV_DIR"
