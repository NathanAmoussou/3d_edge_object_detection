# Orin

# 0) Préparer le repo
```
sudo apt update
sudo apt install -y git wget
mkdir -p ~/src
cd ~/src
git clone https://github.com/NathanAmoussou/3d_edge_object_detection.git
cd 3d_edge_object_detection
```

# 1) Paquets système (TensorRT + PyCUDA + Python 3.8)
```
sudo apt install -y \
  python3.8 python3.8-venv python3-pip \
  python3-libnvinfer python3-libnvinfer-dev \
  python3-pycuda \
  libnvinfer8 libnvinfer-plugin8 \
  libnvonnxparsers8 libnvparsers8
```

# 2) Venv propre (Python 3.8) + pip
```
rm -rf .venv
python3.8 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

# 3) Torch CUDA Jetson (JP 5.1.1 par défaut)
```
mkdir -p wheels
wget -O wheels/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl \
  https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
python -m pip install --no-cache-dir --force-reinstall wheels/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

# (Optionnel) torchvision si tu as un wheel local:
# python -m pip install --no-cache-dir --force-reinstall /path/to/torchvision.whl
```

# 4) Déps projet TRT
```
python -m pip install --prefer-binary -r requirements.orin_trt.txt
```

# 5) PATH + CUDA lazy loading (dans ce terminal)
```
export PATH="/usr/src/tensorrt/bin:$PATH"
export CUDA_MODULE_LOADING=LAZY
```

# 6) Vérifs rapides
```
trtexec --version
python - <<'PY'
import tensorrt as trt, pycuda.driver as cuda, torch
cuda.init()
print("tensorrt", trt.__version__)
print("cuda devices", cuda.Device.count())
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("torch.cuda.is_available()", torch.cuda.is_available())
PY
```

# Pi

# 0) Préparer le repo
```
sudo apt update
sudo apt install -y git python3-venv python3-pip
mkdir -p ~/src
cd ~/src
git clone https://github.com/NathanAmoussou/3d_edge_object_detection.git
cd 3d_edge_object_detection
```

# 1) Venv propre
```
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

# 2) Déps Pi (CPU)
```
python -m pip install --prefer-binary -r requirements.pi.txt
```

# 3) Test rapide
```
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
