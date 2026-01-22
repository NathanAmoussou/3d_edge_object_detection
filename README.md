# 3D EDGE Object Detection

Projet dans le cadre du cours EIEIN909 - ECUE Edge Computing et IA embarquée
de Master 2 de l'Université Côte d'Azur, par le prof Gérald Rocher.
Projet réalisé par **[Noé FLORENCE](https://github.com/NoeFBou)**,
**[Nathan AMOUSSOU](https://github.com/NathanAmoussou)**,
**[Louis MALMASSARY](https://github.com/Kitsunro)**.

Ce dépôt benchmarke YOLO11 sur OAK-D Pro, Jetson Orin Nano et Raspberry Pi 4.
Objectif: comparer des variantes de YOLO11 (élagage n/s/m, résolution, quantification...) avec une boucle de
benchmark identique (préprocess → inference → postprocess → métriques).

## Installer les dépendances nécessaires

Résumé:
- Orin (JetPack 5.x): Python 3.8 requis.
- Pi + OAK (et tout host OAK): utiliser `requirements.txt`.
- Orin (TRT only): utiliser `requirements.orin_trt.txt` + torch Jetson (voir ci-dessous).
- Note: `onnxruntime-gpu` n'est pas dispo sur Pi (CPU only).
- CPU/GPU 4070: utiliser `requirements.txt` (GPU requiert drivers NVIDIA + CUDA/cuDNN).
- Note: OAK et GPU ont besoin d'un host (machine qui lance les scripts et pilote le device).

Pi / OAK (ou host générique):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Orin (TRT only):
```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.orin_trt.txt
```

Torch sur Orin (peut être capricieux). Exemple de séquence qui marche:
```bash
pip3 uninstall -y torch torchvision ultralytics

sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'

wget \
  https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl \
  -O torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip3 install numpy torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

sudo apt-get install -y \
  libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev \
  libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.1
python3 setup.py install --user
cd ../
pip3 install 'pillow<7'

# Reinstaller le reste sans tout casser:
pip3 install ultralytics --no-deps
pip3 install ultralytics onnxruntime-gpu --no-deps
```

## Commandes principales

Toutes les commandes sont à lancer depuis la racine du dépôt.

1. Générer les variantes de YOLO11 nécessaire au benchmark :
    1. Générer les variantes de bases : `python scripts/generate_variants.py`.
    2. Compiler les variantes pour Oak : `python scripts/compile.py --target oak`.
    3. Compiler les variantes pour Orin (à effectuer sur la Orin uniquement) : `python scripts/compile.py --target orin_trt`.
2. Lancer les benchmarks :
    1. Sur Oak : brancher la caméra sur l'host (son PC personnel, ou la Pi par exemple), puis : `python scripts/benchmark.py --target oak`.
    2. Sur Pi : `python scripts/benchmark.py --target pi4`.
    3. Sur Orin : `python scripts/benchmark.py --target orin_trt`.
3. Les résultats des benchmarks sont inscrits dans `benchmark_results.csv`.

## Structure rapide
- `scripts/generate_variants.py`: génère les variantes ONNX (n/s/m × résolutions × fp32/fp16) à partir des poids Ultralytics.
- `scripts/compile.py`: compile les ONNX vers les formats natifs (blob OAK, engine TensorRT Orin) selon la cible.
- `scripts/benchmark.py`: exécute un benchmark multi‑hardware, calcule les métriques et écrit le CSV.
- `scripts/transform.py`: tente d'appliquer des transformations runtime (fusion ORT, INT8 QDQ) pour préparer des variantes optimisées (échec).
- `models/base`: contient les poids Ultralytics (yolo11n/s/m .pt) utilisés comme point de départ.
- `models/variants`: stocke les variantes ONNX générées par `generate_variants.py`.
- `models/transformed`: regroupe les variantes ONNX transformées (fusion ORT / INT8 QDQ).
- `models/oak`: contient les blobs compilés pour OAK‑D.
- `models/orin`: contient les engines TensorRT compilés pour Orin.
