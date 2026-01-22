# 3D EDGE Object Detection

Projet dans le cadre du cours EIEIN909 - ECUE Edge Computing et IA embarquée
de Master 2 de l'Université Côte d'Azur, par le prof Gérald Rocher.
Projet réalisé par **[Noé FLORENCE](https://github.com/NoeFBou)**,
**[Nathan AMOUSSOU](https://github.com/NathanAmoussou)**,
**[Louis MALMASSARY](https://github.com/Kitsunro)**.

Ce dépôt benchmarke YOLO11 sur OAK-D Pro, Jetson Orin Nano et Raspberry Pi 4.
Objectif: comparer des variantes de YOLO11 (élagage n/s/m, résolution, quantification...) avec une boucle de
benchmark identique (préprocess → inference → postprocess → métriques).

## Installation des dépendances

Scripts d'installation (à sourcer pour activer le venv):
```bash
# PC
source scripts/setup_pc.sh

# Pi
source scripts/setup_pi.sh

# Orin (TRT only) - nécessite un wheel torch Jetson
TORCH_WHL=/path/to/torch.whl TORCHVISION_WHL=/path/to/torchvision.whl \
  source scripts/setup_orin_trt.sh
```

Notes:
- `RUN_APT=0` permet de sauter la phase `apt` si déjà fait.
- `PYTHON_BIN=/usr/bin/python3.11` ou autre pour forcer le binaire Python.



## Commandes principales

1. Générer les variantes de YOLO11 nécessaire au benchmark :
    1. Générer les variantes de bases : `python scripts/generate_variants.py`.
    2. Compiler les variantes pour Oak : `python scripts/compile.py --target oak`.
    3. Compiler les variantes pour Orin (à effectuer sur la Orin uniquement) : `python scripts/compile.py --target orin_trt`.
2. Lancer les benchmarks :
    1. Sur Oak : brancher la caméra sur l'host (son PC personnel, ou la Pi par exemple), tirer ce dépôt GitHub et installer les dépendances nécessaires, puis : `python scripts/benchmark.py --target oak`.
    2. Sur Pi : tirer ce dépôt GitHub et installer les dépendances nécessaires, puis : `python scripts/benchmark.py --target pi4`.
    3. Sur Orin : tirer ce dépôt GitHub et installer les dépendances nécessaires, puis : `python scripts/benchmark.py --target orin_trt`.
3. Les résultats des benchmarks sont inscrits dans `benchmark_results.csv`.

## Prérequis
- Python 3.8 sur Jetson Orin (JetPack 5.x). Sur PC, Python 3.11 OK.
- Paquets dans `requirements.txt`
- Accès réseau (download YOLO11 et COCO128/COCO)
- Commandes lancées depuis la racine du repo

Dépendances spécifiques:
- OAK-D Pro: `depthai`, `blobconverter`
- Orin: JetPack + TensorRT + `trtexec` dans le PATH + Python `tensorrt`
- Pi 4: `onnxruntime` CPU (pas `onnxruntime-gpu`)

Installation (machine de génération / host):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- Sur Pi/Orin, `onnxruntime-gpu` peut ne pas exister; installez `onnxruntime` ou les wheels Jetson adaptées.
- `psutil` est optionnel pour le monitoring; `pandas` est requis pour `scripts/summary_tables.py`.
- Sur Orin, utilisez `python3.8` pour le venv si `python3` n'est pas 3.8.

## Structure rapide
- `models/base`: poids Ultralytics (yolo11n/s/m .pt)
- `models/variants`: ONNX générés
- `models/transformed`: ONNX transformés (ORT fusion / INT8 QDQ)
- `models/oak`: blobs OAK
- `models/orin`: engines TensorRT
- `scripts/generate_variants.py`, `scripts/transform.py`, `scripts/compile.py`, `scripts/benchmark.py`

## 1) Re-générer les modèles d'IA

### 1.1 Générer les variantes ONNX
```bash
python scripts/generate_variants.py
```

Exemples:
```bash
python scripts/generate_variants.py --quantizations fp32
python scripts/generate_variants.py --models n --resolutions 640 416
```

Sortie: `models/variants/yolo11{n,s,m}_{imgsz}_{fp32|fp16}.onnx`
Note: l'export FP16 requiert CUDA; sinon les variantes FP16 sont ignorées (utiliser `--quantizations fp32`).

### 1.2 Transformations optionnelles
Fusion ORT (pour bench ORT uniquement, non portable OAK/TRT):
```bash
python scripts/transform.py --pattern "yolo11*_*_fp16.onnx" --fusion all --output models/transformed
```

INT8 QDQ (portable TRT/ORT):
```bash
python scripts/transform.py --pattern "yolo11n_*_fp32.onnx" --int8 --calib-dataset coco128 --calib-size 100 --output models/transformed
```

### 1.3 Compiler pour OAK-D Pro (BLOB)
```bash
python scripts/compile.py --target oak
python scripts/compile.py --target oak --model models/variants/yolo11n_640_fp16.onnx --shaves 8
```

Notes:
- Myriad X = FP16 seulement.
- Les modèles `_ortopt_` ou `_int8_qdq` ne sont pas supportés.

### 1.4 Compiler pour Jetson Orin (TensorRT ENGINE)
A exécuter sur l'Orin (TRT uniquement):
```bash
python scripts/compile.py --target orin_trt
```

Notes:
- `trtexec` doit être présent; l’engine est spécifique au GPU local.
- Les modèles `_ortopt_` ne sont pas supportés.

### 1.5 Copier les artefacts vers les devices
Copier `models/variants/` + `models/oak/` ou `models/orin/` selon la cible.

## 2) Relancer les benchmarks

### 2.1 OAK-D Pro
```bash
python scripts/benchmark.py --target oak
python scripts/benchmark.py --target oak --model models/oak/yolo11n_640_fp16_6shave.blob --repeat 3
```

### 2.2 Raspberry Pi 4
```bash
python scripts/benchmark.py --target pi4 --host-tag PI4 --cpu-threads 4
```

Note: le mode `pi4` fait un sweep `yolo11n` uniquement.

### 2.3 Jetson Orin
```bash
python scripts/benchmark.py --target orin_trt
```

### 2.4 Résultats
- CSV principal: `benchmark_results.csv` (append). Supprimez-le pour un run "clean".
- Power logging (optionnel): `power_timeseries.csv`
- Tableaux: `python scripts/summary_tables.py --csv benchmark_results.csv --out-dir tables` (nécessite `pandas`).

### Options utiles
- Dataset: `--dataset coco` (val2017) ou `--dataset coco128` (rapide, défaut).
- Monitoring: `--no-monitor` si `psutil`/`nvidia-smi`/`tegrastats` indisponibles.
- Power logging: `--tapo-ip ... --tapo-user ... --tapo-password ...` (ou variables `TAPO_*`).

## 3) Re-setup hardware (OAK / Pi / Orin)

### OAK-D Pro
- Installer les dépendances (`depthai`, `blobconverter`).
- Connecter la caméra en USB3.
- Sanity check: compiler un blob puis lancer un bench simple (voir section 2.1).

### Raspberry Pi 4
Matériel: microSD 32 GB minimum, câble Ethernet.

Flash OS + SSH fiable (résumé; voir `flash_the_pi.md` pour les détails):
1) Flasher Raspberry Pi OS (Imager) sur la microSD.
2) Sur Ubuntu, trouver le point de montage `bootfs`:
```bash
lsblk -o NAME,FSTYPE,SIZE,LABEL,MOUNTPOINTS
```
3) Forcer SSH + créer l’utilisateur sur la partition boot:
```bash
sudo touch /media/$USER/bootfs/ssh
HASH=$(echo '0000' | openssl passwd -6 -stdin)
echo "nathan:$HASH" | sudo tee /media/$USER/bootfs/userconf.txt
sync
```
4) Démarrer le Pi en Ethernet, trouver l’IP (éviter `.local`):
```bash
sudo nmap -sn 192.168.1.0/24
ssh nathan@<PI_IP>
```

Installer le projet:
```bash
sudo apt update
sudo apt install -y git
mkdir -p ~/src
cd ~/src
git clone https://github.com/NathanAmoussou/3d_edge_object_detection.git
cd 3d_edge_object_detection
```

Créer un venv et installer les deps (CPU):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Notes Pi:
- `onnxruntime-gpu` n’est pas dispo sur Pi → utiliser `onnxruntime`.
- `opencv-python` 4.12+ force `numpy>=2`; rester sur `opencv-python==4.11.0.86` si vous gardez `numpy==1.26.4`.

Paquets Pi: `requirements.pi.txt` (dans le repo).

Bench:
```bash
python scripts/benchmark.py --target pi4 --host-tag PI4 --cpu-threads 4
```

### Jetson Orin Nano
Objectif: reproduire uniquement avec des `.engine` TensorRT (pas d’ORT).

Vérifications JetPack / TensorRT:
```bash
dpkg -l | grep -E "nvidia-jetpack|nvidia-l4t-core"
cat /etc/nv_tegra_release
which trtexec || export PATH=/usr/src/tensorrt/bin:$PATH
trtexec --help | egrep -i "heuristic|memPoolSize|buildOnly|skipInference"
```

Clean install (from scratch, Python 3.8):
```bash
rm -rf ~/src/3d_edge_object_detection
rm -rf ~/.cache/pip

sudo apt update
sudo apt install -y git python3.8-venv python3-pip

mkdir -p ~/src
cd ~/src
git clone https://github.com/NathanAmoussou/3d_edge_object_detection.git
cd 3d_edge_object_detection
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Installer PyTorch CUDA (Jetson):
- Utiliser une build Jetson (pas `+cpu`). Sinon `torch.cuda.is_available()` sera `False`
  et le benchmark TRT plantera.
- Exemple de vérif:
```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
PY
```

Installer les deps Python (minimum TRT):
```bash
python -m pip install --prefer-binary -r requirements.orin_trt.txt
```

Compilation TRT (sweep):
```bash
export PATH=/usr/src/tensorrt/bin:$PATH
export CUDA_MODULE_LOADING=LAZY
python scripts/compile.py --target orin_trt
```

Benchmark TRT:
```bash
python scripts/benchmark.py --target orin_trt
```

Notes:
- Si `trtexec` se plaint d’options inconnues, faire `git pull` (le script adapte les flags).
- Si certaines combinaisons OOM/tactic échouent, relancer après nettoyage avec `--skip-existing`.

## Notes utiles
- Le dataset COCO128/COCO est téléchargé automatiquement par Ultralytics au premier run.
- Le pipeline de bench utilise un NMS côté host (modèles exportés avec `nms=False`).
- Détails des bins d’optimisation et conventions de noms: `AI_models.md`.
