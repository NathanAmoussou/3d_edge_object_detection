"""
Benchmark d'un modele YOLO sur GPU, CPU, OAK-D ou Jetson Orin.

Usage:
    # GPU (RTX 4070)
    python scripts/benchmark.py --target 4070 --model models/variants/yolo11n_640_fp16.onnx

    # CPU (PC ou Raspberry Pi 4)
    python scripts/benchmark.py --target cpu --model models/variants/yolo11n_640_fp32.onnx --host-tag PC --cpu-threads 8
    python scripts/benchmark.py --target cpu --model models/variants/yolo11n_320_fp32.onnx --host-tag PI4 --cpu-threads 4

    # OAK-D (Myriad X VPU)
    python scripts/benchmark.py --target oak --model models/oak/yolo11n_640_fp16_6shave.blob --host-tag PC
    python scripts/benchmark.py --target oak --model models/oak/yolo11n_640_fp16 --shaves 8 --host-tag PI4

    # Jetson Orin
    python scripts/benchmark.py --target orin --model models/orin/yolo11n_640_fp16.engine --num-classes 80

Options:
    --target 4070|cpu|oak|orin : Cible hardware
    --model PATH               : Chemin vers le modele (.onnx pour GPU/CPU, .blob pour OAK, .engine pour Orin)
    --num-classes N            : Nombre de classes du modele (defaut: 80 pour COCO)
    --dataset coco128|coco     : Dataset a utiliser (defaut: coco128)
    --backend ort|ultralytics  : Backend GPU (defaut: ort pour ONNX Runtime, meme postprocess que OAK)
    --host-tag TAG             : Tag identifiant le host (ex: PC, PI4). Auto-detecte si non fourni.
    --cpu-threads N            : Nombre de threads intra-op pour inference CPU (defaut: auto)
    --cpu-execution-mode       : Mode d'execution ORT CPU: sequential ou parallel (defaut: sequential)
    --shaves N                 : Nombre de shaves OAK (4-8) (auto-complete le nom du blob: *_{N}shave.blob)

===============================================================================
DECISIONS DE CONCEPTION - EQUITE MULTI-HARDWARE
===============================================================================

1. MEME ONNX SOURCE
   -----------------
   Toutes les targets partent du meme ONNX (genere par generate_variants.py).
   - GPU: execute l'ONNX via ONNX Runtime CUDA/TensorRT EP
   - CPU: execute l'ONNX via ONNX Runtime CPUExecutionProvider
   - OAK: ONNX compile en BLOB
   - Orin: ONNX compile en ENGINE

2. MEME PIPELINE DE BENCHMARK
   ---------------------------
   Toutes les branches utilisent la meme boucle:
   - load_coco128_dataset() -> memes images
   - letterbox() -> meme preprocessing (imgsz dynamique)
   - inference -> decode_yolo_output() -> apply_nms() -> boxes_to_predictions()
   - compute_map50_prf1() -> memes metriques

3. PAS DE NMS DANS LES ARTEFACTS
   ------------------------------
   Les modeles sont exportes avec nms=False (generate_variants.py).
   Le NMS est fait cote host avec les memes parametres:
   - NMS_IOU = 0.70
   - MAX_DET = 300
   - CONF_EVAL = 0.001 (pour mAP), CONF_OP = 0.25 (pour P/R/F1)

4. IMGSZ DYNAMIQUE
   ----------------
   La taille d'entree est parsee depuis le nom du fichier modele.
   Format attendu: yolo11{scale}_{imgsz}_{quant}.{ext}
   Exemple: yolo11n_416_fp16.onnx -> imgsz=416

5. TIMING : CE QUI EST MESURE
   ---------------------------
   ATTENTION: Device_Time_ms n'a pas la meme signification selon la target!

   - GPU (ORT): temps session.run() = inference pure
   - CPU (ORT): temps session.run() = inference pure
   - OAK: temps send() -> get() = USB + VPU + USB (pas VPU seul)
   - Orin: temps session.run() ou result.speed["inference"]

   Pour une comparaison "fair", utiliser E2E_Time_ms qui inclut tout
   (preprocess + inference + postprocess) et est mesure de la meme facon.

6. CSV UNIFIE
   -----------
   save_results() ecrit les memes colonnes pour toutes les targets.

7. HOST TAGGING
   -------------
   Pour distinguer les runs sur differents hosts (PC vs Pi4), un tag est
   enregistre dans la colonne Hardware:
   - CPU: CPU_ORT_{host_tag}_{exec_mode} (ex: CPU_ORT_PC_x86_64_SEQ)
   - OAK: OAK_MyriadX_HOST_{host_tag} (ex: OAK_MyriadX_HOST_PI4_aarch64)

   Le host_tag peut etre specifie via --host-tag ou auto-detecte.

===============================================================================
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.metrics import ap_per_class

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent.resolve()
RESULTS_FILE = ROOT_DIR / "benchmark_results.csv"

# --- Seuils de confiance ---
CONF_EVAL = 0.001  # seuil pour la courbe PR complete (mAP)
CONF_OP = 0.25  # seuil "operationnel" pour P/R/F1

# --- Parametres NMS (uniformes sur toutes les targets) ---
NMS_IOU = 0.70
MATCH_IOU = 0.50
MAX_DET = 300

# --- Warmup ---
WARMUP_FRAMES = 10


# =============================================================================
# UTILITAIRES
# =============================================================================


def is_int8_qdq_model(model_path: str) -> bool:
    """
    Detecte si un modele ONNX est un modele INT8 QDQ.

    Detection via:
    1. Nom du fichier contient "_int8_qdq"
    2. Metadata JSON indique int8_format="QDQ"
    3. (Fallback) Scan ONNX pour noeuds QuantizeLinear/DequantizeLinear
    """
    path = Path(model_path)

    # 1. Detection par nom
    if "_int8_qdq" in path.stem:
        return True

    # 2. Detection par metadata JSON
    json_path = path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                metadata = json.load(f)
                if metadata.get("int8_format") == "QDQ":
                    return True
                if metadata.get("int8_calibration") is True:
                    return True
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. Scan ONNX (plus lent, fallback)
    try:
        import onnx

        model = onnx.load(model_path)
        for node in model.graph.node:
            if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                return True
    except Exception:
        pass

    return False


def compute_timing_stats(
    e2e_times: list[float],
    preprocess_times: list[float],
    inference_times: list[float],
    postprocess_times: list[float],
) -> dict:
    """
    Calcule les statistiques de timing etendues.

    Args:
        e2e_times: Liste des temps E2E en ms
        preprocess_times: Liste des temps preprocess en ms
        inference_times: Liste des temps inference en ms
        postprocess_times: Liste des temps postprocess en ms

    Returns:
        Dict avec: e2e_mean, latency_p50/p90/p95/p99, preprocess_ms,
                   inference_ms, postprocess_ms, fps
    """
    if not e2e_times:
        return {
            "e2e_mean": 0.0,
            "latency_p50": 0.0,
            "latency_p90": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "fps": 0.0,
        }

    e2e_arr = np.array(e2e_times)

    e2e_mean = float(np.mean(e2e_arr))
    fps = 1000.0 / e2e_mean if e2e_mean > 0 else 0.0

    return {
        "e2e_mean": e2e_mean,
        "latency_p50": float(np.percentile(e2e_arr, 50)),
        "latency_p90": float(np.percentile(e2e_arr, 90)),
        "latency_p95": float(np.percentile(e2e_arr, 95)),
        "latency_p99": float(np.percentile(e2e_arr, 99)),
        "preprocess_ms": float(np.mean(preprocess_times)) if preprocess_times else 0.0,
        "inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "postprocess_ms": float(np.mean(postprocess_times))
        if postprocess_times
        else 0.0,
        "fps": fps,
    }


def parse_imgsz_from_filename(filepath: str) -> int:
    """
    Parse imgsz depuis le nom du fichier.
    Format attendu: yolo11{scale}_{imgsz}_{quant}.{ext}
    Exemple: yolo11n_416_fp16.onnx -> 416

    Fallback: lit le .json metadata si disponible.
    """
    path = Path(filepath)
    stem = path.stem  # yolo11n_416_fp16

    # Essayer de parser depuis le nom
    parts = stem.split("_")
    if len(parts) >= 2:
        try:
            imgsz = int(parts[1])
            if imgsz % 32 == 0 and 128 <= imgsz <= 1280:
                return imgsz
        except ValueError:
            pass

    # Fallback: lire le .json metadata
    json_path = path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                metadata = json.load(f)
                if "imgsz" in metadata:
                    return int(metadata["imgsz"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback ultime
    print(f"  [Warning] Impossible de parser imgsz depuis {path.name}, utilise 640")
    return 640


def get_model_size_mb(filepath: str) -> float:
    """Retourne la taille du fichier en MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def calculate_f1(precision: float, recall: float) -> float:
    """Calcule le F1-score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def get_versions():
    """Retourne les versions des librairies cles."""
    versions = {}
    try:
        import ultralytics

        versions["ultralytics"] = ultralytics.__version__
    except (ImportError, AttributeError):
        versions["ultralytics"] = "N/A"
    try:
        import depthai

        versions["depthai"] = depthai.__version__
    except (ImportError, AttributeError):
        versions["depthai"] = "N/A"
    try:
        versions["opencv"] = cv2.__version__
    except AttributeError:
        versions["opencv"] = "N/A"
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    except ImportError:
        versions["torch"] = "N/A"
        versions["cuda"] = "N/A"
    try:
        import tensorrt

        versions["tensorrt"] = tensorrt.__version__
    except (ImportError, AttributeError):
        versions["tensorrt"] = "N/A"
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except (ImportError, AttributeError):
        versions["onnxruntime"] = "N/A"
    return versions


def detect_host_tag() -> str:
    """
    Auto-detecte un tag identifiant le host.

    Retourne un tag du style:
    - PC_x86_64 : PC standard
    - PI4_aarch64 : Raspberry Pi 4

    Detection basee sur platform.machine() + heuristique /proc/cpuinfo pour Pi4.
    """
    import platform

    machine = platform.machine()

    # Heuristique pour Raspberry Pi 4
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
            if "raspberry pi 4" in cpuinfo or "bcm2711" in cpuinfo:
                return f"PI4_{machine}"
    except (FileNotFoundError, PermissionError):
        pass

    # Heuristique basee sur l'architecture
    if machine in ("aarch64", "arm64"):
        # Potentiellement un Pi ou un Jetson
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                if "raspberry" in model:
                    return f"PI4_{machine}"
                if "jetson" in model:
                    return f"JETSON_{machine}"
        except (FileNotFoundError, PermissionError):
            pass
        return f"ARM_{machine}"

    # PC standard
    return f"PC_{machine}"


def save_results(
    hardware: str,
    model_name: str,
    size_mb: float,
    imgsz: int,
    e2e_time_ms: float,
    device_time_ms: float,
    map50: float,
    precision: float,
    recall: float,
    f1: float,
    dataset: str = "coco128",
    backend: str = "N/A",
    # Extended timing metrics (Phase 3)
    latency_p50: float = 0.0,
    latency_p90: float = 0.0,
    latency_p95: float = 0.0,
    latency_p99: float = 0.0,
    preprocess_ms: float = 0.0,
    inference_ms: float = 0.0,
    postprocess_ms: float = 0.0,
    fps: float = 0.0,
    # ORT config tracking (Phase 4)
    ort_level_requested: str = "N/A",
    providers_used: str = "N/A",
):
    """
    Sauvegarde les resultats dans le CSV avec metadonnees completes.

    Note sur Device_Time_ms:
    - GPU (ORT): inference pure (session.run)
    - OAK: USB + VPU + USB (pas VPU seul)
    - Orin: inference pure (session.run)

    Pour comparaison fair, utiliser E2E_Time_ms.

    Extended metrics (Phase 3):
    - latency_p50/p90/p95/p99: Percentiles E2E latence (ms)
    - preprocess_ms: Temps moyen preprocessing (ms)
    - inference_ms: Temps moyen inference (ms)
    - postprocess_ms: Temps moyen postprocessing (ms)
    - fps: Throughput (images/sec)
    """
    file_exists = RESULTS_FILE.exists()
    versions = get_versions()

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Hardware",
                    "Model_Name",
                    "Size_MB",
                    "ImgSz",
                    "E2E_Time_ms",
                    "Device_Time_ms",
                    "mAP50",
                    "Precision",
                    "Recall",
                    "F1",
                    # Extended timing (Phase 3)
                    "Latency_p50",
                    "Latency_p90",
                    "Latency_p95",
                    "Latency_p99",
                    "Preprocess_ms",
                    "Inference_ms",
                    "Postprocess_ms",
                    "FPS",
                    # ORT config tracking (Phase 4)
                    "ORT_Level_Requested",
                    "Providers_Used",
                    # Metadonnees
                    "Dataset",
                    "Backend",
                    "CONF_EVAL",
                    "CONF_OP",
                    "NMS_IOU",
                    "MATCH_IOU",
                    "MAX_DET",
                    "WARMUP",
                    # Versions
                    "ultralytics",
                    "depthai",
                    "opencv",
                    "torch",
                    "cuda",
                    "tensorrt",
                    "onnxruntime",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                hardware,
                model_name,
                f"{size_mb:.2f}",
                imgsz,
                f"{e2e_time_ms:.2f}",
                f"{device_time_ms:.2f}",
                f"{map50:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                # Extended timing (Phase 3)
                f"{latency_p50:.2f}",
                f"{latency_p90:.2f}",
                f"{latency_p95:.2f}",
                f"{latency_p99:.2f}",
                f"{preprocess_ms:.2f}",
                f"{inference_ms:.2f}",
                f"{postprocess_ms:.2f}",
                f"{fps:.2f}",
                # ORT config tracking (Phase 4)
                ort_level_requested,
                providers_used,
                # Metadonnees
                dataset,
                backend,
                CONF_EVAL,
                CONF_OP,
                NMS_IOU,
                MATCH_IOU,
                MAX_DET,
                WARMUP_FRAMES,
                # Versions
                versions["ultralytics"],
                versions["depthai"],
                versions["opencv"],
                versions["torch"],
                versions["cuda"],
                versions["tensorrt"],
                versions["onnxruntime"],
            ]
        )

    print(f"\nResultats sauvegardes dans: {RESULTS_FILE}")


def load_coco128_dataset(dataset_name: str = "coco128"):
    """Charge le dataset COCO128 ou COCO val2017 (images + labels)."""
    from ultralytics.data.utils import check_det_dataset

    if dataset_name == "coco":
        data = check_det_dataset("coco.yaml")
        val_images_dir = Path(data["path"]) / "images" / "val2017"
        val_labels_dir = Path(data["path"]) / "labels" / "val2017"
        expected_count = 5000
        print("[Dataset] COCO val2017 (baseline scientifique)")
    else:
        data = check_det_dataset("coco128.yaml")
        val_images_dir = Path(data["path"]) / "images" / "train2017"
        val_labels_dir = Path(data["path"]) / "labels" / "train2017"
        expected_count = 128
        print("[Dataset] coco128/train2017 (equite multi-hardware)")

    print(f"[Dataset] Chemin images: {val_images_dir}")
    print(f"[Dataset] Chemin labels: {val_labels_dir}")

    image_files = sorted(val_images_dir.glob("*.jpg"))
    actual_count = len(image_files)

    print(f"[Dataset] Images trouvees: {actual_count} (attendu: {expected_count})")

    if actual_count != expected_count:
        raise AssertionError(
            f"[ERREUR DATASET] Nombre d'images incorrect!\n"
            f"  - Attendu: {expected_count}\n"
            f"  - Trouve: {actual_count}\n"
            f"  - Chemin: {val_images_dir}"
        )

    dataset = []
    for img_path in image_files:
        label_path = val_labels_dir / (img_path.stem + ".txt")

        gt_boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        gt_boxes.append(
                            {
                                "class_id": cls_id,
                                "x_center": x_center,
                                "y_center": y_center,
                                "width": width,
                                "height": height,
                            }
                        )

        dataset.append(
            {
                "image_path": str(img_path),
                "gt_boxes": gt_boxes,
            }
        )

    return dataset


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Redimensionne l'image en gardant le ratio (ajoute du padding).
    Utilise la classe LetterBox d'Ultralytics.
    """
    from ultralytics.data.augment import LetterBox

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    transform = LetterBox(new_shape, auto=False, stride=32, center=True)

    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    img_lb = transform(image=img)

    return img_lb, ratio, (dw, dh)


def calculate_iou(box1, box2):
    """Calcule l'IoU entre deux boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_tp_fp_fn(preds, gts, iou_thr=0.5):
    """Associe predictions et GT (class-aware, une GT max par prediction)."""
    preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)
    tp = np.zeros((len(preds), 1), dtype=bool)
    conf = np.array([p["confidence"] for p in preds], dtype=float)
    pred_cls = np.array([p["class_id"] for p in preds], dtype=int)
    target_cls = np.array([g["class_id"] for g in gts], dtype=int)

    gt_matched = [False] * len(gts)
    for i, p in enumerate(preds):
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if gt_matched[j] or p["class_id"] != g["class_id"]:
                continue
            iou = calculate_iou(p["box"], g["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i, 0] = True
            gt_matched[best_j] = True
    return tp, conf, pred_cls, target_cls, gt_matched


def compute_map50_prf1(all_predictions, all_ground_truths, conf_op=0.25, iou_thr=0.5):
    """Calcule mAP50 (AP integrale) + P/R/F1 au seuil conf_op."""
    tp_all, conf_all, pred_cls_all, target_cls_all = [], [], [], []
    TP = FP = FN = 0

    for preds, gts in zip(all_predictions, all_ground_truths):
        tp, conf, pred_cls, target_cls, _ = match_tp_fp_fn(preds, gts, iou_thr=iou_thr)
        if len(conf):
            tp_all.append(tp)
            conf_all.append(conf)
            pred_cls_all.append(pred_cls)
        if len(target_cls):
            target_cls_all.append(target_cls)

        preds_op = [p for p in preds if p["confidence"] >= conf_op]
        tp_op, _, _, _, gt_matched = match_tp_fp_fn(preds_op, gts, iou_thr=iou_thr)

        tp_count = int(tp_op.sum())
        TP += tp_count
        FP += len(preds_op) - tp_count
        FN += sum(1 for m in gt_matched if not m)

    if not tp_all or not target_cls_all:
        return {"map50": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp_all = np.concatenate(tp_all, axis=0)
    conf_all = np.concatenate(conf_all, axis=0)
    pred_cls_all = np.concatenate(pred_cls_all, axis=0)
    target_cls_all = np.concatenate(target_cls_all, axis=0)

    _, _, _, _, _, ap_c, _, *_ = ap_per_class(
        tp_all, conf_all, pred_cls_all, target_cls_all
    )
    map50 = float(ap_c[:, 0].mean()) if ap_c.size else 0.0

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = calculate_f1(precision, recall)

    return {"map50": map50, "precision": precision, "recall": recall, "f1": f1}


def decode_yolo_output(raw_output, num_classes, imgsz, output_shape=None):
    """
    Decode la sortie brute YOLO (v8/v11 anchor-free ou v5/v7 avec objectness).

    Detecte automatiquement le layout:
    - (1, 84, n_anchors) -> standard Ultralytics v8/v11
    - (1, n_anchors, 84) -> transposed (certains exports TRT)
    - flatten -> reshape selon n_anchors attendu

    Args:
        raw_output: numpy array (flatten ou shaped)
        num_classes: nombre de classes
        imgsz: taille d'entree du modele (pour calculer n_anchors)
        output_shape: shape originale si disponible (tuple), sinon None

    Returns:
        boxes: (N, 4) en format [cx, cy, w, h] pixels
        scores: (N,) confiances
        class_ids: (N,) indices de classe
    """
    n_anchors = (imgsz // 8) ** 2 + (imgsz // 16) ** 2 + (imgsz // 32) ** 2
    n_attrs_v8 = num_classes + 4  # v8/v11: cx, cy, w, h, cls0..cls79
    n_attrs_v5 = num_classes + 5  # v5/v7: cx, cy, w, h, obj, cls0..cls79

    # Si on a la shape originale, l'utiliser pour detecter le layout
    if output_shape is not None and len(output_shape) >= 2:
        # Enlever batch dim si present
        shape = output_shape[-2:]  # (dim1, dim2)

        # Detecter layout: (84, n_anchors) vs (n_anchors, 84)
        if shape[0] == n_attrs_v8 or shape[0] == n_attrs_v5:
            # Layout standard: (n_attrs, n_anchors) -> pas de transpose
            pass
        elif shape[1] == n_attrs_v8 or shape[1] == n_attrs_v5:
            # Layout transpose: (n_anchors, n_attrs) -> deja dans le bon sens pour row-major
            raw_output = raw_output.reshape(shape).T.flatten()

    # Flatten pour traitement uniforme
    raw_output = raw_output.flatten()

    # Detecter format v5/v7 (objectness) vs v8/v11 (anchor-free)
    has_objectness = (raw_output.size % n_attrs_v5 == 0) and (
        raw_output.size // n_attrs_v5 == n_anchors
    )

    if has_objectness:
        # Format v5/v7: [cx, cy, w, h, obj, cls0, cls1, ...]
        out = raw_output.reshape(n_attrs_v5, -1).T
        boxes = out[:, :4]
        obj = out[:, 4]
        cls = out[:, 5:]

        # Auto-sigmoid si logits
        if obj.min() < 0 or obj.max() > 1:
            obj = 1 / (1 + np.exp(-np.clip(obj, -50, 50)))
        if cls.min() < 0 or cls.max() > 1:
            cls = 1 / (1 + np.exp(-np.clip(cls, -50, 50)))

        class_ids = cls.argmax(axis=1)
        scores = obj * cls.max(axis=1)
    else:
        # Format v8/v11: [cx, cy, w, h, cls0, cls1, ...]
        out = raw_output.reshape(n_attrs_v8, -1).T
        boxes = out[:, :4]
        cls = out[:, 4:]

        # Auto-sigmoid si logits
        if cls.min() < 0 or cls.max() > 1:
            cls = 1 / (1 + np.exp(-np.clip(cls, -50, 50)))

        class_ids = cls.argmax(axis=1)
        scores = cls.max(axis=1)

    return boxes, scores, class_ids


def apply_nms(boxes, scores, class_ids, conf_thresh, iou_thresh, max_det):
    """
    Applique le NMS class-aware avec OpenCV.

    Args:
        boxes: (N, 4) en format [cx, cy, w, h]
        scores: (N,)
        class_ids: (N,)

    Returns:
        indices, boxes_filtered, scores_filtered, class_ids_filtered
    """
    mask = scores > conf_thresh
    boxes_f = boxes[mask]
    scores_f = scores[mask]
    class_ids_f = class_ids[mask]

    if len(scores_f) == 0:
        return np.array([], dtype=int), boxes_f, scores_f, class_ids_f

    # Convertir cx,cy,w,h -> x,y,w,h pour OpenCV
    boxes_xywh = []
    for cx, cy, w, h in boxes_f:
        boxes_xywh.append([cx - w / 2, cy - h / 2, w, h])

    # NMS class-aware
    if hasattr(cv2.dnn, "NMSBoxesBatched"):
        indices = cv2.dnn.NMSBoxesBatched(
            boxes_xywh,
            scores_f.tolist(),
            class_ids_f.tolist(),
            conf_thresh,
            iou_thresh,
        )
    else:
        keep = []
        for c in np.unique(class_ids_f):
            ids = np.where(class_ids_f == c)[0]
            idx_c = cv2.dnn.NMSBoxes(
                [boxes_xywh[i] for i in ids],
                [float(scores_f[i]) for i in ids],
                conf_thresh,
                iou_thresh,
            )
            if len(idx_c):
                keep.extend(ids[idx_c.flatten()].tolist())
        indices = np.array(keep, dtype=int)

    indices = np.array(indices).reshape(-1)

    # Cap MAX_DET
    if len(indices) > max_det:
        idx_scores = [(idx, scores_f[idx]) for idx in indices]
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        indices = np.array([x[0] for x in idx_scores[:max_det]])

    return indices, boxes_f, scores_f, class_ids_f


def boxes_to_predictions(
    indices, boxes, scores, class_ids, ratio, dw, dh, orig_w, orig_h
):
    """
    Convertit les boxes en predictions normalisees.
    Applique le reverse letterbox.
    """
    predictions = []

    for idx in indices:
        cx, cy, w, h = boxes[idx]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Reverse letterbox
        x1 = (x1 - dw) / ratio[0]
        y1 = (y1 - dh) / ratio[1]
        x2 = (x2 - dw) / ratio[0]
        y2 = (y2 - dh) / ratio[1]

        # Normalisation + clamping
        predictions.append(
            {
                "box": [
                    max(0, min(1, x1 / orig_w)),
                    max(0, min(1, y1 / orig_h)),
                    max(0, min(1, x2 / orig_w)),
                    max(0, min(1, y2 / orig_h)),
                ],
                "class_id": int(class_ids[idx]),
                "confidence": float(scores[idx]),
            }
        )

    return predictions


# =============================================================================
# BRANCHE GPU - ONNX Runtime (meme postprocess que OAK)
# =============================================================================


def benchmark_gpu_ort(
    model_path: str,
    num_classes: int,
    dataset: str = "coco128",
    ort_opt_level: str | None = None,
):
    """
    Benchmark sur GPU via ONNX Runtime.

    Utilise le MEME postprocess que OAK (decode_yolo_output + apply_nms)
    pour garantir une comparaison "fair".

    Args:
        ort_opt_level: Niveau d'optimisation ORT au chargement.
            - None (auto): DISABLE pour modeles _ortopt_*, sinon ALL
            - "disable": ORT_DISABLE_ALL (pas d'optimisation)
            - "basic": ORT_ENABLE_BASIC
            - "extended": ORT_ENABLE_EXTENDED
            - "all": ORT_ENABLE_ALL (defaut ORT)

        Pour comparer les fusion levels offline, utiliser "disable"
        pour eviter qu'ORT re-optimise et gomme les differences.

    Notes INT8 QDQ:
        - Si le modele est INT8 QDQ, TensorRT EP est utilise en priorite
        - TensorRT EP accelere les modeles QDQ mieux que CUDA EP
        - Cache TensorRT engine active pour eviter re-build a chaque run
    """
    import onnxruntime as ort

    print("=" * 60)
    print("BENCHMARK GPU - ONNX Runtime (meme postprocess que OAK)")
    print("=" * 60)

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)
    imgsz = parse_imgsz_from_filename(model_path)

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")
    print(f"ImgSz: {imgsz}")

    # Detecter si modele INT8 QDQ
    is_int8 = is_int8_qdq_model(model_path)
    if is_int8:
        print("  [Detected] Modele INT8 QDQ")

    # Configurer SessionOptions avec niveau d'optimisation
    sess_options = ort.SessionOptions()

    # Auto-detection: DISABLE pour modeles _ortopt_*, sinon defaut ORT
    ort_level_requested = ort_opt_level or "auto"
    if ort_opt_level is None:
        if "_ortopt_" in model_name:
            ort_opt_level = "disable"
            print("  [Auto] Modele _ortopt_* detecte -> ORT_DISABLE_ALL")
        else:
            ort_opt_level = "all"
    # Niveau effectif apres auto-detection (pour le CSV)
    ort_level_effective = ort_opt_level

    opt_level_map = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    sess_options.graph_optimization_level = opt_level_map[ort_opt_level]
    print(f"ORT Opt Level: {ort_opt_level.upper()}")

    # --- Choix des providers selon le type de modele ---
    # Pour INT8 QDQ: TensorRT EP en priorite (meilleure acceleration)
    # Pour FP16/FP32: CUDA EP standard
    #
    # TODO (BUG INT8): CUDA EP ne supporte pas bien les ops QDQ -> fallback CPU massif
    #   - CUDA EP execute chaque noeud individuellement, si pas capable -> CPU
    #   - Sur modeles INT8 QDQ, beaucoup d'ops quantifiees non supportees -> CPU spike
    #   - Fix: Verifier TensorrtExecutionProvider ou NvTensorRtRtxExecutionProvider (RTX)
    #   - Si pas de TRT EP dispo, l'INT8 sera majoritairement CPU meme avec CUDA EP en tete
    #   - Ref: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
    #
    available_providers = ort.get_available_providers()
    trt_available = "TensorrtExecutionProvider" in available_providers
    cuda_available = "CUDAExecutionProvider" in available_providers

    if is_int8 and trt_available:
        # TensorRT EP avec cache pour INT8 QDQ
        # Configuration du cache TensorRT pour eviter re-build
        cache_dir = Path(model_path).parent / ".trt_cache"
        cache_dir.mkdir(exist_ok=True)

        trt_provider_options = {
            "trt_fp16_enable": True,
            "trt_int8_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_timing_cache_enable": True,
        }

        providers = [
            ("TensorrtExecutionProvider", trt_provider_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        print(f"  [INT8 QDQ] Using TensorRT EP (cache: {cache_dir})")
    elif cuda_available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
        print("  [Warning] Ni TensorRT ni CUDA disponibles, fallback CPU")

    # Creer la session ONNX Runtime
    try:
        session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        actual_providers = session.get_providers()
        actual_provider = actual_providers[0]
        providers_used = ",".join(actual_providers)
        print(f"Provider: {actual_provider}")
        print(f"All providers: {providers_used}")

        # Warning si INT8 sans TensorRT
        if is_int8 and "TensorrtExecutionProvider" not in actual_providers:
            print(
                "  [WARNING] INT8 QDQ sans TensorRT EP -> acceleration peut etre limitee"
            )
    except Exception as e:
        print(f"  [Warning] Erreur creation session: {e}")
        print("  [Fallback] CPU uniquement")
        session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        actual_provider = "CPUExecutionProvider"
        providers_used = "CPUExecutionProvider"

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_name = input_info.name
    output_name = output_info.name

    # Detecter le dtype d'entree (FP16 ou FP32)
    input_type = input_info.type  # 'tensor(float)' ou 'tensor(float16)'
    if "float16" in input_type:
        input_dtype = np.float16
        print("Input dtype: float16")
    else:
        input_dtype = np.float32
        print("Input dtype: float32")

    print(f"\nChargement du dataset {dataset.upper()}...")
    dataset_items = load_coco128_dataset(dataset)

    # Warmup
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    for sample in dataset_items[:WARMUP_FRAMES]:
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue
        img_lb, _, _ = letterbox(img, (imgsz, imgsz))
        # Preprocess: BGR->RGB, HWC->CHW, normalize, cast to model dtype
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_chw, axis=0).astype(input_dtype)
        _ = session.run([output_name], {input_name: img_batch})

    print("\nInference...")
    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    first_frame = True

    for idx, sample in enumerate(dataset_items):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        t0 = time.perf_counter()

        # Preprocess (meme que OAK conceptuellement)
        img_lb, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_chw, axis=0).astype(input_dtype)

        t1 = time.perf_counter()

        # Inference ONNX Runtime
        outputs = session.run([output_name], {input_name: img_batch})
        output_shape = outputs[0].shape
        raw_output = (
            outputs[0].astype(np.float32).flatten()
        )  # Cast to float32 for postprocess

        t2 = time.perf_counter()

        if first_frame:
            print(
                f"  [Sanity-check] Output shape: {output_shape}, "
                f"min={raw_output.min():.4f}, max={raw_output.max():.4f}"
            )
            first_frame = False

        # Decode + NMS (MEME code que OAK)
        try:
            boxes, scores, class_ids = decode_yolo_output(
                raw_output, num_classes, imgsz, output_shape=output_shape
            )
            indices, boxes_f, scores_f, class_ids_f = apply_nms(
                boxes, scores, class_ids, CONF_EVAL, NMS_IOU, MAX_DET
            )
            predictions = boxes_to_predictions(
                indices, boxes_f, scores_f, class_ids_f, ratio, dw, dh, orig_w, orig_h
            )
        except ValueError as e:
            if idx == 0:
                print(f"  [ERREUR DECODE] {e}")
            predictions = []

        t3 = time.perf_counter()

        e2e_times.append((t3 - t0) * 1000)
        preprocess_times.append((t1 - t0) * 1000)
        inference_times.append((t2 - t1) * 1000)
        postprocess_times.append((t3 - t2) * 1000)

        all_predictions.append(predictions)

        gt_list = []
        for gt in sample["gt_boxes"]:
            x_center, y_center, w, h = (
                gt["x_center"],
                gt["y_center"],
                gt["width"],
                gt["height"],
            )
            gt_list.append(
                {
                    "box": [
                        x_center - w / 2,
                        y_center - h / 2,
                        x_center + w / 2,
                        y_center + h / 2,
                    ],
                    "class_id": gt["class_id"],
                }
            )
        all_ground_truths.append(gt_list)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(dataset_items)} images...")

    print("\nCalcul des metriques...")
    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    # Compute extended timing stats
    timing = compute_timing_stats(
        e2e_times, preprocess_times, inference_times, postprocess_times
    )

    print("\n" + "-" * 40)
    print("RESULTATS GPU (ONNX Runtime)")
    print("-" * 40)
    print(f"Taille modele       : {size_mb:.2f} MB")
    print(f"ImgSz               : {imgsz}")
    print(f"Temps E2E moyen     : {timing['e2e_mean']:.2f} ms")
    print(f"  Preprocess        : {timing['preprocess_ms']:.2f} ms")
    print(f"  Inference         : {timing['inference_ms']:.2f} ms")
    print(f"  Postprocess       : {timing['postprocess_ms']:.2f} ms")
    print(
        f"Latence p50/p90/p99 : {timing['latency_p50']:.2f} / {timing['latency_p90']:.2f} / {timing['latency_p99']:.2f} ms"
    )
    print(f"FPS                 : {timing['fps']:.1f}")
    print(f"mAP50               : {metrics['map50']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-Score            : {metrics['f1']:.4f}")

    # -------------------------------------------------------------------------
    # MICRO-PATCH: rendre le CSV "sweep-friendly"
    # -------------------------------------------------------------------------
    # On encode le niveau d'optimisation ORT dans la colonne Hardware afin
    # de distinguer clairement disable/basic/extended/all dans le CSV.
    #
    # Rappel ORT: les niveaux d'optimisation sont définis par GraphOptimizationLevel
    # (disable/basic/extended/all).
    if "TensorrtExecutionProvider" in actual_provider:
        provider_tag = "TRT"
    elif "CUDA" in actual_provider:
        provider_tag = "CUDA"
    else:
        provider_tag = "CPU"
    level_tag = (ort_opt_level or "all").upper()
    hardware = f"GPU_ORT_{provider_tag}_{level_tag}"

    save_results(
        hardware=hardware,
        model_name=model_name,
        size_mb=size_mb,
        imgsz=imgsz,
        e2e_time_ms=timing["e2e_mean"],
        device_time_ms=timing["inference_ms"],
        map50=metrics["map50"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        dataset=dataset,
        backend="onnxruntime",
        # Extended timing (Phase 3)
        latency_p50=timing["latency_p50"],
        latency_p90=timing["latency_p90"],
        latency_p95=timing["latency_p95"],
        latency_p99=timing["latency_p99"],
        preprocess_ms=timing["preprocess_ms"],
        inference_ms=timing["inference_ms"],
        postprocess_ms=timing["postprocess_ms"],
        fps=timing["fps"],
        # ORT config tracking (Phase 4)
        ort_level_requested=f"{ort_level_requested}->{ort_level_effective}",
        providers_used=providers_used,
    )


# =============================================================================
# BRANCHE CPU - ONNX Runtime CPUExecutionProvider
# =============================================================================


def benchmark_cpu_ort(
    model_path: str,
    num_classes: int,
    dataset: str = "coco128",
    host_tag: str | None = None,
    cpu_threads: int | None = None,
    cpu_execution_mode: str = "sequential",
):
    """
    Benchmark sur CPU via ONNX Runtime CPUExecutionProvider.

    Utilise le MEME postprocess que OAK/GPU-ORT (decode_yolo_output + apply_nms)
    pour garantir une comparaison "fair".

    Args:
        host_tag: Tag identifiant le host (ex: PC, PI4). Auto-detecte si None.
        cpu_threads: Nombre de threads intra-op. None = auto ORT.
        cpu_execution_mode: "sequential" (ORT_SEQUENTIAL) ou "parallel" (ORT_PARALLEL).
    """
    import onnxruntime as ort

    print("=" * 60)
    print("BENCHMARK CPU - ONNX Runtime (CPUExecutionProvider)")
    print("=" * 60)

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)
    imgsz = parse_imgsz_from_filename(model_path)

    # Auto-detect host tag si non fourni
    if host_tag is None:
        host_tag = detect_host_tag()
    print(f"Host tag: {host_tag}")

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")
    print(f"ImgSz: {imgsz}")

    # Configurer SessionOptions pour CPU
    sess_options = ort.SessionOptions()

    # Niveau d'optimisation: ALL par defaut pour CPU
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_level = "ALL"

    # Configurer les threads CPU
    # TODO (THREADS): inter_op_num_threads=1 est un choix de stabilite pour le benchmark.
    #   - En mode sequentiel (ORT_SEQUENTIAL), ORT ignore de toute façon l'inter-op.
    #   - En mode parallele (ORT_PARALLEL), inter_op_num_threads peut redevenir pertinent
    #     selon le graphe / ORT. On privilegie ici la stabilite vs perf max en mode parallele.
    #   - Ref: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
    if cpu_threads is not None:
        sess_options.intra_op_num_threads = cpu_threads
        sess_options.inter_op_num_threads = 1  # Plus stable pour benchmark
        print(f"CPU threads: intra={cpu_threads}, inter=1")
    else:
        print("CPU threads: auto (ORT default)")

    # Configurer le mode d'execution
    if cpu_execution_mode == "parallel":
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        exec_mode_tag = "PAR"
    else:
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        exec_mode_tag = "SEQ"
    print(f"Execution mode: {cpu_execution_mode.upper()}")

    # CPUExecutionProvider uniquement
    providers = ["CPUExecutionProvider"]

    # Creer la session ONNX Runtime
    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers
    )
    actual_providers = session.get_providers()
    providers_used = ",".join(actual_providers)
    print(f"Provider: {actual_providers[0]}")

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_name = input_info.name
    output_name = output_info.name

    # Detecter le dtype d'entree (FP16 ou FP32)
    input_type = input_info.type
    if "float16" in input_type:
        input_dtype = np.float16
        print("Input dtype: float16")
    else:
        input_dtype = np.float32
        print("Input dtype: float32")

    print(f"\nChargement du dataset {dataset.upper()}...")
    dataset_items = load_coco128_dataset(dataset)

    # Warmup
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    for sample in dataset_items[:WARMUP_FRAMES]:
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue
        img_lb, _, _ = letterbox(img, (imgsz, imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_chw, axis=0).astype(input_dtype)
        _ = session.run([output_name], {input_name: img_batch})

    print("\nInference...")
    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    first_frame = True

    for idx, sample in enumerate(dataset_items):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        t0 = time.perf_counter()

        # Preprocess (meme que OAK/GPU)
        img_lb, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_chw, axis=0).astype(input_dtype)

        t1 = time.perf_counter()

        # Inference ONNX Runtime CPU
        outputs = session.run([output_name], {input_name: img_batch})
        output_shape = outputs[0].shape
        raw_output = outputs[0].astype(np.float32).flatten()

        t2 = time.perf_counter()

        if first_frame:
            print(
                f"  [Sanity-check] Output shape: {output_shape}, "
                f"min={raw_output.min():.4f}, max={raw_output.max():.4f}"
            )
            first_frame = False

        # Decode + NMS (MEME code que OAK/GPU)
        try:
            boxes, scores, class_ids = decode_yolo_output(
                raw_output, num_classes, imgsz, output_shape=output_shape
            )
            indices, boxes_f, scores_f, class_ids_f = apply_nms(
                boxes, scores, class_ids, CONF_EVAL, NMS_IOU, MAX_DET
            )
            predictions = boxes_to_predictions(
                indices, boxes_f, scores_f, class_ids_f, ratio, dw, dh, orig_w, orig_h
            )
        except ValueError as e:
            if idx == 0:
                print(f"  [ERREUR DECODE] {e}")
            predictions = []

        t3 = time.perf_counter()

        e2e_times.append((t3 - t0) * 1000)
        preprocess_times.append((t1 - t0) * 1000)
        inference_times.append((t2 - t1) * 1000)
        postprocess_times.append((t3 - t2) * 1000)

        all_predictions.append(predictions)

        gt_list = []
        for gt in sample["gt_boxes"]:
            x_center, y_center, w, h = (
                gt["x_center"],
                gt["y_center"],
                gt["width"],
                gt["height"],
            )
            gt_list.append(
                {
                    "box": [
                        x_center - w / 2,
                        y_center - h / 2,
                        x_center + w / 2,
                        y_center + h / 2,
                    ],
                    "class_id": gt["class_id"],
                }
            )
        all_ground_truths.append(gt_list)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(dataset_items)} images...")

    print("\nCalcul des metriques...")
    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    # Compute extended timing stats
    timing = compute_timing_stats(
        e2e_times, preprocess_times, inference_times, postprocess_times
    )

    print("\n" + "-" * 40)
    print("RESULTATS CPU (ONNX Runtime)")
    print("-" * 40)
    print(f"Host                : {host_tag}")
    print(f"Taille modele       : {size_mb:.2f} MB")
    print(f"ImgSz               : {imgsz}")
    print(f"Temps E2E moyen     : {timing['e2e_mean']:.2f} ms")
    print(f"  Preprocess        : {timing['preprocess_ms']:.2f} ms")
    print(f"  Inference         : {timing['inference_ms']:.2f} ms")
    print(f"  Postprocess       : {timing['postprocess_ms']:.2f} ms")
    print(
        f"Latence p50/p90/p99 : {timing['latency_p50']:.2f} / {timing['latency_p90']:.2f} / {timing['latency_p99']:.2f} ms"
    )
    print(f"FPS                 : {timing['fps']:.1f}")
    print(f"mAP50               : {metrics['map50']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-Score            : {metrics['f1']:.4f}")

    # Hardware tag: CPU_ORT_{host_tag}_{exec_mode}
    hardware = f"CPU_ORT_{host_tag}_{exec_mode_tag}"

    # Thread config pour le CSV
    thread_config = f"intra={cpu_threads or 'auto'},inter=1"

    save_results(
        hardware=hardware,
        model_name=model_name,
        size_mb=size_mb,
        imgsz=imgsz,
        e2e_time_ms=timing["e2e_mean"],
        device_time_ms=timing["inference_ms"],
        map50=metrics["map50"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        dataset=dataset,
        backend="onnxruntime-cpu",
        # Extended timing (Phase 3)
        latency_p50=timing["latency_p50"],
        latency_p90=timing["latency_p90"],
        latency_p95=timing["latency_p95"],
        latency_p99=timing["latency_p99"],
        preprocess_ms=timing["preprocess_ms"],
        inference_ms=timing["inference_ms"],
        postprocess_ms=timing["postprocess_ms"],
        fps=timing["fps"],
        # ORT config tracking
        ort_level_requested=f"{ort_level}_{exec_mode_tag}_{thread_config}",
        providers_used=providers_used,
    )


# =============================================================================
# BRANCHE GPU - Ultralytics (pour compatibilite .pt)
# =============================================================================


def benchmark_gpu_ultralytics(
    model_path: str,
    num_classes: int,
    dataset: str = "coco128",
):
    """
    Benchmark sur GPU via Ultralytics.

    NOTE: Le postprocess est fait par Ultralytics (torchvision.nms),
    pas exactement le meme que OAK. Utiliser benchmark_gpu_ort() pour
    une comparaison plus "fair".
    """
    import torch
    from ultralytics import YOLO

    print("=" * 60)
    print("BENCHMARK GPU - Ultralytics")
    print("=" * 60)
    print("  [Note] Postprocess Ultralytics (torchvision.nms)")
    print("  [Note] Pour comparaison fair avec OAK, utiliser --backend ort")

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)
    imgsz = parse_imgsz_from_filename(model_path)

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")
    print(f"ImgSz: {imgsz}")

    model = YOLO(model_path)
    print(f"Classes: {len(model.names)}")

    print(f"\nChargement du dataset {dataset.upper()}...")
    dataset_items = load_coco128_dataset(dataset)

    # Warmup
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    with torch.inference_mode():
        for sample in dataset_items[:WARMUP_FRAMES]:
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue
            img_lb, _, _ = letterbox(img, (imgsz, imgsz))
            _ = model.predict(
                img_lb, imgsz=imgsz, conf=CONF_EVAL, device=0, verbose=False, half=True
            )

    print("\nInference...")
    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    with torch.inference_mode():
        for idx, sample in enumerate(dataset_items):
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            t0 = time.perf_counter()

            img_lb, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
            result = model.predict(
                img_lb,
                imgsz=imgsz,
                conf=CONF_EVAL,
                iou=NMS_IOU,
                max_det=MAX_DET,
                device=0,
                verbose=False,
                half=True,
            )[0]

            preds = []
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1 = (x1 - dw) / ratio[0]
                y1 = (y1 - dh) / ratio[1]
                x2 = (x2 - dw) / ratio[0]
                y2 = (y2 - dh) / ratio[1]
                preds.append(
                    {
                        "box": [
                            max(0, min(1, x1 / orig_w)),
                            max(0, min(1, y1 / orig_h)),
                            max(0, min(1, x2 / orig_w)),
                            max(0, min(1, y2 / orig_h)),
                        ],
                        "class_id": int(b.cls.item()),
                        "confidence": float(b.conf.item()),
                    }
                )

            t1 = time.perf_counter()
            e2e_times.append((t1 - t0) * 1000)

            # Extract timing from Ultralytics result.speed if available
            if hasattr(result, "speed") and isinstance(result.speed, dict):
                preprocess_times.append(float(result.speed.get("preprocess", 0.0)))
                inference_times.append(float(result.speed.get("inference", 0.0)))
                postprocess_times.append(float(result.speed.get("postprocess", 0.0)))
            else:
                # Fallback: attribute all time to inference
                preprocess_times.append(0.0)
                inference_times.append((t1 - t0) * 1000)
                postprocess_times.append(0.0)

            all_predictions.append(preds)

            gt_list = []
            for gt in sample["gt_boxes"]:
                x_center, y_center, w, h = (
                    gt["x_center"],
                    gt["y_center"],
                    gt["width"],
                    gt["height"],
                )
                gt_list.append(
                    {
                        "box": [
                            x_center - w / 2,
                            y_center - h / 2,
                            x_center + w / 2,
                            y_center + h / 2,
                        ],
                        "class_id": gt["class_id"],
                    }
                )
            all_ground_truths.append(gt_list)

    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    # Compute extended timing stats
    timing = compute_timing_stats(
        e2e_times, preprocess_times, inference_times, postprocess_times
    )

    print("\n" + "-" * 40)
    print("RESULTATS GPU (Ultralytics)")
    print("-" * 40)
    print(f"Taille modele       : {size_mb:.2f} MB")
    print(f"ImgSz               : {imgsz}")
    print(f"Temps E2E moyen     : {timing['e2e_mean']:.2f} ms")
    print(f"  Preprocess        : {timing['preprocess_ms']:.2f} ms")
    print(f"  Inference         : {timing['inference_ms']:.2f} ms")
    print(f"  Postprocess       : {timing['postprocess_ms']:.2f} ms")
    print(
        f"Latence p50/p90/p99 : {timing['latency_p50']:.2f} / {timing['latency_p90']:.2f} / {timing['latency_p99']:.2f} ms"
    )
    print(f"FPS                 : {timing['fps']:.1f}")
    print(f"mAP50               : {metrics['map50']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-Score            : {metrics['f1']:.4f}")

    save_results(
        hardware="GPU_Ultralytics",
        model_name=model_name,
        size_mb=size_mb,
        imgsz=imgsz,
        e2e_time_ms=timing["e2e_mean"],
        device_time_ms=timing["inference_ms"],
        map50=metrics["map50"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        dataset=dataset,
        backend="ultralytics",
        # Extended timing (Phase 3)
        latency_p50=timing["latency_p50"],
        latency_p90=timing["latency_p90"],
        latency_p95=timing["latency_p95"],
        latency_p99=timing["latency_p99"],
        preprocess_ms=timing["preprocess_ms"],
        inference_ms=timing["inference_ms"],
        postprocess_ms=timing["postprocess_ms"],
        fps=timing["fps"],
    )


# =============================================================================
# BRANCHE OAK-D (blob + DepthAI)
# =============================================================================


def benchmark_oak(
    model_path: str,
    num_classes: int,
    dataset: str = "coco128",
    host_tag: str | None = None,
):
    """
    Benchmark sur OAK-D (Myriad X VPU).

    Args:
        host_tag: Tag identifiant le host (ex: PC, PI4). Auto-detecte si None.
    """

    import depthai as dai

    print("=" * 60)
    print("BENCHMARK OAK-D (Myriad X VPU)")
    print("=" * 60)

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)
    imgsz = parse_imgsz_from_filename(model_path)

    # Auto-detect host tag si non fourni
    if host_tag is None:
        host_tag = detect_host_tag()
    print(f"Host tag: {host_tag}")

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")
    print(f"ImgSz: {imgsz}")
    print(f"Classes: {num_classes}")

    print(f"\nChargement du dataset {dataset.upper()}...")
    dataset_items = load_coco128_dataset(dataset)

    # Pipeline DepthAI
    pipeline = dai.Pipeline()

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName("input")
    xin.setMaxDataSize(imgsz * imgsz * 3)  # imgsz dynamique
    xin.setNumFrames(4)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(model_path)
    nn.setNumInferenceThreads(2)
    xin.out.link(nn.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("nn")
    nn.out.link(xout.input)

    print("\nInference sur OAK-D...")
    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    with dai.Device(pipeline) as device:
        # TODO (OAK_QUEUE): blocking=False + maxSize=1 = risque de drop/variabilite.
        #   Pour un benchmark latence, il serait plus "propre" d'eviter toute logique de
        #   drop/backpressure implicite. Options:
        #   - Mettre blocking=True (ou trySend() + check retour si dispo)
        #   - Augmenter maxSize > 1 pour eviter les drops
        q_in = device.getInputQueue(name="input", maxSize=1, blocking=False)
        q_out = device.getOutputQueue(name="nn", maxSize=1, blocking=True)

        # TODO (PREPROCESS_FAIRNESS): OAK vs GPU/CPU preprocessing mismatch!
        #   GPU/CPU: BGR->RGB, normalize /255, CHW, float16/float32
        #   OAK: BGR888p uint8 (pas de RGB, pas de /255)
        #
        #   Si le BLOB n'integre pas le meme pretraitement que l'ONNX, on risque
        #   d'avoir des mAP/PRF1 non comparables (drop cote OAK) meme si le postprocess
        #   est "aligne".
        #
        #   Actions possibles:
        #   1. Verifier ce que le blob attend (U8 vs FP16, RGB vs BGR, normalisation)
        #   2. Ajouter le preprocess dans le graphe ONNX avant compilation en BLOB
        #   3. Preprocesser cote host avant send() (envoyer NNData/FP16 si necessaire)
        #   4. Utiliser un noeud ImageManip cote device pour rester "propre"

        # Warmup
        print(f"Warmup ({WARMUP_FRAMES} frames)...")
        for sample in dataset_items[:WARMUP_FRAMES]:
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue
            img_lb, _, _ = letterbox(img, (imgsz, imgsz))
            img_chw = np.ascontiguousarray(img_lb.transpose(2, 0, 1))
            dai_frame = dai.ImgFrame()
            dai_frame.setWidth(imgsz)
            dai_frame.setHeight(imgsz)
            dai_frame.setType(dai.ImgFrame.Type.BGR888p)
            dai_frame.setData(img_chw.reshape(-1))
            q_in.send(dai_frame)
            _ = q_out.get()

        output_layers = None
        first_frame = True

        for i, sample in enumerate(dataset_items):
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            t0 = time.perf_counter()

            img_lb, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
            img_chw = np.ascontiguousarray(img_lb.transpose(2, 0, 1))

            t1 = time.perf_counter()

            dai_frame = dai.ImgFrame()
            dai_frame.setWidth(imgsz)
            dai_frame.setHeight(imgsz)
            dai_frame.setType(dai.ImgFrame.Type.BGR888p)
            dai_frame.setData(img_chw.reshape(-1))

            q_in.send(dai_frame)
            in_nn = q_out.get()

            t2 = time.perf_counter()

            if output_layers is None:
                output_layers = in_nn.getAllLayerNames()
            raw_data = np.array(in_nn.getLayerFp16(output_layers[0]))

            if first_frame:
                print(
                    f"  [Sanity-check] Output shape: {raw_data.shape}, "
                    f"min={raw_data.min():.4f}, max={raw_data.max():.4f}"
                )
                first_frame = False

            try:
                boxes, scores, class_ids = decode_yolo_output(
                    raw_data, num_classes, imgsz
                )
                indices, boxes_f, scores_f, class_ids_f = apply_nms(
                    boxes, scores, class_ids, CONF_EVAL, NMS_IOU, MAX_DET
                )
                predictions = boxes_to_predictions(
                    indices,
                    boxes_f,
                    scores_f,
                    class_ids_f,
                    ratio,
                    dw,
                    dh,
                    orig_w,
                    orig_h,
                )
            except ValueError as e:
                if i == 0:
                    print(f"  [ERREUR DECODE] {e}")
                predictions = []

            t3 = time.perf_counter()
            e2e_times.append((t3 - t0) * 1000)
            preprocess_times.append((t1 - t0) * 1000)
            inference_times.append((t2 - t1) * 1000)  # Note: inclut USB, pas VPU seul
            postprocess_times.append((t3 - t2) * 1000)

            all_predictions.append(predictions)

            gt_list = []
            for gt in sample["gt_boxes"]:
                x_center, y_center, w, h = (
                    gt["x_center"],
                    gt["y_center"],
                    gt["width"],
                    gt["height"],
                )
                gt_list.append(
                    {
                        "box": [
                            x_center - w / 2,
                            y_center - h / 2,
                            x_center + w / 2,
                            y_center + h / 2,
                        ],
                        "class_id": gt["class_id"],
                    }
                )
            all_ground_truths.append(gt_list)

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(dataset_items)} images...")

    print("\nCalcul des metriques...")
    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    # Compute extended timing stats
    timing = compute_timing_stats(
        e2e_times, preprocess_times, inference_times, postprocess_times
    )

    print("\n" + "-" * 40)
    print("RESULTATS OAK-D")
    print("-" * 40)
    print(f"Taille blob         : {size_mb:.2f} MB")
    print(f"ImgSz               : {imgsz}")
    print(f"Temps E2E moyen     : {timing['e2e_mean']:.2f} ms")
    print(f"  Preprocess        : {timing['preprocess_ms']:.2f} ms")
    print(f"  Inference (USB+VPU): {timing['inference_ms']:.2f} ms")
    print(f"  Postprocess       : {timing['postprocess_ms']:.2f} ms")
    print(
        f"Latence p50/p90/p99 : {timing['latency_p50']:.2f} / {timing['latency_p90']:.2f} / {timing['latency_p99']:.2f} ms"
    )
    print(f"FPS                 : {timing['fps']:.1f}")
    print(f"mAP50               : {metrics['map50']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-Score            : {metrics['f1']:.4f}")

    # Hardware tag: OAK_MyriadX_HOST_{host_tag}[_SHAVES_N]
    shaves_match = re.search(r"_(\d+)shave$", model_name)
    shaves_tag = f"_SHAVES_{shaves_match.group(1)}" if shaves_match else ""
    hardware = f"OAK_MyriadX_HOST_{host_tag}{shaves_tag}"

    save_results(
        hardware=hardware,
        model_name=model_name,
        size_mb=size_mb,
        imgsz=imgsz,
        e2e_time_ms=timing["e2e_mean"],
        device_time_ms=timing["inference_ms"],
        map50=metrics["map50"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        dataset=dataset,
        backend="depthai",
        # Extended timing (Phase 3)
        latency_p50=timing["latency_p50"],
        latency_p90=timing["latency_p90"],
        latency_p95=timing["latency_p95"],
        latency_p99=timing["latency_p99"],
        preprocess_ms=timing["preprocess_ms"],
        inference_ms=timing["inference_ms"],
        postprocess_ms=timing["postprocess_ms"],
        fps=timing["fps"],
    )


# =============================================================================
# BRANCHE ORIN (TensorRT ENGINE)
# =============================================================================


def benchmark_orin(model_path: str, num_classes: int, dataset: str = "coco128"):
    """
    Benchmark sur Jetson Orin (TensorRT ENGINE).

    Utilise le MEME postprocess que OAK/GPU-ORT pour garantir l'equite.
    """
    print("=" * 60)
    print("BENCHMARK ORIN (Jetson TensorRT)")
    print("=" * 60)

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)
    imgsz = parse_imgsz_from_filename(model_path)

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")
    print(f"ImgSz: {imgsz}")
    print(f"Classes: {num_classes}")

    # Essayer TensorRT Python API d'abord, sinon Ultralytics
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import tensorrt as trt

        use_trt_api = True
        print("  Backend: TensorRT Python API")
    except ImportError:
        use_trt_api = False
        print("  Backend: Ultralytics (TensorRT via torch)")

    print(f"\nChargement du dataset {dataset.upper()}...")
    dataset_items = load_coco128_dataset(dataset)

    if use_trt_api:
        # TensorRT Python API - meme postprocess que OAK
        _benchmark_orin_trt(
            model_path, num_classes, imgsz, dataset_items, model_name, size_mb, dataset
        )
    else:
        # Fallback Ultralytics
        _benchmark_orin_ultralytics(
            model_path, num_classes, imgsz, dataset_items, model_name, size_mb, dataset
        )


def _benchmark_orin_trt(
    model_path, num_classes, imgsz, dataset_items, model_name, size_mb, dataset
):
    """Benchmark Orin avec TensorRT Python API (meme postprocess que OAK)."""
    import pycuda.driver as cuda
    import tensorrt as trt

    # Mapping TensorRT dtype -> numpy dtype
    TRT_TO_NP_DTYPE = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
    }

    # Charger le moteur TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(model_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Detecter dynamiquement les bindings input/output
    input_idx = None
    output_idx = None
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            input_idx = i
        else:
            output_idx = i

    if input_idx is None or output_idx is None:
        raise RuntimeError("Impossible de trouver les bindings input/output")

    # Recuperer shapes et dtypes depuis le moteur
    input_shape = tuple(context.get_binding_shape(input_idx))
    output_shape = tuple(context.get_binding_shape(output_idx))

    # Gerer les dims dynamiques (-1) : forcer batch=1
    if -1 in input_shape:
        input_shape = (1, 3, imgsz, imgsz)
        context.set_binding_shape(input_idx, input_shape)
        output_shape = tuple(context.get_binding_shape(output_idx))

    input_trt_dtype = engine.get_binding_dtype(input_idx)
    output_trt_dtype = engine.get_binding_dtype(output_idx)

    input_np_dtype = TRT_TO_NP_DTYPE.get(input_trt_dtype, np.float32)
    output_np_dtype = TRT_TO_NP_DTYPE.get(output_trt_dtype, np.float32)

    print(f"  Input shape: {input_shape}, dtype: {input_np_dtype.__name__}")
    print(f"  Output shape: {output_shape}, dtype: {output_np_dtype.__name__}")

    # Allouer les buffers avec les bons dtypes
    input_size = int(np.prod(input_shape) * np.dtype(input_np_dtype).itemsize)
    output_size = int(np.prod(output_shape) * np.dtype(output_np_dtype).itemsize)

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    stream = cuda.Stream()

    # Buffer host pour output avec le bon dtype
    h_output = np.empty(output_shape, dtype=output_np_dtype)

    # Creer le tableau bindings indexe correctement pour TensorRT
    # TensorRT attend une liste de taille engine.num_bindings
    bindings = [0] * engine.num_bindings
    bindings[input_idx] = int(d_input)
    bindings[output_idx] = int(d_output)

    # Warmup
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    for sample in dataset_items[:WARMUP_FRAMES]:
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue
        img_lb, _, _ = letterbox(img, (imgsz, imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.ascontiguousarray(
            np.expand_dims(img_chw, axis=0).astype(input_np_dtype)
        )

        cuda.memcpy_htod_async(d_input, img_batch, stream)
        context.execute_async_v2(bindings, stream.handle)
        stream.synchronize()

    print("\nInference...")
    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    first_frame = True

    for idx, sample in enumerate(dataset_items):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        t0 = time.perf_counter()

        # Preprocess avec cast au bon dtype
        img_lb, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_batch = np.ascontiguousarray(
            np.expand_dims(img_chw, axis=0).astype(input_np_dtype)
        )

        t1 = time.perf_counter()

        # Inference TensorRT
        cuda.memcpy_htod_async(d_input, img_batch, stream)
        context.execute_async_v2(bindings, stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        t2 = time.perf_counter()

        # Cast output to float32 pour le postprocess
        raw_output = h_output.astype(np.float32).flatten()

        if first_frame:
            print(
                f"  [Sanity-check] Output shape: {output_shape}, "
                f"min={raw_output.min():.4f}, max={raw_output.max():.4f}"
            )
            first_frame = False

        # Decode + NMS (MEME code que OAK)
        try:
            boxes, scores, class_ids = decode_yolo_output(
                raw_output, num_classes, imgsz, output_shape=output_shape
            )
            indices, boxes_f, scores_f, class_ids_f = apply_nms(
                boxes, scores, class_ids, CONF_EVAL, NMS_IOU, MAX_DET
            )
            predictions = boxes_to_predictions(
                indices, boxes_f, scores_f, class_ids_f, ratio, dw, dh, orig_w, orig_h
            )
        except ValueError as e:
            if idx == 0:
                print(f"  [ERREUR DECODE] {e}")
            predictions = []

        t3 = time.perf_counter()

        e2e_times.append((t3 - t0) * 1000)
        preprocess_times.append((t1 - t0) * 1000)
        inference_times.append((t2 - t1) * 1000)
        postprocess_times.append((t3 - t2) * 1000)

        all_predictions.append(predictions)

        gt_list = []
        for gt in sample["gt_boxes"]:
            x_center, y_center, w, h = (
                gt["x_center"],
                gt["y_center"],
                gt["width"],
                gt["height"],
            )
            gt_list.append(
                {
                    "box": [
                        x_center - w / 2,
                        y_center - h / 2,
                        x_center + w / 2,
                        y_center + h / 2,
                    ],
                    "class_id": gt["class_id"],
                }
            )
        all_ground_truths.append(gt_list)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(dataset_items)} images...")

    print("\nCalcul des metriques...")
    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    # Compute extended timing stats
    timing = compute_timing_stats(
        e2e_times, preprocess_times, inference_times, postprocess_times
    )

    print("\n" + "-" * 40)
    print("RESULTATS ORIN (TensorRT API)")
    print("-" * 40)
    print(f"Taille engine       : {size_mb:.2f} MB")
    print(f"ImgSz               : {imgsz}")
    print(f"Temps E2E moyen     : {timing['e2e_mean']:.2f} ms")
    print(f"  Preprocess        : {timing['preprocess_ms']:.2f} ms")
    print(f"  Inference         : {timing['inference_ms']:.2f} ms")
    print(f"  Postprocess       : {timing['postprocess_ms']:.2f} ms")
    print(
        f"Latence p50/p90/p99 : {timing['latency_p50']:.2f} / {timing['latency_p90']:.2f} / {timing['latency_p99']:.2f} ms"
    )
    print(f"FPS                 : {timing['fps']:.1f}")
    print(f"mAP50               : {metrics['map50']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-Score            : {metrics['f1']:.4f}")

    save_results(
        hardware="Jetson_Orin_TRT",
        model_name=model_name,
        size_mb=size_mb,
        imgsz=imgsz,
        e2e_time_ms=timing["e2e_mean"],
        device_time_ms=timing["inference_ms"],
        map50=metrics["map50"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        dataset=dataset,
        backend="tensorrt",
        # Extended timing (Phase 3)
        latency_p50=timing["latency_p50"],
        latency_p90=timing["latency_p90"],
        latency_p95=timing["latency_p95"],
        latency_p99=timing["latency_p99"],
        preprocess_ms=timing["preprocess_ms"],
        inference_ms=timing["inference_ms"],
        postprocess_ms=timing["postprocess_ms"],
        fps=timing["fps"],
    )


def _benchmark_orin_ultralytics(
    model_path, num_classes, imgsz, dataset_items, model_name, size_mb, dataset
):
    """Fallback: Benchmark Orin avec Ultralytics (postprocess different)."""
    import torch
    from ultralytics import YOLO

    print("  [Note] Postprocess Ultralytics (pas identique a OAK)")

    model = YOLO(model_path, task="detect")

    # Injecter metadata si manquantes
    try:
        if model.metadata is None or not model.metadata.get("names"):
            model.metadata = {
                "names": {i: f"class_{i}" for i in range(num_classes)},
                "batch": 1,
                "stride": 32,
                "imgsz": [imgsz, imgsz],
                "task": "detect",
            }
    except Exception:
        pass

    # Warmup
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    with torch.inference_mode():
        for sample in dataset_items[:WARMUP_FRAMES]:
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue
            img_lb, _, _ = letterbox(img, (imgsz, imgsz))
            _ = model.predict(
                img_lb, imgsz=imgsz, conf=CONF_EVAL, device=0, verbose=False, half=True
            )

    print("\nInference...")
    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    with torch.inference_mode():
        for idx, sample in enumerate(dataset_items):
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            t0 = time.perf_counter()

            img_lb, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
            result = model.predict(
                img_lb,
                imgsz=imgsz,
                conf=CONF_EVAL,
                iou=NMS_IOU,
                max_det=MAX_DET,
                device=0,
                verbose=False,
                half=True,
            )[0]

            preds = []
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1 = (x1 - dw) / ratio[0]
                y1 = (y1 - dh) / ratio[1]
                x2 = (x2 - dw) / ratio[0]
                y2 = (y2 - dh) / ratio[1]
                preds.append(
                    {
                        "box": [
                            max(0, min(1, x1 / orig_w)),
                            max(0, min(1, y1 / orig_h)),
                            max(0, min(1, x2 / orig_w)),
                            max(0, min(1, y2 / orig_h)),
                        ],
                        "class_id": int(b.cls.item()),
                        "confidence": float(b.conf.item()),
                    }
                )

            t1 = time.perf_counter()
            e2e_times.append((t1 - t0) * 1000)

            # Extract timing from Ultralytics result.speed if available
            if hasattr(result, "speed") and isinstance(result.speed, dict):
                preprocess_times.append(float(result.speed.get("preprocess", 0.0)))
                inference_times.append(float(result.speed.get("inference", 0.0)))
                postprocess_times.append(float(result.speed.get("postprocess", 0.0)))
            else:
                # Fallback: attribute all time to inference
                preprocess_times.append(0.0)
                inference_times.append((t1 - t0) * 1000)
                postprocess_times.append(0.0)

            all_predictions.append(preds)

            gt_list = []
            for gt in sample["gt_boxes"]:
                x_center, y_center, w, h = (
                    gt["x_center"],
                    gt["y_center"],
                    gt["width"],
                    gt["height"],
                )
                gt_list.append(
                    {
                        "box": [
                            x_center - w / 2,
                            y_center - h / 2,
                            x_center + w / 2,
                            y_center + h / 2,
                        ],
                        "class_id": gt["class_id"],
                    }
                )
            all_ground_truths.append(gt_list)

            if (idx + 1) % 20 == 0:
                print(f"  {idx + 1}/{len(dataset_items)} images...")

    print("\nCalcul des metriques...")
    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    # Compute extended timing stats
    timing = compute_timing_stats(
        e2e_times, preprocess_times, inference_times, postprocess_times
    )

    print("\n" + "-" * 40)
    print("RESULTATS ORIN (Ultralytics)")
    print("-" * 40)
    print(f"Taille engine       : {size_mb:.2f} MB")
    print(f"ImgSz               : {imgsz}")
    print(f"Temps E2E moyen     : {timing['e2e_mean']:.2f} ms")
    print(f"  Preprocess        : {timing['preprocess_ms']:.2f} ms")
    print(f"  Inference         : {timing['inference_ms']:.2f} ms")
    print(f"  Postprocess       : {timing['postprocess_ms']:.2f} ms")
    print(
        f"Latence p50/p90/p99 : {timing['latency_p50']:.2f} / {timing['latency_p90']:.2f} / {timing['latency_p99']:.2f} ms"
    )
    print(f"FPS                 : {timing['fps']:.1f}")
    print(f"mAP50               : {metrics['map50']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-Score            : {metrics['f1']:.4f}")

    save_results(
        hardware="Jetson_Orin_Ultralytics",
        model_name=model_name,
        size_mb=size_mb,
        imgsz=imgsz,
        e2e_time_ms=timing["e2e_mean"],
        device_time_ms=timing["inference_ms"],
        map50=metrics["map50"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        dataset=dataset,
        backend="ultralytics",
        # Extended timing (Phase 3)
        latency_p50=timing["latency_p50"],
        latency_p90=timing["latency_p90"],
        latency_p95=timing["latency_p95"],
        latency_p99=timing["latency_p99"],
        preprocess_ms=timing["preprocess_ms"],
        inference_ms=timing["inference_ms"],
        postprocess_ms=timing["postprocess_ms"],
        fps=timing["fps"],
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark d'un modele YOLO sur GPU, CPU, OAK-D ou Jetson Orin"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["4070", "cpu", "oak", "orin"],
        help="Cible hardware: '4070' (GPU), 'cpu' (ONNX Runtime CPU), 'oak' (Myriad X), 'orin' (Jetson)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le modele (.onnx pour GPU, .blob pour OAK, .engine pour Orin)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Nombre de classes du modele (defaut: 80 pour COCO)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco128",
        choices=["coco128", "coco"],
        help="Dataset: 'coco128' (rapide) ou 'coco' (val2017, baseline)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ort",
        choices=["ort", "ultralytics"],
        help="Backend GPU: 'ort' (ONNX Runtime, meme postprocess que OAK) ou 'ultralytics'",
    )
    parser.add_argument(
        "--ort-opt-level",
        type=str,
        default=None,
        choices=["disable", "basic", "extended", "all"],
        help="Niveau d'optimisation ORT au chargement (defaut: auto-detect, DISABLE pour _ortopt_*)",
    )
    parser.add_argument(
        "--host-tag",
        type=str,
        default=None,
        help="Tag identifiant le host (ex: PC, PI4). Auto-detecte si non fourni.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Nombre de threads intra-op pour inference CPU (defaut: auto)",
    )
    parser.add_argument(
        "--cpu-execution-mode",
        type=str,
        default="sequential",
        choices=["sequential", "parallel"],
        help="Mode d'execution ONNX Runtime CPU: sequential ou parallel (defaut: sequential)",
    )
    parser.add_argument(
        "--shaves",
        type=int,
        default=None,
        choices=range(4, 9),
        metavar="[4-8]",
        help="Nombre de shaves pour OAK (4-8). Si fourni, selectionne le blob *_{shaves}shave.blob",
    )

    args = parser.parse_args()

    # Resoudre le chemin
    model_path = Path(args.model)

    # Pour OAK avec --shaves: construire le chemin du blob automatiquement
    # Ex: --model models/oak/yolo11n_640_fp16 --shaves 8 -> yolo11n_640_fp16_8shave.blob
    # TODO: gerer le cas --model deja suffixe "_6shave" sans extension pour eviter "_6shave_8shave.blob".
    if args.target == "oak" and args.shaves is not None:
        s = str(model_path)
        if s.endswith(".blob"):
            # Remplace le suffixe _{N}shave.blob si present, sinon ajoute _{N}shave.
            base = re.sub(r"_\d+shave\.blob$", "", s)
            if base == s:
                base = s[: -len(".blob")]
            model_path = Path(f"{base}_{args.shaves}shave.blob")
        else:
            model_path = Path(f"{s}_{args.shaves}shave.blob")

    if not model_path.exists():
        model_path = ROOT_DIR / model_path

    if not model_path.exists():
        print(f"Erreur: Modele introuvable: {args.model}")
        if args.shaves:
            print(f"  (avec --shaves {args.shaves}, chemin tente: {model_path})")
        return 1

    model_path = str(model_path.resolve())

    # Verifier l'extension
    ext = Path(model_path).suffix.lower()
    if args.target == "4070" and ext not in [".pt", ".onnx"]:
        print(f"Attention: Pour GPU, .pt ou .onnx attendu (recu: {ext})")
    elif args.target == "cpu" and ext != ".onnx":
        print(f"Attention: Pour CPU, .onnx attendu (recu: {ext})")
    elif args.target == "oak" and ext != ".blob":
        print(f"Attention: Pour OAK, .blob attendu (recu: {ext})")
    elif args.target == "orin" and ext != ".engine":
        print(f"Attention: Pour Orin, .engine attendu (recu: {ext})")

    # Benchmark
    if args.target == "4070":
        if args.backend == "ort" and ext == ".onnx":
            benchmark_gpu_ort(
                model_path,
                args.num_classes,
                args.dataset,
                ort_opt_level=args.ort_opt_level,
            )
        else:
            benchmark_gpu_ultralytics(model_path, args.num_classes, args.dataset)
    elif args.target == "cpu":
        benchmark_cpu_ort(
            model_path,
            args.num_classes,
            args.dataset,
            host_tag=args.host_tag,
            cpu_threads=args.cpu_threads,
            cpu_execution_mode=args.cpu_execution_mode,
        )
    elif args.target == "oak":
        benchmark_oak(
            model_path,
            args.num_classes,
            args.dataset,
            host_tag=args.host_tag,
        )
    elif args.target == "orin":
        benchmark_orin(model_path, args.num_classes, args.dataset)

    return 0


if __name__ == "__main__":
    exit(main())
