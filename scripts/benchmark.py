"""
Benchmark d'un modele YOLO sur GPU ou OAK-D.

Usage:
    python scripts/benchmark.py --target 4070 --model models/base/yolo11n.pt
    python scripts/benchmark.py --target oak --model models/base/yolo11n_openvino_2022.1_6shave.blob --num-classes 80

Options:
    --target 4070|oak        : Cible hardware (GPU RTX 4070 ou OAK-D Myriad X)
    --model PATH             : Chemin vers le modele (.pt/.onnx pour GPU, .blob pour OAK)
    --num-classes N          : Nombre de classes du modele (defaut: 80 pour COCO)
    --dataset coco128|coco   : Dataset a utiliser (defaut: coco128)
                               - coco128: rapide, 128 images (equite GPU vs OAK)
                               - coco: COCO val2017, 5000 images (baseline scientifique)
    --pixel-perfect          : Meme letterbox GPU/OAK pour equite (defaut: active)
    --no-pixel-perfect       : Laisser Ultralytics gerer le preprocess GPU

Features:
    - Warmup de 10 frames exclu des stats pour stabilite
    - Decodage auto-robuste : detecte objectness (v5/v7) vs anchor-free (v8/v11)
    - Sigmoid auto si logits detectes hors [0,1]
    - NMS class-aware unifie (MAX_DET=300)
    - LetterBox Ultralytics officiel pour pixel-perfect
    - CSV enrichi : parametres, versions libs, metadonnees
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.metrics import ap_per_class

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
RESULTS_FILE = ROOT_DIR / "benchmark_results.csv"
IMGSZ = 640
CONF_EVAL = 0.001  # seuil très bas pour la courbe PR (mAP)
CONF_OP = 0.25  # seuil d'exploitation pour P/R/F1
NMS_IOU = 0.70  # IoU du NMS (class-aware)
MATCH_IOU = 0.50  # IoU pour le matching TP/GT (mAP50)
MAX_DET = 300  # max detections après NMS (unifié GPU/OAK pour équité)
WARMUP_FRAMES = 10  # frames de warmup exclues des stats (GPU et OAK)


def get_model_size_mb(filepath: str) -> float:
    """Retourne la taille du fichier en MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def calculate_f1(precision: float, recall: float) -> float:
    """Calcule le F1-score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def get_versions():
    """Retourne les versions des librairies clés."""
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
    return versions


def save_results(
    hardware: str,
    model_name: str,
    size_mb: float,
    e2e_time_ms: float,
    device_time_ms: float,
    map50: float,
    precision: float,
    recall: float,
    f1: float,
    dataset: str = "coco128",
    pixel_perfect: bool = True,
):
    """Sauvegarde les resultats dans le CSV avec métadonnées complètes."""
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
                    "E2E_Time_ms",
                    "Device_Time_ms",
                    "mAP50",
                    "Precision",
                    "Recall",
                    "F1",
                    # Métadonnées
                    "Dataset",
                    "Pixel_Perfect",
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
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                hardware,
                model_name,
                f"{size_mb:.2f}",
                f"{e2e_time_ms:.2f}",
                f"{device_time_ms:.2f}",
                f"{map50:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                # Métadonnées
                dataset,
                pixel_perfect,
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
            ]
        )

    print(f"\nResultats sauvegardes dans: {RESULTS_FILE}")


def load_coco128_dataset(dataset_name: str = "coco128"):
    """Charge le dataset COCO128 ou COCO val2017 (images + labels).

    Args:
        dataset_name: "coco128" pour coco128/train2017 (rapide, 128 images)
                      "coco" pour COCO val2017 (baseline scientifique, 5000 images)
    """
    from ultralytics.data.utils import check_det_dataset

    if dataset_name == "coco":
        # COCO val2017 - vrai set de validation pour baseline scientifique
        data = check_det_dataset("coco.yaml")
        val_images_dir = Path(data["path"]) / "images" / "val2017"
        val_labels_dir = Path(data["path"]) / "labels" / "val2017"
        expected_count = 5000
        print("[Dataset] COCO val2017 (baseline scientifique)")
    else:
        # coco128 - rapide pour tests GPU vs OAK
        data = check_det_dataset("coco128.yaml")
        val_images_dir = Path(data["path"]) / "images" / "train2017"
        val_labels_dir = Path(data["path"]) / "labels" / "train2017"
        expected_count = 128
        print("[Dataset] coco128/train2017 (equite GPU vs OAK, pas baseline)")

    print(f"[Dataset] Chemin images: {val_images_dir}")
    print(f"[Dataset] Chemin labels: {val_labels_dir}")

    image_files = sorted(val_images_dir.glob("*.jpg"))
    actual_count = len(image_files)

    print(f"[Dataset] Images trouvees: {actual_count} (attendu: {expected_count})")

    # Assertion stricte pour garantir l'équité GPU vs OAK
    if actual_count != expected_count:
        raise AssertionError(
            f"[ERREUR DATASET] Nombre d'images incorrect!\n"
            f"  - Attendu: {expected_count}\n"
            f"  - Trouve: {actual_count}\n"
            f"  - Chemin: {val_images_dir}\n"
            f"Verifiez que le dataset est correctement telecharge."
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
    """Redimensionne l'image en gardant le ratio (ajoute du padding).

    Utilise la classe LetterBox d'Ultralytics pour garantir un comportement
    identique au pipeline Ultralytics natif.

    Returns:
        img: image letterboxée
        ratio: (ratio_w, ratio_h) facteurs de scale
        (dw, dh): padding ajouté (gauche, haut)
    """
    from ultralytics.data.augment import LetterBox

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Créer le transformer Ultralytics (center=True pour padding symétrique)
    transform = LetterBox(new_shape, auto=False, stride=32, center=True)

    # Calculer ratio et padding manuellement (LetterBox ne les retourne pas)
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    # Appliquer la transformation
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
    """
    Associe prédictions et GT (class-aware, une GT max par prédiction).
    Retourne en une seule passe: tp, conf, pred_cls, target_cls, gt_matched

    Returns:
        tp: (N,1) bool - True si la prédiction est un TP
        conf: (N,) float - confiances des prédictions
        pred_cls: (N,) int - classes prédites
        target_cls: (M,) int - classes des GT
        gt_matched: list[bool] - True si la GT a été matchée
    """
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
    """Calcule mAP50 (AP intégrale) + P/R/F1 au seuil conf_op.

    Le matching est fait une seule fois par image pour éviter les divergences.
    """
    tp_all, conf_all, pred_cls_all, target_cls_all = [], [], [], []
    TP = FP = FN = 0

    for preds, gts in zip(all_predictions, all_ground_truths):
        # --- Matching pour mAP (toutes les prédictions, conf >= CONF_EVAL) ---
        tp, conf, pred_cls, target_cls, _ = match_tp_fp_fn(preds, gts, iou_thr=iou_thr)
        if len(conf):
            tp_all.append(tp)
            conf_all.append(conf)
            pred_cls_all.append(pred_cls)
        if len(target_cls):
            target_cls_all.append(target_cls)

        # --- Matching pour P/R/F1 au seuil conf_op (une seule passe) ---
        preds_op = [p for p in preds if p["confidence"] >= conf_op]
        tp_op, _, _, _, gt_matched = match_tp_fp_fn(preds_op, gts, iou_thr=iou_thr)

        # TP = nombre de prédictions matchées
        # FP = prédictions non matchées
        # FN = GT non matchées
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


# =============================================================================
# BRANCHE GPU (Ultralytics)
# =============================================================================


def benchmark_gpu(
    model_path: str,
    num_classes: int,
    pixel_perfect: bool = True,
    dataset: str = "coco128",
):
    """Benchmark sur GPU NVIDIA via Ultralytics.

    Args:
        pixel_perfect: Si True, applique le même letterbox que OAK pour équité.
        dataset: "coco128" ou "coco" pour COCO val2017.
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("BENCHMARK GPU (RTX 4070) - Ultralytics")
    print("=" * 60)

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")
    print(f"Pixel-perfect (meme preprocess que OAK): {pixel_perfect}")

    # Charger le modele
    model = YOLO(model_path)
    print(f"Classes: {len(model.names)}")

    print(f"\nInference sur {dataset.upper()} (GPU)...")
    dataset_items = load_coco128_dataset(dataset)

    # Warmup (exclure des stats)
    print(f"Warmup ({WARMUP_FRAMES} frames)...")
    for sample in dataset_items[:WARMUP_FRAMES]:
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue
        img_lb, _, _ = letterbox(img, (IMGSZ, IMGSZ))
        _ = model.predict(img_lb, imgsz=IMGSZ, conf=CONF_EVAL, device=0, verbose=False)

    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    device_times = []

    for idx, sample in enumerate(dataset_items):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        t0 = time.perf_counter()

        if pixel_perfect:
            # Appliquer le même letterbox que OAK pour équité pixel-perfect
            img_lb, ratio, (dw, dh) = letterbox(img, (IMGSZ, IMGSZ))
            result = model.predict(
                img_lb,
                imgsz=IMGSZ,
                conf=CONF_EVAL,
                iou=NMS_IOU,
                max_det=MAX_DET,
                device=0,
                verbose=False,
            )[0]

            preds = []
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                # Reverse letterbox (même logique que OAK)
                x1 = (x1 - dw) / ratio[0]
                y1 = (y1 - dh) / ratio[1]
                x2 = (x2 - dw) / ratio[0]
                y2 = (y2 - dh) / ratio[1]
                # Normalisation et clamping
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
        else:
            # Mode original: Ultralytics gère le preprocessing
            result = model.predict(
                img,
                imgsz=IMGSZ,
                conf=CONF_EVAL,
                iou=NMS_IOU,
                max_det=MAX_DET,
                device=0,
                verbose=False,
            )[0]

            preds = []
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds.append(
                    {
                        "box": [
                            x1 / orig_w,
                            y1 / orig_h,
                            x2 / orig_w,
                            y2 / orig_h,
                        ],
                        "class_id": int(b.cls.item()),
                        "confidence": float(b.conf.item()),
                    }
                )

        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000)

        nn_ms = (
            float(result.speed.get("inference", 0.0))
            if hasattr(result, "speed") and isinstance(result.speed, dict)
            else 0.0
        )
        if nn_ms <= 0.0:
            nn_ms = (t1 - t0) * 1000
        device_times.append(nn_ms)

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

    e2e_time_ms = float(np.mean(e2e_times)) if e2e_times else 0.0
    device_time_ms = float(np.mean(device_times)) if device_times else 0.0
    map50 = metrics["map50"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    # Resultats
    print("\n" + "-" * 40)
    print("RESULTATS GPU")
    print("-" * 40)
    print(f"Taille modele  : {size_mb:.2f} MB")
    print(f"Temps inference (end-to-end): {e2e_time_ms:.2f} ms")
    print(f"Temps NN-only (device):       {device_time_ms:.2f} ms")
    print(f"mAP50          : {map50:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")

    save_results(
        hardware="GPU_RTX4070",
        model_name=model_name,
        size_mb=size_mb,
        e2e_time_ms=e2e_time_ms,
        device_time_ms=device_time_ms,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
        dataset=dataset,
        pixel_perfect=pixel_perfect,
    )


# =============================================================================
# BRANCHE OAK-D (blob + DepthAI)
# =============================================================================


def benchmark_oak(model_path: str, num_classes: int, dataset: str = "coco128"):
    """Benchmark sur OAK-D (Myriad X VPU).

    Args:
        dataset: "coco128" ou "coco" pour COCO val2017.
    """
    import depthai as dai

    print("=" * 60)
    print("BENCHMARK OAK-D (Myriad X VPU)")
    print("=" * 60)

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")
    print(f"Classes: {num_classes}")

    # Charger le dataset
    print(f"\nChargement du dataset {dataset.upper()}...")
    dataset_items = load_coco128_dataset(dataset)
    print(f"Images: {len(dataset_items)}")

    # Pipeline DepthAI
    pipeline = dai.Pipeline()

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName("input")
    xin.setMaxDataSize(IMGSZ * IMGSZ * 3)
    xin.setNumFrames(4)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(model_path)
    nn.setNumInferenceThreads(2)
    xin.out.link(nn.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("nn")
    nn.out.link(xout.input)

    # Inference
    print("\nInference sur OAK-D...")
    all_predictions = []
    all_ground_truths = []
    e2e_times = []
    device_times = []
    preprocess_times = []
    postprocess_times = []

    with dai.Device(pipeline) as device:
        q_in = device.getInputQueue("input")
        q_out = device.getOutputQueue("nn", maxSize=1, blocking=True)

        # Warmup (exclure des stats)
        print(f"Warmup ({WARMUP_FRAMES} frames)...")
        for sample in dataset_items[:WARMUP_FRAMES]:
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue
            img_lb, _, _ = letterbox(img, (IMGSZ, IMGSZ))
            img_chw = np.ascontiguousarray(img_lb.transpose(2, 0, 1))
            dai_frame = dai.ImgFrame()
            dai_frame.setWidth(IMGSZ)
            dai_frame.setHeight(IMGSZ)
            dai_frame.setType(dai.ImgFrame.Type.BGR888p)
            dai_frame.setData(img_chw.reshape(-1))
            q_in.send(dai_frame)
            _ = q_out.get()

        output_layers = None

        for i, sample in enumerate(dataset_items):
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            t0 = time.perf_counter()

            # Letterbox en BGR (pas de conversion RGB, le preprocess est dans le blob)
            img_lb, ratio, (dw, dh) = letterbox(img, (IMGSZ, IMGSZ))

            # Planar CHW uint8 contigu (pas de float, pas de /255 - fait dans le blob)
            img_chw = np.ascontiguousarray(
                img_lb.transpose(2, 0, 1)
            )  # shape (3, H, W), uint8

            t1 = time.perf_counter()

            # Frame DepthAI
            dai_frame = dai.ImgFrame()
            dai_frame.setWidth(IMGSZ)
            dai_frame.setHeight(IMGSZ)
            dai_frame.setType(dai.ImgFrame.Type.BGR888p)
            dai_frame.setData(img_chw.reshape(-1))  # PAS de .tolist()

            # Inference
            q_in.send(dai_frame)
            in_nn = q_out.get()

            t2 = time.perf_counter()

            # Decoder
            if output_layers is None:
                output_layers = in_nn.getAllLayerNames()
            raw_data = np.array(in_nn.getLayerFp16(output_layers[0]))

            # --- DETECTION AUTO DU FORMAT (v5/v7 avec objectness vs v8/v11 sans) ---
            n_anchors = (IMGSZ // 8) ** 2 + (IMGSZ // 16) ** 2 + (IMGSZ // 32) ** 2
            has_objectness = (raw_data.size % (num_classes + 5) == 0) and (
                raw_data.size // (num_classes + 5) == n_anchors
            )

            # --- SANITY-CHECK DU FORMAT DE SORTIE OAK ---
            if i == 0:
                print("\n[Sanity-check] Format de sortie OAK:")
                print(f"  - Forme brute        : {raw_data.shape}")
                print(
                    f"  - Min / Max          : {raw_data.min():.4f} / {raw_data.max():.4f}"
                )
                expected_v8 = (num_classes + 4) * n_anchors
                expected_v5 = (num_classes + 5) * n_anchors
                print(f"  - Taille attendue v8 : {expected_v8} | v5: {expected_v5}")
                print(
                    f"  - Objectness détecté : {'OUI (v5/v7 style)' if has_objectness else 'NON (v8/v11 style)'}"
                )

            # --- RESHAPE ET DECODAGE AUTO-ROBUSTE ---
            try:
                if has_objectness:
                    # Format v5/v7: [cx, cy, w, h, obj, cls0, cls1, ...]
                    out = raw_data.reshape(num_classes + 5, -1).transpose()
                    boxes = out[:, :4]
                    obj = out[:, 4]
                    cls = out[:, 5:]

                    # Auto-sigmoid si logits détectés
                    if obj.min() < 0 or obj.max() > 1:
                        if i == 0:
                            print("  [AUTO] Sigmoid appliqué sur objectness")
                        obj = 1 / (1 + np.exp(-np.clip(obj, -50, 50)))
                    if cls.min() < 0 or cls.max() > 1:
                        if i == 0:
                            print("  [AUTO] Sigmoid appliqué sur classes")
                        cls = 1 / (1 + np.exp(-np.clip(cls, -50, 50)))

                    class_ids_all = cls.argmax(axis=1)
                    scores = obj * cls.max(axis=1)
                else:
                    # Format v8/v11: [cx, cy, w, h, cls0, cls1, ...]
                    out = raw_data.reshape(num_classes + 4, -1).transpose()
                    boxes = out[:, :4]
                    cls = out[:, 4:]

                    # Auto-sigmoid si logits détectés
                    if cls.min() < 0 or cls.max() > 1:
                        if i == 0:
                            print("  [AUTO] Sigmoid appliqué sur classes")
                        cls = 1 / (1 + np.exp(-np.clip(cls, -50, 50)))

                    class_ids_all = cls.argmax(axis=1)
                    scores = cls.max(axis=1)

            except ValueError as e:
                if i == 0:
                    print(f"  [ERREUR RESHAPE] {e}")
                    print(f"  - Taille réelle: {raw_data.size}")
                continue

            # Sanity-check des scores (première image uniquement)
            if i == 0:
                print(
                    f"  - Score range (post) : {scores.min():.4f} - {scores.max():.4f}"
                )

            mask = scores > CONF_EVAL
            boxes_filtered = boxes[mask]
            scores_filtered = scores[mask]
            class_ids = class_ids_all[mask]

            predictions = []
            if len(scores_filtered) > 0:
                # --- PREPARATION POUR NMS (Format xywh requis par OpenCV) ---
                # boxes_filtered est en [cx, cy, w, h] (format YOLO brut)
                # OpenCV NMS demande [x_top_left, y_top_left, w, h]

                boxes_nms = []
                boxes_xyxy = []  # On garde aussi la version xyxy pour apres le NMS

                for j in range(len(scores_filtered)):
                    cx, cy, w, h = boxes_filtered[j, 0:4]

                    # 1. Format pour NMS [x, y, w, h]
                    x = cx - (w / 2)
                    y = cy - (h / 2)
                    boxes_nms.append([x, y, w, h])

                    # 2. Format pour nous (xyxy) pour le calcul final
                    boxes_xyxy.append([x, y, x + w, y + h])

                # --- NMS class-aware ---
                if hasattr(cv2.dnn, "NMSBoxesBatched"):
                    indices = cv2.dnn.NMSBoxesBatched(
                        boxes_nms,
                        scores_filtered.tolist(),
                        class_ids.tolist(),
                        CONF_EVAL,
                        NMS_IOU,
                    )
                else:
                    keep = []
                    for c in np.unique(class_ids):
                        ids = np.where(class_ids == c)[0]
                        idx_c = cv2.dnn.NMSBoxes(
                            [boxes_nms[i] for i in ids],
                            [float(scores_filtered[i]) for i in ids],
                            CONF_EVAL,
                            NMS_IOU,
                        )
                        if len(idx_c):
                            keep.extend(ids[idx_c.flatten()].tolist())
                    indices = np.array(keep, dtype=int)

                indices = np.array(indices).reshape(-1)

                # --- CAP MAX_DET (équité avec GPU) ---
                # Trier par score décroissant et garder les top-K
                if len(indices) > MAX_DET:
                    idx_scores = [(idx, scores_filtered[idx]) for idx in indices]
                    idx_scores.sort(key=lambda x: x[1], reverse=True)
                    indices = np.array([x[0] for x in idx_scores[:MAX_DET]])

                if len(indices) > 0:
                    for idx in indices.flatten():
                        # On recupere la boite correspondante en xyxy
                        box = list(boxes_xyxy[idx])

                        # --- REVERSE LETTERBOX ---
                        box[0] = (box[0] - dw) / ratio[0]  # x1
                        box[1] = (box[1] - dh) / ratio[1]  # y1
                        box[2] = (box[2] - dw) / ratio[0]  # x2
                        box[3] = (box[3] - dh) / ratio[1]  # y2

                        # --- NORMALISATION ET CLAMPING ---
                        box[0] = max(0, min(1, box[0] / orig_w))
                        box[1] = max(0, min(1, box[1] / orig_h))
                        box[2] = max(0, min(1, box[2] / orig_w))
                        box[3] = max(0, min(1, box[3] / orig_h))

                        predictions.append(
                            {
                                "box": box,
                                "class_id": int(class_ids[idx]),
                                "confidence": float(scores_filtered[idx]),
                            }
                        )

            t3 = time.perf_counter()
            e2e_times.append((t3 - t0) * 1000)
            device_times.append((t2 - t1) * 1000)
            preprocess_times.append((t1 - t0) * 1000)
            postprocess_times.append((t3 - t2) * 1000)

            all_predictions.append(predictions)

            # Ground truth
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

    # Metriques
    print("\nCalcul des metriques...")
    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    avg_e2e_time = float(np.mean(e2e_times)) if e2e_times else 0.0
    avg_device_time = float(np.mean(device_times)) if device_times else 0.0
    avg_preprocess_time = float(np.mean(preprocess_times)) if preprocess_times else 0.0
    avg_postprocess_time = (
        float(np.mean(postprocess_times)) if postprocess_times else 0.0
    )
    precision = metrics["precision"]
    recall = metrics["recall"]
    map50 = metrics["map50"]
    f1 = metrics["f1"]

    # Resultats
    print("\n" + "-" * 40)
    print("RESULTATS OAK-D")
    print("-" * 40)
    print(f"Taille blob    : {size_mb:.2f} MB")
    print(f"Temps end-to-end : {avg_e2e_time:.2f} ms")
    print(f"Temps send→get  : {avg_device_time:.2f} ms")
    print(f"Temps preprocess : {avg_preprocess_time:.2f} ms")
    print(f"Temps postprocess: {avg_postprocess_time:.2f} ms")
    print(f"mAP50          : {map50:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")

    save_results(
        hardware="OAK_MyriadX",
        model_name=model_name,
        size_mb=size_mb,
        e2e_time_ms=avg_e2e_time,
        device_time_ms=avg_device_time,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
        dataset=dataset,
        pixel_perfect=True,  # OAK utilise toujours le même letterbox
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark d'un modele YOLO compile sur GPU ou OAK-D"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["4070", "oak"],
        help="Cible hardware: '4070' pour GPU (ONNX), 'oak' pour OAK-D (blob)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le modele (.pt/.onnx pour GPU, .blob pour OAK)",
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
        help="Dataset: 'coco128' (rapide, 128 images) ou 'coco' (val2017, baseline scientifique)",
    )
    parser.add_argument(
        "--pixel-perfect",
        dest="pixel_perfect",
        action="store_true",
        help="Appliquer le meme letterbox GPU/OAK pour equite (defaut)",
    )
    parser.add_argument(
        "--no-pixel-perfect",
        dest="pixel_perfect",
        action="store_false",
        help="Desactiver le mode pixel-perfect (Ultralytics gere le preprocess GPU)",
    )
    parser.set_defaults(pixel_perfect=True)

    args = parser.parse_args()

    # Resoudre le chemin
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = ROOT_DIR / args.model

    if not model_path.exists():
        print(f"Erreur: Modele introuvable: {args.model}")
        return 1

    model_path = str(model_path.resolve())

    # Verifier l'extension
    ext = Path(model_path).suffix.lower()
    if args.target == "4070" and ext not in [".pt", ".onnx"]:
        print(f"Attention: Pour GPU, un fichier .pt ou .onnx est attendu (recu: {ext})")
    elif args.target == "oak" and ext != ".blob":
        print(f"Attention: Pour OAK, un fichier .blob est attendu (recu: {ext})")

    # Benchmark
    if args.target == "4070":
        benchmark_gpu(
            model_path,
            args.num_classes,
            pixel_perfect=args.pixel_perfect,
            dataset=args.dataset,
        )
    elif args.target == "oak":
        benchmark_oak(model_path, args.num_classes, dataset=args.dataset)

    return 0


if __name__ == "__main__":
    exit(main())
