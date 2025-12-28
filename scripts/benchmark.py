"""
Benchmark d'un modele YOLO sur GPU ou OAK-D.

Usage:
    python scripts/benchmark.py --target 4070 --model models/base/yolo11n.pt
    python scripts/benchmark.py --target oak --model models/base/yolo11n_openvino_2022.1_6shave.blob --num-classes 80
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


def get_model_size_mb(filepath: str) -> float:
    """Retourne la taille du fichier en MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def calculate_f1(precision: float, recall: float) -> float:
    """Calcule le F1-score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def save_results(
    hardware: str,
    model_name: str,
    size_mb: float,
    inference_time_ms: float,
    map50: float,
    precision: float,
    recall: float,
    f1: float,
):
    """Sauvegarde les resultats dans le CSV."""
    file_exists = RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Timestamp",
                    "Hardware",
                    "Model_Name",
                    "Size_MB",
                    "Inference_Time_ms",
                    "mAP50",
                    "Precision",
                    "Recall",
                    "F1",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                hardware,
                model_name,
                f"{size_mb:.2f}",
                f"{inference_time_ms:.2f}",
                f"{map50:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
            ]
        )

    print(f"\nResultats sauvegardes dans: {RESULTS_FILE}")


def load_coco128_dataset():
    """Charge le dataset COCO128 (images + labels)."""
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset("coco128.yaml")
    val_images_dir = Path(data["path"]) / "images" / "train2017"
    val_labels_dir = Path(data["path"]) / "labels" / "train2017"

    image_files = sorted(val_images_dir.glob("*.jpg"))

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
    """Redimensionne l'image en gardant le ratio (ajoute du padding)."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, ratio, (dw, dh)


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


def match_tp_fp(preds, gts, iou_thr=0.5):
    """
    Associe prédictions et GT (class-aware, une GT max par prédiction).
    Retourne: tp (N,1) bool, conf (N,), pred_cls (N,), target_cls (M,)
    """
    preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)
    tp = np.zeros((len(preds), 1), dtype=bool)
    conf = np.array([p["confidence"] for p in preds], dtype=float)
    pred_cls = np.array([p["class_id"] for p in preds], dtype=int)
    target_cls = np.array([g["class_id"] for g in gts], dtype=int)

    matched = [False] * len(gts)
    for i, p in enumerate(preds):
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if matched[j] or p["class_id"] != g["class_id"]:
                continue
            iou = calculate_iou(p["box"], g["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i, 0] = True
            matched[best_j] = True
    return tp, conf, pred_cls, target_cls


def compute_map50_prf1(all_predictions, all_ground_truths, conf_op=0.25, iou_thr=0.5):
    """Calcule mAP50 (AP intégrale) + P/R/F1 au seuil conf_op."""
    tp_all, conf_all, pred_cls_all, target_cls_all = [], [], [], []
    for preds, gts in zip(all_predictions, all_ground_truths):
        tp, conf, pred_cls, target_cls = match_tp_fp(preds, gts, iou_thr=iou_thr)
        if len(conf):
            tp_all.append(tp)
            conf_all.append(conf)
            pred_cls_all.append(pred_cls)
        if len(target_cls):
            target_cls_all.append(target_cls)

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

    TP = FP = FN = 0
    for preds, gts in zip(all_predictions, all_ground_truths):
        preds_op = [p for p in preds if p["confidence"] >= conf_op]
        m = match_tp_fp(preds_op, gts, iou_thr=iou_thr)[0].reshape(-1)
        TP += int(m.sum())
        FP += int((~m).sum())

        gt_matched = [False] * len(gts)
        preds_sorted = sorted(preds_op, key=lambda x: x["confidence"], reverse=True)
        for p in preds_sorted:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if gt_matched[j] or p["class_id"] != g["class_id"]:
                    continue
                iou = calculate_iou(p["box"], g["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                gt_matched[best_j] = True
        FN += int(sum(1 for x in gt_matched if not x))

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = calculate_f1(precision, recall)

    return {"map50": map50, "precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# BRANCHE GPU (Ultralytics)
# =============================================================================


def benchmark_gpu(model_path: str, num_classes: int):
    """Benchmark sur GPU NVIDIA via Ultralytics."""
    from ultralytics import YOLO

    print("=" * 60)
    print("BENCHMARK GPU (RTX 4070) - Ultralytics")
    print("=" * 60)

    model_name = Path(model_path).stem
    size_mb = get_model_size_mb(model_path)

    print(f"Modele: {model_path}")
    print(f"Taille: {size_mb:.2f} MB")

    # Charger le modele
    model = YOLO(model_path)
    print(f"Classes: {len(model.names)}")

    print("\nInference sur COCO128 (GPU)...")
    dataset = load_coco128_dataset()
    all_predictions = []
    all_ground_truths = []
    inference_times = []

    for sample in dataset:
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]

        start_time = time.perf_counter()
        result = model.predict(
            img,
            imgsz=IMGSZ,
            conf=CONF_EVAL,
            iou=NMS_IOU,
            max_det=3000,
            device=0,
            verbose=False,
        )[0]
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)

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

    inference_time_ms = float(np.mean(inference_times)) if inference_times else 0.0
    map50 = metrics["map50"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    # Resultats
    print("\n" + "-" * 40)
    print("RESULTATS GPU")
    print("-" * 40)
    print(f"Taille modele  : {size_mb:.2f} MB")
    print(f"Temps inference: {inference_time_ms:.2f} ms")
    print(f"mAP50          : {map50:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")

    save_results(
        hardware="GPU_RTX4070",
        model_name=model_name,
        size_mb=size_mb,
        inference_time_ms=inference_time_ms,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# =============================================================================
# BRANCHE OAK-D (blob + DepthAI)
# =============================================================================


def benchmark_oak(model_path: str, num_classes: int):
    """Benchmark sur OAK-D (Myriad X VPU)."""
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
    print("\nChargement du dataset COCO128...")
    dataset = load_coco128_dataset()
    print(f"Images: {len(dataset)}")

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
    inference_times = []

    with dai.Device(pipeline) as device:
        q_in = device.getInputQueue("input")
        q_out = device.getOutputQueue("nn", maxSize=4, blocking=True)

        for i, sample in enumerate(dataset):
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]

            # Letterbox en BGR (pas de conversion RGB, le preprocess est dans le blob)
            img_lb, ratio, (dw, dh) = letterbox(img, (IMGSZ, IMGSZ))

            # Planar CHW uint8 contigu (pas de float, pas de /255 - fait dans le blob)
            img_chw = np.ascontiguousarray(
                img_lb.transpose(2, 0, 1)
            )  # shape (3, H, W), uint8

            # Frame DepthAI
            dai_frame = dai.ImgFrame()
            dai_frame.setWidth(IMGSZ)
            dai_frame.setHeight(IMGSZ)
            dai_frame.setType(dai.ImgFrame.Type.BGR888p)
            dai_frame.setData(img_chw.reshape(-1))  # PAS de .tolist()

            # Inference
            start_time = time.perf_counter()
            q_in.send(dai_frame)
            in_nn = q_out.get()
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)

            # Decoder
            output_layers = in_nn.getAllLayerNames()
            raw_data = np.array(in_nn.getLayerFp16(output_layers[0]))

            try:
                data = raw_data.reshape(num_classes + 4, -1).transpose()
            except ValueError:
                continue

            scores = np.max(data[:, 4:], axis=1)
            mask = scores > CONF_EVAL
            data_filtered = data[mask]
            scores_filtered = scores[mask]

            predictions = []
            if len(scores_filtered) > 0:
                class_ids = np.argmax(data_filtered[:, 4:], axis=1)

                # --- PREPARATION POUR NMS (Format xywh requis par OpenCV) ---
                # data_filtered est en [cx, cy, w, h] (format YOLO brut)
                # OpenCV NMS demande [x_top_left, y_top_left, w, h]

                boxes_nms = []
                boxes_xyxy = []  # On garde aussi la version xyxy pour apres le NMS

                for j in range(len(scores_filtered)):
                    cx, cy, w, h = data_filtered[j, 0:4]

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
                print(f"  {i + 1}/{len(dataset)} images...")

    # Metriques
    print("\nCalcul des metriques...")
    metrics = compute_map50_prf1(
        all_predictions, all_ground_truths, conf_op=CONF_OP, iou_thr=MATCH_IOU
    )

    avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
    precision = metrics["precision"]
    recall = metrics["recall"]
    map50 = metrics["map50"]
    f1 = metrics["f1"]

    # Resultats
    print("\n" + "-" * 40)
    print("RESULTATS OAK-D")
    print("-" * 40)
    print(f"Taille blob    : {size_mb:.2f} MB")
    print(f"Temps inference: {avg_inference_time:.2f} ms")
    print(f"mAP50          : {map50:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")

    save_results(
        hardware="OAK_MyriadX",
        model_name=model_name,
        size_mb=size_mb,
        inference_time_ms=avg_inference_time,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
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
        benchmark_gpu(model_path, args.num_classes)
    elif args.target == "oak":
        benchmark_oak(model_path, args.num_classes)

    return 0


if __name__ == "__main__":
    exit(main())
