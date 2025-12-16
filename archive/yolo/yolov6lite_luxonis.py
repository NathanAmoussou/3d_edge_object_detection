#!/usr/bin/env python3
"""
YOLOv6-Lite to Luxonis OAK-D Pipeline

This script:
1. Converts yolov6lite_m.pt to ONNX (with Luxonis-compatible ops)
2. Converts ONNX to blob using blobconverter
3. Runs spatial detection on OAK-D camera

Usage:
    python yolov6lite_luxonis.py              # Run inference (requires blob)
    python yolov6lite_luxonis.py --convert    # Convert model then run
    python yolov6lite_luxonis.py --convert-only  # Only convert, don't run
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ============== Configuration ==============
MODEL_PT = "yolov6lite_m.pt"
MODEL_ONNX = "yolov6lite_m.onnx"
MODEL_BLOB = "yolov6lite_m.blob"
INPUT_SIZE = (640, 640)  # YOLOv6-Lite default
NN_SHAVES = 6
CONF_THR = 0.5
IOU_THR = 0.45
FPS = 30

# COCO class labels (80 classes)
LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


# ============== Luxonis Compatibility Patches ==============
def patch_hardswish_hardsigmoid():
    """
    Patch PyTorch's HardSigmoid/HardSwish with Luxonis-compatible versions.
    Must be called before loading the model.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class HardSigmoidCompat(nn.Module):
        """Luxonis-compatible HardSigmoid: relu6(x + 3) / 6"""
        def forward(self, x):
            return F.relu6(x + 3.0) / 6.0

    class HardSwishCompat(nn.Module):
        """Luxonis-compatible HardSwish: x * HardSigmoid(x)"""
        def __init__(self):
            super().__init__()
            self.hs = HardSigmoidCompat()

        def forward(self, x):
            return x * self.hs(x)

    def replace_hard_ops(module: nn.Module):
        """Recursively replace nn.Hardsigmoid and nn.Hardswish."""
        for name, child in module.named_children():
            if isinstance(child, nn.Hardsigmoid):
                setattr(module, name, HardSigmoidCompat())
            elif isinstance(child, nn.Hardswish):
                setattr(module, name, HardSwishCompat())
            else:
                replace_hard_ops(child)

    return replace_hard_ops


# ============== Model Conversion ==============
def convert_pt_to_onnx(pt_path: str, onnx_path: str, input_size: tuple):
    """Convert YOLOv6-Lite .pt to ONNX format."""
    import torch

    print(f"[1/3] Loading model from {pt_path}...")

    # Try loading as YOLOv6 checkpoint
    ckpt = torch.load(pt_path, map_location="cpu")

    # YOLOv6 checkpoints can have different structures
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model = ckpt["model"]
        elif "ema" in ckpt and ckpt["ema"] is not None:
            model = ckpt["ema"]
        else:
            model = ckpt
    else:
        model = ckpt

    # If model is state_dict, we need to build architecture first
    if isinstance(model, dict):
        print("  Model is a state_dict. Attempting to load YOLOv6 architecture...")
        try:
            # Try importing YOLOv6
            sys.path.insert(0, str(Path(pt_path).parent))
            from yolov6.models.yolo import Model
            # This requires the YOLOv6 config file
            raise ImportError("Need to specify model config")
        except ImportError:
            print("  ERROR: Cannot load state_dict without YOLOv6 architecture.")
            print("  Please ensure yolov6lite_m.pt contains the full model, not just weights.")
            print("  You can export from YOLOv6 repo: python export.py --weights yolov6lite_m.pt --simplify")
            sys.exit(1)

    # Convert to float and eval mode
    model = model.float()
    model.eval()

    # Apply Luxonis-compatible patches
    print("[2/3] Patching HardSigmoid/HardSwish for Luxonis compatibility...")
    replace_hard_ops = patch_hardswish_hardsigmoid()
    replace_hard_ops(model)

    # Prepare dummy input
    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w)

    # Export to ONNX
    print(f"[3/3] Exporting to ONNX: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=None,  # Fixed size for edge deployment
        do_constant_folding=True,
    )

    print(f"  ONNX model saved to: {onnx_path}")
    return onnx_path


def convert_onnx_to_blob(onnx_path: str, blob_path: str, shaves: int = 6):
    """Convert ONNX to OpenVINO blob using blobconverter."""
    import blobconverter

    print(f"Converting ONNX to blob (shaves={shaves})...")

    blob = blobconverter.from_onnx(
        model=onnx_path,
        data_type="FP16",
        shaves=shaves,
        version="2022.1",
        use_cache=False,
        output_dir=str(Path(blob_path).parent) or ".",
    )

    # blobconverter returns the path, rename if needed
    if Path(blob).name != Path(blob_path).name:
        os.rename(blob, blob_path)
        blob = blob_path

    print(f"  Blob saved to: {blob}")
    return blob


def convert_model(pt_path: str, onnx_path: str, blob_path: str, input_size: tuple):
    """Full conversion pipeline: PT -> ONNX -> Blob"""
    print("=" * 50)
    print("YOLOv6-Lite Model Conversion")
    print("=" * 50)

    # Step 1: PT to ONNX
    if not Path(onnx_path).exists():
        convert_pt_to_onnx(pt_path, onnx_path, input_size)
    else:
        print(f"ONNX already exists: {onnx_path}")

    # Step 2: ONNX to Blob
    if not Path(blob_path).exists():
        convert_onnx_to_blob(onnx_path, blob_path, NN_SHAVES)
    else:
        print(f"Blob already exists: {blob_path}")

    print("=" * 50)
    print("Conversion complete!")
    print("=" * 50)


# ============== YOLO Post-processing ==============
def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


def postprocess_yolov6(output, conf_thres=0.5, iou_thres=0.45, input_size=(640, 640)):
    """
    Post-process YOLOv6 output.

    Args:
        output: Raw network output, shape (1, N, 85) or (1, 85, N) for 80 classes
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        input_size: Input image size (h, w)

    Returns:
        List of detections: [[x1, y1, x2, y2, conf, class_id], ...]
    """
    # Handle different output shapes
    if output.ndim == 3:
        output = output[0]  # Remove batch dim

    # YOLOv6 output: (N, 85) where 85 = 4 (bbox) + 1 (obj_conf) + 80 (class_conf)
    # Or it might be (85, N) - transpose if needed
    if output.shape[0] == 85 or output.shape[0] < output.shape[1]:
        output = output.T

    # Extract components
    boxes = output[:, :4]  # x, y, w, h
    obj_conf = output[:, 4:5] if output.shape[1] > 5 else np.ones((output.shape[0], 1))
    class_conf = output[:, 5:] if output.shape[1] > 5 else output[:, 4:]

    # For YOLOv6, sometimes it's already x1y1x2y2 with class scores
    # Check if it looks like xywh (center format) or xyxy
    if boxes[:, 2:4].max() > 2.0:  # Likely already xyxy in pixel coords
        # Normalize to input size
        boxes[:, [0, 2]] /= input_size[1]
        boxes[:, [1, 3]] /= input_size[0]
    else:
        # Convert xywh to xyxy
        boxes = xywh2xyxy(boxes)

    # Compute final scores
    if class_conf.shape[1] > 1:
        scores = obj_conf * class_conf
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
    else:
        class_ids = np.zeros(output.shape[0], dtype=np.int32)
        confidences = obj_conf.flatten() * class_conf.flatten()

    # Filter by confidence
    mask = confidences > conf_thres
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Apply NMS per class
    detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = confidences[cls_mask]

        keep = nms(cls_boxes, cls_scores, iou_thres)

        for idx in keep:
            detections.append([
                *cls_boxes[idx],  # x1, y1, x2, y2 (normalized)
                cls_scores[idx],  # confidence
                cls_id  # class id
            ])

    return detections


# ============== Luxonis Pipeline ==============
def run_inference(blob_path: str, conf_threshold: float = CONF_THR):
    """Run YOLOv6-Lite inference on OAK-D with spatial detection."""
    import depthai as dai

    if not Path(blob_path).exists():
        print(f"ERROR: Blob not found: {blob_path}")
        print("Run with --convert flag first to generate the blob.")
        sys.exit(1)

    print("=" * 50)
    print("YOLOv6-Lite Spatial Detection")
    print("=" * 50)
    print(f"Blob: {blob_path}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Confidence threshold: {conf_threshold}")
    print("Press 'q' to quit")
    print("=" * 50)

    # Create pipeline
    pipeline = dai.Pipeline()

    # === RGB Camera ===
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewSize(INPUT_SIZE[1], INPUT_SIZE[0])
    cam_rgb.setPreviewKeepAspectRatio(False)
    cam_rgb.setFps(FPS)

    # === Mono Cameras (for depth) ===
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # === Stereo Depth ===
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setOutputSize(INPUT_SIZE[1], INPUT_SIZE[0])

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # === Neural Network ===
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    nn.setNumPoolFrames(4)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)

    cam_rgb.preview.link(nn.input)

    # === Spatial Location Calculator ===
    spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial_calc.inputConfig.setWaitForMessage(False)
    spatial_calc.setWaitForConfigInput(False)

    stereo.depth.link(spatial_calc.inputDepth)

    # === Outputs ===
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    nn.out.link(xout_nn.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    spatial_cfg_in = pipeline.create(dai.node.XLinkIn)
    spatial_cfg_in.setStreamName("spatial_cfg")
    spatial_cfg_in.out.link(spatial_calc.inputConfig)

    xout_spatial = pipeline.create(dai.node.XLinkOut)
    xout_spatial.setStreamName("spatial")
    spatial_calc.out.link(xout_spatial.input)

    # === Run ===
    with dai.Device(pipeline) as device:
        print(f"Device connected: {device.getMxId()}")

        # Try to enable IR laser for OAK-D Pro
        try:
            device.setIrLaserDotProjectorBrightness(800)
            print("IR laser enabled")
        except Exception:
            pass

        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        q_spatial = device.getOutputQueue("spatial", maxSize=4, blocking=False)
        q_spatial_cfg = device.getInputQueue("spatial_cfg")

        while True:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()
            in_depth = q_depth.tryGet()

            if in_rgb is None:
                continue

            frame = in_rgb.getCvFrame()
            h, w = frame.shape[:2]

            # Process NN output
            detections = []
            if in_nn is not None:
                # Get raw output
                layer_names = in_nn.getAllLayerNames()
                if layer_names:
                    output = np.array(in_nn.getLayerFp16(layer_names[0]))
                else:
                    output = np.array(in_nn.getFirstLayerFp16())

                # Reshape based on output size
                # YOLOv6-Lite outputs vary, common shapes:
                # (1, 8400, 85) or (8400, 85) or (1, 85, 8400)
                total = output.size
                num_classes = len(LABELS)
                box_dim = 4 + 1 + num_classes  # 85 for COCO

                if total % box_dim == 0:
                    num_boxes = total // box_dim
                    output = output.reshape(num_boxes, box_dim)
                elif total % (box_dim - 1) == 0:
                    # Maybe no obj_conf, just class scores
                    num_boxes = total // (box_dim - 1)
                    output = output.reshape(num_boxes, box_dim - 1)

                detections = postprocess_yolov6(
                    output, conf_thres=conf_threshold, iou_thres=IOU_THR, input_size=INPUT_SIZE
                )

            # Request spatial data for detections
            if detections and in_depth is not None:
                cfg = dai.SpatialLocationCalculatorConfig()
                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    # Clamp to [0, 1]
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))

                    if x2 > x1 and y2 > y1:
                        cd = dai.SpatialLocationCalculatorConfigData()
                        cd.depthThresholds.lowerThreshold = 100
                        cd.depthThresholds.upperThreshold = 10000
                        cd.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                        cd.roi = dai.Rect(dai.Point2f(x1, y1), dai.Point2f(x2, y2))
                        cfg.addROI(cd)

                if cfg.getConfigData():
                    q_spatial_cfg.send(cfg)

            # Get spatial results
            spatial_data = []
            in_spatial = q_spatial.tryGet()
            if in_spatial is not None:
                spatial_data = in_spatial.getSpatialLocations()

            # Draw detections
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)

                # Convert to pixel coordinates
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)

                # Get label
                label = LABELS[cls_id] if 0 <= cls_id < len(LABELS) else f"id:{cls_id}"

                # Get depth if available
                z_mm = None
                if i < len(spatial_data):
                    z_mm = spatial_data[i].spatialCoordinates.z

                # Draw
                color = (0, 255, 0)
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

                text = f"{label} {conf:.2f}"
                if z_mm is not None and z_mm > 0:
                    text += f" Z={z_mm/1000:.2f}m"

                cv2.putText(frame, text, (px1, max(0, py1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show FPS
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("YOLOv6-Lite Spatial Detection", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description="YOLOv6-Lite to Luxonis OAK-D Pipeline")
    parser.add_argument("--convert", action="store_true",
                        help="Convert model before running inference")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert model, don't run inference")
    parser.add_argument("--pt", type=str, default=MODEL_PT,
                        help=f"Path to .pt model (default: {MODEL_PT})")
    parser.add_argument("--onnx", type=str, default=MODEL_ONNX,
                        help=f"Path to .onnx model (default: {MODEL_ONNX})")
    parser.add_argument("--blob", type=str, default=MODEL_BLOB,
                        help=f"Path to .blob model (default: {MODEL_BLOB})")
    parser.add_argument("--conf", type=float, default=CONF_THR,
                        help=f"Confidence threshold (default: {CONF_THR})")

    args = parser.parse_args()

    conf_threshold = args.conf

    if args.convert or args.convert_only:
        convert_model(args.pt, args.onnx, args.blob, INPUT_SIZE)

    if not args.convert_only:
        run_inference(args.blob, conf_threshold)


if __name__ == "__main__":
    main()
