#!/usr/bin/env python3
import depthai as dai
import cv2
import numpy as np
import os
from depthai_nodes.node import ParsingNeuralNetwork

# --- CONFIG ---
MODEL_ID = "luxonis/yolov8-instance-segmentation-nano:coco-512x288"
NN_SHAVES = 6
CONF_THR = 0.5
MASK_THR = 0.5
OUT_DIR = "oai_output"
os.makedirs(OUT_DIR, exist_ok=True)

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def main():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    model_desc = dai.NNModelDescription(model=MODEL_ID, platform="RVC2")
    blob = dai.getModelFromZoo(model_desc, useCached=True)
    nn_archive = dai.NNArchive(blob)

    nn = pipeline.create(ParsingNeuralNetwork).build(camRgb=cam_rgb, nnArchive=nn_archive, fps=10)
    nn.setNNArchive(nn_archive, numShaves=NN_SHAVES)
    try:
        nn.getParser().setConfThreshold(CONF_THR)
    except:
        pass

    frame_q = nn.passthrough.createOutputQueue()
    det_q   = nn.out.createOutputQueue()

    mono_l = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_r = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    l_out = mono_l.requestOutput((640, 400))
    r_out = mono_r.requestOutput((640, 400))

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setOutputSize(640, 400)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    l_out.link(stereo.left)
    r_out.link(stereo.right)

    spatial = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial.inputConfig.setWaitForMessage(True)
    stereo.depth.link(spatial.inputDepth)
    spatial_cfg_q = spatial.inputConfig.createInputQueue()
    spatial_q     = spatial.out.createOutputQueue()

    pipeline.start()

    frame_idx = 0
    while pipeline.isRunning():
        in_frame = frame_q.tryGet()
        if in_frame is None:
            continue
        frame = in_frame.getCvFrame()
        H, W = frame.shape[:2]

        in_det = det_q.tryGet()
        detections = []
        masks = None
        if in_det:
            detections = in_det.detections
            masks = getattr(in_det, "masks", None)
            if masks is not None:
                masks = np.array(masks)
                if masks.ndim == 3 and masks.shape[0] != len(detections) and masks.shape[-1] == len(detections):
                    masks = np.transpose(masks, (2,0,1))

        spatial_data = []
        if detections:
            cfg = dai.SpatialLocationCalculatorConfig()
            for det in detections:
                rr = det.rotated_rect
                cx, cy = rr.center.x, rr.center.y
                bw, bh = rr.size.width, rr.size.height

                x1 = clamp01(cx - bw/2)
                y1 = clamp01(cy - bh/2)
                x2 = clamp01(cx + bw/2)
                y2 = clamp01(cy + bh/2)

                cd = dai.SpatialLocationCalculatorConfigData()
                cd.depthThresholds.lowerThreshold = 100
                cd.depthThresholds.upperThreshold = 5000
                cd.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                cd.roi = dai.Rect(dai.Point2f(x1, y1), dai.Point2f(x2, y2))
                cfg.addROI(cd)

            spatial_cfg_q.send(cfg)
            res = spatial_q.get()
            spatial_data = res.spatialLocations

        # Dessiner dans la frame et sauvegarder
        for i, det in enumerate(detections):
            rr = det.rotated_rect
            cx, cy = rr.center.x, rr.center.y
            bw, bh = rr.size.width, rr.size.height

            x1 = int(clamp01(cx - bw/2) * W)
            y1 = int(clamp01(cy - bh/2) * H)
            x2 = int(clamp01(cx + bw/2) * W)
            y2 = int(clamp01(cy + bh/2) * H)

            conf = det.confidence
            label = det.label

            z_mm = None
            if i < len(spatial_data):
                z_mm = spatial_data[i].spatialCoordinates.z

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            txt = f"{label} {conf:.2f}"
            if z_mm is not None:
                txt += f" Z={z_mm}mm"
            cv2.putText(frame, txt, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        out_path = os.path.join(OUT_DIR, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        print("Wrote", out_path)
        frame_idx += 1

        if frame_idx >= 10:
            break

    pipeline.stop()

if __name__ == "__main__":
    main()
