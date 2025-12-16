#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np

MODEL = "luxonis/mobilenet-ssd:300x300"
FPS = 30
CONF_THR = 0.5

# Labels VOC (MobileNet-SSD)
LABELS = [
    "background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    d = depth_mm.astype(np.float32)
    d_down = d[::4, ::4]
    nonzero = d_down[d_down > 0]
    if nonzero.size == 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo = np.percentile(nonzero, 1)
    hi = np.percentile(nonzero, 99)
    hi = max(hi, lo + 1.0)
    img = np.interp(d, (lo, hi), (0, 255)).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_HOT)

with dai.Pipeline() as p:
    modelDesc = dai.NNModelDescription(MODEL)

    # Sources
    camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Stereo depth (configuration légère)
    stereo = p.create(dai.node.StereoDepth)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)

    # Recommandation RVC2: largeur divisible par 16 (sinon warnings/erreurs selon config)
    platform = p.getDefaultDevice().getPlatform()
    if platform == dai.Platform.RVC2:
        stereo.setOutputSize(640, 400)

    monoLeft.requestOutput((640, 400)).link(stereo.left)
    monoRight.requestOutput((640, 400)).link(stereo.right)

    # SpatialDetectionNetwork = bbox + (X,Y,Z) à partir de la depth
    sdn = p.create(dai.node.SpatialDetectionNetwork).build(camRgb, stereo, modelDesc, fps=FPS)
    sdn.setConfidenceThreshold(CONF_THR)
    sdn.input.setBlocking(False)
    sdn.setBoundingBoxScaleFactor(0.5)
    sdn.setDepthLowerThreshold(100)    # mm
    sdn.setDepthUpperThreshold(10000)  # mm

    # Queues (v3: output.createOutputQueue())
    qRgb = sdn.passthrough.createOutputQueue()
    qDet = sdn.out.createOutputQueue()
    qDepth = sdn.passthroughDepth.createOutputQueue()

    p.start()  # DepthAI v3: remplacer dai.Device(pipeline) par pipeline.start() :contentReference[oaicite:3]{index=3}

    last_dets = []

    while p.isRunning():
        inDet = qDet.tryGet()
        if inDet is not None:
            last_dets = inDet.detections

        inRgb = qRgb.tryGet()
        inDepth = qDepth.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            h, w = frame.shape[:2]

            for d in last_dets:
                x1 = int(d.xmin * w); y1 = int(d.ymin * h)
                x2 = int(d.xmax * w); y2 = int(d.ymax * h)

                label_id = int(d.label)
                label = LABELS[label_id] if 0 <= label_id < len(LABELS) else str(label_id)

                z_m = d.spatialCoordinates.z / 1000.0
                txt = f"{label} {d.confidence*100:.0f}% Z={z_m:.2f}m"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, txt, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("RGB (NN passthrough)", frame)

        if inDepth is not None:
            depth = inDepth.getCvFrame()  # depth en mm
            depth_col = colorize_depth(depth)

            # Dessiner la bbox remappée sur l'image depth (si disponible)
            for d in last_dets:
                if hasattr(d, "boundingBoxMapping"):
                    roi = d.boundingBoxMapping.roi.denormalize(depth_col.shape[1], depth_col.shape[0])
                    tl = roi.topLeft(); br = roi.bottomRight()
                    cv2.rectangle(depth_col, (int(tl.x), int(tl.y)), (int(br.x), int(br.y)), (255, 255, 255), 1)

            cv2.imshow("Depth (passthroughDepth)", depth_col)

        if cv2.waitKey(1) == ord("q"):
            p.stop()
            break

cv2.destroyAllWindows()
