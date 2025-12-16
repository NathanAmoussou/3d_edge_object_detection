import depthai as dai
import cv2
import numpy as np
import math
import time

import blobconverter  # <-- nouveau

# ========= 1. CRÉATION DU PIPELINE =========
pipeline = dai.Pipeline()

# --- Caméra RGB (Couleur) ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Pour le NN : flux "preview" redimensionné (plus léger)
cam_rgb.setPreviewSize(300, 300)  # taille classique pour MobileNet-SSD

# --- Caméras Mono (Gauche / Droite) pour la profondeur ---
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Gauche

mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Droite

# --- Noeud StereoDepth (calcul de la carte de profondeur) ---
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # aligner la depth sur la vue couleur
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# ========= 2. RÉSEAU DE DÉTECTION SPATIALE =========

spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)

# Récupération automatique d'un blob MobileNet-SSD depuis le model zoo DepthAI
blob_path = blobconverter.from_zoo(
    name="mobilenet-ssd",
    zoo_type="depthai",  # model zoo DepthAI
    shaves=6,            # optionnel, 6 shaves est courant sur OAK-D
)
spatial_nn.setBlobPath(blob_path)
spatial_nn.setConfidenceThreshold(0.5)

# Zone de depth considérée dans la bbox
spatial_nn.setBoundingBoxScaleFactor(0.5)
spatial_nn.setDepthLowerThreshold(100)    # 100 mm
spatial_nn.setDepthUpperThreshold(10000)  # 10 000 mm

# Connexions : image + depth vers le NN
cam_rgb.preview.link(spatial_nn.input)
stereo.depth.link(spatial_nn.inputDepth)

# ========= 3. SORTIES XLink =========

# Sortie RGB pour affichage
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Sortie disparity pour visu (optionnelle)
xout_disp = pipeline.create(dai.node.XLinkOut)
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

# Sortie des détections spatiales
xout_det = pipeline.create(dai.node.XLinkOut)
xout_det.setStreamName("detections")
spatial_nn.out.link(xout_det.input)

# ========= 4. BOUCLE PRINCIPALE =========

# Labels par défaut de ce MobileNet-SSD (COCO-like)
label_map = [
    "background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

with dai.Device(pipeline) as device:
    print("Device connecté :", device.getMxId())

    # OAK-D Pro : activer le laser IR
    device.setIrLaserDotProjectorBrightness(800)  # de 0 à 1200 mA

    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_disp = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    start_time = time.time()
    frame_count = 0

    while True:
        in_rgb = q_rgb.tryGet()
        in_disp = q_disp.tryGet()
        in_det = q_det.tryGet()

        frame_rgb = None

        # --- RGB ---
        if in_rgb is not None:
            frame_rgb = in_rgb.getCvFrame()
            frame_rgb = cv2.resize(frame_rgb, (640, 360))

        # --- Disparity (visu) ---
        if in_disp is not None:
            frame_disp = in_disp.getFrame()
            frame_disp = (frame_disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
            frame_disp_color = cv2.applyColorMap(frame_disp, cv2.COLORMAP_JET)
            frame_disp_color = cv2.resize(frame_disp_color, (640, 360))
            cv2.imshow("Depth (Disparity)", frame_disp_color)

        # --- Détections spatiales ---
        if in_det is not None and frame_rgb is not None:
            detections = in_det.detections
            height, width = frame_rgb.shape[:2]

            for det in detections:
                # Bbox normalisée [0..1] -> pixels
                x1 = int(det.xmin * width)
                y1 = int(det.ymin * height)
                x2 = int(det.xmax * width)
                y2 = int(det.ymax * height)

                # Coordonnées spatiales en mm
                x = det.spatialCoordinates.x
                y = det.spatialCoordinates.y
                z = det.spatialCoordinates.z

                distance = math.sqrt(x*x + y*y + z*z) / 1000.0  # en mètres
                distance_z = z / 1000.0

                # Label
                label_id = det.label
                if 0 <= label_id < len(label_map):
                    label = label_map[label_id]
                else:
                    label = f"id:{label_id}"

                # Dessin bbox + texte
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {det.confidence*100:.1f}% Z={distance_z:.2f}m"
                cv2.putText(frame_rgb, text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Log console
                print(f"[DET] {label} conf={det.confidence:.2f} "
                      f"X={x/1000:.2f}m Y={y/1000:.2f}m Z={distance_z:.2f}m | dist={distance:.2f}m")

        if frame_rgb is not None:
            cv2.imshow("RGB + Spatial detections", frame_rgb)

        frame_count += 1
        now = time.time()
        if (now - start_time) > 1.0:
            fps = frame_count / (now - start_time)
            frame_count = 0
            start_time = now
            # print(f"FPS: {fps:.1f}")

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
