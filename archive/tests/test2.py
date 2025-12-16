import depthai as dai
import cv2
import blobconverter
import numpy as np

# Création du pipeline
pipeline = dai.Pipeline()

# --- 1. Caméra Couleur ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300) # Taille requise par MobileNetSSD
cam_rgb.setInterleaved(False)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# --- 2. Réseau de Neurones (Detection) ---
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
# Télécharge automatiquement le modèle compilé pour la caméra
nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
nn.setConfidenceThreshold(0.5)

# --- 3. Sorties (Flux vidéo + Données IA) ---
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# --- 4. Liaisons (Linking) ---
# Caméra -> Preview (300x300) -> Réseau de Neurones
cam_rgb.preview.link(nn.input)
# Caméra -> Video (HD) -> Sortie PC (pour l'affichage)
cam_rgb.video.link(xout_rgb.input)
# Réseau de Neurones -> Sortie PC (Données de détection)
nn.out.link(xout_nn.input)

# Liste des labels de MobileNetSSD
labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
          "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
          "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Fonction utilitaire pour normaliser les boites englobantes
def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# --- 5. Boucle principale ---
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

    frame = None
    detections = []

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            detections = in_nn.detections

        if frame is not None:
            # Affichage des détections
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                
                label_index = detection.label
                label_text = labels[label_index] if label_index < len(labels) else f"ID {label_index}"
                cv2.putText(frame, f"{label_text} {int(detection.confidence * 100)}%", 
                            (bbox[0] + 10, bbox[1] + 20), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))

            cv2.imshow("OAK-D Pro - AI Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break