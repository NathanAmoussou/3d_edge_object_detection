"""
Launches YOLOv11 on the OAK D PRO Camera.

Supports headless mode (no display) for embedded systems like Raspberry Pi.
In headless mode, prints detection results (detected, x, y, z) to stdout.

Usage:
    python detect_oak_redcube.py
    python detect_oak_redcube.py --model models/oak/yolo11n_640_fp16_8shave.blob
    python detect_oak_redcube.py --shaves 8
"""

from pathlib import Path
import argparse
import os
import cv2
import depthai as dai
import numpy as np


def is_headless() -> bool:
    """
    Detecte si on est en mode headless (pas de display disponible).

    Returns:
        True si pas de display, False sinon.
    """
    # Methode 1: Verifier DISPLAY (Linux/X11)
    # display = os.environ.get("DISPLAY")
    # if display is None or display == "":
    #     return True

    # # Methode 2: Essayer d'ouvrir une fenetre OpenCV (plus robuste)
    # try:
    #     test_img = np.zeros((1, 1, 3), dtype=np.uint8)
    #     cv2.imshow("__test__", test_img)
    #     cv2.waitKey(1)
    #     cv2.destroyWindow("__test__")
    #     return False
    # except cv2.error:
    #     return True

    return True


# --- Configuration par defaut ---
ROOT_DIR = Path(__file__).parent.parent
DEFAULT_BLOB_PATH = ROOT_DIR / "models/red_cube_01/best_openvino_2022.1_6shave.blob"
INPUT_SIZE = 640
NUM_CLASSES = 1
CLASS_NAME = "red_cube"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4


def create_pipeline(blob_path: str) -> dai.Pipeline:
    """Cree le pipeline DepthAI avec le blob specifie."""
    pipeline = dai.Pipeline()

    # --- 1. Caméra Couleur (RGB) ---
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(INPUT_SIZE, INPUT_SIZE)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # --- 2. Caméras Mono (Stéréo pour la profondeur) ---
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Gauche

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Droite

    # --- 3. Nœud de Profondeur (StereoDepth) ---
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # IMPORTANT: Aligner la profondeur sur la caméra RGB
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 360)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # --- 4. Réseau de neurones (NeuralNetwork) ---
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    cam_rgb.preview.link(nn.input)

    # --- 5. Sorties (XLinkOut) ---
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    nn.out.link(xout_nn.input)

    # Sortie Profondeur
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


# --- Fonction de décodage YOLO (La même que précédemment) ---
def decode_yolo_v11(output_layer, conf_thres, iou_thres):
    data = np.array(output_layer).reshape(NUM_CLASSES + 4, -1).transpose()
    scores = np.max(data[:, 4:], axis=1)
    mask = scores > conf_thres
    data_filtered = data[mask]
    scores_filtered = scores[mask]
    
    if len(scores_filtered) == 0:
        return [], [], []

    class_ids = np.argmax(data_filtered[:, 4:], axis=1)
    boxes = data_filtered[:, 0:4]
    boxes[:, 0] -= 0.5 * boxes[:, 2]
    boxes[:, 1] -= 0.5 * boxes[:, 3]
    
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores_filtered.tolist(), conf_thres, iou_thres)
    
    final_boxes = []
    final_scores = []
    final_ids = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores_filtered[i])
            final_ids.append(class_ids[i])
            
    return final_boxes, final_scores, final_ids

# --- Fonction utilitaire pour calculer la distance moyenne ---
def get_object_depth(depth_frame, box, scale_x, scale_y):
    # box est [x, y, w, h] dans l'espace INPUT_SIZE x INPUT_SIZE
    # On doit convertir vers la taille de la depth_frame
    
    # Dimensions de la frame de profondeur
    dh, dw = depth_frame.shape

    # Conversion des coordonnées
    # On utilise les scales inverses car scale_x était (W_ecran / INPUT_SIZE)
    # Ici on veut projeter INPUT_SIZE vers W_depth
    
    x = int(box[0] * (dw / INPUT_SIZE))
    y = int(box[1] * (dh / INPUT_SIZE))
    w = int(box[2] * (dw / INPUT_SIZE))
    h = int(box[3] * (dh / INPUT_SIZE))

    # Sécurité pour ne pas sortir de l'image
    x = max(0, x)
    y = max(0, y)
    w = min(dw - x, w)
    h = min(dh - y, h)

    if w <= 0 or h <= 0:
        return 0

    # On prend une petite zone au centre de la boîte (ROI)
    # Prendre toute la boîte risque d'inclure le fond
    center_x = x + w // 2
    center_y = y + h // 2
    roi_size = 10 # Rayon de 10 pixels
    
    roi_x1 = max(0, center_x - roi_size)
    roi_y1 = max(0, center_y - roi_size)
    roi_x2 = min(dw, center_x + roi_size)
    roi_y2 = min(dh, center_y + roi_size)

    roi = depth_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Filtrer les zéros (valeurs invalides)
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) == 0:
        return 0
    
    # Retourne la médiane en mm (plus robuste que la moyenne)
    return np.median(valid_depths)


# --- Boucle Principale ---
def main():
    """
    Boucle principale de detection.

    En mode headless (pas de display), retourne les resultats sous forme:
        detected: bool, x: float, y: float, z: float

    x, y sont les coordonnees normalisees (0-1) du centre de la detection.
    z est la distance en metres.
    """
    parser = argparse.ArgumentParser(
        description="Detection YOLOv11 sur OAK-D avec profondeur"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin vers le fichier .blob (defaut: models/red_cube_01/...)",
    )
    parser.add_argument(
        "--shaves",
        type=int,
        default=None,
        choices=range(1, 17),
        metavar="[1-16]",
        help="Nombre de shaves (selectionne le blob correspondant si --model non specifie)",
    )
    args = parser.parse_args()

    # Determiner le chemin du blob
    if args.model:
        blob_path = Path(args.model)
        if not blob_path.exists():
            blob_path = ROOT_DIR / args.model
    elif args.shaves:
        # Convention de nommage: *_{shaves}shave.blob
        blob_path = ROOT_DIR / f"models/red_cube_01/best_openvino_2022.1_{args.shaves}shave.blob"
    else:
        blob_path = DEFAULT_BLOB_PATH

    if not blob_path.exists():
        print(f"[ERREUR] Blob introuvable: {blob_path}")
        return 1

    print(f"[INFO] Blob: {blob_path}")

    headless = is_headless()

    if headless:
        print("[INFO] Mode headless detecte - pas d'affichage")
        print("[INFO] Format sortie: detected,x,y,z")
    else:
        print("[INFO] Mode display actif")

    pipeline = create_pipeline(str(blob_path))

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        frame = None
        depth_frame = None

        try:
            while True:
                in_rgb = q_rgb.tryGet()
                in_nn = q_nn.tryGet()
                in_depth = q_depth.tryGet()

                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()

                if in_depth is not None:
                    depth_frame = in_depth.getFrame()  # Format uint16 en millimètres

                if in_nn is not None and frame is not None:
                    output_layers = in_nn.getAllLayerNames()
                    raw_data = in_nn.getLayerFp16(output_layers[0])

                    boxes, scores, class_ids = decode_yolo_v11(
                        raw_data, CONF_THRESHOLD, IOU_THRESHOLD
                    )

                    h, w = frame.shape[:2]
                    scale_x = w / INPUT_SIZE
                    scale_y = h / INPUT_SIZE

                    if headless:
                        # --- Mode headless: retourne detected, x, y, z ---
                        if len(boxes) > 0:
                            # Prendre la detection avec le score le plus eleve
                            best_idx = np.argmax(scores)
                            box = boxes[best_idx]

                            # Centre de la bbox normalise (0-1)
                            cx = (box[0] + box[2] / 2) / INPUT_SIZE
                            cy = (box[1] + box[3] / 2) / INPUT_SIZE

                            # Distance Z en metres
                            z_mm = 0
                            if depth_frame is not None:
                                z_mm = get_object_depth(
                                    depth_frame, box, scale_x, scale_y
                                )
                            z_m = z_mm / 1000.0 if z_mm > 0 else 0.0

                            print(f"True,{cx:.4f},{cy:.4f},{z_m:.3f}")
                        else:
                            print("False,0,0,0")
                    else:
                        # --- Mode display: affichage OpenCV ---
                        for i in range(len(boxes)):
                            box = boxes[i]  # [x, y, w, h] brut (INPUT_SIZE px)

                            # Calcul de la distance si la frame de profondeur est dispo
                            distance_mm = 0
                            if depth_frame is not None:
                                distance_mm = get_object_depth(
                                    depth_frame, box, scale_x, scale_y
                                )

                            # Coordonnées pour l'affichage RGB
                            x_disp = int(box[0] * scale_x)
                            y_disp = int(box[1] * scale_y)
                            bw_disp = int(box[2] * scale_x)
                            bh_disp = int(box[3] * scale_y)

                            # Texte : Classe + Confiance + Distance
                            dist_str = (
                                f"{distance_mm/1000:.2f}m" if distance_mm > 0 else "??"
                            )
                            label = f"{CLASS_NAME} {scores[i]:.2f} [{dist_str}]"

                            cv2.rectangle(
                                frame,
                                (x_disp, y_disp),
                                (x_disp + bw_disp, y_disp + bh_disp),
                                (0, 255, 0),
                                2,
                            )
                            cv2.putText(
                                frame,
                                label,
                                (x_disp, y_disp - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                        cv2.imshow("OAK-D YOLOv11 + Depth", frame)

                if not headless and cv2.waitKey(1) == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Arret demande (Ctrl+C)")

        finally:
            if not headless:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()