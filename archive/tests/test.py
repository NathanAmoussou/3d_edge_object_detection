import depthai as dai
import cv2
import numpy as np

# 1. Création du pipeline
pipeline = dai.Pipeline()

# --- Configuration Caméra RGB (Couleur) ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# --- Configuration Caméras Mono (Gauche / Droite) pour la profondeur ---
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B) # Gauche

mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C) # Droite

# --- Configuration du Noeud Stereo Depth (Calcul de profondeur) ---
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Alignement : on aligne la profondeur sur la vue couleur (très utile pour superposer les deux)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True) # Nettoie les erreurs de bordure
stereo.setSubpixel(True)       # Plus précis

# Liaison des caméras mono au moteur stéréo
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# --- Sorties (XLinkOut) ---
# Sortie RGB
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

# Sortie Disparité (La visualisation de la profondeur)
xout_disp = pipeline.create(dai.node.XLinkOut)
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

# --- Boucle Principale ---
with dai.Device(pipeline) as device:
    
    # *** SPÉCIAL OAK-D PRO : Activer le Laser IR ***
    # Cela projette des points invisibles pour aider la caméra à voir les murs blancs ou dans le noir
    device.setIrLaserDotProjectorBrightness(800) # De 0 à 1200mA
    
    # Création des queues de réception
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_disp = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.tryGet()
        in_disp = q_disp.tryGet()

        if in_rgb is not None:
            frame_rgb = in_rgb.getCvFrame()
            # On redimensionne un peu pour que ça rentre dans l'écran
            frame_rgb = cv2.resize(frame_rgb, (640, 360))
            cv2.imshow("RGB", frame_rgb)

        if in_disp is not None:
            # Récupération de la frame brute
            frame_disp = in_disp.getFrame()
            
            # La disparité brute n'est pas jolie, on la normalise pour l'affichage (0-255)
            # On multiplie par le facteur de subpixel (si activé)
            frame_disp = (frame_disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
            
            # Application d'une carte de couleur (JET = Bleu vers Rouge)
            frame_disp_color = cv2.applyColorMap(frame_disp, cv2.COLORMAP_JET)
            
            frame_disp_color = cv2.resize(frame_disp_color, (640, 360))
            cv2.imshow("Depth (Disparity)", frame_disp_color)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()