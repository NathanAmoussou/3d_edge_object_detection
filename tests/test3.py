import depthai as dai
import cv2
import blobconverter
import numpy as np

# --- 1. Création du pipeline ---
pipeline = dai.Pipeline()

# --- 2. Caméra Couleur ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(256, 256)
cam_rgb.setInterleaved(False)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewKeepAspectRatio(False)

# --- 3. Réseau de Neurones (Segmentation) ---

nn = pipeline.create(dai.node.NeuralNetwork)

print("Téléchargement/Chargement du modèle DeepLabV3+...")
nn.setBlobPath(blobconverter.from_zoo(name="deeplab_v3_mnv2_256x256", zoo_type="depthai", shaves=6))

# --- 4. Sorties ---
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# --- 5. Liaisons ---
cam_rgb.preview.link(nn.input)
cam_rgb.preview.link(xout_rgb.input)
nn.out.link(xout_nn.input)


# --- Fonction pour colorier le résultat ---
def decode_deeplab(output_tensor):
    # Le modèle sort un tableau de forme (1, 21, 256, 256)
    # 21 = nombre de classes (fond, avion, vélo, oiseau, bateau, bouteille, bus, voiture, chat, chaise, vache, table, chien, cheval, moto, personne, plante, mouton, canapé, train, tv)

    # 1. On récupère les données brutes
    class_predictions = output_tensor.reshape((21, 256, 256))

    class_map = class_predictions.argmax(axis=0).astype(np.uint8)

    output_colors = cv2.applyColorMap(class_map * 12, cv2.COLORMAP_JET)

    return output_colors

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

    frame = None
    print("Démarrage... (Appuie sur 'q' pour quitter)")

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            output_data = in_nn.getFirstLayerInt32()

            # Vérification de la taille (256x256 = 65536)
            if len(output_data) == 65536:
                # 1. Conversion directe en image 256x256
                class_map = np.array(output_data).reshape(256, 256).astype(np.uint8)

                # 2. On applique les couleurs (x10 pour bien séparer les couleurs visuellement)
                colored_mask = cv2.applyColorMap(class_map * 10, cv2.COLORMAP_JET)

                if frame is not None:
                    blended = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)

                    display_frame = cv2.resize(blended, (768, 768), interpolation=cv2.INTER_NEAREST)

                    cv2.imshow("OAK-D Segmentation (Zoom)", display_frame)

        if cv2.waitKey(1) == ord('q'):
            break