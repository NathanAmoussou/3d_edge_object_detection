import cv2
import depthai as dai
import numpy as np
import blobconverter

# 1. Définition du modèle et de la pipeline
# Nous utilisons un modèle DeepLabV3+ pré-entraîné sur la classe "Personne"
model_name = "deeplabv3p_person"
input_size = (512, 512)

pipeline = dai.Pipeline()

# 2. Configuration de la caméra couleur
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(input_size[0], input_size[1])
cam.setInterleaved(False)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(30)
# Pour une inférence propre, on garde le ratio ou on stretch (ici stretch pour simplifier)
cam.setPreviewKeepAspectRatio(False)

# 3. Configuration du Réseau de Neurones
nn = pipeline.create(dai.node.NeuralNetwork)
# Le blobconverter télécharge et compile le modèle automatiquement
print(f"Téléchargement et compilation du modèle {model_name}...")
nn.setBlobPath(blobconverter.from_zoo(name=model_name, zoo_type="depthai", shaves=6))
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Liaison Caméra -> NN
cam.preview.link(nn.input)

# 4. Sorties (XLinkOut)
# Flux vidéo original
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

# Flux NN (Segmentation)
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

# Fonction utilitaire pour coloriser le masque
def decode_deeplabv3p(output_tensor):
    # Le modèle sort généralement une classe par pixel (0=fond, 1=personne)
    class_map = np.array(output_tensor).reshape(input_size[1], input_size[0]).astype(np.uint8)
    
    # Création d'un masque couleur : Fond noir, Personne en bleu/violet
    # On multiplie par 100 pour avoir une valeur de gris visible, puis on applique une colormap
    output_colors = cv2.applyColorMap(class_map * 100, cv2.COLORMAP_JET)
    
    # On rend le fond noir (là où la classe est 0)
    output_colors[class_map == 0] = [0, 0, 0]
    
    return output_colors

# 5. Boucle principale
with dai.Device(pipeline) as device:
    print("Caméra connectée. Démarrage...")
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

    frame = None
    
    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            # Récupération des données brutes du réseau
            layers = in_nn.getAllLayerNames()
            # Généralement la couche de sortie s'appelle 'Output' ou similaire, 
            # mais ici on prend la première couche disponible
            layer1 = in_nn.getLayerInt32(layers[0])
            
            # Décodage
            color_mask = decode_deeplabv3p(layer1)

            # Affichage
            if frame is not None:
                # Superposition (addWeighted)
                mixed = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
                cv2.imshow("Segmentation OAK-D Pro", mixed)
            else:
                cv2.imshow("Segmentation OAK-D Pro", color_mask)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()