import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# 1. Verification du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-" * 30)
print(f"Device utilise : {device}")

if device.type == 'cpu':
    print("ATTENTION : Toujours sur CPU. Verifiez l'installation de NumPy.")
else:
    print("Succes : GPU detecte !")
print("-" * 30)

# 2. Chargement du modele
print("Chargement du modele (ResNet50)...")
# On utilise la nouvelle syntaxe
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = model.to(device)
model.eval()

# 3. Choix de l'image
print("-" * 30)
if len(sys.argv) > 1:
    # L'utilisateur a donne le nom dans la commande
    image_path = sys.argv[1]
else:
    # Sinon on demande
    print("Quelle image voulez-vous analyser ?")
    # Liste des fichiers pour aider
    files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
    print(f"Fichiers disponibles : {files}")
    image_path = input("Entrez le nom du fichier : ")

# Verification si le fichier existe
if not os.path.exists(image_path):
    print(f"ERREUR : Le fichier '{image_path}' est introuvable !")
    sys.exit()

# 4. Lecture de l'image
print(f"Lecture de l'image '{image_path}'...")
try:
    img = Image.open(image_path).convert('RGB')
except Exception as e:
    print(f"Erreur de lecture : {e}")
    sys.exit()

# 5. Preparation
preprocess = weights.transforms()
input_batch = preprocess(img).unsqueeze(0).to(device)

# 6. Analyse
print("Analyse en cours...")
with torch.no_grad():
    output = model(input_batch)

# 7. Resultats
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print("-" * 30)
top3_prob, top3_catid = torch.topk(probabilities, 3)
for i in range(top3_prob.size(0)):
    category_name = weights.meta["categories"][top3_catid[i]]
    print(f"Resultat : {category_name} ({top3_prob[i].item()*100:.2f}%)")
print("-" * 30)
