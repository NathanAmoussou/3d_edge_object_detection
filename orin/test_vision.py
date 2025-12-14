import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# 1. Vérifier le GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"👀 Utilisation du périphérique : {device}")

# 2. Charger un modèle pré-entraîné (ResNet50)
# C'est un cerveau qui a déjà vu des millions d'images
print("📥 Chargement du modèle (ça peut prendre un moment la 1ère fois)...")
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# 3. Télécharger une image d'internet (Un Panda)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG/800px-Giant_Panda_in_Beijing_Zoo_1.JPG"
print("🌐 Téléchargement de l'image...")
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# 4. Préparer l'image pour l'IA (Redimensionner, Normaliser)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0).to(device) # Créer un lot de 1 image

# 5. L'IA regarde l'image
print("🧠 Analyse en cours...")
with torch.no_grad():
    output = model(input_batch)

# 6. Lire le résultat (Télécharger les noms des classes)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.split('\n')

# 7. Afficher le Top 3 des prédictions
print("-" * 30)
top3_prob, top3_catid = torch.topk(probabilities, 3)
for i in range(top3_prob.size(0)):
    print(f"🏆 Prédiction #{i+1} : {labels[top3_catid[i]]} ({top3_prob[i].item()*100:.2f}%)")
print("-" * 30)
