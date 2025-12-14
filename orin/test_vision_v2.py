import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Vérifier le GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-" * 30)
print(f"👀 Périphérique : {device}")
if device.type == 'cpu':
    print("⚠️ ATTENTION : Toujours sur CPU. Vérifiez l'installation de NumPy.")
else:
    print("✅ GPU détecté ! C'est parti.")
print("-" * 30)

# 2. Charger le modèle
print("📥 Chargement du cerveau (ResNet50)...")
# On utilise la nouvelle syntaxe pour éviter les warnings
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = model.to(device)
model.eval()

# 3. Ouvrir l'image locale 'panda.jpg'
print("🖼️ Lecture de l'image 'panda.jpg'...")
try:
    img = Image.open("IMG_0689.jpg").convert('RGB')
except Exception as e:
    print(f"❌ Erreur impossible de lire l'image : {e}")
    exit()

# 4. Préparer l'image
preprocess = weights.transforms()
input_batch = preprocess(img).unsqueeze(0).to(device)

# 5. Prédiction
print("🧠 Analyse en cours...")
with torch.no_grad():
    output = model(input_batch)

# 6. Résultat
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print("-" * 30)
top3_prob, top3_catid = torch.topk(probabilities, 3)
for i in range(top3_prob.size(0)):
    category_name = weights.meta["categories"][top3_catid[i]]
    print(f"🏆 C'est un : {category_name} ({top3_prob[i].item()*100:.2f}%)")
print("-" * 30)
