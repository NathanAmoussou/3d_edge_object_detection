import os
import shutil
import random
from pathlib import Path

# Chemins actuels (basés sur votre 'tree')
base_dir = Path("fine_tunning/red_cube")
src_images = base_dir / "base_images"
src_labels = base_dir / "red_cube_yolo_classification/obj_train_data"

# Chemins cibles (format YOLOv11)
dest_dir = Path("datasets/red_cube")
splits = ['train', 'valid']
subdirs = ['images', 'labels']

# Création des dossiers cibles
for split in splits:
    for subdir in subdirs:
        (dest_dir / split / subdir).mkdir(parents=True, exist_ok=True)

# Récupération des paires images/labels
images = list(src_images.glob("*.[jJ][pP][gG]")) # jpg ou JPG
pairs = []

for img_path in images:
    # On cherche le label avec le même nom (stem)
    label_path = src_labels / f"{img_path.stem}.txt"
    if label_path.exists():
        pairs.append((img_path, label_path))
    else:
        print(f"Attention: Pas de label trouvé pour {img_path.name}")

# Mélange et séparation 80/20
random.shuffle(pairs)
split_idx = int(len(pairs) * 0.8)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

def copy_files(file_pairs, split_name):
    print(f"Copie de {len(file_pairs)} fichiers vers {split_name}...")
    for img, lbl in file_pairs:
        shutil.copy(img, dest_dir / split_name / "images" / img.name)
        shutil.copy(lbl, dest_dir / split_name / "labels" / lbl.name)

copy_files(train_pairs, "train")
copy_files(val_pairs, "valid")

print("\nRéorganisation terminée dans le dossier 'datasets/red_cube' !")