import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)


def freeze_bn(module: nn.Module):
    # Mettre toutes les BatchNorm en mode eval (utilise running_mean/var, ne met pas à jour)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


# -----------------------------
# Dataset Pascal VOC (1 classe)
# -----------------------------


def _read_image_ids(imageset_txt: Path):
    with imageset_txt.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _ensure_train_val_splits(voc_root: Path, val_ratio=0.2, seed=42):
    """
    Créer ImageSets/Main/train.txt et val.txt si absents.
    Utilise tous les .xml présents dans Annotations/.
    """
    main_dir = voc_root / "ImageSets" / "Main"
    main_dir.mkdir(parents=True, exist_ok=True)

    train_txt = main_dir / "train.txt"
    val_txt = main_dir / "val.txt"

    if train_txt.exists() and val_txt.exists():
        return

    ann_dir = voc_root / "Annotations"
    xmls = sorted([p.stem for p in ann_dir.glob("*.xml")])
    if not xmls:
        raise RuntimeError(f"Aucun XML trouvé dans {ann_dir}")

    random.seed(seed)
    random.shuffle(xmls)
    n_val = max(1, int(len(xmls) * val_ratio))
    val_ids = xmls[:n_val]
    train_ids = xmls[n_val:]

    train_txt.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    print(f"[split] train={len(train_ids)} val={len(val_ids)} écrits dans {main_dir}")


class VOCCubeDataset(Dataset):
    """
    Lit un VOC-like :
      voc_root/
        Annotations/*.xml
        JPEGImages/*.jpg|png
        ImageSets/Main/train.txt (liste d'ids sans extension)
    """

    def __init__(self, voc_root: str, split: str, class_name: str = "cube_rouge"):
        self.voc_root = Path(voc_root)
        self.split = split
        self.class_name = class_name

        imageset = self.voc_root / "ImageSets" / "Main" / f"{split}.txt"
        if not imageset.exists():
            raise FileNotFoundError(f"Fichier split introuvable: {imageset}")

        self.ids = _read_image_ids(imageset)

        self.ann_dir = self.voc_root / "Annotations"
        self.img_dir = self.voc_root / "JPEGImages"

    def __len__(self):
        return len(self.ids)

    def _find_image_path(self, image_id: str) -> Path:
        # Essayer extensions courantes
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = self.img_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        # fallback: chercher par préfixe
        matches = list(self.img_dir.glob(f"{image_id}.*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Image introuvable pour id={image_id} dans {self.img_dir}"
        )

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        xml_path = self.ann_dir / f"{image_id}.xml"
        img_path = self._find_image_path(image_id)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        # Parser XML Pascal VOC
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            if name != self.class_name:
                continue

            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))

            # Clamp sécurité
            xmin = max(0.0, min(xmin, w - 1))
            ymin = max(0.0, min(ymin, h - 1))
            xmax = max(0.0, min(xmax, w - 1))
            ymax = max(0.0, min(ymax, h - 1))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # 1 = cube (0 = background)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.numel()
            else torch.zeros((0,), dtype=torch.float32),
        }

        # torchvision detection attend des tensors float [0..1]
        img_t = torchvision.transforms.functional.to_tensor(img)
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# Train loop (torchvision)
# -----------------------------


def train_one_epoch(model, optimizer, loader, device, epoch, print_every=50):
    model.train()
    model.apply(freeze_bn)  # <-- important
    total = 0.0

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item()
        if (step + 1) % print_every == 0:
            print(
                f"[epoch {epoch}] step {step + 1}/{len(loader)} loss={loss.item():.4f} "
                + " ".join([f"{k}={v.item():.3f}" for k, v in loss_dict.items()])
            )

    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.train()  # on garde train() pour que la loss soit calculée
    model.apply(freeze_bn)
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
    return total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--voc_root",
        required=True,
        help="Chemin vers le dossier VOC (Annotations/JPEGImages/ImageSets)",
    )
    ap.add_argument(
        "--class_name", default="cube_rouge", help="Nom exact de la classe dans les XML"
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out", default="cube_ssdlite.pth")
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    _ensure_train_val_splits(voc_root, val_ratio=0.2, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = VOCCubeDataset(str(voc_root), split="train", class_name=args.class_name)
    val_ds = VOCCubeDataset(str(voc_root), split="val", class_name=args.class_name)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 2 classes = background + cube
    num_classes = 2

    # Modèle SSDLite 320 (MobileNetV3 backbone)
    # On charge les poids COCO puis on remplace la tête pour num_classes=2
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    # Remplacer la tête de classification (la régression bbox reste OK)
    from functools import partial

    import torch.nn as nn
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.ssdlite import SSDLiteHead

    num_classes = 2  # background + cube

    # Taille d'entrée fixe du modèle ssdlite320
    size = (320, 320)

    # Récupérer les canaux de sortie réels du backbone (méthode officielle utilisée par torchvision)
    out_channels = det_utils.retrieve_out_channels(model.backbone, size)

    # Anchors par feature-map
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # Même norm_layer que celui utilisé dans la définition du modèle ssdlite torchvision
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # Remplacer la tête complète (évite les soucis de structures internes)
    model.head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.2
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_every=50
        )
        vl = evaluate_loss(model, val_loader, device)
        scheduler.step()

        print(f"[epoch {epoch}] train_loss={tr:.4f} val_loss={vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_name": args.class_name,
                    "num_classes": num_classes,
                },
                args.out,
            )
            print(
                f"[save] meilleur modèle sauvegardé: {args.out} (val_loss={best_val:.4f})"
            )


if __name__ == "__main__":
    main()


from functools import partial

import torch
import torch.nn as nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteHead

device = "cuda" if torch.cuda.is_available() else "cpu"

# Recréer le modèle EXACTEMENT comme à l'entraînement
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)

num_classes = 2
size = (320, 320)
out_channels = det_utils.retrieve_out_channels(model.backbone, size)
num_anchors = model.anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
model.head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)

ckpt = torch.load("cube_ssdlite.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

img = Image.open("finetuning_dataset/JPEGImages/20251209_190950.jpg").convert("RGB")
x = torchvision.transforms.functional.to_tensor(img).to(device)

with torch.no_grad():
    preds = model([x])[0]  # en eval(), torchvision renvoie des prédictions

# Filtrer
keep = preds["scores"] > 0.3
print("scores:", preds["scores"][keep].detach().cpu().numpy())
print("boxes:", preds["boxes"][keep].detach().cpu().numpy())
print("labels:", preds["labels"][keep].detach().cpu().numpy())
