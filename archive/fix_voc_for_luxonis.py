from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def find_image_for_stem(dirpath: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = dirpath / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: chercher n'importe quelle extension
    matches = list(dirpath.glob(stem + ".*"))
    return matches[0] if matches else None

def ensure_child(parent, tag):
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    return el

def sanitize_one(xml_path: Path):
    stem = xml_path.stem
    img_path = find_image_for_stem(xml_path.parent, stem)
    if img_path is None:
        print(f"[SKIP] image introuvable pour {xml_path}")
        return 0

    # Lire dimensions réelles
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        depth = 3

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # filename cohérent
    fn = ensure_child(root, "filename")
    fn.text = img_path.name

    # size cohérent
    size = ensure_child(root, "size")
    ensure_child(size, "width").text = str(w)
    ensure_child(size, "height").text = str(h)
    ensure_child(size, "depth").text = str(depth)

    # corriger bndbox (int + clamp) et supprimer objets invalides
    removed = 0
    for obj in list(root.findall("object")):
        bnd = obj.find("bndbox")
        if bnd is None:
            root.remove(obj); removed += 1; continue

        def getf(tag):
            t = bnd.findtext(tag)
            return float(t) if t is not None and t.strip() != "" else None

        xmin = getf("xmin"); ymin = getf("ymin"); xmax = getf("xmax"); ymax = getf("ymax")
        if None in (xmin, ymin, xmax, ymax):
            root.remove(obj); removed += 1; continue

        # arrondir puis clamp
        xmin = int(round(xmin)); ymin = int(round(ymin)); xmax = int(round(xmax)); ymax = int(round(ymax))
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(0, min(xmax, w - 1))
        ymax = max(0, min(ymax, h - 1))

        if xmax <= xmin or ymax <= ymin:
            root.remove(obj); removed += 1; continue

        bnd.find("xmin").text = str(xmin)
        bnd.find("ymin").text = str(ymin)
        bnd.find("xmax").text = str(xmax)
        bnd.find("ymax").text = str(ymax)

    tree.write(xml_path, encoding="utf-8", xml_declaration=False)
    return 1

def main():
    root = Path("dataset_cube")
    total = 0
    for split in ["train", "valid", "test"]:
        d = root / split
        if not d.exists():
            continue
        for xml in d.glob("*.xml"):
            total += sanitize_one(xml)
    print(f"[DONE] XML traités: {total}")

if __name__ == "__main__":
    main()
