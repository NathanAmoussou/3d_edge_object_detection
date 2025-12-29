"""
Genere toutes les variantes ONNX de YOLO11 selon les bins definis.

Bins:
    - Modeles (elagage): yolo11n, yolo11s, yolo11m
    - Quantification: fp32, fp16
    - Resolution: 640, 512, 416, 320, 256

Total: 3 x 2 x 5 = 30 variantes ONNX

Note: INT8 n'est pas supporte pour l'export ONNX direct par Ultralytics.
      Il faudrait utiliser onnxruntime.quantization apres l'export (a faire separement).

Usage:
    python generate_variants.py
    python generate_variants.py --output ../models/variants/
    python generate_variants.py --dry-run
    python generate_variants.py --models n s --resolutions 640 320
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "variants"

# Bins selon AI_models.md
MODEL_SCALES = ["n", "s", "m"]
RESOLUTIONS = [640, 512, 416, 320, 256]
QUANTIZATIONS = ["fp32", "fp16"]
OPSET_VERSION = 13


def _check_cuda() -> bool:
    """Verifie si CUDA est disponible (import torch lazy)."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# Cache pour CUDA disponible
_cuda_available: bool | None = None


def cuda_available() -> bool:
    """Retourne True si CUDA est disponible (cache le resultat)."""
    global _cuda_available
    if _cuda_available is None:
        _cuda_available = _check_cuda()
    return _cuda_available


def get_variant_name(scale: str, imgsz: int, quant: str) -> str:
    """Genere le nom de fichier pour une variante."""
    return f"yolo11{scale}_{imgsz}_{quant}.onnx"


def download_base_model(scale: str):
    """
    Telecharge un modele YOLO11 de base depuis Ultralytics.

    Args:
        scale: Echelle du modele (n, s, m)

    Returns:
        Objet YOLO charge
    """
    from ultralytics import YOLO

    model_name = f"yolo11{scale}.pt"
    print(f"  Chargement de {model_name}...")

    # YOLO() telecharge automatiquement si le modele n'existe pas
    model = YOLO(f"yolo11{scale}.pt")

    return model


def export_variant(
    model,
    scale: str,
    imgsz: int,
    quant: str,
    output_dir: Path,
) -> str | None:
    """
    Exporte une variante ONNX.

    Args:
        model: Modele YOLO charge
        scale: Echelle du modele (n, s, m)
        imgsz: Resolution d'entree
        quant: Quantification (fp32, fp16)
        output_dir: Dossier de sortie

    Returns:
        Chemin vers le fichier ONNX genere, ou None si skip (ex: fp16 sans CUDA)
    """
    import ultralytics

    variant_name = get_variant_name(scale, imgsz, quant)
    target_path = output_dir / variant_name

    # Sanity check: imgsz doit etre multiple de 32
    if imgsz % 32 != 0:
        print(f"  [ERREUR] {variant_name} - imgsz={imgsz} n'est pas multiple de 32")
        return None

    # FP16 requiert CUDA pour l'export
    half = quant == "fp16"
    if half and not cuda_available():
        print(f"  [SKIP] {variant_name} - FP16 requiert CUDA")
        return None

    print(f"  Export: {variant_name}")

    # Export ONNX (GPU si disponible pour accelerer, CPU sinon)
    device = 0 if cuda_available() else "cpu"
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=False,  # Pas de fusion, sera fait au runtime par ORT
        opset=OPSET_VERSION,
        dynamic=False,
        device=device,
        batch=1,  # Explicite pour determinisme
        nms=False,  # NMS fait cote host, pas dans le modele
    )

    # Deplacer vers le dossier de sortie avec le bon nom (shutil.move pour cross-device)
    onnx_path = Path(onnx_path)
    if onnx_path != target_path:
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(onnx_path), str(target_path))

    size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"    -> {target_path.name} ({size_mb:.2f} MB)")

    # Sauvegarder metadata JSON
    import torch

    metadata = {
        "model": f"yolo11{scale}",
        "scale": scale,
        "imgsz": imgsz,
        "quantization": quant,
        "opset": OPSET_VERSION,
        "simplify": False,
        "dynamic": False,
        "batch": 1,
        "nms": False,
        "size_mb": round(size_mb, 2),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available(),
        "generated_at": datetime.now().isoformat(),
    }
    metadata_path = target_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(target_path)


def generate_all_variants(
    output_dir: Path,
    models: list[str],
    resolutions: list[int],
    quantizations: list[str],
    skip_existing: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """
    Genere toutes les variantes selon les bins specifies.

    Args:
        output_dir: Dossier de sortie
        models: Liste des echelles (n, s, m)
        resolutions: Liste des resolutions
        quantizations: Liste des quantifications (fp32, fp16)
        skip_existing: Ne pas regenerer les fichiers existants
        dry_run: Afficher sans generer

    Returns:
        Liste des chemins generes
    """
    total = len(models) * len(resolutions) * len(quantizations)
    print("=" * 60)
    print("GENERATION DES VARIANTES YOLO11")
    print("=" * 60)
    print(f"Modeles     : {', '.join([f'yolo11{s}' for s in models])}")
    print(f"Resolutions : {', '.join(map(str, resolutions))}")
    print(f"Quantif.    : {', '.join(quantizations)}")
    print(f"Total       : {total} variantes")
    print(f"Sortie      : {output_dir}")
    print(f"Skip exist. : {skip_existing}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY-RUN] Variantes a generer:")
        for scale in models:
            for imgsz in resolutions:
                for quant in quantizations:
                    name = get_variant_name(scale, imgsz, quant)
                    exists = (output_dir / name).exists()
                    status = " [EXISTE]" if exists else ""
                    print(f"  - {name}{status}")
        print("\n[DRY-RUN] Aucun fichier genere.")
        return []

    # Creer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    skipped = 0
    skipped_no_cuda = 0
    count = 0

    for scale in models:
        print(f"\n{'=' * 40}")
        print(f"Modele: yolo11{scale}")
        print("=" * 40)

        # Charger le modele une seule fois par echelle
        model = download_base_model(scale)

        for imgsz in resolutions:
            for quant in quantizations:
                count += 1
                variant_name = get_variant_name(scale, imgsz, quant)
                target_path = output_dir / variant_name

                # Skip si onnx ET json existent
                metadata_path = target_path.with_suffix(".json")
                if skip_existing and target_path.exists() and metadata_path.exists():
                    print(f"  [{count}/{total}] {variant_name} - SKIP (existe)")
                    skipped += 1
                    continue

                print(f"  [{count}/{total}] {variant_name}")

                path = export_variant(model, scale, imgsz, quant, output_dir)
                if path is None:
                    skipped_no_cuda += 1
                else:
                    generated.append(path)

    # Resume
    print("\n" + "=" * 60)
    print("GENERATION TERMINEE")
    print("=" * 60)
    print(f"Generes      : {len(generated)}")
    print(f"Skip (existe): {skipped}")
    print(f"Skip (CUDA)  : {skipped_no_cuda}")
    print(f"Dossier      : {output_dir}")

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Genere toutes les variantes ONNX de YOLO11"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Dossier de sortie (defaut: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODEL_SCALES,
        choices=MODEL_SCALES,
        help="Echelles a generer (defaut: n s m)",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=RESOLUTIONS,
        help=f"Resolutions a generer (defaut: {RESOLUTIONS})",
    )
    parser.add_argument(
        "--quantizations",
        type=str,
        nargs="+",
        default=QUANTIZATIONS,
        choices=QUANTIZATIONS,
        help="Quantifications a generer (defaut: fp32 fp16)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ne pas regenerer les fichiers existants",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les variantes sans les generer",
    )

    args = parser.parse_args()

    # Dossier de sortie
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    # Generer
    generate_all_variants(
        output_dir=output_dir,
        models=args.models,
        resolutions=args.resolutions,
        quantizations=args.quantizations,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    exit(main())
