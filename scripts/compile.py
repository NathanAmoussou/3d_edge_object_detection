"""
Compile un modele YOLO (.pt) pour une cible hardware specifique.

Usage:
    python compile.py --target 4070 --model ../models/base/yolo11n.pt --output ../models/base/
    python compile.py --target oak --model ../models/base/yolo11n.pt --output ../models/base/
"""

import argparse
import os
from pathlib import Path


# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
IMGSZ = 640


def compile_for_gpu(model_path: str, output_dir: str) -> str:
    """
    Compile le modele pour GPU (NVIDIA).
    Export en ONNX sans optimisations agressives.
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("COMPILATION GPU (ONNX)")
    print("=" * 60)

    model = YOLO(model_path)
    model_name = Path(model_path).stem

    print(f"Modele source: {model_path}")
    print(f"Classes: {len(model.names)}")
    print(f"Input size: {IMGSZ}x{IMGSZ}")

    # Export ONNX
    print("\nExport ONNX (simplify=False, dynamic=False)...")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=13,
        simplify=False,  # Pas de fusion de couches
        dynamic=False,
    )

    # Deplacer vers output_dir si different
    onnx_path = Path(onnx_path)
    target_path = Path(output_dir) / onnx_path.name

    if onnx_path != target_path:
        os.makedirs(output_dir, exist_ok=True)
        onnx_path.rename(target_path)
        onnx_path = target_path

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    print("\n" + "-" * 40)
    print("COMPILATION TERMINEE")
    print("-" * 40)
    print(f"Format : ONNX")
    print(f"Fichier: {onnx_path}")
    print(f"Taille : {size_mb:.2f} MB")

    return str(onnx_path)


def compile_for_oak(model_path: str, output_dir: str) -> str:
    """
    Compile le modele pour OAK-D (Myriad X VPU).
    Export en ONNX puis conversion en blob (FP16).
    """
    import blobconverter
    from ultralytics import YOLO

    print("=" * 60)
    print("COMPILATION OAK-D (BLOB)")
    print("=" * 60)

    model = YOLO(model_path)
    model_name = Path(model_path).stem

    print(f"Modele source: {model_path}")
    print(f"Classes: {len(model.names)}")
    print(f"Input size: {IMGSZ}x{IMGSZ}")

    # Export ONNX (etape intermediaire)
    print("\n1. Export ONNX (simplify=False, dynamic=False)...")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=13,
        simplify=False,
        dynamic=False,
    )

    # Conversion blob avec preprocessing integre
    print("\n2. Conversion blob (FP16, 6 shaves, preprocess integre)...")
    os.makedirs(output_dir, exist_ok=True)

    blob_path = blobconverter.from_onnx(
        model=onnx_path,
        data_type="FP16",
        shaves=6,
        output_dir=output_dir,
        optimizer_params=[
            "--reverse_input_channels",  # BGR -> RGB
            "--scale", "255",            # Normalisation 0-255 -> 0-1
        ],
    )

    size_mb = os.path.getsize(blob_path) / (1024 * 1024)

    print("\n" + "-" * 40)
    print("COMPILATION TERMINEE")
    print("-" * 40)
    print(f"Format : OpenVINO blob (FP16)")
    print(f"Fichier: {blob_path}")
    print(f"Taille : {size_mb:.2f} MB")
    print(f"ONNX intermediaire: {onnx_path}")

    return blob_path


def main():
    parser = argparse.ArgumentParser(
        description="Compile un modele YOLO (.pt) pour GPU ou OAK-D"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["4070", "oak"],
        help="Cible hardware: '4070' pour GPU (ONNX), 'oak' pour OAK-D (blob)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le modele .pt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Dossier de sortie (defaut: meme dossier que le modele)"
    )

    args = parser.parse_args()

    # Resoudre le chemin du modele
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = ROOT_DIR / args.model

    if not model_path.exists():
        print(f"Erreur: Modele introuvable: {args.model}")
        return 1

    model_path = str(model_path.resolve())

    # Dossier de sortie
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(model_path).parent)
    else:
        output_dir = str(Path(output_dir).resolve())

    # Compiler
    if args.target == "4070":
        output_path = compile_for_gpu(model_path, output_dir)
    elif args.target == "oak":
        output_path = compile_for_oak(model_path, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
