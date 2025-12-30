"""
Compile un modele ONNX pour une cible hardware specifique.

Le workflow "fair" part d'un ONNX commun (genere par generate_variants.py)
et compile vers le format natif de chaque hardware.

Usage:
    python compile.py --target 4070 --model models/variants/yolo11n_640_fp16.onnx
    python compile.py --target oak --model models/variants/yolo11n_640_fp16.onnx
    python compile.py --target orin --model models/variants/yolo11n_640_fp16.onnx

Targets:
    - 4070/cpu: Retourne l'ONNX tel quel (inference via ONNX Runtime)
    - oak: Compile ONNX -> BLOB (Myriad X, FP16)
    - orin: Compile ONNX -> ENGINE (TensorRT, sur Jetson)

TODO (Phase 4): Metadata preservation
    - Copier/enrichir le .json du modele source vers l'artefact compile (.blob, .engine)
    - Ajouter les infos de compilation: target, flags (--fp16, --int8), timestamp
    - Preserver la lineage: parent_onnx, transform_type, experiment_id
"""

import argparse
import os
import shutil
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Domaines ONNX internes a ORT (non portables vers TRT/OpenVINO)
ORT_INTERNAL_DOMAINS = [
    "com.ms.internal",
    "com.microsoft",
    "ai.onnx.contrib",
]


def check_onnx_portability(onnx_path: str) -> tuple[bool, list[str]]:
    """
    Verifie si un ONNX contient des ops/domaines internes ORT.

    Les modeles generes par ORT optimized_model_filepath peuvent contenir
    des ops internes (com.ms.internal.nhwc, etc.) non portables.

    Returns:
        (is_portable, list of problematic domains found)
    """
    import onnx

    model = onnx.load(onnx_path)
    found_domains = set()

    # Verifier les opset imports
    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx"
        for internal in ORT_INTERNAL_DOMAINS:
            if domain.startswith(internal):
                found_domains.add(domain)

    # Verifier les ops dans le graphe
    for node in model.graph.node:
        domain = node.domain or "ai.onnx"
        for internal in ORT_INTERNAL_DOMAINS:
            if domain.startswith(internal):
                found_domains.add(domain)

    return len(found_domains) == 0, list(found_domains)


def is_ort_optimized_model(onnx_path: str) -> bool:
    """Detecte si le modele est un ONNX optimise par ORT (via nom de fichier)."""
    return "_ortopt_" in Path(onnx_path).stem


def has_qdq_nodes(onnx_path: str) -> bool:
    """
    Detecte si le modele ONNX contient des nodes QuantizeLinear/DequantizeLinear.

    Si present, TensorRT doit etre invoque avec --int8 pour explicit quantization.
    """
    import onnx

    model = onnx.load(onnx_path)
    qdq_ops = {"QuantizeLinear", "DequantizeLinear"}

    for node in model.graph.node:
        if node.op_type in qdq_ops:
            return True

    return False


def compile_for_gpu(onnx_path: str, output_dir: str) -> str:
    """
    "Compile" pour GPU/CPU = copie l'ONNX tel quel.
    L'inference se fait via ONNX Runtime (CUDA EP ou CPU EP).
    """
    print("=" * 60)
    print("COMPILATION GPU/CPU (ONNX Runtime)")
    print("=" * 60)

    onnx_path = Path(onnx_path)
    print(f"Modele source: {onnx_path}")

    # Verifier que c'est bien un ONNX
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"Attendu un fichier .onnx, recu: {onnx_path.suffix}")

    # Copier vers output_dir si different
    os.makedirs(output_dir, exist_ok=True)
    target_path = Path(output_dir) / onnx_path.name

    if onnx_path.resolve() != target_path.resolve():
        shutil.copy2(str(onnx_path), str(target_path))

    size_mb = os.path.getsize(target_path) / (1024 * 1024)

    print("\n" + "-" * 40)
    print("COMPILATION TERMINEE")
    print("-" * 40)
    print("Format : ONNX (pour ONNX Runtime)")
    print(f"Fichier: {target_path}")
    print(f"Taille : {size_mb:.2f} MB")
    print("\nNote: Aucune transformation appliquee.")
    print("      L'ONNX sera execute via ONNX Runtime CUDA/CPU EP.")

    return str(target_path)


def compile_for_oak(onnx_path: str, output_dir: str) -> str:
    """
    Compile ONNX -> BLOB pour OAK-D (Myriad X VPU).
    Utilise blobconverter avec les memes parametres que le preprocess OAK.

    IMPORTANT: Les modeles _ortopt_* ne sont PAS supportes car ils peuvent
    contenir des ops internes ORT non portables vers OpenVINO.
    """
    import blobconverter

    print("=" * 60)
    print("COMPILATION OAK-D (ONNX -> BLOB)")
    print("=" * 60)

    onnx_path = Path(onnx_path)
    print(f"Modele source: {onnx_path}")

    # Verifier que c'est bien un ONNX
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"Attendu un fichier .onnx, recu: {onnx_path.suffix}")

    # Guard: refuser les modeles ORT-optimized (non portables)
    if is_ort_optimized_model(str(onnx_path)):
        raise ValueError(
            "Les modeles _ortopt_* ne sont pas supportes pour OAK.\n"
            "  Raison: ORT peut inserer des ops internes (com.ms.internal.*)\n"
            "          non supportes par OpenVINO/blobconverter.\n"
            "  Solution: Utiliser le seed FP16 original."
        )

    # Guard: refuser les modeles INT8 QDQ (MyriadX = FP16 only)
    if "_int8_qdq" in onnx_path.stem:
        raise ValueError(
            "Les modeles _int8_qdq ne sont pas supportes pour OAK.\n"
            "  Raison: RVC2/MyriadX supporte uniquement FP16 (INT8 = RVC3+).\n"
            "  Solution: Utiliser le seed FP16 original."
        )

    # Verification supplementaire du contenu ONNX
    is_portable, bad_domains = check_onnx_portability(str(onnx_path))
    if not is_portable:
        raise ValueError(
            f"ONNX contient des domaines non portables: {bad_domains}\n"
            f"  Ce modele ne peut pas etre compile pour OAK/OpenVINO."
        )

    # Extraire imgsz du nom de fichier (ex: yolo11n_640_fp16.onnx -> 640)
    # Format attendu: yolo11{scale}_{imgsz}_{quant}.onnx
    parts = onnx_path.stem.split("_")
    if len(parts) >= 2:
        try:
            imgsz = int(parts[1])
        except ValueError:
            imgsz = 640
            print(f"  Warning: impossible d'extraire imgsz du nom, utilise {imgsz}")
    else:
        imgsz = 640
        print(f"  Warning: format de nom inattendu, utilise imgsz={imgsz}")

    print(f"Input size: {imgsz}x{imgsz}")

    # Conversion blob avec preprocessing integre
    # IMPORTANT: Ces parametres doivent matcher le benchmark OAK
    # - reverse_input_channels: BGR -> RGB (Ultralytics attend RGB)
    # - scale 255: normalisation 0-255 -> 0-1 (fait dans le blob)
    print("\nConversion blob (FP16, 6 shaves, preprocess integre)...")
    os.makedirs(output_dir, exist_ok=True)

    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        data_type="FP16",
        shaves=6,
        output_dir=output_dir,
        optimizer_params=[
            "--reverse_input_channels",  # BGR -> RGB
            "--scale",
            "255",  # Normalisation 0-255 -> 0-1
        ],
    )

    size_mb = os.path.getsize(blob_path) / (1024 * 1024)

    print("\n" + "-" * 40)
    print("COMPILATION TERMINEE")
    print("-" * 40)
    print("Format : OpenVINO blob (FP16)")
    print(f"Fichier: {blob_path}")
    print(f"Taille : {size_mb:.2f} MB")

    return blob_path


def compile_for_orin(
    onnx_path: str,
    output_dir: str,
    fp16: bool = True,
    workspace_mb: int = 4096,
) -> str:
    """
    Compile ONNX -> ENGINE pour Jetson Orin (TensorRT).

    IMPORTANT: Cette fonction doit etre executee SUR LE JETSON.
    TensorRT compile pour l'architecture GPU locale.

    IMPORTANT: Les modeles _ortopt_* ne sont PAS supportes car ils peuvent
    contenir des ops internes ORT non portables vers TensorRT.

    Args:
        onnx_path: Chemin vers le fichier ONNX
        output_dir: Dossier de sortie
        fp16: Utiliser FP16 (recommande pour Orin)
        workspace_mb: Taille du workspace TensorRT en MB (defaut: 4096)
                      Reduire si RAM limitee (ex: 2048 pour Orin Nano 4GB)
    """
    print("=" * 60)
    print("COMPILATION ORIN (ONNX -> TensorRT ENGINE)")
    print("=" * 60)

    onnx_path = Path(onnx_path)
    print(f"Modele source: {onnx_path}")

    # Verifier que c'est bien un ONNX
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"Attendu un fichier .onnx, recu: {onnx_path.suffix}")

    # Guard: refuser les modeles ORT-optimized (non portables)
    if is_ort_optimized_model(str(onnx_path)):
        raise ValueError(
            "Les modeles _ortopt_* ne sont pas supportes pour TensorRT.\n"
            "  Raison: ORT peut inserer des ops internes (com.ms.internal.*)\n"
            "          non supportes par TensorRT.\n"
            "  Solution: Utiliser le seed original ou le modele _int8_qdq."
        )

    # Verification supplementaire du contenu ONNX
    is_portable, bad_domains = check_onnx_portability(str(onnx_path))
    if not is_portable:
        raise ValueError(
            f"ONNX contient des domaines non portables: {bad_domains}\n"
            f"  Ce modele ne peut pas etre compile pour TensorRT."
        )

    # Extraire imgsz du nom de fichier
    parts = onnx_path.stem.split("_")
    if len(parts) >= 2:
        try:
            imgsz = int(parts[1])
        except ValueError:
            imgsz = 640
    else:
        imgsz = 640

    # Detecter si le modele contient des nodes QDQ (INT8 explicit quantization)
    is_qdq_model = has_qdq_nodes(str(onnx_path))

    print(f"Input size: {imgsz}x{imgsz}")
    if is_qdq_model:
        print("Precision: INT8 (QDQ detected)")
    else:
        print(f"Precision: {'FP16' if fp16 else 'FP32'}")
    print(f"Workspace: {workspace_mb} MB")

    # Nom du fichier engine et log
    engine_name = onnx_path.stem + ".engine"
    log_name = onnx_path.stem + ".trtexec.log"
    os.makedirs(output_dir, exist_ok=True)
    engine_path = Path(output_dir) / engine_name
    log_path = Path(output_dir) / log_name

    # Option 1: Utiliser trtexec (CLI TensorRT)
    # C'est la methode recommandee pour les Jetson
    print("\nCompilation TensorRT via trtexec...")
    print("  (Cela peut prendre plusieurs minutes)")
    print(f"  Log: {log_path}")

    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]

    # Si QDQ present, utiliser --int8 (explicit quantization)
    # Ajouter aussi --fp16 pour fallback sur les layers non-quantifies
    if is_qdq_model:
        cmd.append("--int8")
        cmd.append("--fp16")  # Fallback FP16 pour layers non-INT8
    elif fp16:
        cmd.append("--fp16")

    # Pas de NMS integre (on veut le meme postprocess que GPU/OAK)
    # Le modele ONNX a deja nms=False depuis generate_variants.py

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )

        # Sauvegarder stdout/stderr dans un fichier log
        with open(log_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"trtexec command: {' '.join(cmd)}\n")
            f.write("=" * 60 + "\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout or "(empty)\n")
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr or "(empty)\n")
            f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

        if result.returncode != 0:
            print("\n[ERREUR] trtexec a echoue:")
            print(result.stderr)
            print(f"  Voir le log complet: {log_path}")
            raise RuntimeError("Compilation TensorRT echouee")

    except FileNotFoundError:
        print("\n[ERREUR] trtexec non trouve!")
        print("Assurez-vous que TensorRT est installe et trtexec est dans le PATH.")
        print("\nAlternative: utiliser Ultralytics pour l'export:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('model.onnx')")
        print("  model.export(format='engine', half=True)")
        raise

    if not engine_path.exists():
        raise RuntimeError(f"Engine non genere: {engine_path}")

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)

    print("\n" + "-" * 40)
    print("COMPILATION TERMINEE")
    print("-" * 40)
    print(f"Format : TensorRT ENGINE ({'FP16' if fp16 else 'FP32'})")
    print(f"Fichier: {engine_path}")
    print(f"Taille : {size_mb:.2f} MB")

    return str(engine_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compile un modele ONNX pour GPU, OAK-D ou Jetson Orin"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["4070", "cpu", "oak", "orin"],
        help="Cible: '4070'/'cpu' (ONNX Runtime), 'oak' (blob), 'orin' (TensorRT)",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Chemin vers le modele ONNX"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Dossier de sortie (defaut: meme dossier que le modele)",
    )
    parser.add_argument(
        "--fp32", action="store_true", help="Forcer FP32 pour Orin (defaut: FP16)"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4096,
        help="Workspace TensorRT en MB pour Orin (defaut: 4096). Reduire si RAM limitee.",
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

    # Compiler selon la cible
    if args.target in ["4070", "cpu"]:
        output_path = compile_for_gpu(model_path, output_dir)
    elif args.target == "oak":
        output_path = compile_for_oak(model_path, output_dir)
    elif args.target == "orin":
        output_path = compile_for_orin(
            model_path, output_dir, fp16=not args.fp32, workspace_mb=args.workspace
        )

    return 0


if __name__ == "__main__":
    exit(main())
