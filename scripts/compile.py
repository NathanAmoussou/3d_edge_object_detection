"""
Compile un modele ONNX pour une cible hardware specifique.

Le workflow "fair" part d'un ONNX commun (genere par generate_variants.py)
et compile vers le format natif de chaque hardware.

Main usage:
    To generate all yolo11{m,s,n}_{640,512,416,320,256}_fp16_{4,5,6,7,8}shave.blob variants for OAK:
        python scripts/compile.py --target oak
    To generate all Orin TRT variants (m/s/n, 640-256, fp32/fp16, builder 3/4/5, sparsity on/off):
        python scripts/compile.py --target orin_trt

Options (debug only)::
    python scripts/compile.py --target 4070 --model models/variants/yolo11n_640_fp16.onnx
    python scripts/compile.py --target oak --model models/variants/yolo11n_640_fp16.onnx
    python scripts/compile.py --target oak --model models/variants/yolo11n_640_fp16.onnx --shaves 8
    python scripts/compile.py --target oak --model models/variants/yolo11n_640_fp16.onnx --shaves 4 5 6 7 8
    python scripts/compile.py --target oak
    python scripts/compile.py --target orin --model models/variants/yolo11n_640_fp16.onnx
    python scripts/compile.py --target orin_trt

Targets:
    - 4070/cpu: Retourne l'ONNX tel quel (inference via ONNX Runtime)
    - oak: Compile ONNX -> BLOB (Myriad X, FP16, shaves configurable)
    - orin: Compile ONNX -> ENGINE (TensorRT, sur Jetson)
    - orin_trt: Sweep TensorRT sur variants m/s/n, 640-256, fp32/fp16, builder 3/4/5, sparsity on/off

OAK Shaves:
    Le nombre de SHAVE cores est un parametre de compilation (pas modifiable a runtime).
    RVC2 a 16 SHAVEs. Luxonis recommande 8 shaves avec 2 NN threads.
    Generer plusieurs blobs pour benchmark: --shaves 4 5 6 7 8
    Sans --model (target oak), compile par defaut toutes les variantes fp16
    yolo11{m,s,n}_{640,512,416,320,256} avec shaves 4-8.

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
DEFAULT_VARIANTS_DIR = ROOT_DIR / "models" / "variants"
DEFAULT_OAK_OUTPUT_DIR = ROOT_DIR / "models" / "oak"
DEFAULT_ORIN_OUTPUT_DIR = ROOT_DIR / "models" / "orin"
DEFAULT_OAK_SCALES = ["m", "s", "n"]
DEFAULT_OAK_RESOLUTIONS = [640, 512, 416, 320, 256]
DEFAULT_OAK_QUANT = "fp16"
DEFAULT_OAK_SHAVES = [4, 5, 6, 7, 8]
DEFAULT_ORIN_TRT_TRANSFORMED_DIR = ROOT_DIR / "models" / "transformed"
DEFAULT_ORIN_TRT_SCALES = ["m", "s", "n"]
DEFAULT_ORIN_TRT_RESOLUTIONS = [640, 512, 416, 320, 256]
DEFAULT_ORIN_TRT_PRECS = ["fp32", "fp16"]
DEFAULT_ORIN_TRT_BUILDER_LEVELS = [3, 4, 5]
DEFAULT_ORIN_TRT_SPARSITY = [False, True]

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


def compile_for_oak(onnx_path: str, output_dir: str, shaves: int = 6) -> str:
    """
    Compile ONNX -> BLOB pour OAK-D (Myriad X VPU).
    Utilise blobconverter avec les memes parametres que le preprocess OAK.

    IMPORTANT: Les modeles _ortopt_* ne sont PAS supportes car ils peuvent
    contenir des ops internes ORT non portables vers OpenVINO.

    Args:
        shaves: Nombre de SHAVE cores a utiliser (4-8, defaut: 6).
                RVC2 a 16 SHAVEs, Luxonis recommande 8 shaves avec 2 NN threads.
                Le nombre de shaves est fixe a la compilation, pas modifiable a runtime.
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
    print(f"\nConversion blob (FP16, {shaves} shaves, preprocess integre)...")
    os.makedirs(output_dir, exist_ok=True)

    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        data_type="FP16",
        shaves=shaves,
        output_dir=output_dir,
        optimizer_params=[
            "--reverse_input_channels",  # BGR -> RGB
            "--scale",
            "255",  # Normalisation 0-255 -> 0-1
        ],
    )

    # Renommer le blob pour inclure le nombre de shaves
    # Format: yolo11n_640_fp16_6shave.blob
    blob_path = Path(blob_path)
    new_name = f"{onnx_path.stem}_{shaves}shave.blob"
    new_blob_path = blob_path.parent / new_name

    if blob_path != new_blob_path:
        blob_path.rename(new_blob_path)
        blob_path = new_blob_path

    size_mb = os.path.getsize(blob_path) / (1024 * 1024)

    print("\n" + "-" * 40)
    print("COMPILATION TERMINEE")
    print("-" * 40)
    print(f"Format : OpenVINO blob (FP16, {shaves} shaves)")
    print(f"Fichier: {blob_path}")
    print(f"Taille : {size_mb:.2f} MB")

    return str(blob_path)


def compile_for_oak_sweep(
    onnx_path: str,
    output_dir: str,
    shaves_list: list[int],
    skip_existing: bool = False,
) -> list[str]:
    """
    Compile plusieurs blobs OAK pour une liste de shaves.
    """
    generated = []
    onnx_path_p = Path(onnx_path)

    for s in shaves_list:
        expected = Path(output_dir) / f"{onnx_path_p.stem}_{s}shave.blob"
        if skip_existing and expected.exists():
            print(f"[SKIP] {expected.name} (existe deja)")
            generated.append(str(expected))
            continue

        out = compile_for_oak(onnx_path, output_dir, shaves=s)
        generated.append(out)

    print("\n" + "-" * 40)
    print("OAK SWEEP TERMINE")
    print("-" * 40)
    print(f"Shaves : {shaves_list}")
    print(f"Sortie : {output_dir}")
    print(f"Blobs  : {len(generated)}")
    return generated


def find_int8_variant(
    scale: str, imgsz: int, transformed_dir: Path
) -> Path | None:
    pattern = f"yolo11{scale}_{imgsz}_fp32_int8_qdq_*.onnx"
    matches = sorted(transformed_dir.glob(pattern))
    if not matches:
        return None
    for match in matches:
        if "s8s8" in match.name:
            return match
    return matches[0]


def compile_orin_trt_sweep(
    output_dir: str,
    skip_existing: bool = False,
) -> list[str]:
    variants_dir = DEFAULT_VARIANTS_DIR
    generated = []
    missing = 0

    print("=" * 60)
    print("COMPILATION ORIN TRT (SWEEP PAR DEFAUT)")
    print("=" * 60)
    print(f"Variants   : {variants_dir}")
    print(f"Scales     : {DEFAULT_ORIN_TRT_SCALES}")
    print(f"Resolutions: {DEFAULT_ORIN_TRT_RESOLUTIONS}")
    print(f"Precisions : {DEFAULT_ORIN_TRT_PRECS}")
    print(f"Builder    : {DEFAULT_ORIN_TRT_BUILDER_LEVELS}")
    print(f"Sparsity   : {DEFAULT_ORIN_TRT_SPARSITY}")
    print(f"Output     : {output_dir}")

    for scale in DEFAULT_ORIN_TRT_SCALES:
        for imgsz in DEFAULT_ORIN_TRT_RESOLUTIONS:
            for prec in DEFAULT_ORIN_TRT_PRECS:
                if prec == "int8":
                    model_path = find_int8_variant(scale, imgsz, transformed_dir)
                else:
                    name = f"yolo11{scale}_{imgsz}_{prec}.onnx"
                    model_path = variants_dir / name

                if not model_path or not model_path.exists():
                    missing += 1
                    if prec == "int8":
                        print(
                            f"[SKIP] int8 introuvable: yolo11{scale}_{imgsz}_fp32_int8_qdq_*.onnx"
                        )
                    else:
                        print(f"[SKIP] {model_path} (introuvable)")
                    continue

                for level in DEFAULT_ORIN_TRT_BUILDER_LEVELS:
                    for sparsity in DEFAULT_ORIN_TRT_SPARSITY:
                        suffix = f"_trt_b{level}_sp{int(sparsity)}"
                        expected = (
                            Path(output_dir) / f"{model_path.stem}{suffix}.engine"
                        )
                        if skip_existing and expected.exists():
                            print(f"[SKIP] {expected.name} (existe deja)")
                            generated.append(str(expected))
                            continue

                        out = compile_for_orin(
                            str(model_path),
                            output_dir,
                            fp16=prec != "fp32",
                            builder_opt_level=level,
                            sparsity=sparsity,
                            engine_suffix=suffix,
                        )
                        generated.append(out)

    print("\n" + "=" * 60)
    print("SWEEP ORIN TRT TERMINE")
    print("=" * 60)
    print(f"Engines generes: {len(generated)}")
    if missing:
        print(f"Manquants     : {missing}")

    return generated


def get_default_oak_variants(variants_dir: Path) -> list[Path]:
    paths = []
    for scale in DEFAULT_OAK_SCALES:
        for imgsz in DEFAULT_OAK_RESOLUTIONS:
            name = f"yolo11{scale}_{imgsz}_{DEFAULT_OAK_QUANT}.onnx"
            paths.append(variants_dir / name)
    return paths


def compile_oak_default_sweep(
    output_dir: str,
    shaves_list: list[int],
    skip_existing: bool = False,
) -> list[str]:
    variants_dir = DEFAULT_VARIANTS_DIR
    if not variants_dir.exists():
        print(f"Erreur: Dossier variantes introuvable: {variants_dir}")
        return []

    generated = []
    missing = 0
    print("=" * 60)
    print("COMPILATION OAK-D (SWEEP PAR DEFAUT)")
    print("=" * 60)
    print(f"Variantes : {variants_dir}")
    print(f"Shaves    : {shaves_list}")
    print(f"Sortie    : {output_dir}")

    for model_path in get_default_oak_variants(variants_dir):
        if not model_path.exists():
            print(f"[SKIP] {model_path} (introuvable)")
            missing += 1
            continue
        generated.extend(
            compile_for_oak_sweep(
                str(model_path),
                output_dir,
                shaves_list=shaves_list,
                skip_existing=skip_existing,
            )
        )

    print("\n" + "=" * 60)
    print("SWEEP PAR DEFAUT TERMINE")
    print("=" * 60)
    print(f"Blobs generes : {len(generated)}")
    if missing:
        print(f"Manquants     : {missing}")

    return generated


def compile_for_orin(
    onnx_path: str,
    output_dir: str,
    fp16: bool = True,
    workspace_mb: int = 4096,
    builder_opt_level: int | None = None,
    sparsity: bool = False,
    engine_suffix: str | None = None,
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
        builder_opt_level: Niveau d'optimisation builder TensorRT (0-5)
        sparsity: Activer la sparsity TensorRT si disponible
        engine_suffix: Suffixe a ajouter au nom de l'engine/log
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
    suffix = engine_suffix or ""
    engine_name = onnx_path.stem + suffix + ".engine"
    log_name = onnx_path.stem + suffix + ".trtexec.log"
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

    if builder_opt_level is not None:
        cmd.append(f"--builderOptimizationLevel={builder_opt_level}")

    if sparsity:
        cmd.append("--sparsity=enable")

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
        choices=["4070", "cpu", "oak", "orin", "orin_trt"],
        help=(
            "Cible: '4070'/'cpu' (ONNX Runtime), 'oak' (blob), "
            "'orin' (TensorRT), 'orin_trt' (sweep TensorRT)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Chemin vers le modele ONNX (requis pour 4070/cpu/orin, "
            "optionnel pour oak/orin_trt)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Dossier de sortie (defaut: models/oak pour oak, models/orin pour "
            "orin/orin_trt, sinon dossier du modele)"
        ),
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
    parser.add_argument(
        "--shaves",
        type=int,
        nargs="+",
        default=None,
        choices=range(1, 17),
        metavar="[1-16]",
        help=(
            "SHAVEs OAK. Accepte une liste: --shaves 4 5 6 7 8 "
            "(defaut: 6, ou 4-8 si --target oak sans --model)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ne pas regenerer si l'artefact de sortie existe deja.",
    )

    args = parser.parse_args()

    if args.target not in ["oak", "orin_trt"] and not args.model:
        parser.error("--model est requis pour cette cible.")

    shaves_list = args.shaves
    if shaves_list is None:
        if args.target == "oak" and not args.model:
            shaves_list = DEFAULT_OAK_SHAVES
        else:
            shaves_list = [6]

    if args.target == "oak" and not args.model:
        output_dir = (
            str(Path(args.output).resolve())
            if args.output
            else str(DEFAULT_OAK_OUTPUT_DIR)
        )
        generated = compile_oak_default_sweep(
            output_dir=output_dir,
            shaves_list=shaves_list,
            skip_existing=args.skip_existing,
        )
        return 0 if generated else 1

    if args.target == "orin_trt":
        if args.model:
            print("[Warning] --model ignore en mode orin_trt (sweep complet).")
        output_dir = (
            str(Path(args.output).resolve())
            if args.output
            else str(DEFAULT_ORIN_OUTPUT_DIR)
        )
        generated = compile_orin_trt_sweep(
            output_dir=output_dir,
            skip_existing=args.skip_existing,
        )
        return 0 if generated else 1

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
        if args.target == "oak":
            output_dir = str(DEFAULT_OAK_OUTPUT_DIR)
        elif args.target == "orin":
            output_dir = str(DEFAULT_ORIN_OUTPUT_DIR)
        else:
            output_dir = str(Path(model_path).parent)
    else:
        output_dir = str(Path(output_dir).resolve())

    # Compiler selon la cible
    if args.target in ["4070", "cpu"]:
        output_path = compile_for_gpu(model_path, output_dir)
    elif args.target == "oak":
        if len(shaves_list) == 1:
            output_path = compile_for_oak(
                model_path, output_dir, shaves=shaves_list[0]
            )
        else:
            compile_for_oak_sweep(
                model_path,
                output_dir,
                shaves_list=shaves_list,
                skip_existing=args.skip_existing,
            )
            output_path = str(
                Path(output_dir)
                / f"{Path(model_path).stem}_{shaves_list[-1]}shave.blob"
            )
    elif args.target == "orin":
        output_path = compile_for_orin(
            model_path, output_dir, fp16=not args.fp32, workspace_mb=args.workspace
        )

    return 0


if __name__ == "__main__":
    exit(main())
