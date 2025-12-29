"""
Transform ONNX models with runtime optimizations.

Transformations supportees:
    A) ORT Fusion Levels: DISABLE, BASIC, EXTENDED, ALL
    B) INT8 Quantization via QDQ (Post-Training Quantization)

Ce script produit des ONNX derives qui peuvent ensuite etre compiles
par compile.py pour des cibles hardware specifiques.

Usage:
    # Fusion ORT
    python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion all
    python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion basic --runtime-only

    # INT8 QDQ (depuis FP32 recommande)
    python transform.py --model models/variants/yolo11n_640_fp32.onnx --int8
    python transform.py --model models/variants/yolo11n_640_fp32.onnx --int8 --calib-size 200

    # Combine (fusion puis INT8)
    python transform.py --model models/variants/yolo11n_640_fp32.onnx --fusion basic --int8

    # Batch via pattern
    python transform.py --pattern "yolo11n_*_fp16.onnx" --fusion all

    # Dry-run
    python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion all --dry-run
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_INPUT_DIR = ROOT_DIR / "models" / "variants"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "transformed"

# ORT Graph Optimization Levels
ORT_FUSION_LEVELS = ["disable", "basic", "extended", "all"]

# INT8 Quantization Config
DEFAULT_CALIB_DATASET = "coco128"
DEFAULT_CALIB_SIZE = 100


# =============================================================================
# UTILITAIRES
# =============================================================================


def parse_imgsz_from_filename(filepath: str) -> int:
    """
    Parse imgsz depuis le nom du fichier.
    Format attendu: yolo11{scale}_{imgsz}_{quant}.onnx
    """
    path = Path(filepath)
    stem = path.stem
    parts = stem.split("_")

    if len(parts) >= 2:
        try:
            imgsz = int(parts[1])
            if imgsz % 32 == 0 and 128 <= imgsz <= 1280:
                return imgsz
        except ValueError:
            pass

    # Fallback: lire le .json metadata
    json_path = path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                metadata = json.load(f)
                if "imgsz" in metadata:
                    return int(metadata["imgsz"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    print(f"  [Warning] Impossible de parser imgsz depuis {path.name}, utilise 640")
    return 640


def get_transform_suffix(fusion_level: str | None, is_int8: bool) -> str:
    """
    Genere le suffixe de transformation pour le nom de fichier.

    Examples:
        fusion_level="all", is_int8=False -> "_ortopt_all"
        fusion_level=None, is_int8=True   -> "_int8_qdq"
        fusion_level="basic", is_int8=True -> "_ortopt_basic_int8_qdq"
    """
    parts = []
    if fusion_level and fusion_level != "disable":
        parts.append(f"ortopt_{fusion_level}")
    if is_int8:
        parts.append("int8_qdq")
    return "_" + "_".join(parts) if parts else ""


def get_transformed_name(
    source_onnx: str,
    fusion_level: str | None = None,
    is_int8: bool = False,
) -> str:
    """
    Genere le nom du fichier de sortie.

    Example:
        source: yolo11n_640_fp16.onnx
        fusion: "all", int8: False
        output: yolo11n_640_fp16_ortopt_all.onnx
    """
    stem = Path(source_onnx).stem
    suffix = get_transform_suffix(fusion_level, is_int8)
    return f"{stem}{suffix}.onnx"


def generate_experiment_id(
    source_onnx: str,
    fusion_level: str | None,
    is_int8: bool,
    calib_dataset: str | None = None,
    calib_size: int | None = None,
    calib_images: list[str] | None = None,
) -> str:
    """
    Genere un ID d'experience deterministe pour cache/reproductibilite.

    Inclut le hash des images de calibration si disponible.
    """
    components = [
        Path(source_onnx).stem,
        fusion_level or "none",
        "int8" if is_int8 else "fp",
        calib_dataset or "none",
        str(calib_size) if calib_size else "0",
    ]

    # Ajouter hash des images de calibration pour reproductibilite
    if calib_images:
        images_str = "|".join(sorted(calib_images))
        images_hash = hashlib.sha256(images_str.encode()).hexdigest()[:8]
        components.append(images_hash)

    content = "|".join(components)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_source_metadata(onnx_path: str) -> dict:
    """Charge le metadata JSON du modele source."""
    json_path = Path(onnx_path).with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def save_transform_metadata(
    output_path: str,
    source_onnx: str,
    source_metadata: dict,
    fusion_level: str | None = None,
    is_int8: bool = False,
    calib_dataset: str | None = None,
    calib_size: int | None = None,
    calib_images: list[str] | None = None,
) -> None:
    """
    Sauvegarde le metadata JSON etendu pour le modele transforme.
    """
    # Partir du metadata parent
    metadata = source_metadata.copy()

    # Ajouter les champs de transformation
    metadata["parent_onnx"] = Path(source_onnx).name

    # Determiner le type de transformation
    if fusion_level and is_int8:
        transform_type = "ort_fusion+int8_qdq"
    elif fusion_level:
        transform_type = "ort_fusion"
    elif is_int8:
        transform_type = "int8_qdq"
    else:
        transform_type = "none"

    metadata["transform_type"] = transform_type
    metadata["fusion_level"] = fusion_level
    metadata["int8_calibration"] = is_int8
    metadata["calib_dataset"] = calib_dataset if is_int8 else None
    metadata["calib_size"] = calib_size if is_int8 else None

    # Hash des images de calibration pour reproductibilite
    if is_int8 and calib_images:
        images_str = "|".join(sorted(calib_images))
        metadata["calib_images_hash"] = hashlib.sha256(images_str.encode()).hexdigest()[
            :16
        ]
        metadata["calib_images_list"] = sorted(calib_images)
    else:
        metadata["calib_images_hash"] = None
        metadata["calib_images_list"] = None

    # Experiment ID
    metadata["experiment_id"] = generate_experiment_id(
        source_onnx, fusion_level, is_int8, calib_dataset, calib_size, calib_images
    )

    metadata["transformed_at"] = datetime.now().isoformat()
    metadata["size_mb"] = round(os.path.getsize(output_path) / (1024 * 1024), 2)

    # Sauvegarder JSON
    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    Metadata: {json_path.name}")


# =============================================================================
# ORT FUSION
# =============================================================================


def apply_ort_fusion(
    onnx_path: str,
    output_dir: str,
    fusion_level: str,
    runtime_only: bool = False,
) -> str | None:
    """
    Applique ORT graph optimization et serialise le resultat.

    Args:
        onnx_path: Chemin vers le modele ONNX source
        output_dir: Dossier de sortie
        fusion_level: Un de "disable", "basic", "extended", "all"
        runtime_only: Si True, ne pas serialiser (juste valider)

    Returns:
        Chemin vers l'ONNX optimise, ou None si runtime_only=True

    Notes:
        - ORT fusion inclut: Conv+BN fusion, Gelu fusion, Attention fusion, etc.
        - "extended" et "all" peuvent produire des optimisations hardware-specific
        - Pour comparaison fair, utiliser "basic" qui est le plus portable
    """
    import onnxruntime as ort

    print(f"  Applying ORT fusion level: {fusion_level}")

    # Mapping string -> enum
    level_map = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    opt_level = level_map[fusion_level]

    # Configurer les options de session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = opt_level

    if runtime_only:
        # Juste charger le modele pour verifier
        _ = ort.InferenceSession(onnx_path, sess_options)
        print(f"    [runtime-only] Modele charge avec optimisation {fusion_level}")
        return None

    # Serialiser le modele optimise
    output_name = get_transformed_name(onnx_path, fusion_level=fusion_level)
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / output_name

    sess_options.optimized_model_filepath = str(output_path)

    # Creer la session declenche l'optimisation et la serialisation
    try:
        _ = ort.InferenceSession(onnx_path, sess_options)
    except Exception as e:
        print(f"    [ERREUR] Echec optimisation ORT: {e}")
        raise

    if not output_path.exists():
        raise RuntimeError(f"Modele optimise non cree: {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    -> {output_path.name} ({size_mb:.2f} MB)")

    return str(output_path)


# =============================================================================
# INT8 QDQ QUANTIZATION
# =============================================================================


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Redimensionne l'image en gardant le ratio (ajoute du padding)."""
    from ultralytics.data.augment import LetterBox

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    transform = LetterBox(new_shape, auto=False, stride=32, center=True)
    return transform(image=img)


def get_calibration_images(
    dataset_name: str,
    num_samples: int,
) -> list[str]:
    """
    Recupere la liste des images de calibration.

    Returns:
        Liste des chemins d'images (tries pour reproductibilite)
    """
    from ultralytics.data.utils import check_det_dataset

    if dataset_name == "coco":
        data = check_det_dataset("coco.yaml")
        images_dir = Path(data["path"]) / "images" / "val2017"
    else:  # coco128
        data = check_det_dataset("coco128.yaml")
        images_dir = Path(data["path"]) / "images" / "train2017"

    image_files = sorted(images_dir.glob("*.jpg"))[:num_samples]
    return [str(p) for p in image_files]


def apply_int8_quantization(
    onnx_path: str,
    output_dir: str,
    calib_dataset: str = "coco128",
    calib_size: int = 100,
    per_channel: bool = True,
) -> tuple[str, list[str]]:
    """
    Applique INT8 Post-Training Quantization via format QDQ.

    Args:
        onnx_path: Chemin vers le modele ONNX source (FP32 recommande)
        output_dir: Dossier de sortie
        calib_dataset: Dataset pour calibration ("coco128" ou "coco")
        calib_size: Nombre d'images de calibration
        per_channel: Utiliser quantification per-channel (plus precis)

    Returns:
        Tuple (chemin vers ONNX quantifie, liste des images de calibration)

    Notes:
        - Format QDQ est le plus portable pour INT8
        - TensorRT peut parser les modeles QDQ directement
        - ORT peut executer les modeles QDQ sur CPU/CUDA
    """
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    print("  Applying INT8 QDQ quantization")
    print(f"    Calibration: {calib_dataset} ({calib_size} images)")

    # Parser imgsz depuis le nom de fichier
    imgsz = parse_imgsz_from_filename(onnx_path)

    # Recuperer les images de calibration
    calib_images = get_calibration_images(calib_dataset, calib_size)
    print(f"    Images trouvees: {len(calib_images)}")

    # Obtenir le nom de l'input depuis le modele
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name

    # Creer le CalibrationDataReader
    class COCOCalibrationReader(CalibrationDataReader):
        def __init__(self, image_files: list[str], imgsz: int, input_name: str):
            self.image_files = image_files
            self.imgsz = imgsz
            self.input_name = input_name
            self.index = 0

        def get_next(self):
            if self.index >= len(self.image_files):
                return None

            img_path = self.image_files[self.index]
            self.index += 1

            # Preprocess: load, letterbox, normalize
            img = cv2.imread(img_path)
            if img is None:
                # Skip bad images
                return self.get_next()

            img_lb = letterbox(img, (self.imgsz, self.imgsz))
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_chw, axis=0)

            return {self.input_name: img_batch}

        def rewind(self):
            self.index = 0

    calib_reader = COCOCalibrationReader(calib_images, imgsz, input_name)

    # Chemin de sortie
    output_name = get_transformed_name(onnx_path, is_int8=True)
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / output_name

    # Quantifier
    print("    Quantification en cours...")
    try:
        quantize_static(
            model_input=onnx_path,
            model_output=str(output_path),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,  # Activations en uint8 (standard)
            calibrate_method=CalibrationMethod.MinMax,
        )
    except Exception as e:
        print(f"    [ERREUR] Echec quantification: {e}")
        raise

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    -> {output_path.name} ({size_mb:.2f} MB)")

    # Retourner la liste des images pour le metadata
    return str(output_path), [Path(p).name for p in calib_images]


# =============================================================================
# TRANSFORMATION COMBINEE
# =============================================================================


def transform_model(
    onnx_path: str,
    output_dir: str,
    fusion_level: str | None = None,
    int8: bool = False,
    calib_dataset: str = "coco128",
    calib_size: int = 100,
    runtime_only: bool = False,
    dry_run: bool = False,
) -> str | None:
    """
    Applique les transformations demandees sur un modele.

    Args:
        onnx_path: Chemin vers le modele ONNX source
        output_dir: Dossier de sortie
        fusion_level: Niveau de fusion ORT (None = pas de fusion)
        int8: Appliquer quantification INT8 QDQ
        calib_dataset: Dataset pour calibration INT8
        calib_size: Nombre d'images de calibration
        runtime_only: Ne pas serialiser (juste valider)
        dry_run: Afficher sans executer

    Returns:
        Chemin vers le modele transforme, ou None
    """
    source_path = Path(onnx_path)
    if not source_path.exists():
        print(f"  [ERREUR] Fichier introuvable: {onnx_path}")
        return None

    # Determiner le nom de sortie
    output_name = get_transformed_name(onnx_path, fusion_level, int8)
    output_path = Path(output_dir) / output_name

    print(f"\n{'=' * 60}")
    print(f"TRANSFORMATION: {source_path.name}")
    print(f"{'=' * 60}")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")
    if fusion_level:
        print(f"  Fusion: {fusion_level}")
    if int8:
        print(f"  INT8 QDQ: {calib_dataset} ({calib_size} images)")

    if dry_run:
        print("  [dry-run] Aucune action effectuee")
        return str(output_path)

    # Charger metadata source
    source_metadata = load_source_metadata(onnx_path)

    calib_images = None
    current_onnx = onnx_path

    # Etape 1: Fusion ORT (si demandee)
    if fusion_level:
        result = apply_ort_fusion(current_onnx, output_dir, fusion_level, runtime_only)
        if runtime_only:
            return None
        if result:
            current_onnx = result

    # Etape 2: INT8 QDQ (si demandee)
    if int8:
        result, calib_images = apply_int8_quantization(
            current_onnx, output_dir, calib_dataset, calib_size
        )
        current_onnx = result

        # Si on a fait fusion + int8, supprimer l'intermediaire fusion-only
        if fusion_level:
            intermediate = Path(output_dir) / get_transformed_name(
                onnx_path, fusion_level, False
            )
            if intermediate.exists() and intermediate != Path(current_onnx):
                intermediate.unlink()
                intermediate.with_suffix(".json").unlink(missing_ok=True)

    # Sauvegarder metadata
    if current_onnx and Path(current_onnx).exists():
        save_transform_metadata(
            current_onnx,
            onnx_path,
            source_metadata,
            fusion_level=fusion_level,
            is_int8=int8,
            calib_dataset=calib_dataset if int8 else None,
            calib_size=calib_size if int8 else None,
            calib_images=calib_images,
        )

    return current_onnx


# =============================================================================
# CLI
# =============================================================================


def process_single(args) -> int:
    """Traite un seul modele."""
    result = transform_model(
        onnx_path=args.model,
        output_dir=args.output or str(DEFAULT_OUTPUT_DIR),
        fusion_level=args.fusion,
        int8=args.int8,
        calib_dataset=args.calib_dataset,
        calib_size=args.calib_size,
        runtime_only=args.runtime_only,
        dry_run=args.dry_run,
    )
    return 0 if result or args.dry_run or args.runtime_only else 1


def process_pattern(args) -> int:
    """Traite les modeles correspondant a un pattern."""
    import glob

    input_dir = args.input_dir or str(DEFAULT_INPUT_DIR)
    pattern_path = str(Path(input_dir) / args.pattern)
    matches = sorted(glob.glob(pattern_path))

    if not matches:
        print(f"Aucun fichier correspondant a: {pattern_path}")
        return 1

    print(f"Fichiers trouves: {len(matches)}")

    success_count = 0
    for onnx_path in matches:
        output_path = Path(args.output or str(DEFAULT_OUTPUT_DIR))
        expected_name = get_transformed_name(onnx_path, args.fusion, args.int8)

        # Skip si existe deja
        if args.skip_existing and (output_path / expected_name).exists():
            print(f"\n[skip] {Path(onnx_path).name} -> {expected_name} existe deja")
            success_count += 1
            continue

        result = transform_model(
            onnx_path=onnx_path,
            output_dir=str(output_path),
            fusion_level=args.fusion,
            int8=args.int8,
            calib_dataset=args.calib_dataset,
            calib_size=args.calib_size,
            runtime_only=args.runtime_only,
            dry_run=args.dry_run,
        )
        if result or args.dry_run or args.runtime_only:
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"TERMINE: {success_count}/{len(matches)} transformations reussies")
    return 0 if success_count == len(matches) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Transform ONNX models avec ORT fusion et INT8 quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Fusion ORT
  python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion all

  # INT8 QDQ
  python transform.py --model models/variants/yolo11n_640_fp32.onnx --int8

  # Combine
  python transform.py --model models/variants/yolo11n_640_fp32.onnx --fusion basic --int8

  # Batch
  python transform.py --pattern "yolo11n_*_fp16.onnx" --fusion all
        """,
    )

    # Source (mutuellement exclusif)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model",
        type=str,
        help="Chemin vers un seul modele ONNX",
    )
    input_group.add_argument(
        "--pattern",
        type=str,
        help="Pattern glob pour batch processing (ex: 'yolo11n_*.onnx')",
    )

    # Options de transformation
    parser.add_argument(
        "--fusion",
        type=str,
        choices=ORT_FUSION_LEVELS,
        help="Niveau d'optimisation ORT graph",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Appliquer quantification INT8 QDQ",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Pour fusion: ne pas serialiser, juste valider",
    )

    # Options de calibration (pour INT8)
    parser.add_argument(
        "--calib-dataset",
        type=str,
        default=DEFAULT_CALIB_DATASET,
        choices=["coco128", "coco"],
        help=f"Dataset pour calibration INT8 (defaut: {DEFAULT_CALIB_DATASET})",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=DEFAULT_CALIB_SIZE,
        help=f"Nombre d'images de calibration (defaut: {DEFAULT_CALIB_SIZE})",
    )

    # Options de sortie
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Dossier de sortie (defaut: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=f"Dossier d'entree pour --pattern (defaut: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ne pas regenerer si le fichier existe deja",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les actions sans les executer",
    )

    args = parser.parse_args()

    # Valider: au moins une transformation
    if not args.fusion and not args.int8:
        parser.error("Specifier --fusion et/ou --int8")

    # Traiter selon le mode
    if args.pattern:
        return process_pattern(args)
    else:
        return process_single(args)


if __name__ == "__main__":
    exit(main())
