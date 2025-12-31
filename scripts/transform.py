"""
Transform ONNX models with runtime optimizations.

Transformations supportees:
    A) ORT Fusion Levels: DISABLE, BASIC, EXTENDED, ALL
    B) INT8 Quantization via QDQ (Post-Training Quantization)

Ce script produit des ONNX derives qui peuvent ensuite etre compiles
par compile.py pour des cibles hardware specifiques.

Usage:
    # Fusion ORT (pour bench ORT uniquement, non portable vers TRT/OAK)
    python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion all
    python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion basic --runtime-only

    # INT8 QDQ (portable vers TRT avec --int8, depuis FP32 recommande)
    python transform.py --model models/variants/yolo11n_640_fp32.onnx --int8
    python transform.py --model models/variants/yolo11n_640_fp32.onnx --int8 --calib-size 200

    # Batch via pattern
    python transform.py --pattern "yolo11n_*_fp16.onnx" --fusion all

    # Dry-run
    python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion all --dry-run

NOTE: La combinaison --fusion + --int8 n'est PAS supportee.
      Les ONNX optimises par ORT contiennent des domaines internes non portables.
      Utiliser --fusion pour bench ORT, ou --int8 pour compilation TRT.

TODO (Phase 2): Batch processing avec YAML manifest
    - Parser manifest YAML (transforms.yaml)
    - Pattern matching glob avec fusion_levels multiples
    - Exemple:
        defaults:
          input_dir: models/variants
          output_dir: models/transformed
          calib_dataset: coco128
        transforms:
          - pattern: "yolo11*_*_fp16.onnx"
            fusion_levels: [basic, extended, all]
          - pattern: "yolo11n_*_fp32.onnx"
            int8: true
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
DEFAULT_INT8_ACT_TYPE = "qint8"  # S8S8 par defaut pour compatibilite GPU/TRT


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


def get_transform_suffix(
    fusion_level: str | None,
    is_int8: bool,
    int8_act_type: str = "qint8",
) -> str:
    """
    Genere le suffixe de transformation pour le nom de fichier.

    Examples:
        fusion_level="all", is_int8=False -> "_ortopt_all"
        fusion_level=None, is_int8=True, act=qint8  -> "_int8_qdq_s8s8"
        fusion_level=None, is_int8=True, act=quint8 -> "_int8_qdq_u8s8"
        fusion_level="basic", is_int8=True -> "_ortopt_basic_int8_qdq_s8s8"
    """
    parts = []
    if fusion_level and fusion_level != "disable":
        parts.append(f"ortopt_{fusion_level}")
    if is_int8:
        # Encoder le type d'activation dans le nom: s8s8 ou u8s8
        act_suffix = "s8s8" if int8_act_type == "qint8" else "u8s8"
        parts.append(f"int8_qdq_{act_suffix}")
    return "_" + "_".join(parts) if parts else ""


def get_transformed_name(
    source_onnx: str,
    fusion_level: str | None = None,
    is_int8: bool = False,
    int8_act_type: str = "qint8",
) -> str:
    """
    Genere le nom du fichier de sortie.

    Example:
        source: yolo11n_640_fp16.onnx
        fusion: "all", int8: False
        output: yolo11n_640_fp16_ortopt_all.onnx

        source: yolo11n_640_fp32.onnx
        fusion: None, int8: True, act: qint8
        output: yolo11n_640_fp32_int8_qdq_s8s8.onnx
    """
    stem = Path(source_onnx).stem
    suffix = get_transform_suffix(fusion_level, is_int8, int8_act_type)
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
    calib_dataset_root: str | None = None,
    int8_act_type: str | None = None,
    int8_weight_type: str | None = None,
    int8_per_channel: bool | None = None,
) -> None:
    """
    Sauvegarde le metadata JSON etendu pour le modele transforme.

    Inclut les informations INT8 detaillees pour eviter les ambiguites:
    - int8_format: "QDQ"
    - activation_type: "QInt8" ou "QUInt8"
    - weight_type: "QInt8"
    - per_channel: True/False
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
    metadata["calib_dataset_root"] = calib_dataset_root if is_int8 else None

    # INT8 format details (pour eviter l'ambiguite S8S8 vs U8S8)
    if is_int8:
        metadata["int8_format"] = "QDQ"
        metadata["activation_type"] = "QInt8" if int8_act_type == "qint8" else "QUInt8"
        metadata["weight_type"] = "QInt8" if int8_weight_type == "qint8" else "QUInt8"
        metadata["per_channel"] = int8_per_channel
    else:
        metadata["int8_format"] = None
        metadata["activation_type"] = None
        metadata["weight_type"] = None
        metadata["per_channel"] = None

    # Hash et liste des images de calibration (chemins relatifs) pour reproductibilite
    if is_int8 and calib_images:
        images_str = "|".join(sorted(calib_images))
        metadata["calib_images_hash"] = hashlib.sha256(images_str.encode()).hexdigest()[
            :16
        ]
        # Stocker les chemins relatifs (ex: images/train2017/000000000009.jpg)
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
    out = transform(image=img)
    # Selon la version Ultralytics, LetterBox peut retourner un dict ou un np.ndarray
    if isinstance(out, dict):
        return out.get("img", out.get("image", img))
    return out


def get_calibration_images(
    dataset_name: str,
    num_samples: int,
) -> tuple[list[str], list[str], str]:
    """
    Recupere la liste des images de calibration.

    Returns:
        Tuple (chemins absolus, chemins relatifs, racine dataset)
    """
    from ultralytics.data.utils import check_det_dataset

    if dataset_name == "coco":
        data = check_det_dataset("coco.yaml")
        dataset_root = Path(data["path"])
        rel_images_dir = "images/val2017"
    else:  # coco128
        data = check_det_dataset("coco128.yaml")
        dataset_root = Path(data["path"])
        rel_images_dir = "images/train2017"

    images_dir = dataset_root / rel_images_dir
    image_files = sorted(images_dir.glob("*.jpg"))[:num_samples]

    # Chemins absolus pour la calibration
    abs_paths = [str(p) for p in image_files]

    # Chemins relatifs pour la reproductibilite
    rel_paths = [f"{rel_images_dir}/{p.name}" for p in image_files]

    return abs_paths, rel_paths, str(dataset_root)


def apply_int8_quantization(
    onnx_path: str,
    output_dir: str,
    calib_dataset: str = "coco128",
    calib_size: int = 100,
    per_channel: bool = True,
    activation_type: str = "qint8",
) -> tuple[str, list[str], str, str, str, bool]:
    """
    Applique INT8 Post-Training Quantization via format QDQ.

    Args:
        onnx_path: Chemin vers le modele ONNX source (FP32 recommande)
        output_dir: Dossier de sortie
        calib_dataset: Dataset pour calibration ("coco128" ou "coco")
        calib_size: Nombre d'images de calibration
        per_channel: Utiliser quantification per-channel (plus precis)
        activation_type: "qint8" (S8S8, defaut GPU/TRT) ou "quint8" (U8S8)

    Returns:
        Tuple (chemin ONNX quantifie, chemins relatifs calibration, racine dataset,
               activation_type effectif, weight_type effectif, per_channel effectif)

    Notes:
        - Format QDQ est le plus portable pour INT8
        - TensorRT peut parser les modeles QDQ directement
        - ORT sur GPU prefere S8S8 (QInt8/QInt8) pour acceleration
        - Per-channel sur les poids ameliore la qualite
    """
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    print("  Applying INT8 QDQ quantization")
    print(f"    Calibration: {calib_dataset} ({calib_size} images)")

    # Mapper activation_type string -> QuantType
    act_type_map = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
    }
    quant_activation_type = act_type_map.get(activation_type, QuantType.QInt8)
    quant_weight_type = QuantType.QInt8  # Toujours QInt8 pour les poids

    act_label = "QInt8 (S8)" if activation_type == "qint8" else "QUInt8 (U8)"
    print(f"    Activation type: {act_label}")
    print("    Weight type: QInt8 (S8)")

    # Parser imgsz depuis le nom de fichier
    imgsz = parse_imgsz_from_filename(onnx_path)

    # Recuperer les images de calibration
    calib_abs_paths, calib_rel_paths, dataset_root = get_calibration_images(
        calib_dataset, calib_size
    )
    print(f"    Images trouvees: {len(calib_abs_paths)}")
    print(f"    Dataset root: {dataset_root}")

    # --- Pre-process: shape inference avant quantification ---
    # Recommande par ORT pour eviter des ranges manquantes et stabiliser la quantif
    print("    Pre-processing: shape inference...")
    preprocessed_path = Path(output_dir) / f"_temp_preprocess_{Path(onnx_path).name}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        quant_pre_process(
            input_model_path=onnx_path,
            output_model_path=str(preprocessed_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
        )
        onnx_for_quant = str(preprocessed_path)
        print("    Pre-processing: OK")
    except Exception as e:
        print(f"    [Warning] Pre-processing failed: {e}")
        print("    Continuing with original model...")
        onnx_for_quant = onnx_path

    # Obtenir le nom de l'input et verifier l'opset
    model = onnx.load(onnx_for_quant)
    input_name = model.graph.input[0].name

    # Verifier opset >= 13 pour per-channel quantization
    opset_version = 0
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset_version = opset.version
            break

    if per_channel and opset_version < 13:
        print(f"    [Warning] Opset {opset_version} < 13, per-channel desactive")
        per_channel = False

    print(f"    Per-channel: {per_channel} (opset {opset_version})")

    # Creer le CalibrationDataReader (utilise les chemins absolus)
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

    calib_reader = COCOCalibrationReader(calib_abs_paths, imgsz, input_name)

    # Chemin de sortie (inclut s8s8 ou u8s8 dans le nom)
    output_name = get_transformed_name(
        onnx_path, is_int8=True, int8_act_type=activation_type
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / output_name

    # Extra options pour S8S8 symetrique (recommande pour GPU/TRT)
    extra_options = {
        "ActivationSymmetric": activation_type == "qint8",  # Symetrique pour S8
        "WeightSymmetric": True,  # Toujours symetrique pour les poids
    }

    # Quantifier
    print("    Quantification en cours...")
    print(f"    Extra options: {extra_options}")
    try:
        quantize_static(
            model_input=onnx_for_quant,
            model_output=str(output_path),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=quant_weight_type,
            activation_type=quant_activation_type,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options=extra_options,
        )
    except Exception as e:
        print(f"    [ERREUR] Echec quantification: {e}")
        # Nettoyage du fichier temporaire
        if preprocessed_path.exists():
            preprocessed_path.unlink()
        raise

    # Nettoyage du fichier temporaire de preprocessing
    if preprocessed_path.exists():
        try:
            preprocessed_path.unlink()
        except Exception:
            pass

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    -> {output_path.name} ({size_mb:.2f} MB)")

    # Retourner les chemins relatifs, racine, et les types effectifs pour le metadata
    return (
        str(output_path),
        calib_rel_paths,
        dataset_root,
        activation_type,
        "qint8",
        per_channel,
    )


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
    int8_act_type: str = "qint8",
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
        int8_act_type: Type d'activation INT8 ("qint8" pour S8S8, "quint8" pour U8S8)
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
    output_name = get_transformed_name(onnx_path, fusion_level, int8, int8_act_type)
    output_path = Path(output_dir) / output_name

    print(f"\n{'=' * 60}")
    print(f"TRANSFORMATION: {source_path.name}")
    print(f"{'=' * 60}")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_path}")
    if fusion_level:
        print(f"  Fusion: {fusion_level}")
    if int8:
        act_label = "S8S8" if int8_act_type == "qint8" else "U8S8"
        print(f"  INT8 QDQ ({act_label}): {calib_dataset} ({calib_size} images)")

    if dry_run:
        print("  [dry-run] Aucune action effectuee")
        return str(output_path)

    # Charger metadata source
    source_metadata = load_source_metadata(onnx_path)

    calib_images = None
    calib_dataset_root = None
    int8_act_type_effective = None
    int8_weight_type_effective = None
    int8_per_channel_effective = None
    current_onnx = onnx_path

    # Etape 1: Fusion ORT (si demandee)
    if fusion_level:
        result = apply_ort_fusion(current_onnx, output_dir, fusion_level, runtime_only)
        if runtime_only:
            return None
        if result:
            current_onnx = result

    # Etape 2: INT8 QDQ (si demandee)
    # Note: fusion_level et int8 sont mutuellement exclusifs (valide par CLI)
    if int8:
        (
            result,
            calib_images,
            calib_dataset_root,
            int8_act_type_effective,
            int8_weight_type_effective,
            int8_per_channel_effective,
        ) = apply_int8_quantization(
            current_onnx,
            output_dir,
            calib_dataset,
            calib_size,
            activation_type=int8_act_type,
        )
        current_onnx = result

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
            calib_dataset_root=calib_dataset_root,
            int8_act_type=int8_act_type_effective,
            int8_weight_type=int8_weight_type_effective,
            int8_per_channel=int8_per_channel_effective,
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
        int8_act_type=args.int8_act_type,
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
        expected_name = get_transformed_name(
            onnx_path, args.fusion, args.int8, args.int8_act_type
        )

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
            int8_act_type=args.int8_act_type,
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
  # Fusion ORT (bench ORT uniquement, non portable TRT/OAK)
  python transform.py --model models/variants/yolo11n_640_fp16.onnx --fusion all

  # INT8 QDQ (portable vers TRT avec --int8)
  python transform.py --model models/variants/yolo11n_640_fp32.onnx --int8

  # Batch
  python transform.py --pattern "yolo11n_*_fp16.onnx" --fusion all

NOTE: --fusion + --int8 n'est pas supporte (ORT injecte des noeuds non portables)
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
    parser.add_argument(
        "--int8-act-type",
        type=str,
        default=DEFAULT_INT8_ACT_TYPE,
        choices=["qint8", "quint8"],
        help=(
            f"Type d'activation INT8 (defaut: {DEFAULT_INT8_ACT_TYPE}). "
            "qint8 = S8S8 (GPU/TRT compatible), quint8 = U8S8 (CPU legacy)"
        ),
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

    # Traiter --fusion disable comme "pas de fusion"
    if args.fusion == "disable":
        args.fusion = None

    # Valider: au moins une transformation
    if not args.fusion and not args.int8:
        parser.error("Specifier --fusion et/ou --int8")

    # Interdire la combinaison --fusion + --int8
    # Raison: Les ONNX optimises par ORT contiennent des domaines internes
    # (com.ms.internal.*, com.microsoft.*) non portables vers TRT/OpenVINO.
    # Un modele _ortopt_*_int8_qdq.onnx heriterait de ces noeuds non portables.
    # -> Fusion ORT = axe ORT-only (bench runtime, pas compilation TRT/OAK)
    # -> INT8 QDQ = axe portable (peut compiler vers TRT avec --int8)
    if args.fusion and args.int8:
        parser.error(
            "La combinaison --fusion + --int8 n'est pas supportee.\n\n"
            "Raison: Les ONNX optimises par ORT peuvent contenir des domaines internes\n"
            "(com.ms.internal.*, com.microsoft.*) non portables vers TRT/OpenVINO.\n"
            "Un modele _ortopt_*_int8_qdq.onnx heriterait de ces problemes.\n\n"
            "Solutions:\n"
            "  1) Pour bench ORT: utiliser --fusion seul (ex: --fusion all)\n"
            "  2) Pour INT8 portable: utiliser --int8 seul depuis FP32\n"
            "     python transform.py --model yolo11n_640_fp32.onnx --int8\n"
            "     python compile.py --model models/transformed/yolo11n_640_fp32_int8_qdq.onnx --target orin"
        )

    # Traiter selon le mode
    if args.pattern:
        return process_pattern(args)
    else:
        return process_single(args)


if __name__ == "__main__":
    exit(main())
