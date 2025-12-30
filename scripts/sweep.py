#!/usr/bin/env python3
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def resolve_script(root: Path, filename: str) -> Path:
    for p in [root / "scripts" / filename, root / filename]:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Script introuvable: {filename} (cherché dans scripts/ et root/)"
    )


def run_cmd(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"CMD: {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n")
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"RETURN_CODE: {proc.returncode}\n")
    return proc.returncode


def find_models(dir_path: Path, pattern: str) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob(pattern))


def latest_matching_file(dir_path: Path, glob_pattern: str) -> Path | None:
    matches = list(dir_path.glob(glob_pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def is_oak_compatible_onnx(p: Path) -> bool:
    s = p.stem
    if "_ortopt_" in s:
        return False
    if "_int8_qdq" in s:
        return False
    return True


def main():
    root = Path(__file__).resolve().parents[1]

    bench_py = resolve_script(root, "benchmark.py")
    compile_py = resolve_script(root, "compile.py")

    parser = argparse.ArgumentParser("Sweep models across GPU/OAK and log to CSV")
    parser.add_argument(
        "--targets", nargs="+", default=["4070", "oak"], choices=["4070", "oak"]
    )
    parser.add_argument("--dataset", default="coco128", choices=["coco128", "coco"])
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument(
        "--ort-levels",
        nargs="+",
        default=["disable", "basic", "extended", "all"],
        choices=["disable", "basic", "extended", "all"],
    )
    parser.add_argument("--variants-dir", default=str(root / "models" / "variants"))
    parser.add_argument(
        "--transformed-dir", default=str(root / "models" / "transformed")
    )
    parser.add_argument("--oak-out-dir", default=str(root / "models" / "oak"))
    parser.add_argument(
        "--filter", default="*.onnx", help="Glob filter for ONNX discovery (GPU)"
    )
    parser.add_argument(
        "--oak-filter", default="*_fp16.onnx", help="Glob filter for ONNX seeds (OAK)"
    )
    parser.add_argument(
        "--compile-oak", action="store_true", help="Compile missing blobs automatically"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--logs-dir", default=str(root / "runs"))
    args = parser.parse_args()

    variants_dir = Path(args.variants_dir)
    transformed_dir = Path(args.transformed_dir)
    oak_out_dir = Path(args.oak_out_dir)
    logs_dir = Path(args.logs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")

    py = sys.executable

    # -------------------------
    # Build job list
    # -------------------------
    jobs = []

    if "4070" in args.targets:
        gpu_models = find_models(variants_dir, args.filter) + find_models(
            transformed_dir, args.filter
        )
        for m in gpu_models:
            if m.suffix.lower() != ".onnx":
                continue
            # Heuristique: si modèle déjà _ortopt_*, on bench avec ORT disable uniquement
            if "_ortopt_" in m.stem:
                levels = ["disable"]
            else:
                levels = args.ort_levels
            for lvl in levels:
                jobs.append(("gpu", m, lvl))

    if "oak" in args.targets:
        oak_seeds = find_models(variants_dir, args.oak_filter)
        for onnx in oak_seeds:
            if onnx.suffix.lower() != ".onnx":
                continue
            if not is_oak_compatible_onnx(onnx):
                continue
            jobs.append(("oak", onnx, None))

    print(f"[sweep] Jobs: {len(jobs)}")
    if args.dry_run:
        for j in jobs[:50]:
            print("  ", j)
        if len(jobs) > 50:
            print(f"  ... ({len(jobs) - 50} more)")
        return 0

    # -------------------------
    # Run jobs
    # -------------------------
    failures = 0
    for idx, (kind, model_path, lvl) in enumerate(jobs, start=1):
        if kind == "gpu":
            tag = f"gpu_{model_path.stem}_ort_{lvl}"
            log = logs_dir / "gpu" / f"{idx:04d}_{tag}.log"
            cmd = [
                py,
                str(bench_py),
                "--target",
                "4070",
                "--backend",
                "ort",
                "--model",
                str(model_path),
                "--dataset",
                args.dataset,
                "--num-classes",
                str(args.num_classes),
                "--ort-opt-level",
                lvl,
            ]
            rc = run_cmd(cmd, log)
            if rc != 0:
                failures += 1
                print(f"[FAIL] {tag} (see {log})")

        elif kind == "oak":
            # Compile to blob (optional)
            blob = latest_matching_file(oak_out_dir, f"{model_path.stem}*.blob")
            if blob is None and args.compile_oak:
                tagc = f"oak_compile_{model_path.stem}"
                logc = logs_dir / "oak" / f"{idx:04d}_{tagc}.log"
                cmdc = [
                    py,
                    str(compile_py),
                    "--target",
                    "oak",
                    "--model",
                    str(model_path),
                    "--output",
                    str(oak_out_dir),
                ]
                rc = run_cmd(cmdc, logc)
                if rc != 0:
                    failures += 1
                    print(f"[FAIL] {tagc} (see {logc})")
                    continue
                blob = latest_matching_file(oak_out_dir, f"{model_path.stem}*.blob")

            if blob is None:
                failures += 1
                print(
                    f"[FAIL] oak_missing_blob for {model_path.name} (use --compile-oak)"
                )
                continue

            tagb = f"oak_bench_{blob.stem}"
            logb = logs_dir / "oak" / f"{idx:04d}_{tagb}.log"
            cmdb = [
                py,
                str(bench_py),
                "--target",
                "oak",
                "--model",
                str(blob),
                "--dataset",
                args.dataset,
                "--num-classes",
                str(args.num_classes),
            ]
            rc = run_cmd(cmdb, logb)
            if rc != 0:
                failures += 1
                print(f"[FAIL] {tagb} (see {logb})")

        print(f"[{idx}/{len(jobs)}] done")

    print(f"[sweep] Finished with failures={failures}, logs={logs_dir}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
