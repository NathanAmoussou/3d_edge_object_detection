#!/usr/bin/env python3
"""
Generate optimization tables from a benchmark CSV.

Creates one table per optimization (elagage, quantisation, resolution, fusion,
SHAVEs, heuristic, sparsity). Each table has:
- columns = optimization levels
- rows = metrics (E2E p95, mAP50, CPU/GPU/VPU, RAM/VRAM, Power)

CPU/GPU/VPU filtering rules:
- CPU util: only CPU_ORT_PC / CPU_ORT_PI4
- GPU util: only GPU_ORT_* (GPU_Util_Mean)
- Orin GPU util: Jetson_Orin_* (GR3D_Mean)
- OAK VPU proxy: OAK_LeonCSS_CPU_Pct_Mean (if available)

Idle deltas are applied per hardware for CPU/GPU/VPU/RAM/VRAM/Power.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _get_imgsz(row: pd.Series) -> int | None:
    v = row.get("ImgSz", None)
    if pd.notna(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    match = re.search(r"yolo11[nsm]_(\d+)_", str(row.get("Model_Name", "")))
    return int(match.group(1)) if match else None


def _get_quant(row: pd.Series) -> str | None:
    match = re.search(r"_fp(16|32)", str(row.get("Model_Name", "")))
    return f"fp{match.group(1)}" if match else None


def _get_scale(row: pd.Series) -> str | None:
    match = re.search(r"yolo11([nsm])_", str(row.get("Model_Name", "")))
    return match.group(1) if match else None


def _get_fusion(row: pd.Series) -> str | None:
    hardware = str(row.get("Hardware", ""))
    match = re.search(r"_(ALL|BASIC|DISABLE)\b", hardware)
    if match:
        return match.group(1)
    return None


def _get_shaves(row: pd.Series) -> int | None:
    match = re.search(r"_SHAVES_(\d+)", str(row.get("Hardware", "")))
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)shave", str(row.get("Model_Name", "")))
    return int(match.group(1)) if match else None


def _get_heuristic(row: pd.Series) -> str | None:
    match = re.search(r"_H(\d)", str(row.get("Hardware", "")))
    return f"H{match.group(1)}" if match else None


def _get_sparsity(row: pd.Series) -> str | None:
    match = re.search(r"_SP(\d)", str(row.get("Hardware", "")))
    return f"SP{match.group(1)}" if match else None


def _add_derived(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    hw = df["Hardware"].astype(str)

    df["CPU_Util"] = np.where(
        hw.str.startswith(("CPU_ORT_PC", "CPU_ORT_PI4"), na=False),
        pd.to_numeric(df.get("CPU_Proc_Norm_Mean"), errors="coerce"),
        np.nan,
    )

    accel = pd.Series(np.nan, index=df.index)
    accel = accel.where(
        ~hw.str.startswith("GPU_ORT", na=False),
        pd.to_numeric(df.get("GPU_Util_Mean"), errors="coerce"),
    )
    accel = accel.where(
        ~hw.str.startswith("Jetson_Orin_", na=False),
        pd.to_numeric(df.get("GR3D_Mean"), errors="coerce"),
    )
    if "OAK_LeonCSS_CPU_Pct_Mean" in df.columns:
        accel = accel.where(
            ~hw.str.startswith("OAK_", na=False),
            pd.to_numeric(df.get("OAK_LeonCSS_CPU_Pct_Mean"), errors="coerce"),
        )
    df["Accel_Util"] = accel

    df["RAM_MB"] = pd.to_numeric(df.get("RAM_RSS_MB_Mean"), errors="coerce")
    df["VRAM_MB"] = pd.to_numeric(df.get("VRAM_Used_MB_Mean"), errors="coerce")
    df["Power_W"] = pd.to_numeric(df.get("Power_W_Mean"), errors="coerce")
    return df


def _apply_idle_delta(run_df: pd.DataFrame, idle_df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = run_df.copy()
    if idle_df.empty or col not in idle_df.columns:
        out[f"{col}_delta"] = np.nan
        return out

    def _hw_group(hw: str) -> str:
        if hw.startswith("CPU_ORT_PC"):
            return "CPU_ORT_PC"
        if hw.startswith("CPU_ORT_PI4"):
            return "CPU_ORT_PI4"
        if hw.startswith("GPU_ORT"):
            return "GPU_ORT"
        if hw.startswith("Jetson_Orin_"):
            return "Jetson_Orin_"
        if hw.startswith("OAK_"):
            return "OAK_"
        return hw.split("_")[0] if "_" in hw else hw

    idle_map = idle_df.groupby("Hardware")[col].mean(numeric_only=True)
    idle_df = idle_df.copy()
    idle_df["HW_GROUP"] = idle_df["Hardware"].astype(str).map(_hw_group)
    idle_group_map = idle_df.groupby("HW_GROUP")[col].mean(numeric_only=True)

    hw_series = out["Hardware"].astype(str)
    idle_exact = hw_series.map(idle_map)
    idle_group = hw_series.map(lambda h: idle_group_map.get(_hw_group(h), np.nan))
    idle_val = idle_exact.fillna(idle_group)

    out[f"{col}_delta"] = out[col] - idle_val
    return out


def _levels_sorted(values, numeric=False):
    vals = sorted(set(values))
    if numeric:
        vals = sorted([v for v in vals if pd.notna(v)])
    return vals


def _build_table(
    df_run: pd.DataFrame,
    df_idle: pd.DataFrame,
    opt_fn,
    levels,
    *,
    numeric=False,
) -> pd.DataFrame:
    work = df_run.copy()
    work["opt"] = work.apply(opt_fn, axis=1)
    work = work[work["opt"].notna()]
    if work.empty:
        return pd.DataFrame()

    if numeric:
        work["opt"] = pd.to_numeric(work["opt"], errors="coerce")

    if levels is None:
        levels = _levels_sorted(work["opt"].dropna().tolist(), numeric=numeric)
    else:
        levels = [lvl for lvl in levels if lvl in work["opt"].unique()]

    if not levels:
        return pd.DataFrame()

    work = _add_derived(work)
    idle = _add_derived(df_idle)

    for col in ["CPU_Util", "Accel_Util", "RAM_MB", "VRAM_MB", "Power_W"]:
        work = _apply_idle_delta(work, idle, col)

    metrics = [
        ("E2E p95 (ms)", "Latency_p95", False),
        ("mAP@50 (%)", "mAP50", True),
        ("CPU util (%)", "CPU_Util_delta", False),
        ("GPU/VPU util (%)", "Accel_Util_delta", False),
        ("RAM RSS (MB)", "RAM_MB_delta", False),
        ("VRAM (MB)", "VRAM_MB_delta", False),
        ("Power mean (W)", "Power_W_delta", False),
    ]

    rows = {}
    for label, col, to_percent in metrics:
        if col not in work.columns:
            continue
        vals = (
            work.groupby("opt")[col]
            .mean(numeric_only=True)
            .reindex(levels)
        )
        if to_percent:
            vals = vals * 100.0
        rows[label] = vals

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).T
    table.columns = [str(c) for c in levels]
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize optimization tables from a benchmark CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("all_benchmark_results.csv"),
        help="Path to benchmark CSV (default: all_benchmark_results.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory for CSV tables.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    run_all = df[df.get("Phase") == "run"].copy()
    idle_all = df[df.get("Phase") == "idle"].copy()

    opt_tables = [
        ("Elagage", _get_scale, ["n", "s", "m"], False),
        ("Quantisation", _get_quant, ["fp16", "fp32"], False),
        ("Resolution", _get_imgsz, None, True),
        ("Fusion", _get_fusion, ["ALL", "BASIC", "DISABLE"], False),
        ("SHAVEs", _get_shaves, None, True),
        ("Heuristic", _get_heuristic, ["H0", "H1"], False),
        ("Sparsity", _get_sparsity, ["SP0", "SP1"], False),
    ]

    out_dir = args.out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    hardware_groups = [
        ("4070", lambda s: s.str.startswith("GPU_ORT", na=False)),
        ("i9", lambda s: s.str.startswith("CPU_ORT_PC", na=False)),
        ("Pi", lambda s: s.str.startswith("CPU_ORT_PI4", na=False)),
        ("Orin", lambda s: s.str.startswith("Jetson_Orin_", na=False)),
        ("Oak", lambda s: s.str.startswith("OAK_", na=False)),
    ]

    metric_order = [
        "E2E p95 (ms)",
        "mAP@50 (%)",
        "CPU util (%)",
        "GPU/VPU util (%)",
        "RAM RSS (MB)",
        "VRAM (MB)",
        "Power mean (W)",
    ]

    for title, opt_fn, levels, numeric in opt_tables:
        rows = []
        for hw_label, hw_filter in hardware_groups:
            hw_run = run_all[hw_filter(run_all["Hardware"].astype(str))].copy()
            hw_idle = idle_all[hw_filter(idle_all["Hardware"].astype(str))].copy()
            if hw_run.empty:
                continue
            tbl = _build_table(hw_run, hw_idle, opt_fn, levels, numeric=numeric)
            if tbl.empty:
                continue
            tbl = tbl.reset_index().rename(columns={"index": "Metric"})
            tbl.insert(1, "Hardware", hw_label)
            rows.append(tbl)

        if not rows:
            print(f"{title}: no data")
            continue

        combined = pd.concat(rows, ignore_index=True)
        combined["Metric"] = pd.Categorical(combined["Metric"], categories=metric_order, ordered=True)

        # Drop metric/hardware combos known to be non-applicable.
        drop_map = {
            "CPU util (%)": {"4070", "Orin", "Oak"},
            "GPU/VPU util (%)": {"i9", "Pi"},
            "VRAM (MB)": {"i9", "Pi", "Oak"},
        }
        for metric, hw_set in drop_map.items():
            mask = (combined["Metric"] == metric) & (combined["Hardware"].isin(hw_set))
            combined = combined[~mask]

        # Drop rows that are all-NaN across optimization levels.
        value_cols = [c for c in combined.columns if c not in {"Metric", "Hardware"}]
        if value_cols:
            combined = combined.dropna(subset=value_cols, how="all")
            combined[value_cols] = combined[value_cols].round(2)

        combined = combined.sort_values(["Metric", "Hardware"])

        # Display without repeating the metric label
        display_tbl = combined.copy()
        display_tbl["Metric"] = display_tbl["Metric"].astype(str)
        dup_mask = display_tbl["Metric"].duplicated()
        display_tbl.loc[dup_mask, "Metric"] = ""

        print(f"\n{title}")
        print(display_tbl.to_string(index=False))

        if out_dir:
            out_path = out_dir / f"{title.lower()}_by_hardware.csv"
            combined.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

    print("\nNote: VPU % is not directly measured. OAK uses LeonCSS CPU % as proxy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
