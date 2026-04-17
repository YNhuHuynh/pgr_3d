#!/usr/bin/env python3
"""
Batch evaluation for PGR-3D on the GSO-30 subset.

Examples
--------
python scripts/eval_gso.py \
    --method_name wonder3d_baseline \
    --pred_root /scratch/s224696943/pgr_3d/outputs/wonder3d_baseline \
    --gt_root /scratch/s224696943/datasets/GSO \
    --output_csv results/wonder3d_baseline.csv \
    --num_novel_views 24

python scripts/eval_gso.py \
    --method_name pgr_sem_eta1.0 \
    --pred_root outputs/pgr_sem_eta1.0 \
    --gt_root /scratch/s224696943/datasets/GSO \
    --output_csv results/pgr_sem_eta1.0.csv \
    --num_novel_views 24 \
    --skip_missing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

    tqdm.write = print

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metrics import evaluate_object  # noqa: E402

METRIC_COLUMNS = ["cd", "iou", "psnr", "ssim", "lpips"]
CSV_COLUMNS = ["object_id", *METRIC_COLUMNS, "method_name"]
LOGGER = logging.getLogger("eval_gso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PGR-3D methods on GSO-30.")
    parser.add_argument("--method_name", required=True, help="Name written to CSV rows.")
    parser.add_argument("--pred_root", required=True, type=Path, help="Root with {object_id}/mesh.obj and renders/.")
    parser.add_argument("--gt_root", required=True, type=Path, help="Root with {object_id}/meshes/model.obj and renders/.")
    parser.add_argument("--output_csv", required=True, type=Path, help="Destination CSV path.")
    parser.add_argument("--num_novel_views", type=int, default=24, help="Number of view_XXX.png render pairs.")
    parser.add_argument("--skip_missing", action="store_true", help="Write NaN rows instead of stopping on missing files.")
    parser.add_argument(
        "--object_file",
        type=Path,
        default=REPO_ROOT / "configs" / "gso30_objects.txt",
        help="Text file containing one GSO object id per line.",
    )
    return parser.parse_args()


def resolve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def read_object_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"object list not found: {path}")
    object_ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            object_ids.append(line)
    if not object_ids:
        raise ValueError(f"object list is empty: {path}")
    return object_ids


def object_paths(args: argparse.Namespace, object_id: str) -> dict[str, Path | list[Path]]:
    pred_dir = args.pred_root / object_id
    gt_dir = args.gt_root / object_id
    pred_render_dir = pred_dir / "renders"
    gt_render_dir = gt_dir / "renders"
    return {
        "pred_mesh": pred_dir / "mesh.obj",
        "gt_mesh": gt_dir / "meshes" / "model.obj",
        "pred_render_dir": pred_render_dir,
        "gt_render_dir": gt_render_dir,
        "pred_views": [pred_render_dir / f"view_{idx:03d}.png" for idx in range(args.num_novel_views)],
        "gt_views": [gt_render_dir / f"view_{idx:03d}.png" for idx in range(args.num_novel_views)],
    }


def missing_paths(paths: dict[str, Path | list[Path]]) -> list[Path]:
    required: list[Path] = [
        paths["pred_mesh"],
        paths["gt_mesh"],
        paths["pred_render_dir"],
        paths["gt_render_dir"],
        *paths["pred_views"],
        *paths["gt_views"],
    ]
    return [path for path in required if not path.exists()]


def nan_row(object_id: str, method_name: str) -> dict[str, object]:
    return {"object_id": object_id, **{key: np.nan for key in METRIC_COLUMNS}, "method_name": method_name}


def format_metrics(row: dict[str, object]) -> str:
    parts = []
    for key in METRIC_COLUMNS:
        value = row[key]
        try:
            parts.append(f"{key}={float(value):.6g}" if np.isfinite(float(value)) else f"{key}=nan")
        except (TypeError, ValueError):
            parts.append(f"{key}=nan")
    return "  ".join(parts)


def with_summary(rows: Iterable[dict[str, object]], method_name: str) -> pd.DataFrame:
    import pandas as pd

    df = pd.DataFrame(list(rows), columns=CSV_COLUMNS)
    summary = {"object_id": "MEAN", "method_name": method_name}
    for key in METRIC_COLUMNS:
        summary[key] = pd.to_numeric(df[key], errors="coerce").mean(skipna=True)
    return pd.concat([df, pd.DataFrame([summary], columns=CSV_COLUMNS)], ignore_index=True)


def save_results(rows: list[dict[str, object]], output_csv: Path, method_name: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with_summary(rows, method_name).to_csv(output_csv, index=False)
    LOGGER.info("saved %d object rows to %s", len(rows), output_csv)


def print_mean_table(rows: list[dict[str, object]], method_name: str) -> None:
    mean_df = with_summary(rows, method_name).tail(1)
    print("\nMean metrics")
    print(mean_df.to_string(index=False, columns=CSV_COLUMNS))


def evaluate_one(args: argparse.Namespace, object_id: str) -> dict[str, object]:
    paths = object_paths(args, object_id)
    missing = missing_paths(paths)
    if missing:
        message = f"{object_id}: missing {len(missing)} required path(s); first missing: {missing[0]}"
        if not args.skip_missing:
            raise FileNotFoundError(message)
        LOGGER.warning(message)
        return nan_row(object_id, args.method_name)

    metrics = evaluate_object(
        str(paths["gt_mesh"]),
        str(paths["pred_mesh"]),
        [str(path) for path in paths["gt_views"]],
        [str(path) for path in paths["pred_views"]],
    )
    return {
        "object_id": object_id,
        "cd": metrics.get("cd", np.nan),
        "iou": metrics.get("iou", np.nan),
        "psnr": metrics.get("psnr", np.nan),
        "ssim": metrics.get("ssim", np.nan),
        "lpips": metrics.get("lpips", np.nan),
        "method_name": args.method_name,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
    args = parse_args()
    output_csv = resolve_output_path(args.output_csv)
    if output_csv != args.output_csv:
        LOGGER.warning("output exists; writing to %s instead", output_csv)

    object_ids = read_object_ids(args.object_file)
    rows: list[dict[str, object]] = []

    try:
        for object_id in tqdm(object_ids, desc=f"Evaluating {args.method_name}", unit="object"):
            start = time.perf_counter()
            row = evaluate_one(args, object_id)
            rows.append(row)
            elapsed = time.perf_counter() - start
            tqdm.write(f"{object_id}: {format_metrics(row)}  time={elapsed:.1f}s")
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl-C; saving partial results before exit.", flush=True)
        if rows:
            save_results(rows, output_csv, args.method_name)
            print_mean_table(rows, args.method_name)
        return 130

    save_results(rows, output_csv, args.method_name)
    print_mean_table(rows, args.method_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
