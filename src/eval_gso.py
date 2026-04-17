"""
PGR-3D GSO Evaluation
----------------------
End-to-end evaluation on the GSO-30 subset used in the Wonder3D/SyncDreamer
papers.  Computes Chamfer Distance (L1, L2), Volume IoU, PSNR, SSIM, LPIPS
for a single run configuration (baseline Wonder3D or PGR-guided).

Pipeline
--------
1. For each GSO object:
   a. Run multi-view generation (Wonder3D ± PGR guidance) if needed.
   b. Run 3D reconstruction (instant-nsr-pl via 3DRAV's run_3d_pipeline_zero.py).
   c. Load predicted mesh and GT mesh; align via ICP.
   d. Compute geometry metrics (CD, IoU).
   e. Compute novel-view-synthesis metrics (PSNR/SSIM/LPIPS) on rendered views
      vs GT rendered views.
   f. Append all rows to a CSV.

Design
------
- This script is intentionally a thin orchestration layer over:
    guidance_inference.py    (multi-view generation)
    metrics.py               (all metric computation)
    3DRAV/run_3d_pipeline_zero.py  (reconstruction — called as subprocess)
- For the baseline, generation is already done in slurm_baseline_gso.sh; this
  script can also be used standalone to regenerate with guidance.
- All angle/rotation corrections are taken from 000.run_won3d.sh in 3DRAV.

Usage
-----
Baseline (images already generated):
    python src/eval_gso.py \
        --mode metrics_only \
        --mesh_dir outputs/meshes_baseline \
        --gt_dir /scratch/s224696943/3DRAV/evaluation/gso \
        --setup baseline \
        --csv_out outputs/metrics/baseline_gso30.csv

Full run (generate + reconstruct + eval):
    python src/eval_gso.py \
        --mode full \
        --head_type semantic \
        --head_ckpt outputs/checkpoints/pgr_semantic_200_YYYYMMDD_final.pt \
        --eta 1.0 \
        --t_guidance_max 800 \
        --gt_dir /scratch/s224696943/3DRAV/evaluation/gso \
        --setup pgr_sem_eta1.0 \
        --csv_out outputs/metrics/pgr_sem_eta1.0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

from metrics import mesh_metrics, image_metrics, append_mesh_csv, append_nvs_csv

# ---------------------------------------------------------------------------
# GSO objects in our eval set (30 objects matching SyncDreamer/Wonder3D paper)
# ---------------------------------------------------------------------------
# Canonical GSO-30 from /scratch/s224696943/GSO/gso/ (official Google Scanned Objects).
# Real-world captured (1st–15th) excluded — no GT mesh, not reproducible.
GSO_OBJECTS = [
    "alarm", "backpack", "bell", "blocks", "chicken",
    "cream", "elephant", "grandfather", "grandmother", "hat",
    "leather", "lion", "lunch_bag", "mario", "oil",
    "school_bus1", "school_bus2", "shoe", "shoe1", "shoe2",
    "shoe3", "soap", "sofa", "sorter", "sorting_board",
    "stucking_cups", "teapot", "toaster", "train", "turtle",
]

# Per-object pre-rotation for GT mesh alignment (from 000.run_won3d.sh)
_GT_ANGLE_DEG: Dict[str, int] = {
    "alarm": 270, "backpack": 270, "chicken": 270, "lion": 270,
    "lunch_bag": 270, "mario": 270, "oil": 270, "sofa": 270,
    "elephant": 180, "school_bus1": 180,
    "school_bus2": 225, "shoe": 225, "train": 225, "turtle": 225,
}
PRED_ROT = "x:90,y:0,z:0"   # constant pred pre-rotation (matches Wonder3D output frame)


def gt_rot_for_object(obj: str) -> str:
    angle = _GT_ANGLE_DEG.get(obj, 0)
    # GT pre-rotation = z-axis by (90 - angle) degrees
    z_deg = 90 - angle
    return f"x:0,y:0,z:{z_deg}"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
PGR_DIR       = Path("/scratch/s224696943/pgr_3d")
GSO_GT_DIR    = Path("/scratch/s224696943/3DRAV/evaluation/gso")
GSO_FRONTS    = PGR_DIR / "data" / "gso_fronts"
RECON_SCRIPT  = Path("/scratch/s224696943/3DRAV/run_3d_pipeline.py")
TRAV_DIR      = Path("/scratch/s224696943/3DRAV")


# ---------------------------------------------------------------------------
# Multi-view generation (Wonder3D ± guidance)
# ---------------------------------------------------------------------------

def generate_views(
    obj: str,
    front_pil: Image.Image,
    output_dir: Path,
    pipe,
    head,
    head_type: str,
    eta: float,
    guidance_scale: float,
    num_steps: int,
    t_min: int,
    t_max: int,
    device: str,
    seed: int,
) -> bool:
    """
    Generate 6 RGB views + 6 normal views using PGR guidance.
    Saves to output_dir/rgb_{i}.png and normal_{i}.png.
    Returns True on success.
    """
    from guidance_inference import run_guided_inference

    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already generated
    if all((output_dir / f"rgb_{i:02d}.png").exists() for i in range(6)):
        print(f"  [SKIP] Views already exist for {obj}")
        return True

    try:
        result = run_guided_inference(
            pipe, head, front_pil,
            head_type=head_type,
            eta=eta,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            t_guidance_min=t_min,
            t_guidance_max=t_max,
            device=device,
            seed=seed,
        )
        for v, img in enumerate(result["images"]):
            img.save(output_dir / f"rgb_{v:02d}.png")
        for v, img in enumerate(result["normal_images"]):
            img.save(output_dir / f"normal_{v:02d}.png")
        return True
    except Exception as e:
        print(f"  [ERROR] Generation failed for {obj}: {e}")
        return False


# ---------------------------------------------------------------------------
# 3D Reconstruction (calls 3DRAV's instant-nsr-pl pipeline)
# ---------------------------------------------------------------------------

def reconstruct_mesh(
    views_dir: Path,
    obj: str,
    mesh_out_dir: Path,
    suffix: str = "",
) -> Optional[Path]:
    """
    Call 3DRAV's reconstruction pipeline on a directory of multi-view images.
    Returns path to output .obj file, or None on failure.

    Expected output naming: {obj}{suffix}.obj in mesh_out_dir.
    """
    mesh_out_dir.mkdir(parents=True, exist_ok=True)
    out_mesh = mesh_out_dir / f"{obj}{suffix}.obj"

    if out_mesh.exists():
        print(f"  [SKIP] Mesh already exists: {out_mesh}")
        return out_mesh

    # 3DRAV's pipeline script expects wonder3D-style output folder
    cmd = [
        sys.executable,
        str(RECON_SCRIPT),
        "--run_dir", str(views_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TRAV_DIR}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env
        )
        if result.returncode != 0:
            print(f"  [ERROR] Reconstruction failed for {obj}: {result.stderr[:300]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Reconstruction timed out for {obj}")
        return None

    # 3DRAV typically writes to a fixed path; we copy to our dir
    candidate_paths = [
        Path("/scratch/s224696943/3DRAV_ext/evaluation/results_zero") / f"{obj}_B.obj",
        Path("/scratch/s224696943/3DRAV_ext/evaluation/results_zero") / f"{obj}.obj",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            import shutil
            shutil.copy(candidate, out_mesh)
            return out_mesh

    print(f"  [WARN] Mesh output not found at expected paths for {obj}")
    return None


# ---------------------------------------------------------------------------
# Mesh alignment + geometry metrics
# ---------------------------------------------------------------------------

def load_and_align_meshes(
    pred_path: Path,
    gt_path: Path,
    pred_rot_str: str = PRED_ROT,
    gt_rot_str: str   = "",
    normalize: str    = "unit-diag",
    icp_iters: int    = 400,
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """
    Load and roughly align pred/GT meshes using the same rotation convention
    as 3DRAV's evaluation scripts.

    Returns (pred_mesh, gt_mesh) both in consistent coordinate frame, or
    (None, None) if either file doesn't exist.
    """
    if not pred_path.exists():
        print(f"  [WARN] Pred mesh missing: {pred_path}")
        return None, None
    if not gt_path.exists():
        print(f"  [WARN] GT mesh missing: {gt_path}")
        return None, None

    def parse_rot(rot_str: str) -> np.ndarray:
        """Parse 'x:90,y:0,z:270' → 4×4 rotation matrix."""
        import transforms3d as t3d
        angles = {"x": 0.0, "y": 0.0, "z": 0.0}
        if rot_str:
            for part in rot_str.split(","):
                axis, deg = part.split(":")
                angles[axis.strip()] = float(deg)
        Rx = t3d.euler.euler2mat(np.radians(angles["x"]), 0, 0)
        Ry = t3d.euler.euler2mat(0, np.radians(angles["y"]), 0)
        Rz = t3d.euler.euler2mat(0, 0, np.radians(angles["z"]))
        R  = Rz @ Ry @ Rx
        T  = np.eye(4)
        T[:3, :3] = R
        return T

    try:
        pred = trimesh.load(str(pred_path), force="mesh")
        gt   = trimesh.load(str(gt_path),   force="mesh")

        # Apply pre-rotations
        if pred_rot_str:
            pred.apply_transform(parse_rot(pred_rot_str))
        if gt_rot_str:
            gt.apply_transform(parse_rot(gt_rot_str))

        # Normalise both to unit diagonal bounding box, centred at origin
        if normalize == "unit-diag":
            diag = np.linalg.norm(gt.bounds[1] - gt.bounds[0])
            scale = 1.0 / diag if diag > 0 else 1.0
            centre = gt.centroid
            for mesh in (pred, gt):
                mesh.apply_translation(-centre)
                mesh.apply_scale(scale)

        return pred, gt

    except Exception as e:
        print(f"  [ERROR] Mesh loading/alignment failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# NVS metrics (PSNR/SSIM/LPIPS on rendered views vs GT)
# ---------------------------------------------------------------------------

def nvs_metrics_from_dirs(
    pred_dir: Path,
    gt_render_dir: Path,
    view_indices: List[int] = list(range(6)),
) -> Optional[Dict]:
    """
    Compute per-view NVS metrics.
    Expects pred_dir/rgb_{i:02d}.png and gt_render_dir/{i}.png (or similar).

    Returns dict with per-view results, or None if files not found.
    """
    results = []
    for v in view_indices:
        pred_path = pred_dir / f"rgb_{v:02d}.png"
        # GT renders: try several naming patterns
        gt_path = None
        for pattern in [f"{v}.png", f"view_{v}.png", f"rgb_{v}.png", f"{v:04d}.png"]:
            p = gt_render_dir / pattern
            if p.exists():
                gt_path = p
                break

        if not pred_path.exists() or gt_path is None:
            continue

        pred_img = np.array(Image.open(pred_path).convert("RGB").resize((256, 256)))
        gt_img   = np.array(Image.open(gt_path).convert("RGB").resize((256, 256)))

        m = image_metrics(pred_img, gt_img)
        m["view_idx"] = v
        results.append(m)

    if not results:
        return None

    return {
        "per_view":    results,
        "mean_psnr":   np.mean([r["psnr"]  for r in results]),
        "mean_ssim":   np.mean([r["ssim"]  for r in results]),
        "mean_lpips":  np.mean([r["lpips"] for r in results]),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def eval_object(
    obj: str,
    args,
    pipe=None,
    head=None,
    device: str = "cuda",
) -> Optional[Dict]:
    """Evaluate a single GSO object. Returns metrics dict or None on failure."""
    print(f"\n--- {obj} ---")

    # Paths
    views_dir   = PGR_DIR / "outputs" / args.setup / obj
    mesh_dir    = PGR_DIR / "outputs" / "meshes" / args.setup
    gt_mesh_path = GSO_GT_DIR / obj / "meshes" / "model.obj"
    gt_render_dir = GSO_GT_DIR / obj / "renders"  # may not exist

    # ---- Step 1: Generate views (if mode == full) ----
    if args.mode == "full":
        front_path = GSO_FRONTS / f"{obj}.png"
        if not front_path.exists():
            print(f"  [SKIP] No front image: {front_path}")
            return None
        front_pil = Image.open(front_path).convert("RGB").resize((256, 256), Image.BILINEAR)
        ok = generate_views(
            obj, front_pil, views_dir,
            pipe=pipe, head=head,
            head_type=args.head_type, eta=args.eta,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            t_min=args.t_guidance_min, t_max=args.t_guidance_max,
            device=device, seed=args.seed,
        )
        if not ok:
            return None

    # ---- Step 2: Reconstruct mesh (if mode == full or reconstruct) ----
    if args.mode in ("full", "reconstruct"):
        pred_mesh_path = reconstruct_mesh(views_dir, obj, mesh_dir, suffix=f"_{args.setup}")
    else:
        # metrics_only: look for mesh with naming convention
        pred_mesh_path = None
        for suffix in [f"_{args.setup}", "_baseline", ""]:
            p = Path(args.mesh_dir) / f"{obj}{suffix}.obj"
            if p.exists():
                pred_mesh_path = p
                break
        if pred_mesh_path is None:
            print(f"  [SKIP] No mesh found in {args.mesh_dir} for {obj}")
            return None

    # ---- Step 3: Geometry metrics ----
    pred_mesh, gt_mesh = load_and_align_meshes(
        pred_mesh_path, gt_mesh_path,
        pred_rot_str=PRED_ROT,
        gt_rot_str=gt_rot_for_object(obj),
    )

    geo_metrics = None
    if pred_mesh is not None and gt_mesh is not None:
        try:
            geo_metrics = mesh_metrics(pred_mesh, gt_mesh)
            print(f"  CD-L1={geo_metrics['cd_l1']:.6f}  CD-L2={geo_metrics['cd_l2']:.6f}  IoU={geo_metrics['iou']:.4f}")
        except Exception as e:
            print(f"  [WARN] Geometry metric failed: {e}")

    # ---- Step 4: NVS metrics ----
    nvs = None
    if gt_render_dir.exists():
        nvs = nvs_metrics_from_dirs(views_dir, gt_render_dir)
        if nvs:
            print(f"  PSNR={nvs['mean_psnr']:.2f}  SSIM={nvs['mean_ssim']:.4f}  LPIPS={nvs['mean_lpips']:.4f}")

    # ---- Step 5: Write CSV rows ----
    if geo_metrics is not None:
        append_mesh_csv(args.csv_out, {
            "object": obj,
            "setup":  args.setup,
            "name":   f"{obj}_{args.setup}",
            "cd_l1":  geo_metrics["cd_l1"],
            "cd_l2":  geo_metrics["cd_l2"],
            "iou":    geo_metrics["iou"],
            "pred":   str(pred_mesh_path),
            "gt":     str(gt_mesh_path),
        })

    if nvs is not None:
        for view_row in nvs["per_view"]:
            append_nvs_csv(args.csv_out.replace(".csv", "_nvs.csv"), {
                "object":   obj,
                "setup":    args.setup,
                "name":     f"{obj}_{args.setup}",
                "psnr":     view_row["psnr"],
                "ssim":     view_row["ssim"],
                "lpips":    view_row["lpips"],
                "view_idx": view_row["view_idx"],
            })

    return {
        "object": obj,
        "geo":    geo_metrics,
        "nvs":    nvs,
    }


def print_aggregate_results(all_results: List[Dict], setup: str) -> None:
    valid_geo = [r for r in all_results if r.get("geo") is not None]
    valid_nvs = [r for r in all_results if r.get("nvs") is not None]

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS — {setup}")
    print(f"Objects evaluated: {len(all_results)}")

    if valid_geo:
        cd_l1s = [r["geo"]["cd_l1"] for r in valid_geo]
        cd_l2s = [r["geo"]["cd_l2"] for r in valid_geo]
        ious   = [r["geo"]["iou"]   for r in valid_geo]
        print(f"\nGeometry ({len(valid_geo)} objects):")
        print(f"  CD-L1 : mean={np.mean(cd_l1s):.6f}  std={np.std(cd_l1s):.6f}")
        print(f"  CD-L2 : mean={np.mean(cd_l2s):.6f}  std={np.std(cd_l2s):.6f}")
        print(f"  IoU   : mean={np.mean(ious):.4f}    std={np.std(ious):.4f}")

    if valid_nvs:
        psnrs  = [r["nvs"]["mean_psnr"]  for r in valid_nvs]
        ssims  = [r["nvs"]["mean_ssim"]  for r in valid_nvs]
        lpipss = [r["nvs"]["mean_lpips"] for r in valid_nvs]
        print(f"\nNVS ({len(valid_nvs)} objects):")
        print(f"  PSNR  : mean={np.mean(psnrs):.2f}   std={np.std(psnrs):.2f}")
        print(f"  SSIM  : mean={np.mean(ssims):.4f}  std={np.std(ssims):.4f}")
        print(f"  LPIPS : mean={np.mean(lpipss):.4f}  std={np.std(lpipss):.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PGR-3D GSO Evaluation")

    # Mode
    parser.add_argument("--mode", choices=["full", "reconstruct", "metrics_only"],
                        default="metrics_only",
                        help="full=generate+reconstruct+eval; reconstruct=recon+eval; metrics_only=eval existing meshes")

    # Generation (only for full mode)
    parser.add_argument("--head_type",       default="semantic", choices=["semantic", "depth"])
    parser.add_argument("--head_ckpt",       default=None, help="Path to trained head checkpoint")
    parser.add_argument("--eta",             type=float, default=0.0,
                        help="Guidance strength (0=baseline)")
    parser.add_argument("--guidance_scale",  type=float, default=3.0)
    parser.add_argument("--num_steps",       type=int,   default=50)
    parser.add_argument("--t_guidance_min",  type=int,   default=0)
    parser.add_argument("--t_guidance_max",  type=int,   default=800)
    parser.add_argument("--seed",            type=int,   default=42)

    # Evaluation
    parser.add_argument("--objects",    nargs="+",  default=GSO_OBJECTS)
    parser.add_argument("--mesh_dir",   default=str(PGR_DIR / "outputs" / "meshes_baseline"),
                        help="Directory with pred meshes (metrics_only mode)")
    parser.add_argument("--gt_dir",     default=str(GSO_GT_DIR))
    parser.add_argument("--setup",      required=True,
                        help="Experiment name, e.g. baseline or pgr_sem_eta1.0")
    parser.add_argument("--csv_out",    required=True,
                        help="Path to output CSV file")
    parser.add_argument("--device",     default="cuda")

    args = parser.parse_args()

    # Override gt_dir if provided
    global GSO_GT_DIR
    GSO_GT_DIR = Path(args.gt_dir)

    # ---- Load pipeline + head (only for full mode) ----
    pipe = head = None
    if args.mode == "full":
        from guidance_inference import build_guided_pipeline
        from readout_heads import build_semantic_head, build_depth_head, load_head

        print("Loading Wonder3D pipeline ...")
        pipe = build_guided_pipeline(args.device)

        if args.head_ckpt:
            print(f"Loading {args.head_type} head from {args.head_ckpt} ...")
            if args.head_type == "semantic":
                head = build_semantic_head(device=args.device)
            else:
                head = build_depth_head(device=args.device)
            load_head(args.head_ckpt, head, device=args.device)
        else:
            print("[INFO] No head checkpoint — running baseline Wonder3D (eta=0)")
            args.eta = 0.0

    # ---- Evaluate all objects ----
    all_results = []
    for obj in args.objects:
        result = eval_object(obj, args, pipe=pipe, head=head, device=args.device)
        if result is not None:
            all_results.append(result)

    print_aggregate_results(all_results, args.setup)
    print(f"Results written to: {args.csv_out}")


if __name__ == "__main__":
    main()
