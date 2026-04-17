"""
Block A-extension: Render 16 GT novel views per GSO object.

Purpose:
    Ground-truth renders for PSNR/SSIM/LPIPS evaluation of the CaptionHead
    readout against Wonder3D baselines.  16 views at 22.5° azimuth spacing,
    elevation 0°, 256×256 orthographic, using the IPS3D blender_script_v2.py.

Output layout:
    data/gso30_eval_novelviews/{obj}/view_{00..15}.png   (16 PNGs per object)
    data/gso30_eval_novelviews/{obj}/meta.pkl            (camera intrinsics/extrinsics)

Canonical GT rotations applied via --rotate_z (sourced from IPS3D get_angle_deg()):
    270°: alarm backpack blocks chicken grandfather grandmother lion
          lunch_bag mario oil soap sofa
    225°: school_bus2 shoe train turtle
    180°: elephant school_bus1
    112.5°: sorter
    0°:  bell cream hat leather shoe1 shoe2 shoe3 sorting_board
         stucking_cups teapot toaster

Usage (standalone):
    python scripts/render_gso30_eval_novelviews.py
    python scripts/render_gso30_eval_novelviews.py --dry_run
    python scripts/render_gso30_eval_novelviews.py --objects alarm sofa
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PGR_DIR        = Path("/scratch/s224696943/pgr_3d")
GSO_MESH_ROOT  = Path("/scratch/s224696943/GSO/gso")
BLENDER_BIN    = Path("/scratch/s224696943/blender/blender-3.3.0-linux-x64/blender")
RENDER_SCRIPT  = Path("/scratch/s224696943/IPS3D/16views/blender_script_v2.py")
OUTPUT_ROOT    = PGR_DIR / "data" / "gso30_eval_novelviews"

# ---------------------------------------------------------------------------
# Canonical 30 GSO objects (alphabetical)
# ---------------------------------------------------------------------------
GSO_OBJECTS_30 = [
    "alarm", "backpack", "bell", "blocks", "chicken",
    "cream", "elephant", "grandfather", "grandmother", "hat",
    "leather", "lion", "lunch_bag", "mario", "oil",
    "school_bus1", "school_bus2", "shoe", "shoe1", "shoe2",
    "shoe3", "soap", "sofa", "sorter", "sorting_board",
    "stucking_cups", "teapot", "toaster", "train", "turtle",
]

# Per-object (rotate_x, rotate_z) for canonical front-facing pose.
# Formula: rotate_z = 90 - IPS3D_base_angle  (NOT the base_angle itself).
# rotate_x = -90 for ALL GSO OBJ meshes (Y-up -> Z-up correction).
# Source: IPS3D/00.IMPORTANT_runIPS3D_mesh_B_plus_2.sh  get_angle_deg()
GT_ROTATION_X: float = -90.0   # constant for all GSO objects

GT_ROTATION_Z: dict[str, float] = {
    # base 270 -> rotate_z = 90 - 270 = -180
    "alarm": -180.0, "backpack": -180.0, "blocks": -180.0, "chicken": -180.0,
    "grandfather": -180.0, "grandmother": -180.0, "lion": -180.0,
    "lunch_bag": -180.0, "mario": -180.0, "oil": -180.0,
    "soap": -180.0, "sofa": -180.0,
    # base 225 -> rotate_z = 90 - 225 = -135
    "school_bus2": -135.0, "shoe": -135.0, "train": -135.0, "turtle": -135.0,
    # base 180 -> rotate_z = 90 - 180 = -90
    "elephant": -90.0, "school_bus1": -90.0,
    # base 112.5 -> rotate_z = 90 - 112.5 = -22.5
    "sorter": -22.5,
    # base 0 -> rotate_z = 90 - 0 = 90  (default for unlisted objects)
}

NUM_VIEWS    = 16
ELEVATION    = 0.0      # degrees
CAMERA_DIST  = 1.35
RESOLUTION   = 256
ENGINE       = "CYCLES"
CAMERA_TYPE  = "fixed"


# ---------------------------------------------------------------------------
# Per-object render
# ---------------------------------------------------------------------------

def render_object(obj: str, dry_run: bool = False) -> bool:
    """
    Render 16 views for `obj`.  Returns True on success.
    Renames blender output {i:03d}.png → view_{i:02d}.png.
    Idempotent: skips if view_15.png already exists.
    """
    mesh_path = GSO_MESH_ROOT / obj / "meshes" / "model.obj"
    if not mesh_path.exists():
        print(f"  SKIP  {obj}  (mesh not found: {mesh_path})", flush=True)
        return False

    out_dir = OUTPUT_ROOT / obj
    done_marker = out_dir / "view_15.png"
    if done_marker.exists():
        print(f"  SKIP  {obj}  (already rendered)", flush=True)
        return True

    rot_z = GT_ROTATION_Z.get(obj, 90.0)   # default 90 = 90 - 0 for base=0 objects

    cmd = [
        str(BLENDER_BIN), "-b", "-P", str(RENDER_SCRIPT), "--",
        "--object_path",  str(mesh_path),
        "--object_name",  obj,
        "--output_dir",   str(OUTPUT_ROOT),
        "--camera_type",  CAMERA_TYPE,
        "--num_images",   str(NUM_VIEWS),
        "--elevation",    str(ELEVATION),
        "--camera_dist",  str(CAMERA_DIST),
        "--resolution",   str(RESOLUTION),
        "--engine",       ENGINE,
        "--rotate_x",     str(GT_ROTATION_X),
        "--rotate_z",     str(rot_z),
    ]

    if dry_run:
        print(f"  DRY   {obj}  rx={GT_ROTATION_X}° rz={rot_z}°  cmd={' '.join(cmd)}", flush=True)
        return True

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT  {obj}", flush=True)
        return False

    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAIL  {obj}  ({elapsed:.1f}s)", flush=True)
        print(result.stderr[-500:] if result.stderr else "(no stderr)", flush=True)
        return False

    # Rename {i:03d}.png → view_{i:02d}.png
    obj_dir = OUTPUT_ROOT / obj
    renamed = 0
    for i in range(NUM_VIEWS):
        src = obj_dir / f"{i:03d}.png"
        dst = obj_dir / f"view_{i:02d}.png"
        if src.exists() and not dst.exists():
            src.rename(dst)
            renamed += 1
        elif dst.exists():
            renamed += 1  # already renamed (re-run safety)

    if renamed < NUM_VIEWS:
        print(f"  PARTIAL  {obj}  ({renamed}/{NUM_VIEWS} views, {elapsed:.1f}s)", flush=True)
        return False

    print(f"  PASS  {obj}  ({elapsed:.1f}s  rx={GT_ROTATION_X}° rz={rot_z}°)", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render 16 GT novel views per GSO object")
    parser.add_argument("--objects", nargs="+", default=None,
                        help="Subset of objects (default: all 30)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    objects = args.objects if args.objects else GSO_OBJECTS_30

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"=== GT novel-view render: {len(objects)} objects, {NUM_VIEWS} views each ===")
    print(f"    Blender:      {BLENDER_BIN}")
    print(f"    Render script:{RENDER_SCRIPT}")
    print(f"    Output:       {OUTPUT_ROOT}")
    print(f"    elevation={ELEVATION}°  camera_dist={CAMERA_DIST}  res={RESOLUTION}", flush=True)

    n_pass = n_fail = 0
    for obj in objects:
        ok = render_object(obj, dry_run=args.dry_run)
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\nDone: {n_pass} pass, {n_fail} fail out of {len(objects)} objects.")

    if not args.dry_run and n_fail > 0:
        # Tolerate up to 2 failures (mesh loading edge cases)
        if n_fail > 2:
            print(f"ERROR: {n_fail} failures exceed tolerance (>2). Exiting 1.", flush=True)
            sys.exit(1)
        else:
            print(f"WARNING: {n_fail} failure(s) within tolerance.", flush=True)


if __name__ == "__main__":
    main()
