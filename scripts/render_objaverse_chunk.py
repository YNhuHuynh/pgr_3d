#!/usr/bin/env python3
"""
PGR-3D Objaverse rendering chunk worker
----------------------------------------
For each object in the assigned manifest chunk:

  1. BlenderProc renders 6 orthographic views (CPU) → rgb_0.png … rgb_5.png
     Uses the exact same camera config as Wonder3D (azimuth 0/45/90/135/180/225,
     elevation 0, ortho_scale=1.35, 256×256).

  2. CLIP ViT-L/14 encodes view 0 (GPU) → clip_emb.pt  [1, 768]

Output per object: {render_dir}/{uid}/rgb_0.png … rgb_5.png + clip_emb.pt

Naming is compatible with ObjaverseRenderDataset in src/data_pipeline.py.

NOTE: Depth and camera_params are NOT pre-computed here.
- Depth (MiDaS): computed on-the-fly during training (see data_pipeline.compute_midas_depth).
- Camera params: fixed constants in data_pipeline.WONDER3D_AZIMUTHS / ELEVATIONS.

Usage (via SLURM array):
    python scripts/render_objaverse_chunk.py \\
        --manifest  configs/objaverse_train_8500.txt \\
        --chunk_id  $SLURM_ARRAY_TASK_ID \\
        --chunk_size 200 \\
        --render_dir /scratch/s224696943/pgr_3d/data/objaverse_8500 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from PIL import Image

from data_pipeline import render_object, get_clip, N_VIEWS   # N_VIEWS = 6

RENDER_DIR_DEFAULT = "/scratch/s224696943/pgr_3d/data/objaverse_8500"
MANIFEST_DEFAULT   = "/scratch/s224696943/pgr_3d/configs/objaverse_train_8500.txt"


# ---------------------------------------------------------------------------
# CLIP encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_clip_view0(uid: str, render_dir: Path, device: str) -> bool:
    """Load rgb_0.png, CLIP-encode, save clip_emb.pt.  Returns True on success."""
    clip_path = render_dir / uid / "clip_emb.pt"
    if clip_path.exists():
        return True
    rgb_path = render_dir / uid / "rgb_0.png"
    if not rgb_path.exists():
        return False
    try:
        model, preproc = get_clip(device)
        pil    = Image.open(rgb_path).convert("RGB")
        tensor = preproc(pil).unsqueeze(0).to(device)
        emb    = F.normalize(model.encode_image(tensor).float(), dim=-1)  # [1, 768]
        torch.save(emb.cpu(), clip_path)
        return True
    except Exception as e:
        print(f"  [WARN] CLIP encode failed for {uid}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render Objaverse chunk")
    parser.add_argument("--manifest",    default=MANIFEST_DEFAULT)
    parser.add_argument("--chunk_id",    type=int, required=True,
                        help="SLURM_ARRAY_TASK_ID (0-indexed)")
    parser.add_argument("--chunk_size",  type=int, default=200,
                        help="Objects per chunk (default 200)")
    parser.add_argument("--render_dir",  default=RENDER_DIR_DEFAULT)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    # ---- Read manifest ----
    with open(args.manifest) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    entries = [l.split("\t") for l in lines]

    start = args.chunk_id * args.chunk_size
    end   = min(start + args.chunk_size, len(entries))
    my_entries = entries[start:end]

    if not my_entries:
        print(f"Chunk {args.chunk_id}: no entries (start={start} >= {len(entries)}). Nothing to do.")
        return

    render_dir = Path(args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Chunk {args.chunk_id:03d}: objects {start}–{end-1} ({len(my_entries)} items)")
    print(f"Manifest:   {args.manifest}")
    print(f"Render dir: {render_dir}")
    print(f"Device:     {args.device}")
    print(f"{'='*60}\n")

    # Pre-load CLIP once per chunk (amortises model load over all objects)
    print("Loading CLIP ViT-L/14 ...", end="", flush=True)
    get_clip(args.device)
    print(" done.\n")

    n_ok = n_render_fail = n_clip_fail = n_cached = 0
    log_lines: list[str] = []

    for rank, row in enumerate(my_entries):
        if len(row) < 2:
            print(f"[{rank+1}/{len(my_entries)}] malformed manifest row: {row}")
            continue
        uid, glb_path = row[0], row[1]
        t0 = time.time()

        obj_dir    = render_dir / uid
        has_rgb    = all((obj_dir / f"rgb_{i}.png").exists() for i in range(N_VIEWS))
        has_clip   = (obj_dir / "clip_emb.pt").exists()

        if has_rgb and has_clip:
            n_ok += 1
            n_cached += 1
            log_lines.append(f"CACHED\t{uid}")
            if (rank + 1) % 50 == 0:
                print(f"[{rank+1}/{len(my_entries)}] {uid} — cached (showing every 50)")
            continue

        print(f"[{rank+1}/{len(my_entries)}] {uid}", end=" ... ", flush=True)

        # ---- Step 1: BlenderProc render ----
        if not has_rgb:
            ok = render_object(glb_path, str(render_dir), uid)
            if not ok:
                print("RENDER FAILED")
                n_render_fail += 1
                log_lines.append(f"RENDER_FAIL\t{uid}\t{glb_path}")
                continue

        # ---- Step 2: CLIP embed view 0 ----
        if not has_clip:
            ok = encode_clip_view0(uid, render_dir, args.device)
            if not ok:
                print("CLIP FAILED")
                n_clip_fail += 1
                log_lines.append(f"CLIP_FAIL\t{uid}")
                continue

        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s)")
        n_ok += 1
        log_lines.append(f"OK\t{uid}\t{elapsed:.1f}s")

    # ---- Summary ----
    total = len(my_entries)
    print(f"\n{'='*60}")
    print(f"Chunk {args.chunk_id:03d} complete")
    print(f"  OK (incl. cached): {n_ok}/{total}")
    print(f"  Newly rendered:    {n_ok - n_cached}")
    print(f"  Cached:            {n_cached}")
    print(f"  Render failures:   {n_render_fail}")
    print(f"  CLIP failures:     {n_clip_fail}")
    print(f"{'='*60}")

    # ---- Per-chunk log ----
    log_path = render_dir / f"chunk_{args.chunk_id:03d}_log.txt"
    with open(log_path, "w") as f:
        f.write(f"# chunk={args.chunk_id}  entries={start}-{end-1}\n")
        f.write(f"# ok={n_ok}  render_fail={n_render_fail}  clip_fail={n_clip_fail}  cached={n_cached}\n")
        for l in log_lines:
            f.write(l + "\n")
    print(f"Log written: {log_path}")

    # ---- Fail-fast if success rate too low ----
    if n_ok < total * 0.85:
        print(f"\n[ERROR] Success rate {n_ok/total:.1%} < 85% — inspect log and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
