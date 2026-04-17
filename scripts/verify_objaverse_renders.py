#!/usr/bin/env python3
"""
PGR-3D — Verify Objaverse render cache completeness.

Checks every UID in the manifest for:
  - 6 RGB views:   rgb_0.png … rgb_5.png
  - CLIP embedding: clip_emb.pt

Prints a summary and flags any UIDs that need re-rendering.
Target: ≥ 8000 fully complete objects out of 8500.

Usage:
    python scripts/verify_objaverse_renders.py \\
        --render_dir data/objaverse_8500 \\
        --manifest   configs/objaverse_train_8500.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

N_VIEWS = 6


def check_object(obj_dir: Path) -> tuple[bool, str]:
    """Returns (complete, reason).  reason is empty string if complete."""
    missing = []
    for i in range(N_VIEWS):
        p = obj_dir / f"rgb_{i}.png"
        if not p.exists():
            missing.append(p.name)
    clip_p = obj_dir / "clip_emb.pt"
    if not clip_p.exists():
        missing.append("clip_emb.pt")
    if missing:
        return False, "missing: " + ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_dir", default="/scratch/s224696943/pgr_3d/data/objaverse_8500")
    parser.add_argument("--manifest",   default="/scratch/s224696943/pgr_3d/configs/objaverse_train_8500.txt")
    parser.add_argument("--show_incomplete", type=int, default=20,
                        help="Print up to N incomplete UIDs (0 = all)")
    args = parser.parse_args()

    render_dir = Path(args.render_dir)

    with open(args.manifest) as f:
        lines  = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    entries    = [l.split("\t") for l in lines]
    total      = len(entries)

    complete   = []
    incomplete = []

    for uid, glb_path in entries:
        ok, reason = check_object(render_dir / uid)
        if ok:
            complete.append(uid)
        else:
            incomplete.append((uid, reason))

    n_ok = len(complete)
    print(f"Render verification: {render_dir}")
    print(f"  Total in manifest: {total}")
    print(f"  Complete:          {n_ok}  ({100*n_ok/total:.1f}%)")
    print(f"  Incomplete:        {len(incomplete)}")

    if n_ok >= 8000:
        print(f"\n  STATUS: OK ✓  ({n_ok}/8500 ≥ 8000 threshold)")
    else:
        print(f"\n  STATUS: BELOW THRESHOLD ✗  ({n_ok}/8500 < 8000)")

    if incomplete:
        limit = len(incomplete) if args.show_incomplete == 0 else args.show_incomplete
        print(f"\nFirst {min(limit, len(incomplete))} incomplete UIDs:")
        for uid, reason in incomplete[:limit]:
            print(f"  {uid}  [{reason}]")

        # Write re-run manifest
        rerun_path = Path(args.manifest).parent / "objaverse_rerun_incomplete.txt"
        with open(rerun_path, "w") as f:
            f.write("# Incomplete objects — re-run targets\n")
            uid_set = {uid for uid, _ in incomplete}
            for uid, glb_path in entries:
                if uid in uid_set:
                    f.write(f"{uid}\t{glb_path}\n")
        print(f"\nRe-run manifest written: {rerun_path}")
        print("To re-render: replace configs/objaverse_train_8500.txt with this file")
        print("and re-submit the array job (cached objects will be skipped).")


if __name__ == "__main__":
    main()
