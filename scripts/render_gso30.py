#!/usr/bin/env python3
"""
Render the canonical FRONT VIEW for all 30 GSO objects (single image per object).

Wonder3D takes 1 image as input; GSO objects are used as test cases,
not training data.  We render only azimuth=0 (the front) after applying
per-object (X, Z) rotations from configs/gso30_per_object_angles.yaml.

Source:  /scratch/s224696943/GSO/gso/{obj}/meshes/model.obj
Output:  /scratch/s224696943/pgr_3d/data/gso30_renders/{obj}/rgb_0.png
         (single 256×256 front view; any stale rgb_{1-5}.png are deleted)

Rotation formula:
    rotate_x = -90            (Y-up OBJ -> Blender Z-up)
    rotate_z = -base_angle    (aligns front face with camera at azimuth=0)
    Source: IPS3D get_angle_deg() — see configs/gso30_per_object_angles.yaml
"""
import sys, time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_pipeline import render_object

PGR_DIR    = Path(__file__).parent.parent
GSO_ROOT   = Path("/scratch/s224696943/GSO/gso")
OUTPUT_DIR = PGR_DIR / "data" / "gso30_renders"
ANGLE_CFG  = PGR_DIR / "configs" / "gso30_per_object_angles.yaml"

OBJECTS    = [d.name for d in sorted(GSO_ROOT.iterdir()) if d.is_dir()]

# Load per-object rotation config
with open(ANGLE_CFG) as fh:
    _cfg = yaml.safe_load(fh)
OBJECT_ANGLES: dict = _cfg.get("objects", {})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Rendering front view for {len(OBJECTS)} GSO objects → {OUTPUT_DIR}", flush=True)
print(f"Rotation config: {ANGLE_CFG}", flush=True)
print("=" * 60, flush=True)

results = []
t_start = time.time()
for i, obj in enumerate(OBJECTS):
    obj_path = GSO_ROOT / obj / "meshes" / "model.obj"
    if not obj_path.exists():
        print(f"[{i+1:02d}/{len(OBJECTS)}] SKIP  {obj}  (no model.obj)", flush=True)
        results.append((obj, False, 0.0))
        continue

    obj_dir  = OUTPUT_DIR / obj
    angles   = OBJECT_ANGLES.get(obj, {"rotate_x": -90.0, "rotate_z": 0.0})
    rotate_x = float(angles.get("rotate_x", -90.0))
    rotate_z = float(angles.get("rotate_z",   0.0))

    # Delete stale multi-view renders from earlier passes (views 1-5 are not used)
    for v in range(1, 6):
        stale = obj_dir / f"rgb_{v}.png"
        if stale.exists():
            stale.unlink()

    t0 = time.time()
    ok = render_object(str(obj_path), str(OUTPUT_DIR), obj,
                       rotate_x=rotate_x, rotate_z=rotate_z,
                       num_views=1)
    elapsed = time.time() - t0
    status = "PASS" if ok else "FAIL"
    print(f"[{i+1:02d}/{len(OBJECTS)}] {status}  {obj}  rx={rotate_x}° rz={rotate_z}°  ({elapsed:.1f}s)", flush=True)
    results.append((obj, ok, elapsed))

total = time.time() - t_start
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(OBJECTS) - n_pass

print("=" * 60)
print(f"SUMMARY: {n_pass}/{len(OBJECTS)} passed, {n_fail} failed  (total {total:.0f}s)")
if n_fail:
    print("Failed:")
    for obj, ok, _ in results:
        if not ok:
            print(f"  {obj}")
sys.exit(0 if n_pass >= 28 else 1)
