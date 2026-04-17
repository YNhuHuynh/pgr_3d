#!/usr/bin/env python3
"""
Render sanity check: first 30 objects from objaverse_train_8500.txt manifest.
Reports pass/fail per object and final summary.
"""
import sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_pipeline import render_object

MANIFEST = "/scratch/s224696943/pgr_3d/configs/objaverse_train_8500.txt"
CACHE_DIR = "/tmp/sanity_render_30"
N = 30

os.makedirs(CACHE_DIR, exist_ok=True)

entries = []
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        uid, glb_path = line.split("\t")
        entries.append((uid, glb_path))
        if len(entries) == N:
            break

print(f"Rendering {len(entries)} objects → {CACHE_DIR}", flush=True)
print("=" * 60, flush=True)

results = []
t_start = time.time()
for i, (uid, glb_path) in enumerate(entries):
    t0 = time.time()
    ok = render_object(glb_path, CACHE_DIR, uid)
    elapsed = time.time() - t0
    status = "PASS" if ok else "FAIL"
    print(f"[{i+1:02d}/{N}] {status}  {uid}  ({elapsed:.1f}s)", flush=True)
    results.append((uid, ok, elapsed))

total = time.time() - t_start
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = N - n_pass

print("=" * 60)
print(f"SUMMARY: {n_pass}/{N} passed, {n_fail}/{N} failed  (total {total:.0f}s)")
if n_fail:
    print("Failed UIDs:")
    for uid, ok, _ in results:
        if not ok:
            print(f"  {uid}")
sys.exit(0 if n_fail <= 2 else 1)
