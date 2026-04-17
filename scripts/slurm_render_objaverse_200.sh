#!/bin/bash
#SBATCH --job-name=pgr_render_200
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/render_200_%A_%a.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/render_200_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -p gpu
#SBATCH --time=0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# Render 200 Objaverse objects in 4 chunks of 50.
# Array task $SLURM_ARRAY_TASK_ID ∈ {0,1,2,3} → objects [i*50 : (i+1)*50]

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"

CHUNK_ID=${SLURM_ARRAY_TASK_ID}
CHUNK_SIZE=50
START=$(( CHUNK_ID * CHUNK_SIZE ))

MANIFEST="${PGR_DIR}/configs/objaverse_train_200.txt"
OUTPUT_DIR="${PGR_DIR}/data/objaverse_200"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D Objaverse Render — chunk ${CHUNK_ID} (objects ${START}..$(( START + CHUNK_SIZE - 1 ))) ==="
echo "Started: $(date)   Node: $(hostname)"

/scratch/s224696943/.conda/envs/pgr3d/bin/python - << PYEOF
import sys, time
from pathlib import Path

sys.path.insert(0, "${PGR_DIR}/src")
from data_pipeline import render_object, N_VIEWS

MANIFEST   = Path("${MANIFEST}")
OUTPUT_DIR = Path("${OUTPUT_DIR}")
CHUNK_ID   = ${CHUNK_ID}
CHUNK_SIZE = ${CHUNK_SIZE}
START      = CHUNK_ID * CHUNK_SIZE

entries = []
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        uid, glb_path = line.split("\t")
        entries.append((uid, glb_path))

chunk = entries[START : START + CHUNK_SIZE]
print(f"Chunk {CHUNK_ID}: {len(chunk)} objects", flush=True)

n_pass, n_fail = 0, 0
for i, (uid, glb_path) in enumerate(chunk):
    t0 = time.time()
    ok = render_object(glb_path, str(OUTPUT_DIR), uid)
    elapsed = time.time() - t0
    status = "PASS" if ok else "FAIL"
    print(f"  [{START+i+1:03d}/200] {status}  {uid}  ({elapsed:.1f}s)", flush=True)
    if ok: n_pass += 1
    else:  n_fail += 1

print(f"Chunk {CHUNK_ID} done: {n_pass} pass, {n_fail} fail", flush=True)
success_rate = n_pass / len(chunk)
sys.exit(0 if success_rate >= 0.85 else 1)
PYEOF

echo "=== Done: $(date) ==="
