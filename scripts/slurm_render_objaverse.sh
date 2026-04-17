#!/bin/bash
#SBATCH --job-name=pgr_objaverse_render
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/objaverse_render_%A_%a.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/objaverse_render_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -p gpu-large
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=32GB
#SBATCH --array=0-42
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# ---------------------------------------------------------------------------
# PGR-3D Objaverse Rendering — SLURM Array Job  (43 tasks × 200 objects)
#
# Renders 8500 Objaverse objects for readout head training.
# 43 array tasks × 200 objects = 8600 slots; the last task (ID 42)
# processes objects 8400-8499 (100 items).  All others process 200.
#
# Per object:
#   1. BlenderProc (CPU, 8 cores) renders 6 orthographic views
#      → {uid}/rgb_0.png … rgb_5.png
#      Camera: azimuths 0/45/90/135/180/225°, elevation 0°,
#              ortho_scale=1.35, 256×256, white background.
#   2. CLIP ViT-L/14 (GPU) encodes view 0
#      → {uid}/clip_emb.pt  [1, 768] L2-normalised
#
# Output: /scratch/s224696943/pgr_3d/data/objaverse_8500/{uid}/
# Compatible with ObjaverseRenderDataset in src/data_pipeline.py.
# MiDaS depth is NOT pre-computed (done on-the-fly during training).
#
# Manifest: configs/objaverse_train_8500.txt
#   Rows 0-29:    motivation-30 objects (superset containment enforced)
#   Rows 30-8499: training-only objects (seed 123, sampled from remainder)
#
# Timing:  BlenderProc ~30-120s/object on 8 CPU cores.
#          200 objects × 60s avg ≈ 3.3h; worst-case ~6.7h → safe under 10h.
#
# Each task writes a per-chunk log:
#   data/objaverse_8500/chunk_NNN_log.txt
#
# Verification after all tasks complete:
#   python scripts/verify_objaverse_renders.py \
#       --render_dir data/objaverse_8500 \
#       --manifest   configs/objaverse_train_8500.txt
#
# Re-run failed chunks (rendered objects are cached — no recomputation):
#   sbatch --array=<failed_ids> scripts/slurm_render_objaverse.sh
#
# Submit from project root:
#   cd /scratch/s224696943/pgr_3d && sbatch scripts/slurm_render_objaverse.sh
# ---------------------------------------------------------------------------

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"
mkdir -p "${PGR_DIR}/data/objaverse_8500"

CHUNK_START=$(( SLURM_ARRAY_TASK_ID * 200 ))
CHUNK_END=$(( (SLURM_ARRAY_TASK_ID + 1) * 200 - 1 ))

echo "=== PGR-3D Objaverse Render  [array task ${SLURM_ARRAY_TASK_ID}/42] ==="
echo "Started at:  $(date)"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "Objects:     ${CHUNK_START}–${CHUNK_END} (manifest rows)"
echo ""

python "${PGR_DIR}/scripts/render_objaverse_chunk.py" \
    --manifest    "${PGR_DIR}/configs/objaverse_train_8500.txt" \
    --chunk_id    "${SLURM_ARRAY_TASK_ID}" \
    --chunk_size  200 \
    --render_dir  "${PGR_DIR}/data/objaverse_8500" \
    --device      cuda

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete at: $(date) ==="
