#!/bin/bash
#SBATCH --job-name=pgr_render_gso30_eval
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/render_gso30_eval_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/render_gso30_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -p gpu
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# Block A-extension: Render 16 GT novel views per GSO object.
# 30 objects × 16 views × ~10s each ≈ 80 min → 2h walltime is safe.
#
# Output: /scratch/s224696943/pgr_3d/data/gso30_eval_novelviews/{obj}/view_{00..15}.png
# Purpose: PSNR/SSIM/LPIPS ground truth for NVS evaluation.
#
# Prerequisites:
#   - Block C4 caption sanity gate passed (user approval obtained)
#   - BlenderProc/Blender environment verified (Block A GSO renders: 30/30 PASS)
# DO NOT SUBMIT before C4 gate.

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D GT Novel-View Render (GSO-30, 16 views) ==="
echo "Started: $(date)   Node: $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

python "${PGR_DIR}/scripts/render_gso30_eval_novelviews.py"

echo ""
echo "=== Done: $(date) ==="
echo "Outputs: ${PGR_DIR}/data/gso30_eval_novelviews/"

# Count completed objects
N_COMPLETE=$(find "${PGR_DIR}/data/gso30_eval_novelviews" -name "view_15.png" 2>/dev/null | wc -l)
echo "Objects with all 16 views: ${N_COMPLETE}/30"

if [ "${N_COMPLETE}" -lt 28 ]; then
    echo "ERROR: Only ${N_COMPLETE}/30 objects complete (threshold: 28). Exiting 1."
    exit 1
fi
