#!/bin/bash
#SBATCH --job-name=pgr_eval_gso
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/eval_gso_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/eval_gso_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# ---------------------------------------------------------------------------
# PGR-3D GSO Evaluation
#
# Mode 1 — baseline (meshes already built by slurm_baseline_gso.sh):
#   sbatch scripts/slurm_eval_gso.sh
#   (uses defaults: MODE=metrics_only, SETUP=baseline)
#
# Mode 2 — PGR full run (generate -> reconstruct -> eval):
#   sbatch --export=ALL,MODE=full,SETUP=pgr_sem_eta1.0,ETA=1.0,\
#          HEAD_CKPT=/scratch/s224696943/pgr_3d/outputs/checkpoints/pgr_semantic_200_YYYYMMDD_final.pt \
#          scripts/slurm_eval_gso.sh
# ---------------------------------------------------------------------------

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
GT_DIR="/scratch/s224696943/3DRAV/evaluation/gso"
DATE=$(date +%Y%m%d)

# Defaults — override with --export=ALL,VAR=val at sbatch time
MODE="${MODE:-metrics_only}"
SETUP="${SETUP:-baseline}"
MESH_DIR="${MESH_DIR:-${PGR_DIR}/outputs/meshes_baseline}"
HEAD_TYPE="${HEAD_TYPE:-semantic}"
HEAD_CKPT="${HEAD_CKPT:-}"
ETA="${ETA:-0.0}"
T_GUIDANCE_MAX="${T_GUIDANCE_MAX:-800}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
NUM_STEPS="${NUM_STEPS:-50}"

CSV_OUT="${PGR_DIR}/outputs/metrics/${SETUP}_gso30_${DATE}.csv"

export PYTHONPATH="${PGR_DIR}/src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"
mkdir -p "${PGR_DIR}/outputs/metrics"

echo "=== PGR-3D GSO Evaluation ==="
echo "Mode:  ${MODE}"
echo "Setup: ${SETUP}"
echo "ETA:   ${ETA}"
echo "CSV:   ${CSV_OUT}"
echo "Started: $(date)"
echo ""

# Build Python argument list
PYARGS=(
    --mode           "${MODE}"
    --setup          "${SETUP}"
    --gt_dir         "${GT_DIR}"
    --csv_out        "${CSV_OUT}"
    --eta            "${ETA}"
    --head_type      "${HEAD_TYPE}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --num_steps      "${NUM_STEPS}"
    --t_guidance_max "${T_GUIDANCE_MAX}"
    --device         cuda
)
[[ "${MODE}" == "metrics_only" ]] && PYARGS+=(--mesh_dir "${MESH_DIR}")
[[ -n "${HEAD_CKPT}" ]]           && PYARGS+=(--head_ckpt "${HEAD_CKPT}")

python "${PGR_DIR}/src/eval_gso.py" "${PYARGS[@]}"

echo ""
echo "=== Done at: $(date) ==="
echo "Results: ${CSV_OUT}"
if [[ -f "${CSV_OUT}" ]]; then
    echo ""
    echo "--- Geometry metrics ---"
    column -t -s',' "${CSV_OUT}" | head -35
fi
