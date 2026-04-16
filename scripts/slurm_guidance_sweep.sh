#!/bin/bash
#SBATCH --job-name=pgr_eta_sweep
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/eta_sweep_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/eta_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# ---------------------------------------------------------------------------
# PGR-3D Guidance Eta Sweep
#
# Runs Wonder3D + PGR guidance at eta in {0.0, 0.1, 0.5, 1.0, 2.0} for each
# GSO object, saves multi-view images.  Then calls eval_gso.py in
# metrics_only mode on the pre-built meshes (reconstruct separately).
#
# Prerequisite: trained head checkpoint at HEAD_CKPT.
#
# Submit:
#   sbatch --export=ALL,\
#     HEAD_CKPT=/scratch/s224696943/pgr_3d/outputs/checkpoints/pgr_semantic_200_YYYYMMDD_final.pt,\
#     HEAD_TYPE=semantic \
#     scripts/slurm_guidance_sweep.sh
# ---------------------------------------------------------------------------

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
DATE=$(date +%Y%m%d)

HEAD_TYPE="${HEAD_TYPE:-semantic}"
HEAD_CKPT="${HEAD_CKPT:?Must set HEAD_CKPT}"   # fail-fast if not set
T_MIN="${T_MIN:-0}"
T_MAX="${T_MAX:-800}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-42}"

ETA_VALUES=(0.0 0.1 0.5 1.0 2.0)

# Objects for the sweep (start with 5, expand after Day-10 gate)
OBJECTS="mario alarm chicken elephant turtle"

export PYTHONPATH="${PGR_DIR}/src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D Eta Sweep ==="
echo "Head type: ${HEAD_TYPE}"
echo "Head ckpt: ${HEAD_CKPT}"
echo "Eta sweep: ${ETA_VALUES[*]}"
echo "Objects:   ${OBJECTS}"
echo "Started:   $(date)"
echo ""

for OBJ in ${OBJECTS}; do
    FRONT="${PGR_DIR}/data/gso_fronts/${OBJ}.png"
    if [[ ! -f "${FRONT}" ]]; then
        echo "[WARN] Missing front image for ${OBJ}: ${FRONT}"
        continue
    fi

    echo "--- ${OBJ} ---"
    for ETA in "${ETA_VALUES[@]}"; do
        ETA_TAG=$(echo "${ETA}" | sed 's/\./p/')   # 1.0 -> 1p0
        OUT_DIR="${PGR_DIR}/outputs/eta_sweep/${OBJ}/eta_${ETA_TAG}"
        echo "  eta=${ETA} -> ${OUT_DIR}"

        python "${PGR_DIR}/src/guidance_inference.py" \
            --front_image    "${FRONT}" \
            --head_ckpt      "${HEAD_CKPT}" \
            --head_type      "${HEAD_TYPE}" \
            --eta            "${ETA}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --num_steps      "${NUM_STEPS}" \
            --t_min          "${T_MIN}" \
            --t_max          "${T_MAX}" \
            --output_dir     "${OUT_DIR}" \
            --device         cuda \
            --seed           "${SEED}"
    done
    echo ""
done

echo "=== Sweep complete at: $(date) ==="
echo "Generated views in: ${PGR_DIR}/outputs/eta_sweep/"
echo ""
echo "Next step: submit 3D reconstruction for each eta dir, then run:"
echo "  sbatch --export=ALL,MODE=metrics_only,SETUP=eta_X.X scripts/slurm_eval_gso.sh"
