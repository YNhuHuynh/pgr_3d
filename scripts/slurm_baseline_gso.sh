#!/bin/bash
#SBATCH --job-name=pgr_baseline_gso
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/baseline_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/baseline_%j.err
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
# PGR-3D: Wonder3D Baseline on GSO-5 (Day 1 sanity check)
# Then expand to GSO-30 once confirmed working.
# ---------------------------------------------------------------------------

set -euo pipefail

# Activate conda env
source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

# Paths
PGR_DIR="/scratch/s224696943/pgr_3d"
WONDER3D_DIR="/scratch/s224696943/Wonder3D"
TRAV_DIR="/scratch/s224696943/3DRAV"
MODEL_DIR="/scratch/s224696943/wonder3d-v1.0"
RESULTS_DIR="${PGR_DIR}/outputs/wonder3d_baseline"
MESH_RESULTS_DIR="${PGR_DIR}/outputs/meshes_baseline"
METRICS_CSV="${PGR_DIR}/outputs/metrics/baseline_gso5.csv"

# Objects to run (Day 1: 5 objects for sanity check)
OBJECTS="mario alarm chicken elephant turtle"

# Ensure Python can find mvdiffusion package from 3DRAV
export PYTHONPATH="${TRAV_DIR}:${WONDER3D_DIR}:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"
mkdir -p "${MESH_RESULTS_DIR}"
mkdir -p "$(dirname ${METRICS_CSV})"

# ---------------------------------------------------------------------------
# Step 1: Generate per-object configs
# ---------------------------------------------------------------------------
echo "=== Step 1: Generating GSO configs ==="
python "${PGR_DIR}/scripts/gen_gso_configs.py" --objects ${OBJECTS}

# ---------------------------------------------------------------------------
# Step 2: Run Wonder3D inference (multi-view generation)
# ---------------------------------------------------------------------------
cd "${WONDER3D_DIR}"

for OBJ in ${OBJECTS}; do
    echo ""
    echo "=== Step 2: Wonder3D inference for ${OBJ} ==="
    CONFIG="${PGR_DIR}/configs/gso_${OBJ}.yaml"
    SAVE_DIR="${RESULTS_DIR}/${OBJ}"
    mkdir -p "${SAVE_DIR}"

    accelerate launch \
        --config_file "${WONDER3D_DIR}/configs/1gpu.yaml" \
        "${TRAV_DIR}/test_mvdiffusion_seq_add.py" \
        --config "${CONFIG}"

    echo "  Done Wonder3D inference for ${OBJ}"
done

# ---------------------------------------------------------------------------
# Step 3: 3D Reconstruction (instant-nsr-pl)
# ---------------------------------------------------------------------------
for OBJ in ${OBJECTS}; do
    echo ""
    echo "=== Step 3: 3D reconstruction for ${OBJ} ==="
    RUN_DIR="${RESULTS_DIR}/${OBJ}"

    python "${TRAV_DIR}/scripts/run_3d_pipeline_zero.py" \
        --run_dir "${RUN_DIR}"

    # Expected output: /scratch/s224696943/3DRAV_ext/evaluation/results_zero/${OBJ}_B.obj
    # Copy to our project results dir
    SRC_MESH="/scratch/s224696943/3DRAV_ext/evaluation/results_zero/${OBJ}_B.obj"
    if [ -f "${SRC_MESH}" ]; then
        cp "${SRC_MESH}" "${MESH_RESULTS_DIR}/${OBJ}_baseline.obj"
        echo "  Mesh saved: ${MESH_RESULTS_DIR}/${OBJ}_baseline.obj"
    else
        echo "  [WARN] Mesh not found at ${SRC_MESH}"
    fi
done

# ---------------------------------------------------------------------------
# Step 4: Evaluate geometry metrics (Chamfer Distance, Volume IoU)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Geometry metrics ==="
cd "${TRAV_DIR}"

# Rotation adjustments (from original 000.run_won3d.sh)
get_angle_deg() {
    local name="$1"
    case "$name" in
        alarm|backpack|chicken|lion|lunch_bag|mario|oil) echo "270" ;;
        elephant|school_bus1) echo "180" ;;
        school_bus2|shoe|train|turtle) echo "225" ;;
        sofa) echo "270" ;;
        *) echo "0" ;;
    esac
}

for OBJ in ${OBJECTS}; do
    PRED="${MESH_RESULTS_DIR}/${OBJ}_baseline.obj"
    GT="/scratch/s224696943/3DRAV/evaluation/gso/${OBJ}/meshes/model.obj"

    if [[ ! -f "${PRED}" ]]; then
        echo "[WARN] Missing pred mesh: ${PRED}"
        continue
    fi

    BASE_ANGLE="$(get_angle_deg ${OBJ})"
    ANGLE=$(awk -v a="${BASE_ANGLE}" 'BEGIN{printf "%.10g", 90-a}')

    python evaluation/iou.py \
        --pred "${PRED}" \
        --gt "${GT}" \
        --object "${OBJ}" --setup "baseline" --name "${OBJ}_baseline" \
        --pre_rot_pred "x:90,y:0,z:0" \
        --pre_rot_gt   "x:0,y:0,z:${ANGLE}" \
        --normalize unit-diag \
        --eval_space gt_bbox --vox_res 256 --pad_vox 0 \
        --icp_iters 400 \
        --csv_out "${METRICS_CSV}"

    echo "  Metrics written for ${OBJ}"
done

echo ""
echo "=== Baseline complete. Results in ${METRICS_CSV} ==="
cat "${METRICS_CSV}" 2>/dev/null || echo "(CSV not found)"
