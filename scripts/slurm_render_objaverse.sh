#!/bin/bash
#SBATCH --job-name=pgr_render_obja
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/render_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/render_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# ---------------------------------------------------------------------------
# Render Objaverse GLB → 6 orthographic views (256×256, white BG)
# Adjust MAX_OBJECTS and START_IDX for batched parallel rendering.
# ---------------------------------------------------------------------------

set -euo pipefail
source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
GLB_DIR="/scratch/s224696943/objaverse_30k/objaverse/hf-objaverse-v1/glbs"
CACHE_DIR="${PGR_DIR}/data/objaverse_renders"
BLENDER_BIN="/scratch/s224696943/blender/blender-3.3.0-linux-x64/blender"

mkdir -p "${CACHE_DIR}"
mkdir -p "${PGR_DIR}/outputs/logs"

export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"

MAX_OBJECTS=${MAX_OBJECTS:-200}   # override with: sbatch --export=MAX_OBJECTS=1000 ...
START_IDX=${START_IDX:-0}

echo "Rendering ${MAX_OBJECTS} objects starting at index ${START_IDX}"
echo "GLB source: ${GLB_DIR}"
echo "Cache dest: ${CACHE_DIR}"

python "${PGR_DIR}/src/data_pipeline.py" \
    --mode render \
    --glb_dir "${GLB_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --max_objects "${MAX_OBJECTS}" \
    --start_idx "${START_IDX}"

echo "Rendering complete."
