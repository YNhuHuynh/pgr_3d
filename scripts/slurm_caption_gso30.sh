#!/bin/bash
#SBATCH --job-name=pgr_caption_gso30
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/caption_gso30_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/caption_gso30_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# Caption + T5 for 30 GSO objects only.
# Input: data/gso30_renders/{obj}/rgb_0.png  (pre-verified front views from gso_fronts/)
# Output: data/gso30_renders/{obj}/caption.txt + t5_emb.pt
# Idempotent: skips objects with existing non-empty caption.txt.
#
# DO NOT touch data/objaverse_200/ — those captions are handled separately.
#
# Walltime estimate: 30 objects × ~8s/caption + model load ~2min = ~8 min; 1h is safe.

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate unique3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/s224696943/.cache/huggingface"

mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D GSO-30 Caption + T5 ==="
echo "Started: $(date)   Node: $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# --- Step 1: Caption generation ---
echo "--- Step 1: Caption generation (Qwen2.5-VL-72B) ---"
python "${PGR_DIR}/src/caption_pipeline.py" \
    --image_dir  "${PGR_DIR}/data/gso30_renders" \
    --output_dir "${PGR_DIR}/data/gso30_renders" \
    --view 0 \
    --device_map "auto"

echo ""

# --- Step 2: T5 embedding ---
echo "--- Step 2: T5 encoding (google-t5/t5-base) ---"
python "${PGR_DIR}/src/text_encoder.py" \
    --caption_dir "${PGR_DIR}/data/gso30_renders" \
    --device cuda

echo ""
echo "--- Validation ---"
N_CAP=$(find "${PGR_DIR}/data/gso30_renders" -name "caption.txt" -size +0c 2>/dev/null | wc -l)
N_T5=$( find "${PGR_DIR}/data/gso30_renders" -name "t5_emb.pt"   2>/dev/null | wc -l)
echo "GSO-30: ${N_CAP}/30 captions,  ${N_T5}/30 t5_emb.pt"

echo ""
echo "=== Done: $(date) ==="

if [ "${N_CAP}" -lt 28 ] || [ "${N_T5}" -lt 28 ]; then
    echo "ERROR: Too few outputs (cap=${N_CAP}, t5=${N_T5}). Expected ≥28/30."
    exit 1
fi
echo "GSO-30 caption + T5 complete."
