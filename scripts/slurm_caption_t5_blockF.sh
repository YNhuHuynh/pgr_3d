#!/bin/bash
#SBATCH --job-name=pgr_caption_blockF
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/caption_blockF_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/caption_blockF_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH -p gpu-large
#SBATCH --time=0-03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# Block F: Generate captions + T5 embeddings for all 230 objects.
#   - 30 GSO objects  (data/gso30_renders/{obj}/rgb_0.png)
#   - 200 Objaverse   (data/objaverse_200/{uid}/rgb_0.png)
#
# Outputs (per object):
#   data/gso30_renders/{obj}/caption.txt       + t5_emb.pt
#   data/objaverse_200/{uid}/caption.txt       + t5_emb.pt
#
# Both caption generation and T5 encoding run in unique3d env.
# Idempotent: skips objects with existing non-empty caption.txt / t5_emb.pt.
#
# Walltime estimate:
#   230 objects × ~8-10s/caption = ~32 min caption
#   230 × ~1s T5 encoding        = ~4  min T5
#   Model load                   = ~2  min (cached from Block C)
#   Total                        = ~40 min; 3h is safe.
#
# PREREQUISITE: Block C4 caption sanity gate passed (user-approved).
# DO NOT SUBMIT before gate.

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate unique3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/s224696943/.cache/huggingface"

mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D Block F: Caption + T5 (230 objects) ==="
echo "Started: $(date)   Node: $(hostname)"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Caption generation
# ---------------------------------------------------------------------------

echo "--- Step 1: Caption generation ---"
echo ""

echo "[GSO-30] Captioning from data/gso30_renders/ …"
python "${PGR_DIR}/src/caption_pipeline.py" \
    --image_dir  "${PGR_DIR}/data/gso30_renders" \
    --output_dir "${PGR_DIR}/data/gso30_renders" \
    --view 0 \
    --device_map "auto"

echo ""
echo "[Objaverse-200] Captioning from data/objaverse_200/ …"
python "${PGR_DIR}/src/caption_pipeline.py" \
    --image_dir  "${PGR_DIR}/data/objaverse_200" \
    --output_dir "${PGR_DIR}/data/objaverse_200" \
    --view 0 \
    --device_map "auto"

echo ""

# ---------------------------------------------------------------------------
# Step 2: T5 embedding
# ---------------------------------------------------------------------------

echo "--- Step 2: T5 embedding (google-t5/t5-base) ---"
echo ""

echo "[GSO-30] Encoding captions …"
python "${PGR_DIR}/src/text_encoder.py" \
    --caption_dir "${PGR_DIR}/data/gso30_renders" \
    --device cuda

echo ""
echo "[Objaverse-200] Encoding captions …"
python "${PGR_DIR}/src/text_encoder.py" \
    --caption_dir "${PGR_DIR}/data/objaverse_200" \
    --device cuda

echo ""

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

echo "--- Validation ---"

N_GSO_CAP=$(find "${PGR_DIR}/data/gso30_renders" -name "caption.txt" -size +0c 2>/dev/null | wc -l)
N_GSO_T5=$( find "${PGR_DIR}/data/gso30_renders" -name "t5_emb.pt"   2>/dev/null | wc -l)
N_OBJ_CAP=$(find "${PGR_DIR}/data/objaverse_200" -name "caption.txt" -size +0c 2>/dev/null | wc -l)
N_OBJ_T5=$( find "${PGR_DIR}/data/objaverse_200" -name "t5_emb.pt"   2>/dev/null | wc -l)

echo "GSO-30:       ${N_GSO_CAP}/30 captions,  ${N_GSO_T5}/30 t5_emb.pt"
echo "Objaverse-200: ${N_OBJ_CAP}/200 captions, ${N_OBJ_T5}/200 t5_emb.pt"

TOTAL_CAP=$(( N_GSO_CAP + N_OBJ_CAP ))
TOTAL_T5=$(( N_GSO_T5 + N_OBJ_T5 ))
echo "Total:        ${TOTAL_CAP}/230 captions, ${TOTAL_T5}/230 t5_emb.pt"
echo ""

echo "=== Done: $(date) ==="

# Fail loudly if too many missing (tolerate up to 5 per dataset for edge cases)
if [ "${N_GSO_CAP}" -lt 25 ] || [ "${N_GSO_T5}" -lt 25 ]; then
    echo "ERROR: GSO-30 incomplete (cap=${N_GSO_CAP}, t5=${N_GSO_T5}). Exiting 1."
    exit 1
fi
if [ "${N_OBJ_CAP}" -lt 195 ] || [ "${N_OBJ_T5}" -lt 195 ]; then
    echo "ERROR: Objaverse-200 incomplete (cap=${N_OBJ_CAP}, t5=${N_OBJ_T5}). Exiting 1."
    exit 1
fi

echo "Block F complete. Proceed to Block G (sanity training run)."
