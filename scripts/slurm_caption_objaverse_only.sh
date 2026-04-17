#!/bin/bash
#SBATCH --job-name=pgr_caption_obj200
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/caption_obj200_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/caption_obj200_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH -p gpu-large
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# Caption + T5 for Objaverse-200 ONLY (not GSO).
# Run AFTER slurm_caption_gso30.sh (job 71835) completes.
#
# Idempotent: skips objects with existing non-empty caption.txt.
# Blacklisted UIDs (configs/objaverse_blacklist.txt) have no rgb_0.png
# and will be skipped automatically with a FAIL message (harmless).
#
# Input:  data/objaverse_200/{uid}/rgb_0.png  (174 valid, 11 blacklisted)
# Output: data/objaverse_200/{uid}/caption.txt + t5_emb.pt
#
# Expected: ~105 new captions (174 valid - 69 existing), 11 graceful FAILs
# Walltime: 105 × ~8s + 2min load + T5 ≈ ~20 min; 2h is safe.

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate unique3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/s224696943/.cache/huggingface"

mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D Objaverse-200 Caption + T5 ==="
echo "Started: $(date)   Node: $(hostname)"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
echo ""

# Pre-flight: verify we have valid renders to caption
N_VALID=$(find "${PGR_DIR}/data/objaverse_200" -name "rgb_0.png" -size +5k 2>/dev/null | wc -l)
echo "Valid rgb_0.png (>5KB): ${N_VALID}/200"
if [ "${N_VALID}" -lt 150 ]; then
    echo "ERROR: Too few valid renders (${N_VALID}). Re-run render job first."
    exit 1
fi

# --- Step 1: Caption generation ---
echo ""
echo "--- Step 1: Caption generation (Qwen2.5-VL-72B) ---"
python "${PGR_DIR}/src/caption_pipeline.py" \
    --image_dir  "${PGR_DIR}/data/objaverse_200" \
    --output_dir "${PGR_DIR}/data/objaverse_200" \
    --view 0 \
    --device_map "auto"

echo ""

# --- Step 2: T5 embedding ---
echo "--- Step 2: T5 encoding (google-t5/t5-base) ---"
python "${PGR_DIR}/src/text_encoder.py" \
    --caption_dir "${PGR_DIR}/data/objaverse_200" \
    --device cuda

echo ""
echo "--- Validation ---"
N_CAP=$(find "${PGR_DIR}/data/objaverse_200" -name "caption.txt" -size +0c 2>/dev/null | wc -l)
N_T5=$( find "${PGR_DIR}/data/objaverse_200" -name "t5_emb.pt"   2>/dev/null | wc -l)
echo "Objaverse-200: ${N_CAP}/174 captions,  ${N_T5}/174 t5_emb.pt"
echo "  (11 blacklisted UIDs are excluded from counts)"
echo ""
echo "=== Done: $(date) ==="

# Expect ≥165 (174 valid - up to 9 tolerance for edge cases)
if [ "${N_T5}" -lt 165 ]; then
    echo "ERROR: Insufficient t5_emb.pt files (${N_T5}/174). Check logs."
    exit 1
fi
echo "Objaverse caption + T5 complete. Ready for Block G."
