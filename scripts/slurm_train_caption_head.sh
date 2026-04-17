#!/bin/bash
#SBATCH --job-name=pgr_train_caption
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/train_caption_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/train_caption_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# Block H: Full CaptionHead overnight training.
#
# Data:    data/objaverse_200/ — 200 objects, each with:
#            rgb_{0..5}.png  (6 rendered views, 256×256)
#            t5_emb.pt       (mean-pooled T5-base [768] embedding)
# Model:   Wonder3D UNet FROZEN; trains AggregationNetwork + CaptionHead
# Steps:   5000  (≈25 epochs over 200 objects at batch_size=4)
# Output:  checkpoints/caption_head_v1/
#            pgr_caption_v1_step500.pt  ... step5000.pt  (every 500 steps)
#            pgr_caption_v1_final.pt
#            pgr_caption_v1_log.csv     (step, loss, lr, time)
#
# Walltime estimate:
#   Wonder3D UNet forward (frozen): ~0.4s/batch on H100
#   5000 steps × 0.4s             = ~33 min
#   Data loading overhead          = ~10 min
#   Total                          = ~45 min; 8h is generous for safety.
#
# Resume: set RESUME_FROM to a checkpoint path to continue interrupted run.
#
# PREREQUISITES:
#   - Block F complete: t5_emb.pt exists in data/objaverse_200/ for all objects
#   - Verify with: find data/objaverse_200 -name t5_emb.pt | wc -l  (expect 200)
#   - Block G sanity run passed (10 objects, 100 steps, loss decreasing)
# DO NOT SUBMIT before Block G is approved.

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/s224696943/.cache/huggingface"
export WANDB_DIR="${PGR_DIR}/outputs/wandb"

mkdir -p "${PGR_DIR}/outputs/logs"
mkdir -p "${PGR_DIR}/outputs/wandb"
mkdir -p "${PGR_DIR}/outputs/checkpoints/caption_head_v1"

echo "=== PGR-3D Block H: CaptionHead Training ==="
echo "Started: $(date)   Node: $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Verify t5_emb.pt files exist before starting
N_T5=$(find "${PGR_DIR}/data/objaverse_200" -name "t5_emb.pt" 2>/dev/null | wc -l)
echo "t5_emb.pt files found: ${N_T5}/200"
if [ "${N_T5}" -lt 195 ]; then
    echo "ERROR: Insufficient t5_emb.pt files (${N_T5}/200). Run Block F first."
    exit 1
fi

# Optional resume
RESUME_FROM=""
# RESUME_FROM="${PGR_DIR}/outputs/checkpoints/caption_head_v1/pgr_caption_v1_step2500.pt"

RESUME_ARG=""
if [ -n "${RESUME_FROM}" ] && [ -f "${RESUME_FROM}" ]; then
    RESUME_ARG="--resume_from ${RESUME_FROM}"
    echo "Resuming from: ${RESUME_FROM}"
fi

echo ""

python "${PGR_DIR}/src/train_readout_caption.py" \
    --cache_dir        "${PGR_DIR}/data/objaverse_200" \
    --blacklist        "${PGR_DIR}/configs/objaverse_blacklist.txt" \
    --steps            5000 \
    --batch_size       4 \
    --lr               1e-4 \
    --weight_decay     0.01 \
    --checkpoint_every 500 \
    --run_name         "pgr_caption_v1" \
    --ckpt_dir         "${PGR_DIR}/outputs/checkpoints/caption_head_v1" \
    --wandb_mode       "online" \
    ${RESUME_ARG}

echo ""
echo "=== Done: $(date) ==="

# Validate final checkpoint exists
FINAL_CKPT="${PGR_DIR}/outputs/checkpoints/caption_head_v1/pgr_caption_v1_final.pt"
if [ ! -f "${FINAL_CKPT}" ]; then
    echo "ERROR: Final checkpoint not found at ${FINAL_CKPT}"
    exit 1
fi
echo "Final checkpoint: ${FINAL_CKPT}"

CSV_LOG="${PGR_DIR}/outputs/checkpoints/caption_head_v1/pgr_caption_v1_log.csv"
N_ROWS=$(tail -n +2 "${CSV_LOG}" 2>/dev/null | wc -l)
echo "CSV log rows: ${N_ROWS} (expect 5000)"
