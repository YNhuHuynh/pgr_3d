#!/bin/bash
#SBATCH --job-name=pgr_train_dep
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/train_dep_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/train_dep_%j.err
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
# Train DepthHead (Day 6)
# ---------------------------------------------------------------------------

set -euo pipefail
source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
CACHE_DIR="${PGR_DIR}/data/objaverse_renders"
CKPT_DIR="${PGR_DIR}/outputs/checkpoints"
DATE=$(date +%Y%m%d)

mkdir -p "${PGR_DIR}/outputs/logs"
export PYTHONPATH="${PGR_DIR}/src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D:${PYTHONPATH:-}"

python "${PGR_DIR}/src/train_readout.py" \
    --head depth \
    --cache_dir "${CACHE_DIR}" \
    --max_objects 200 \
    --steps 5000 \
    --batch_size 4 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --run_name "pgr_depth_200_${DATE}" \
    --ckpt_dir "${CKPT_DIR}" \
    --wandb_mode online

echo "DepthHead training complete."
