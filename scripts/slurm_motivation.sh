#!/bin/bash
#SBATCH --job-name=pgr_motivation
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/motivation_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/motivation_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=60GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# ---------------------------------------------------------------------------
# PGR-3D Day-3 Motivation Experiment
# Measures CLIP-cosine-similarity drift during Wonder3D denoising on 5 GSO
# objects.  Read outputs/motivation/clip_drift_summary.png after completion.
#
# Submit from project root:
#   cd /scratch/s224696943/pgr_3d && sbatch scripts/slurm_motivation.sh
#
# Gate (Day 3 — Apr 19):
#   Check outputs/motivation/clip_drift_data.csv
#   If total drift > 0.05 → hypothesis supported, continue.
#   If drift ≤ 0.01 → ESCALATE TO USER before proceeding.
# ---------------------------------------------------------------------------

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"

export PYTHONPATH="${PGR_DIR}/src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"
mkdir -p "${PGR_DIR}/outputs/motivation"

echo "=== PGR-3D Motivation Experiment ==="
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

python "${PGR_DIR}/src/motivation_experiment.py" \
    --objects mario alarm chicken elephant turtle \
    --gso_fronts "${PGR_DIR}/data/gso_fronts" \
    --output_dir "${PGR_DIR}/outputs/motivation" \
    --num_steps 50 \
    --guidance_scale 3.0 \
    --device cuda

echo ""
echo "=== Experiment complete at: $(date) ==="
echo ""
echo "Check the gate decision above (DRIFT DETECTED / MARGINAL / NO DRIFT)."
echo "Summary plot: ${PGR_DIR}/outputs/motivation/clip_drift_summary.png"
echo "Raw data CSV: ${PGR_DIR}/outputs/motivation/clip_drift_data.csv"
echo ""
echo "ACTION: Review and make go/no-go decision before Day 3 (Apr 19) deadline."
