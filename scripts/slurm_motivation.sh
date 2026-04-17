#!/bin/bash
#SBATCH --job-name=pgr_motivation
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/motivation_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/motivation_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=80GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# ---------------------------------------------------------------------------
# PGR-3D Day-3 Motivation Experiment  (v3 — dual pass + bootstrap CIs)
#
# Two-pass experiment:
#   Pass 1 — GSO-30 (primary evaluation set, ~3h)
#     Probes pre-CD and post-CD at 3 up_blocks, 50 steps, 30 objects.
#     Computes bootstrap 95% CIs (200 resamples) on input-front CKA.
#
#   Pass 2 — Objaverse-30 (OOD robustness check, ~4h incl. BlenderProc render)
#     Samples 30 Objaverse GLBs (seed=42), renders with BlenderProc,
#     runs the same probe — confirms drift is not GSO-specific.
#
# Produces:
#   outputs/motivation/gso30/
#     clip_drift_summary.png          — CLIP cosine sim vs timestep
#     cka_input_front.png             — PRIMARY metric with bootstrap CI bands
#     cka_decoded_all.png / cka_per_view.png — secondary metrics
#     cka_data.csv / clip_drift_data.csv
#   outputs/motivation/objaverse30/   — same structure for OOD objects
#
# Gate logic (see docs/motivation_evidence.md):
#   SCENARIO A or B (both/post-CD drift) → green light, proceed
#   SCENARIO C or D (pre drifts more, or no drift) → ESCALATE before proceeding
#
# Quick test (5 objects, skip Objaverse, no CIs):
#   sbatch --export=OBJECTS="mario alarm chicken elephant turtle" \
#          scripts/slurm_motivation.sh \
#          -- --skip_objaverse --n_bootstrap 0
#
# Submit from project root:
#   cd /scratch/s224696943/pgr_3d && sbatch scripts/slurm_motivation.sh
# ---------------------------------------------------------------------------

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"

export PYTHONPATH="${PGR_DIR}/src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"
mkdir -p "${PGR_DIR}/outputs/motivation"

echo "=== PGR-3D Motivation Experiment v3 (dual pass + bootstrap CIs) ==="
echo "Started at: $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Canonical GSO-30 from /scratch/s224696943/GSO/gso/ — real-world objects removed.
# Override with --export=OBJECTS="obj1 obj2 ..." at sbatch time for quick tests.
OBJECTS="${OBJECTS:-alarm backpack bell blocks chicken cream elephant grandfather grandmother hat leather lion lunch_bag mario oil school_bus1 school_bus2 shoe shoe1 shoe2 shoe3 soap sofa sorter sorting_board stucking_cups teapot toaster train turtle}"

python "${PGR_DIR}/src/motivation_experiment.py" \
    --objects ${OBJECTS} \
    --gso_fronts "${PGR_DIR}/data/gso_fronts" \
    --output_dir "${PGR_DIR}/outputs/motivation_v2_clean_gso30" \
    --num_steps 50 \
    --n_bootstrap 1000 \
    --guidance_scale 3.0 \
    --skip_objaverse \
    --device cuda

echo ""
echo "=== Experiment complete at: $(date) ==="
echo ""
echo "KEY OUTPUTS (v2 clean GSO-30):"
echo "  Primary CKA metric (with CI): ${PGR_DIR}/outputs/motivation_v2_clean_gso30/gso30/cka_input_front.png"
echo "  CLIP cosine sim:              ${PGR_DIR}/outputs/motivation_v2_clean_gso30/gso30/clip_drift_summary.png"
echo "  CKA raw data:                 ${PGR_DIR}/outputs/motivation_v2_clean_gso30/gso30/cka_data.csv"
echo ""
echo "INTERPRETATION: see docs/motivation_evidence.md"
echo "  Flat elevated CKA (≥0.4) = stable encoding = readout feasibility (GREEN)"
