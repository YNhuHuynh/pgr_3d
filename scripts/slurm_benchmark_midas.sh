#!/bin/bash
#SBATCH --job-name=pgr_bench_midas
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/bench_midas_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/bench_midas_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH -p gpu-large
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=40GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate pgr3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D:${PYTHONPATH:-}"

mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D MiDaS Overhead Benchmark ==="
echo "Started at: $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

python "${PGR_DIR}/scripts/benchmark_midas_overhead.py"

echo ""
echo "=== Benchmark complete at: $(date) ==="
