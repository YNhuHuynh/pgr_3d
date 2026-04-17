#!/bin/bash
#SBATCH --job-name=pgr_caption_sanity
#SBATCH --output=/scratch/%u/pgr_3d/outputs/logs/caption_sanity_%j.out
#SBATCH --error=/scratch/%u/pgr_3d/outputs/logs/caption_sanity_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH -p gpu-large
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cuda-12.6
#SBATCH --mem=120GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s224696943@deakin.edu.au

# Sanity check: generate captions for 5 GSO objects using Qwen2.5-VL-72B.
# Saves caption.txt + copies rgb_0.png for user review.
# STOP GATE: user must approve caption quality before full batch (Block F).

set -euo pipefail

source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate unique3d

PGR_DIR="/scratch/s224696943/pgr_3d"
export PYTHONPATH="${PGR_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/s224696943/.cache/huggingface"

SAMPLE_OBJECTS="alarm chicken mario sofa turtle"
IMAGE_DIR="${PGR_DIR}/data/gso30_renders"
OUTPUT_DIR="${PGR_DIR}/outputs/caption_samples"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PGR_DIR}/outputs/logs"

echo "=== PGR-3D Caption Sanity (5 GSO objects, Qwen2.5-VL-72B) ==="
echo "Started: $(date)   Node: $(hostname)"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
echo ""

# Generate captions (also saves caption.txt in each render dir)
python "${PGR_DIR}/src/caption_pipeline.py" \
    --image_dir  "${IMAGE_DIR}" \
    --output_dir "${IMAGE_DIR}" \
    --objects    ${SAMPLE_OBJECTS} \
    --view       0 \
    --device_map "auto"

# Copy captions + reference images to caption_samples/ for easy review
echo ""
echo "=== Caption results ==="
for obj in ${SAMPLE_OBJECTS}; do
    cap_file="${IMAGE_DIR}/${obj}/caption.txt"
    img_file="${IMAGE_DIR}/${obj}/rgb_0.png"
    if [ -f "${cap_file}" ]; then
        cp "${cap_file}" "${OUTPUT_DIR}/caption_${obj}.txt"
        [ -f "${img_file}" ] && cp "${img_file}" "${OUTPUT_DIR}/rgb_${obj}.png"
        echo "--- ${obj} ---"
        cat "${cap_file}"
        echo ""
    else
        echo "MISSING: ${obj}/caption.txt"
    fi
done

echo "=== Done: $(date) ==="
echo "Review outputs: ${OUTPUT_DIR}"
echo "If captions look good → approve Block F (full 230-object batch)."
