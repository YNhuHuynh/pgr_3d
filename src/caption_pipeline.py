"""
PGR-3D Caption Pipeline
-----------------------
Generates structured 3D-focused captions for rendered object front views
using Qwen2.5-VL-72B-Instruct.

Run with the unique3d conda environment (transformers >= 4.45):
    /scratch/s224696943/.conda/envs/unique3d/bin/python src/caption_pipeline.py \
        --image_dir  data/gso30_renders \
        --output_dir data/gso30_renders \
        --objects    alarm backpack ...

Saves per-object {output_dir}/{obj}/caption.txt.
Idempotent: skips objects with existing caption.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

CAPTION_PROMPT = (
    "Describe this 3D object in detail for the purpose of 3D reconstruction. Focus on:\n"
    "1. What the object is (category, identity)\n"
    "2. Key visual features (colors, textures, materials)\n"
    "3. Geometric shape (rigid vs soft, angular vs round, proportions)\n"
    "4. Distinctive parts and their positions (e.g., 'hat on top', 'arm raised to the right')\n"
    "5. Pose or configuration (if character/articulated)\n\n"
    "Write a single concise paragraph of 2-4 sentences."
)

_model = None
_processor = None


def get_model_and_processor(device_map: str = "auto"):
    global _model, _processor
    if _model is None:
        print(f"Loading {MODEL_ID} (FP16, device_map={device_map}) …", flush=True)
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        _model.eval()
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("Model loaded.", flush=True)
    return _model, _processor


def generate_caption(image_path: str, prompt: str = CAPTION_PROMPT) -> str:
    """Generate a single caption for the image at image_path."""
    model, processor = get_model_and_processor()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    # Decode only the newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return caption


def generate_captions_batch(image_paths: List[str], prompt: str = CAPTION_PROMPT) -> List[str]:
    """Generate captions for a list of images. Processes one at a time (memory safety)."""
    return [generate_caption(p, prompt) for p in image_paths]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Qwen2.5-VL captions for GSO/Objaverse")
    parser.add_argument("--image_dir",  required=True, help="Root dir containing {obj}/rgb_0.png")
    parser.add_argument("--output_dir", required=True, help="Root dir to write {obj}/caption.txt")
    parser.add_argument("--objects",    nargs="+", default=None, help="Object names to process (default: all subdirs)")
    parser.add_argument("--view",       type=int, default=0, help="Which view index to caption (default: 0 = front)")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if args.objects:
        objects = args.objects
    else:
        objects = sorted(d.name for d in image_dir.iterdir() if d.is_dir())

    print(f"Captioning {len(objects)} objects (view={args.view}) …", flush=True)

    # Warm up model
    get_model_and_processor(args.device_map)

    n_done, n_skip, n_fail = 0, 0, 0
    for obj in objects:
        out_path = output_dir / obj / "caption.txt"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  SKIP  {obj}", flush=True)
            n_skip += 1
            continue
        # Remove 0-byte stub (created if a previous write_text call failed mid-write)
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
            print(f"  RETRY {obj}  (removed 0-byte stub)", flush=True)

        img_path = image_dir / obj / f"rgb_{args.view}.png"
        if not img_path.exists():
            print(f"  FAIL  {obj}  (no image at {img_path})", flush=True)
            n_fail += 1
            continue

        try:
            caption = generate_caption(str(img_path))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(caption + "\n", encoding="utf-8")
            print(f"  OK    {obj}: {caption[:80]}…", flush=True)
            n_done += 1
        except Exception as e:
            print(f"  FAIL  {obj}: {e}", flush=True)
            n_fail += 1

    print(f"\nDone: {n_done} generated, {n_skip} skipped, {n_fail} failed.")


if __name__ == "__main__":
    main()
