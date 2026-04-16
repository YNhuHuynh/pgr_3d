#!/usr/bin/env python3
"""
Generate per-object Wonder3D inference configs for the GSO baseline evaluation.

Usage:
    python scripts/gen_gso_configs.py [--objects mario alarm chicken ...]

Writes configs to /scratch/s224696943/pgr_3d/configs/gso_{object}.yaml
"""
import argparse
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PRETRAINED_MODEL   = "/scratch/s224696943/wonder3d-v1.0"
FRONT_IMG_TEMPLATE = "/weka/s224696943/3DRAV/data/expB/{object}/B/front.png"
SAVE_DIR_TEMPLATE  = "/scratch/s224696943/pgr_3d/outputs/wonder3d_baseline/{object}"
WONDER3D_OUT_ROOT  = "/scratch/s224696943/Wonder3D/outputs/wonder3D-finetune/vis_0"
CONFIG_OUT_DIR     = Path("/scratch/s224696943/pgr_3d/configs")

# Full GSO-30 eval set (can override via --objects)
GSO_OBJECTS = [
    "alarm", "backpack", "blocks", "chicken", "elephant",
    "grandfather", "grandmother", "lion", "lunch_bag", "mario",
    "oil", "school_bus1", "school_bus2", "shoe", "soap",
    "sofa", "sorter", "train", "turtle",
    # numbered objects (from existing gso folder)
    "1st", "2nd", "3rd", "4th", "5th",
    "6th", "11st", "12nd", "13rd", "14th",
    "15th",
]

TEMPLATE = {
    "pretrained_model_name_or_path": PRETRAINED_MODEL,
    "revision": None,
    "validation_dataset": {
        "root_dir": WONDER3D_OUT_ROOT,
        "num_views": 6,
        "bg_color": "white",
        "img_wh": [256, 256],
        "num_validation_samples": 1000,
        "crop_size": 192,
        "single_image": {"front": None},   # filled per object
        "input_view_pattern": ["front"] * 6,
    },
    "save_dir": None,                       # filled per object
    "pred_type": "joint",
    "seed": 42,
    "validation_batch_size": 1,
    "dataloader_num_workers": 1,
    "local_rank": -1,
    "pipe_kwargs": {"camera_embedding_type": "e_de_da_sincos", "num_views": 6},
    "validation_guidance_scales": [1.0],
    "pipe_validation_kwargs": {"eta": 1.0},
    "validation_grid_nrow": 6,
    "unet_from_pretrained_kwargs": {
        "camera_embedding_type": "e_de_da_sincos",
        "projection_class_embeddings_input_dim": 10,
        "num_views": 6,
        "sample_size": 32,
        "cd_attention_mid": True,
        "zero_init_conv_in": False,
        "zero_init_camera_projection": False,
    },
    "num_views": 6,
    "camera_embedding_type": "e_de_da_sincos",
    "enable_xformers_memory_efficient_attention": True,
}


def gen_config(obj: str) -> Path:
    cfg = dict(TEMPLATE)
    cfg["validation_dataset"] = dict(TEMPLATE["validation_dataset"])
    cfg["validation_dataset"]["single_image"] = {
        "front": FRONT_IMG_TEMPLATE.format(object=obj)
    }
    cfg["save_dir"] = SAVE_DIR_TEMPLATE.format(object=obj)

    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG_OUT_DIR / f"gso_{obj}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", default=GSO_OBJECTS)
    args = parser.parse_args()

    print(f"Generating configs for {len(args.objects)} objects...")
    for obj in args.objects:
        front = Path(FRONT_IMG_TEMPLATE.format(object=obj))
        if not front.exists():
            print(f"  [WARN] Missing front image: {front} — skipping {obj}")
            continue
        gen_config(obj)
    print("Done.")


if __name__ == "__main__":
    main()
