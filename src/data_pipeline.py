"""
PGR-3D Data Pipeline
--------------------
Two stages:

Stage 1 — Rendering (offline, once):
    render_objaverse_batch()  →  runs blenderproc on GLB files
    Outputs per-object cache: {uid}/rgb_{0..5}.png + depth_{0..5}.npy

Stage 2 — Training DataLoader:
    ObjaverseRenderDataset   →  reads cached renders, returns batch with
        rgb:   [6, 3, 256, 256]  (float32, [0,1])
        depth: [6, 1, 256, 256]  (float32, MiDaS-scale)
        clip_emb: [768]          (CLIP ViT-L/14 image embedding of view-0)

Wonder3D camera convention (6 views, orthographic, from configs):
    Elevations: 0, 0, 0, 0, 0, 0   (all front-horizontal)
    Azimuths:   0, 45, 90, 135, 180, 225  degrees
(Same as Wonder3D's default 6-view orthographic config)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Camera parameters matching Wonder3D 6-view orthographic config
# ---------------------------------------------------------------------------
WONDER3D_AZIMUTHS   = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0]
WONDER3D_ELEVATIONS = [0.0,  0.0,  0.0,   0.0,   0.0,   0.0]
ORTHO_SCALE         = 1.35
N_VIEWS             = 6
RENDER_RES          = 256   # render at 256×256 to match Wonder3D
BG_COLOR            = (1.0, 1.0, 1.0, 1.0)   # white background


# ---------------------------------------------------------------------------
# Stage 1: Rendering via BlenderProc
# ---------------------------------------------------------------------------

RENDER_SCRIPT = Path(__file__).parent.parent / "scripts" / "blender_render_6views.py"
BLENDER_BIN   = Path("/scratch/s224696943/blender/blender-3.3.0-linux-x64/blender")


def render_object(
    glb_path: str,
    output_dir: str,
    uid: str,
    resolution: int = RENDER_RES,
    ortho_scale: float = ORTHO_SCALE,
    blender_bin: str = str(BLENDER_BIN),
) -> bool:
    """
    Render 6 orthographic views of a GLB object using BlenderProc.
    Saves to {output_dir}/{uid}/rgb_{i}.png and depth_{i}.npy.

    Returns True on success, False on failure.
    """
    out_path = Path(output_dir) / uid
    out_path.mkdir(parents=True, exist_ok=True)

    # Check if already rendered
    if all((out_path / f"rgb_{i}.png").exists() for i in range(N_VIEWS)):
        return True   # cached

    cmd = [
        "blenderproc", "run",
        f"--blender-install-path={blender_bin}",
        str(RENDER_SCRIPT),
        "--object_path", str(glb_path),
        "--output_dir",  str(out_path),
        "--uid",         uid,
        "--resolution",  str(resolution),
        "--ortho_scale", str(ortho_scale),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[WARN] Render failed for {uid}: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[WARN] Render timed out for {uid}")
        return False


def render_objaverse_batch(
    glb_dir: str   = "/scratch/s224696943/objaverse_30k/objaverse/hf-objaverse-v1/glbs",
    cache_dir: str = "/scratch/s224696943/pgr_3d/data/objaverse_renders",
    max_objects: Optional[int] = None,
    start_idx: int = 0,
) -> List[str]:
    """
    Render a batch of Objaverse GLB files.
    Returns list of successfully rendered UIDs.
    """
    glb_dir   = Path(glb_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(glb_dir.glob("**/*.glb"))
    if max_objects is not None:
        glb_files = glb_files[start_idx: start_idx + max_objects]
    else:
        glb_files = glb_files[start_idx:]

    successful = []
    for glb_path in glb_files:
        uid = glb_path.stem
        ok  = render_object(str(glb_path), str(cache_dir), uid)
        if ok:
            successful.append(uid)

    uid_list_path = cache_dir / "rendered_uids.json"
    with open(uid_list_path, "w") as f:
        json.dump(successful, f)
    print(f"Rendered {len(successful)} objects → {uid_list_path}")
    return successful


# ---------------------------------------------------------------------------
# MiDaS depth estimation (online during training)
# ---------------------------------------------------------------------------
_midas_model = None
_midas_transform = None


def get_midas(device: str = "cuda"):
    global _midas_model, _midas_transform
    if _midas_model is None:
        import timm   # already in env
        # Use DPT-Large for quality; MiDaS v3
        _midas_model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_Large",
            trust_repo=True, verbose=False,
        ).to(device).eval()
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms",
            trust_repo=True, verbose=False,
        )
        _midas_transform = midas_transforms.dpt_transform
    return _midas_model, _midas_transform


@torch.no_grad()
def compute_midas_depth(
    images: torch.Tensor,   # [B, 3, H, W] float [0,1]
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns depth maps [B, 1, H, W] in MiDaS relative scale (larger = closer).
    """
    model, transform = get_midas(device)
    B, C, H, W = images.shape

    depths = []
    for i in range(B):
        img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        inp    = transform(img_np).to(device)
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)
        depth = model(inp)                                     # [1, H', W']
        depth = F.interpolate(depth.unsqueeze(1).float(),
                               size=(H, W), mode="bilinear",
                               align_corners=False)            # [1, 1, H, W]
        depths.append(depth.squeeze(0))                        # [1, H, W]
    return torch.stack(depths, dim=0)                          # [B, 1, H, W]


# ---------------------------------------------------------------------------
# CLIP embedding helper
# ---------------------------------------------------------------------------
_clip_model   = None
_clip_preproc = None


def get_clip(device: str = "cuda"):
    global _clip_model, _clip_preproc
    if _clip_model is None:
        import clip as clip_lib
        _clip_model, _clip_preproc = clip_lib.load("ViT-L/14", device=device)
        _clip_model = _clip_model.eval()
    return _clip_model, _clip_preproc


@torch.no_grad()
def compute_clip_embedding(
    image: Image.Image,   # PIL front-view image
    device: str = "cuda",
) -> torch.Tensor:
    """Returns normalised CLIP ViT-L/14 image embedding [1, 768]."""
    model, preproc = get_clip(device)
    tensor = preproc(image).unsqueeze(0).to(device)
    emb    = model.encode_image(tensor)                 # [1, 768]
    return F.normalize(emb.float(), dim=-1)


# ---------------------------------------------------------------------------
# Stage 2: Training Dataset
# ---------------------------------------------------------------------------

class ObjaverseRenderDataset(Dataset):
    """
    Reads pre-rendered Objaverse views from {cache_dir}/{uid}/rgb_{i}.png.

    Returns dict:
        uid:      str
        rgb:      [6, 3, 256, 256]   float32 in [0,1]
        clip_emb: [768]              CLIP embedding of view-0 (front)

    Note: MiDaS depth is computed ONLINE in the training loop (on-the-fly on
    the clean rendered RGB) rather than pre-cached, to save disk space and
    match RG's approach.
    """

    def __init__(
        self,
        cache_dir: str     = "/scratch/s224696943/pgr_3d/data/objaverse_renders",
        uid_list: Optional[List[str]] = None,
        max_objects: Optional[int]    = None,
        image_size: int = 256,
        clip_device: str = "cpu",   # load CLIP on CPU for DataLoader workers
    ):
        self.cache_dir   = Path(cache_dir)
        self.image_size  = image_size
        self.clip_device = clip_device

        if uid_list is not None:
            self.uids = uid_list
        else:
            uid_list_path = self.cache_dir / "rendered_uids.json"
            if uid_list_path.exists():
                with open(uid_list_path) as f:
                    self.uids = json.load(f)
            else:
                # Fallback: scan directory
                self.uids = [
                    d.name for d in sorted(self.cache_dir.iterdir())
                    if d.is_dir() and (d / "rgb_0.png").exists()
                ]

        if max_objects is not None:
            self.uids = self.uids[:max_objects]

        print(f"ObjaverseRenderDataset: {len(self.uids)} objects in {cache_dir}")

    def __len__(self) -> int:
        return len(self.uids)

    def _load_rgb(self, uid: str) -> torch.Tensor:
        """Load 6 RGB views → [6, 3, H, W] float32 [0,1]."""
        obj_dir = self.cache_dir / uid
        frames  = []
        for i in range(N_VIEWS):
            img = Image.open(obj_dir / f"rgb_{i}.png").convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))   # [3, H, W]
        return torch.stack(frames, dim=0)   # [6, 3, H, W]

    def _load_clip_emb(self, uid: str, view0_rgb: torch.Tensor) -> torch.Tensor:
        """
        Returns CLIP embedding of view-0 (front face).
        Tries to load cached .pt first; computes and caches otherwise.
        """
        clip_path = self.cache_dir / uid / "clip_emb.pt"
        if clip_path.exists():
            return torch.load(clip_path, map_location="cpu").squeeze(0)

        # Compute on-the-fly (CPU-based for DataLoader compatibility)
        pil_img = Image.fromarray(
            (view0_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        emb = compute_clip_embedding(pil_img, device=self.clip_device)   # [1, 768]
        torch.save(emb.cpu(), clip_path)
        return emb.squeeze(0)   # [768]

    def __getitem__(self, idx: int) -> dict:
        uid = self.uids[idx]
        rgb = self._load_rgb(uid)             # [6, 3, H, W]
        clip_emb = self._load_clip_emb(uid, rgb[0])   # [768]
        return {
            "uid":      uid,
            "rgb":      rgb,               # [6, 3, H, W]
            "clip_emb": clip_emb,          # [768]
        }


# ---------------------------------------------------------------------------
# CLI entry point (for slurm_render_objaverse.sh)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["render"], default="render")
    parser.add_argument("--glb_dir",     default="/scratch/s224696943/objaverse_30k/objaverse/hf-objaverse-v1/glbs")
    parser.add_argument("--cache_dir",   default="/scratch/s224696943/pgr_3d/data/objaverse_renders")
    parser.add_argument("--max_objects", type=int, default=200)
    parser.add_argument("--start_idx",   type=int, default=0)
    args = parser.parse_args()

    if args.mode == "render":
        render_objaverse_batch(
            glb_dir=args.glb_dir,
            cache_dir=args.cache_dir,
            max_objects=args.max_objects,
            start_idx=args.start_idx,
        )


def make_dataloader(
    cache_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    max_objects: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = ObjaverseRenderDataset(
        cache_dir=cache_dir,
        max_objects=max_objects,
        clip_device="cpu",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
