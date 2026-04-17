"""
PGR-3D Training Loop — CaptionHead
-----------------------------------
Trains CaptionHead on frozen Wonder3D UNet features.
Wonder3D UNet is FROZEN throughout; only AggregationNetwork + CaptionHead
parameters receive gradients.

Data requirement per object:
    {cache_dir}/{uid}/rgb_{0..5}.png   — rendered views
    {cache_dir}/{uid}/t5_emb.pt        — mean-pooled T5 [768] embedding

Usage:
    python src/train_readout_caption.py \
        --cache_dir /scratch/s224696943/pgr_3d/data/objaverse_200 \
        --steps 5000 \
        --batch_size 4 \
        --run_name pgr_caption_200_20260418 \
        --ckpt_dir /scratch/s224696943/pgr_3d/outputs/checkpoints

Quick sanity run (10 objects, 100 steps):
    python src/train_readout_caption.py \
        --cache_dir /scratch/s224696943/pgr_3d/data/objaverse_200 \
        --max_objects 10 \
        --steps 100 \
        --checkpoint_every 50 \
        --run_name pgr_caption_sanity
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

import wandb

from feature_extractor import Wonder3DFeatureExtractor
from readout_heads import build_caption_head, CaptionHead, save_head

WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
NUM_VIEWS     = 6


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_blacklist(blacklist_path: str | None) -> set:
    """Return set of blacklisted UIDs (lines starting with '#' are comments)."""
    if blacklist_path is None:
        return set()
    p = Path(blacklist_path)
    if not p.exists():
        return set()
    uids = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            uids.add(line)
    return uids


class CaptionTrainDataset(Dataset):
    """
    Reads pre-rendered views + pre-computed T5 embeddings.
    Returns:
        rgb:     [6, 3, 256, 256]  float32 [0,1]
        t5_emb:  [768]             float32, L2-normalised

    Filtering (applied in order):
        1. uid_list: explicit allow-list (skips auto-scan)
        2. blacklist_path: UIDs to exclude unconditionally
        3. Defensive: skip dirs missing any rgb_{0..5}.png or t5_emb.pt
    """
    def __init__(self, cache_dir: str, uid_list=None, max_objects=None,
                 image_size=256, blacklist_path: str | None = None):
        self.cache_dir  = Path(cache_dir)
        self.image_size = image_size

        blacklist = _load_blacklist(blacklist_path)
        if blacklist:
            print(f"CaptionTrainDataset: blacklisting {len(blacklist)} UIDs from {blacklist_path}")

        if uid_list is not None:
            candidates = [u for u in uid_list if u not in blacklist]
        else:
            candidates = [
                d.name for d in sorted(self.cache_dir.iterdir())
                if d.is_dir() and d.name not in blacklist
            ]

        # Defensive: require all 6 views + t5_emb.pt
        self.uids = []
        n_skip = 0
        for uid in candidates:
            d = self.cache_dir / uid
            missing_views = [i for i in range(NUM_VIEWS)
                             if not (d / f"rgb_{i}.png").exists()]
            if missing_views or not (d / "t5_emb.pt").exists():
                n_skip += 1
                continue
            self.uids.append(uid)

        if n_skip:
            print(f"CaptionTrainDataset: skipped {n_skip} incomplete objects "
                  f"(missing views or t5_emb.pt)")

        if max_objects is not None:
            self.uids = self.uids[:max_objects]

        print(f"CaptionTrainDataset: {len(self.uids)} objects in {cache_dir}")
        if len(self.uids) == 0:
            raise ValueError(
                f"No valid objects found in {cache_dir}. "
                "Each object dir needs rgb_0..5.png + t5_emb.pt."
            )

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid     = self.uids[idx]
        obj_dir = self.cache_dir / uid

        frames = []
        for i in range(NUM_VIEWS):
            img = Image.open(obj_dir / f"rgb_{i}.png").convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        rgb = torch.stack(frames, dim=0)   # [6, 3, H, W]

        t5_emb = torch.load(obj_dir / "t5_emb.pt", map_location="cpu")
        if t5_emb.dim() > 1:
            t5_emb = t5_emb.squeeze(0)    # [768]

        return {"uid": uid, "rgb": rgb, "t5_emb": t5_emb}


# ---------------------------------------------------------------------------
# Wonder3D pipeline loading (same as train_readout.py)
# ---------------------------------------------------------------------------

def load_wonder3d_pipeline(device="cuda"):
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image_joint import MVDiffusionImagePipeline
    pipe = MVDiffusionImagePipeline.from_pretrained(
        WONDER3D_CKPT, torch_dtype=torch.float32, local_files_only=True,
    ).to(device)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.eval(); pipe.vae.eval(); pipe.image_encoder.eval()
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


# ---------------------------------------------------------------------------
# UNet input preparation (identical to train_readout.py)
# ---------------------------------------------------------------------------

def prepare_unet_inputs(rgb, pipe, noise_scheduler, device="cuda"):
    """
    Build UNet inputs for a batch.
    rgb:  [B, 6, 3, H, W]

    Image conditioning uses Wonder3D's own CLIP encoder (pipe.image_encoder)
    applied to the front view (view 0).  This preserves Wonder3D's conditioning
    contract — t5_emb is the CaptionHead prediction TARGET (loss), not UNet input.

    Returns: (latent_model_input, timesteps, image_embeddings, camera_embeddings)
      latent_model_input  [2BV, 8, H/8, W/8]
      timesteps           [2BV]
      image_embeddings    [2BV, 1, 768]   CLIP image embeds
      camera_embeddings   [2BV, 10]
    """
    B, V, C, H, W = rgb.shape
    BV = B * V

    rgb_flat     = rgb.view(BV, C, H, W).to(device)
    rgb_for_vae  = rgb_flat * 2.0 - 1.0
    with torch.no_grad():
        latents = pipe.vae.encode(rgb_for_vae).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    latents_normal = latents.clone()
    latents_rgb    = latents

    t    = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device)
    t_bv = t.unsqueeze(1).expand(B, V).reshape(BV)
    t_2bv = t_bv.repeat(2)

    noisy_normal = noise_scheduler.add_noise(latents_normal, torch.randn_like(latents_normal), t_bv)
    noisy_rgb    = noise_scheduler.add_noise(latents_rgb,    torch.randn_like(latents_rgb),    t_bv)

    view0 = rgb[:, 0].to(device)   # [B, 3, H, W]  front view
    with torch.no_grad():
        cond_lat  = pipe.vae.encode(view0 * 2.0 - 1.0).latent_dist.sample()
        cond_lat  = cond_lat * pipe.vae.config.scaling_factor
    cond_bv = cond_lat.unsqueeze(1).expand(B, V, *cond_lat.shape[1:]).reshape(BV, *cond_lat.shape[1:])

    input_normal = torch.cat([noisy_normal, cond_bv], dim=1)
    input_rgb    = torch.cat([noisy_rgb,    cond_bv], dim=1)
    latent_model_input = torch.cat([input_normal, input_rgb], dim=0)

    # Correct CLIP image conditioning: encode front view via Wonder3D's own encoder.
    # Matches _encode_image() in pipeline_mvdiffusion_image_joint.py:
    #   feature_extractor(pil_images) → pixel_values
    #   image_encoder(pixel_values).image_embeds  → [B, 768]
    with torch.no_grad():
        view0_pil = [
            Image.fromarray(
                (view0[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            for i in range(B)
        ]
        clip_dtype = next(pipe.image_encoder.parameters()).dtype
        clip_pixel = pipe.feature_extractor(images=view0_pil, return_tensors="pt").pixel_values
        clip_pixel = clip_pixel.to(device=device, dtype=clip_dtype)
        clip_embed = pipe.image_encoder(clip_pixel).image_embeds.float()   # [B, 768]

    # Expand to [BV, 1, 768] then 2× for normal+rgb branches → [2BV, 1, 768]
    clip_bv = clip_embed.unsqueeze(1).expand(B, V, 768).reshape(BV, 768).unsqueeze(1)
    image_embeddings = clip_bv.repeat(2, 1, 1)   # [2BV, 1, 768]

    raw_cam  = pipe.camera_embedding.to(device=device, dtype=torch.float32)
    cam_10d  = torch.cat([torch.sin(raw_cam), torch.cos(raw_cam)], dim=-1)
    cam_norm_bv = cam_10d[:V].unsqueeze(0).expand(B, V, 10).reshape(BV, 10)
    cam_rgb_bv  = cam_10d[V:2*V].unsqueeze(0).expand(B, V, 10).reshape(BV, 10)
    camera_embeddings = torch.cat([cam_norm_bv, cam_rgb_bv], dim=0)

    return latent_model_input, t_2bv, image_embeddings, camera_embeddings


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda"

    run = wandb.init(
        project="pgr-3d",
        name=args.run_name,
        config=vars(args),
        mode=args.wandb_mode,
    )

    print("Loading Wonder3D pipeline …")
    pipe = load_wonder3d_pipeline(device)

    print("Building CaptionHead …")
    head = build_caption_head(num_views=NUM_VIEWS, device=device)
    print(f"  Trainable params: {sum(p.numel() for p in head.parameters() if p.requires_grad):,}")

    extractor = Wonder3DFeatureExtractor(pipe.unet, num_views=NUM_VIEWS)

    from diffusers import DDIMScheduler
    noise_scheduler = DDIMScheduler.from_pretrained(WONDER3D_CKPT, subfolder="scheduler")
    noise_scheduler.set_timesteps(1000)

    dataset = CaptionTrainDataset(
        cache_dir=args.cache_dir,
        max_objects=args.max_objects,
        blacklist_path=args.blacklist,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint ---
    global_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        head.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume_from} at step {global_step}")

    # --- CSV log ---
    csv_path = ckpt_dir / f"{args.run_name}_log.csv"
    csv_file  = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_path.stat().st_size == 0:
        csv_writer.writerow(["step", "loss", "lr", "time"])

    t_start = time.time()

    head.train()
    pbar = tqdm(total=args.steps, desc="Training CaptionHead", initial=global_step)

    while global_step < args.steps:
        for batch in loader:
            if global_step >= args.steps:
                break

            rgb    = batch["rgb"]      # [B, 6, 3, 256, 256]
            t5_emb = batch["t5_emb"]   # [B, 768]
            B      = rgb.shape[0]

            latent_input, timesteps, img_embs, cam_embs = prepare_unet_inputs(
                rgb, pipe, noise_scheduler, device
            )

            with extractor:
                with torch.no_grad():
                    _ = pipe.unet(
                        latent_input, timesteps,
                        encoder_hidden_states=img_embs,
                        class_labels=cam_embs,
                    )
            rgb_features = extractor.get_rgb_features_batched(B, use_cfg=False)

            optimizer.zero_grad()
            pred = head(rgb_features, batch_size=B)              # [B, 768]
            loss = CaptionHead.loss(pred, t5_emb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - t_start
            wandb.log({"train/loss": loss.item(), "train/lr": current_lr,
                       "train/step": global_step})
            csv_writer.writerow([global_step, f"{loss.item():.6f}",
                                 f"{current_lr:.2e}", f"{elapsed:.1f}"])
            csv_file.flush()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            if global_step > 0 and global_step % args.checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"{args.run_name}_step{global_step}.pt"
                save_head(str(ckpt_path), head, optimizer, global_step, vars(args))
                print(f"\nSaved: {ckpt_path}")

            global_step += 1

    ckpt_path = ckpt_dir / f"{args.run_name}_final.pt"
    save_head(str(ckpt_path), head, optimizer, global_step, vars(args))
    print(f"Training complete. Final checkpoint: {ckpt_path}")
    csv_file.close()
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PGR-3D CaptionHead")
    parser.add_argument("--cache_dir",         default="/scratch/s224696943/pgr_3d/data/objaverse_200")
    parser.add_argument("--max_objects",        type=int, default=None)
    parser.add_argument("--steps",              type=int, default=5000)
    parser.add_argument("--batch_size",         type=int, default=4)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight_decay",       type=float, default=0.01)
    parser.add_argument("--checkpoint_every",   type=int, default=500)
    parser.add_argument("--run_name",           type=str, default="pgr_caption")
    parser.add_argument("--ckpt_dir",           default="/scratch/s224696943/pgr_3d/outputs/checkpoints")
    parser.add_argument("--wandb_mode",         default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--resume_from",        type=str, default=None,
                        help="Path to checkpoint .pt to resume from")
    parser.add_argument("--blacklist",           type=str, default=None,
                        help="Path to text file of UIDs to exclude from training "
                             "(one UID per line; lines starting with # are comments). "
                             "Default: configs/objaverse_blacklist.txt if it exists.")
    args = parser.parse_args()

    # Auto-discover blacklist if not set
    if args.blacklist is None:
        default_bl = Path(__file__).parent.parent / "configs" / "objaverse_blacklist.txt"
        if default_bl.exists():
            args.blacklist = str(default_bl)

    train(args)


if __name__ == "__main__":
    main()
