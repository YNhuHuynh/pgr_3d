"""
PGR-3D Training Loop for Readout Heads
--------------------------------------
Trains SemanticHead and DepthHead on frozen Wonder3D UNet features.
Closely follows Readout Guidance's training recipe (Luo et al., CVPR 2024).

Usage (from cluster, after `sbatch scripts/slurm_train_readout.sh`):
    python src/train_readout.py \
        --head semantic \
        --cache_dir /scratch/s224696943/pgr_3d/data/objaverse_renders \
        --max_objects 200 \
        --steps 5000 \
        --batch_size 4 \
        --lr 1e-4 \
        --run_name pgr_semantic_200_20260418 \
        --ckpt_dir /scratch/s224696943/pgr_3d/outputs/checkpoints

Key design:
  - Wonder3D UNet is FROZEN — zero grad throughout.
  - For each batch: sample random timestep t, add noise to rendered latents,
    run UNet forward (both domains), capture features, extract RGB features,
    pass through AggregationNetwork + head, compute loss, backprop through head.
  - Domain layout (no CFG): [normal×V, rgb×V]. RGB features = slice[V:2V].
  - Task embeddings: domain indicator prepended to camera embeddings.
  - Checkpoint saved every 500 steps.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# Make project importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

import wandb

from data_pipeline    import make_dataloader, compute_midas_depth
from feature_extractor import Wonder3DFeatureExtractor
from readout_heads    import (
    build_semantic_head, build_depth_head,
    SemanticHead, DepthHead, save_head,
)


# ---------------------------------------------------------------------------
# Wonder3D UNet loading
# ---------------------------------------------------------------------------
WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
NUM_VIEWS      = 6


def load_wonder3d_pipeline(device: str = "cuda"):
    """
    Load the frozen Wonder3D pipeline (MVDiffusionImagePipeline).
    Returns (pipeline, unet, vae, scheduler, image_encoder).
    """
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image_joint import MVDiffusionImagePipeline
    pipe = MVDiffusionImagePipeline.from_pretrained(
        WONDER3D_CKPT,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)

    # Freeze everything
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.image_encoder.eval()

    # Enable memory-efficient attention
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe


# ---------------------------------------------------------------------------
# Prepare Wonder3D UNet inputs for a training batch
# ---------------------------------------------------------------------------

def prepare_unet_inputs(
    rgb: torch.Tensor,         # [B, 6, 3, H, W]   rendered views
    clip_emb: torch.Tensor,    # [B, 768]           CLIP embedding of view-0
    pipe,
    noise_scheduler,
    device: str = "cuda",
) -> tuple:
    """
    Construct (latent_model_input, timesteps, image_embeddings, camera_embeddings)
    ready for pipe.unet(latent_model_input, t, encoder_hidden_states=..., class_labels=...).

    Batch layout (no CFG, both domains concatenated):
        [normal_latents_BV, rgb_latents_BV]  where BV = B × 6

    Returns:
        latent_model_input : [2*B*V, 8, H/8, W/8]
        timesteps          : [2*B*V]   (same t for each object, repeated across views/domains)
        image_embeddings   : [2*B*V, 1, 768]
        camera_embeddings  : [2*B*V, 10]
    """
    B, V, C, H, W = rgb.shape
    BV = B * V

    # --- Encode RGB views to latents ---
    rgb_flat = rgb.view(BV, C, H, W).to(device)            # [BV, 3, 256, 256]
    rgb_for_vae = rgb_flat * 2.0 - 1.0                     # [-1, 1]
    with torch.no_grad():
        latents = pipe.vae.encode(rgb_for_vae).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor  # [BV, 4, 32, 32]

    # For normal domain, we use the same latents (simplification for readout training;
    # the cross-domain attention still shapes the RGB features we extract).
    # In real Wonder3D, normal and RGB domains are generated jointly.
    latents_normal = latents.clone()   # [BV, 4, 32, 32]
    latents_rgb    = latents           # [BV, 4, 32, 32]

    # --- Sample random timestep ---
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                      (B,), device=device)
    # Broadcast to all views and both domains
    t_bv = t.unsqueeze(1).expand(B, V).reshape(BV)      # [BV]
    t_2bv = t_bv.repeat(2)                              # [2*BV]

    # --- Add noise ---
    noise_normal = torch.randn_like(latents_normal)
    noise_rgb    = torch.randn_like(latents_rgb)
    noisy_normal = noise_scheduler.add_noise(latents_normal, noise_normal, t_bv)
    noisy_rgb    = noise_scheduler.add_noise(latents_rgb,    noise_rgb,    t_bv)

    # --- Image conditioning latents (concatenated on channel dim with noisy latents) ---
    # Wonder3D UNet expects 8-channel input: [noisy_4ch, cond_4ch]
    # Use the VAE latent of the input image (view-0) as conditioning for all views.
    view0 = rgb[:, 0, :, :, :]                             # [B, 3, H, W]
    view0_for_vae = (view0 * 2.0 - 1.0).to(device)
    with torch.no_grad():
        cond_latent_b = pipe.vae.encode(view0_for_vae).latent_dist.sample()
        cond_latent_b = cond_latent_b * pipe.vae.config.scaling_factor   # [B, 4, h, w]
    cond_latent_bv = cond_latent_b.unsqueeze(1).expand(B, V, *cond_latent_b.shape[1:])
    cond_latent_bv = cond_latent_bv.reshape(BV, *cond_latent_b.shape[1:])  # [BV, 4, h, w]

    # 8-channel inputs
    input_normal = torch.cat([noisy_normal, cond_latent_bv], dim=1)   # [BV, 8, h, w]
    input_rgb    = torch.cat([noisy_rgb,    cond_latent_bv], dim=1)   # [BV, 8, h, w]

    # Concat domains: [normal, rgb]
    latent_model_input = torch.cat([input_normal, input_rgb], dim=0)  # [2*BV, 8, h, w]

    # --- CLIP image embeddings (encoder_hidden_states) ---
    # Shape expected by Wonder3D: [2*BV, 1, 768] (one token per sample)
    clip_emb = clip_emb.to(device)                                 # [B, 768]
    clip_bv  = clip_emb.unsqueeze(1).expand(B, V, 768).reshape(BV, 768)  # [BV, 768]
    clip_bv  = clip_bv.unsqueeze(1)                                # [BV, 1, 768]
    image_embeddings = clip_bv.repeat(2, 1, 1)                    # [2*BV, 1, 768]

    # --- Camera embeddings (class_labels) ---
    # Wonder3D uses sincos of elevation/azimuth for 6 views, dim=10
    # We load the pre-computed camera embedding from the pipeline
    cam_emb = pipe.camera_embedding.to(device)                     # [6, 10]
    cam_bv  = cam_emb.unsqueeze(0).expand(B, V, 10).reshape(BV, 10)  # [BV, 10]
    camera_embeddings = cam_bv.repeat(2, 1)                         # [2*BV, 10]

    return latent_model_input, t_2bv, image_embeddings, camera_embeddings


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step_semantic(
    head: SemanticHead,
    features,          # list of [V, C, H, W] RGB features from 3 blocks
    clip_target,       # [B, 768]
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """One training step for SemanticHead."""
    pred = head(features, batch_size=batch_size)           # [B, 768]
    loss = SemanticHead.loss(pred, clip_target.to(device))
    return loss


def train_step_depth(
    head: DepthHead,
    features,          # list of [B*V, C, H, W]
    rgb_clean,         # [B*V, 3, H, W]
    device: str,
) -> torch.Tensor:
    """One training step for DepthHead. Computes MiDaS online."""
    with torch.no_grad():
        depth_target = compute_midas_depth(rgb_clean.to(device), device=device)   # [BV, 1, H, W]
    pred = head(features)                                   # [BV, 1, target_size, target_size]
    loss = DepthHead.loss(pred, depth_target)
    return loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda"

    # --- wandb init ---
    run = wandb.init(
        project="pgr-3d",
        name=args.run_name,
        config=vars(args),
        mode=args.wandb_mode,
    )

    # --- Load Wonder3D pipeline (frozen) ---
    print("Loading Wonder3D pipeline...")
    pipe = load_wonder3d_pipeline(device)

    # --- Build readout head ---
    if args.head == "semantic":
        head = build_semantic_head(num_views=NUM_VIEWS, device=device)
        print(f"SemanticHead params: {sum(p.numel() for p in head.parameters() if p.requires_grad):,}")
    elif args.head == "depth":
        head = build_depth_head(target_size=256, device=device)
        print(f"DepthHead params: {sum(p.numel() for p in head.parameters() if p.requires_grad):,}")
    else:
        raise ValueError(f"Unknown head type: {args.head}")

    # --- Feature extractor ---
    extractor = Wonder3DFeatureExtractor(pipe.unet, num_views=NUM_VIEWS)

    # --- Scheduler (for noise addition) ---
    from diffusers import DDIMScheduler
    noise_scheduler = DDIMScheduler.from_pretrained(WONDER3D_CKPT, subfolder="scheduler")
    noise_scheduler.set_timesteps(1000)

    # --- DataLoader ---
    loader = make_dataloader(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=2,
        max_objects=args.max_objects,
        shuffle=True,
    )

    # --- Optimizer (same as RG: AdamW, train only head params) ---
    optimizer = AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- Training loop ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    head.train()

    pbar = tqdm(total=args.steps, desc=f"Training {args.head} head")
    while global_step < args.steps:
        for batch in loader:
            if global_step >= args.steps:
                break

            rgb      = batch["rgb"]        # [B, 6, 3, 256, 256]
            clip_emb = batch["clip_emb"]   # [B, 768]
            B = rgb.shape[0]
            V = NUM_VIEWS
            BV = B * V

            # Prepare UNet inputs
            latent_input, timesteps, img_embs, cam_embs = prepare_unet_inputs(
                rgb, clip_emb, pipe, noise_scheduler, device
            )

            # Run frozen UNet with feature hooks
            with extractor:
                with torch.no_grad():
                    _ = pipe.unet(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=img_embs,
                        class_labels=cam_embs,
                    )
            # RGB features: second half of batch (domain=rgb, no CFG)
            rgb_features = extractor.get_rgb_features_batched(B, use_cfg=False)
            # Each: [B*V, C, H, W]

            # Head-specific loss
            optimizer.zero_grad()

            if args.head == "semantic":
                loss = train_step_semantic(head, rgb_features, clip_emb, B, device)
            else:  # depth
                rgb_flat = rgb.view(BV, 3, 256, 256)
                loss = train_step_depth(head, rgb_features, rgb_flat, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            wandb.log({"train/loss": loss.item(), "train/step": global_step})
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            # Checkpoint
            if global_step % 500 == 0 and global_step > 0:
                ckpt_path = ckpt_dir / f"{args.run_name}_step{global_step}.pt"
                save_head(str(ckpt_path), head, optimizer, global_step, vars(args))
                print(f"\nSaved checkpoint: {ckpt_path}")

            global_step += 1

    # Final checkpoint
    ckpt_path = ckpt_dir / f"{args.run_name}_final.pt"
    save_head(str(ckpt_path), head, optimizer, global_step, vars(args))
    print(f"Training complete. Final checkpoint: {ckpt_path}")
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PGR-3D readout heads")
    parser.add_argument("--head",         choices=["semantic", "depth"], required=True)
    parser.add_argument("--cache_dir",    default="/scratch/s224696943/pgr_3d/data/objaverse_renders")
    parser.add_argument("--max_objects",  type=int, default=200)
    parser.add_argument("--steps",        type=int, default=5000)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--run_name",     type=str, required=True)
    parser.add_argument("--ckpt_dir",     default="/scratch/s224696943/pgr_3d/outputs/checkpoints")
    parser.add_argument("--wandb_mode",   default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
