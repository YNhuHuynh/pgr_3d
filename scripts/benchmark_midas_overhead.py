#!/usr/bin/env python3
"""
PGR-3D — MiDaS training-step overhead benchmark.

Answers: what fraction of per-step training time does online MiDaS add?

Two conditions, each 3 warmup + 10 timed steps (B=4, V=6):

  A) FULL DEPTH STEP  : UNet forward + MiDaS BV=24 images + DepthHead + backward
  B) CLIP-ONLY STEP   : UNet forward + SemanticHead + backward  (no MiDaS)

Overhead = (median_A - median_B) / median_A * 100%

Uses synthetic random tensors — no real renders needed.

Decision rule (from user spec):
  overhead < 15%  → keep online  → 7 files/object
  overhead ≥ 15%  → pre-compute  → 13 files/object, add depth to rendering job
"""

from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

import torch
import torch.nn.functional as F
import numpy as np

WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
DEVICE        = "cuda"
B, V          = 4, 6
BV            = B * V
H, W          = 256, 256
N_WARMUP      = 3
N_TIMED       = 10


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_wonder3d():
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image_joint import MVDiffusionImagePipeline
    pipe = MVDiffusionImagePipeline.from_pretrained(
        WONDER3D_CKPT, torch_dtype=torch.float32, local_files_only=True,
    ).to(DEVICE)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.eval(); pipe.vae.eval(); pipe.image_encoder.eval()
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


def load_midas():
    from data_pipeline import get_midas
    model, transform = get_midas(DEVICE)
    return model, transform


def load_heads(pipe):
    from readout_heads import build_semantic_head, build_depth_head
    sem   = build_semantic_head(num_views=V, device=DEVICE)
    depth = build_depth_head(target_size=H, device=DEVICE)
    return sem, depth


def load_extractor(pipe):
    from feature_extractor import Wonder3DFeatureExtractor
    return Wonder3DFeatureExtractor(pipe.unet, num_views=V)


# ---------------------------------------------------------------------------
# Synthetic batch
# ---------------------------------------------------------------------------

def make_synthetic_batch(pipe):
    """
    Returns everything prepare_unet_inputs would return, using random data.
    Avoids loading any actual rendered images.
    """
    # Synthetic: random noisy latents [2*BV, 8, 32, 32] + random camera/CLIP
    h_lat, w_lat = H // 8, W // 8
    latent_input = torch.randn(2 * BV, 8, h_lat, w_lat, device=DEVICE)
    t_2bv        = torch.randint(0, 1000, (2 * BV,), device=DEVICE)
    img_embs     = torch.randn(2 * BV, 1, 768, device=DEVICE)
    # Camera embeddings: sincos of raw camera embedding
    raw_cam = pipe.camera_embedding.to(device=DEVICE, dtype=torch.float32)   # [2V, 5]
    cam_10d = torch.cat([torch.sin(raw_cam), torch.cos(raw_cam)], dim=-1)    # [2V, 10]
    cam_norm_bv = cam_10d[:V].unsqueeze(0).expand(B, V, 10).reshape(BV, 10)
    cam_rgb_bv  = cam_10d[V:2*V].unsqueeze(0).expand(B, V, 10).reshape(BV, 10)
    cam_embs    = torch.cat([cam_norm_bv, cam_rgb_bv], dim=0)                # [2*BV, 10]

    # Synthetic clean RGB for MiDaS: [BV, 3, H, W] in [0,1]
    rgb_clean = torch.rand(BV, 3, H, W, device=DEVICE)
    # Synthetic CLIP target: [B, 768] L2-normalised
    clip_target = F.normalize(torch.randn(B, 768, device=DEVICE), dim=-1)

    return latent_input, t_2bv, img_embs, cam_embs, rgb_clean, clip_target


# ---------------------------------------------------------------------------
# Timed step functions
# ---------------------------------------------------------------------------

def step_with_midas(pipe, extractor, depth_head, midas_fn, batch):
    """Full depth-head step: UNet + MiDaS (24 images) + DepthHead + backward."""
    latent_input, t_2bv, img_embs, cam_embs, rgb_clean, _ = batch

    # UNet forward (frozen, with hooks)
    with extractor:
        with torch.no_grad():
            _ = pipe.unet(
                latent_input, t_2bv,
                encoder_hidden_states=img_embs,
                class_labels=cam_embs,
            )
    rgb_features = extractor.get_rgb_features_batched(B, use_cfg=False)

    # MiDaS (online, sequential loop — exactly as in train_step_depth)
    with torch.no_grad():
        depth_target = midas_fn(rgb_clean, device=DEVICE)   # [BV, 1, H, W]

    # DepthHead forward + backward
    depth_head.train()
    optimizer_depth = torch.optim.AdamW(depth_head.parameters(), lr=1e-4)
    optimizer_depth.zero_grad()
    pred = depth_head(rgb_features)
    from readout_heads import DepthHead
    loss = DepthHead.loss(pred, depth_target)
    loss.backward()
    optimizer_depth.step()

    torch.cuda.synchronize()


def step_without_midas(pipe, extractor, sem_head, batch):
    """Clip-only step: UNet + SemanticHead + backward (no MiDaS)."""
    latent_input, t_2bv, img_embs, cam_embs, _, clip_target = batch

    # UNet forward (frozen, with hooks)
    with extractor:
        with torch.no_grad():
            _ = pipe.unet(
                latent_input, t_2bv,
                encoder_hidden_states=img_embs,
                class_labels=cam_embs,
            )
    rgb_features = extractor.get_rgb_features_batched(B, use_cfg=False)

    # SemanticHead forward + backward
    sem_head.train()
    optimizer_sem = torch.optim.AdamW(sem_head.parameters(), lr=1e-4)
    optimizer_sem.zero_grad()
    from readout_heads import SemanticHead
    pred = sem_head(rgb_features, batch_size=B)
    loss = SemanticHead.loss(pred, clip_target)
    loss.backward()
    optimizer_sem.step()

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def timed_steps(fn, n_warmup, n_timed, label):
    print(f"  {label}: {n_warmup} warmup + {n_timed} timed steps ...", flush=True)
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    step {i+1:02d}/{n_timed}: {elapsed*1000:.0f} ms", flush=True)

    return np.array(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("PGR-3D MiDaS Training-Step Overhead Benchmark")
    print(f"  B={B}, V={V}, BV={BV}, resolution={H}×{W}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 65, "\n")

    print("Loading Wonder3D pipeline ...", flush=True)
    pipe = load_wonder3d()
    print("Loading MiDaS DPT-Large ...", flush=True)
    load_midas()   # warms up cache
    from data_pipeline import compute_midas_depth
    print("Building readout heads ...", flush=True)
    sem_head, depth_head = load_heads(pipe)
    extractor = load_extractor(pipe)
    print("Building synthetic batch ...\n", flush=True)
    batch = make_synthetic_batch(pipe)

    # ---- Condition A: WITH MiDaS ----
    fn_a = lambda: step_with_midas(
        pipe, extractor, depth_head, compute_midas_depth, batch
    )
    times_a = timed_steps(fn_a, N_WARMUP, N_TIMED, "WITH  MiDaS (depth step)")
    print()

    # ---- Condition B: WITHOUT MiDaS ----
    fn_b = lambda: step_without_midas(
        pipe, extractor, sem_head, batch
    )
    times_b = timed_steps(fn_b, N_WARMUP, N_TIMED, "WITHOUT MiDaS (CLIP-only step)")
    print()

    # ---- Report ----
    med_a = np.median(times_a) * 1000
    med_b = np.median(times_b) * 1000
    midas_time = med_a - med_b
    overhead_pct = midas_time / med_a * 100

    print("=" * 65)
    print("RESULTS")
    print(f"  WITH  MiDaS  — median: {med_a:.0f} ms  (min {times_a.min()*1000:.0f}  max {times_a.max()*1000:.0f})")
    print(f"  WITHOUT MiDaS — median: {med_b:.0f} ms  (min {times_b.min()*1000:.0f}  max {times_b.max()*1000:.0f})")
    print(f"\n  MiDaS overhead: {midas_time:.0f} ms  ({overhead_pct:.1f}% of full step)")
    print()
    if overhead_pct < 15.0:
        print("  DECISION: overhead < 15% → KEEP ONLINE")
        print("  → 7 files/object (rgb_0-5 + clip_emb.pt)")
        print("  → data_pipeline.py design unchanged")
    else:
        print("  DECISION: overhead ≥ 15% → PRE-COMPUTE DEPTH")
        print("  → Add MiDaS to render pipeline")
        print("  → 13 files/object (rgb_0-5 + depth_0-5.pt + clip_emb.pt)")
        print("  → Update ObjaverseRenderDataset to load from cache")
    print("=" * 65)


if __name__ == "__main__":
    main()
