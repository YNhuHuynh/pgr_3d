"""
PGR-3D Feature Shape Sanity-Check
----------------------------------
Run this BEFORE submitting training jobs to verify:

  1. Wonder3D pipeline loads correctly.
  2. Forward hooks fire on up_blocks[1,2,3] with expected shapes.
  3. AggregationNetwork output shape is correct.
  4. SemanticHead and DepthHead forward pass produce expected shapes.
  5. Loss values are finite (not NaN/Inf).
  6. Gradient flows back through head to latents (no autograd errors).

Expected output (no errors):
  [1/6] Pipeline loaded
  [2/6] UNet hooks: 3 feature maps captured
        up_blocks[1]: torch.Size([12, 1280, 16, 16])
        up_blocks[2]: torch.Size([12,  640, 32, 32])
        up_blocks[3]: torch.Size([12,  320, 32, 32])
  [3/6] RGB features (training layout): 3 tensors
        feat[0]: torch.Size([6, 1280, 16, 16])
        feat[1]: torch.Size([6,  640, 32, 32])
        feat[2]: torch.Size([6,  320, 32, 32])
  [4/6] AggregationNetwork: torch.Size([6, 384, 32, 32])
  [5/6] SemanticHead output: torch.Size([1, 768])  loss: 0.XXXX (finite)
        DepthHead output:    torch.Size([6, 1, 256, 256])  loss: X.XXXX (finite)
  [6/6] Gradient check: grad w.r.t. latents  shape=torch.Size([12, 4, 32, 32])  norm=X.XXXX

Usage:
  conda activate pgr3d
  cd /scratch/s224696943/pgr_3d
  PYTHONPATH=src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D \
      python src/test_feature_shapes.py

  # On a CPU-only login node (slow but no GPU needed for shape checks):
  PYTHONPATH=src:/scratch/s224696943/3DRAV:/scratch/s224696943/Wonder3D \
      python src/test_feature_shapes.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
NUM_VIEWS     = 6
BATCH_SIZE    = 1


def check(cond: bool, msg: str):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  OK:   {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype",  default="float32", choices=["float32", "float16"])
    args = parser.parse_args()

    device = args.device
    dtype  = torch.float32 if args.dtype == "float32" else torch.float16
    V      = NUM_VIEWS
    B      = BATCH_SIZE
    BV     = B * V

    # ------------------------------------------------------------------
    print("\n[1/6] Loading Wonder3D pipeline ...")
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image_joint import MVDiffusionImagePipeline
    pipe = MVDiffusionImagePipeline.from_pretrained(
        WONDER3D_CKPT, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.eval()
    pipe.vae.eval()
    print("  OK:  pipeline loaded")

    # ------------------------------------------------------------------
    print("\n[2/6] Testing UNet forward hooks ...")
    from feature_extractor import Wonder3DFeatureExtractor

    extractor = Wonder3DFeatureExtractor(pipe.unet, num_views=V)

    # Build a minimal synthetic batch (2 domains × V views)
    # latent shape: [2*V, 8, 32, 32]  (4 noisy + 4 image-cond channels)
    latent_input = torch.randn(2 * V, 8, 32, 32, dtype=dtype, device=device)
    timesteps    = torch.zeros(2 * V, dtype=torch.long, device=device)

    # image embeddings: [2*V, 1, 768]
    img_embs  = torch.randn(2 * V, 1, 768, dtype=dtype, device=device)
    # camera embeddings: [2*V, 10]
    cam_embs  = pipe.camera_embedding[:V].repeat(2, 1).to(device=device, dtype=dtype)

    with extractor:
        with torch.no_grad():
            _ = pipe.unet(
                latent_input, timesteps,
                encoder_hidden_states=img_embs,
                class_labels=cam_embs,
            )

    raw = extractor.get_features()
    check(len(raw) == 3, f"3 hooks fired (got {len(raw)})")
    expected_shapes = {
        1: (2*V, 1280, 16, 16),
        2: (2*V,  640, 32, 32),
        3: (2*V,  320, 32, 32),
    }
    for idx, exp_shape in expected_shapes.items():
        actual = tuple(raw[idx].shape)
        check(actual == exp_shape,
              f"up_blocks[{idx}] shape {actual} == {exp_shape}")
        print(f"        up_blocks[{idx}]: {raw[idx].shape}")

    # ------------------------------------------------------------------
    print("\n[3/6] Testing RGB feature extraction (training layout) ...")
    rgb_features = extractor.get_rgb_features(use_cfg=False)
    check(len(rgb_features) == 3, f"3 RGB feature maps (got {len(rgb_features)})")
    expected_rgb = [
        (V, 1280, 16, 16),
        (V,  640, 32, 32),
        (V,  320, 32, 32),
    ]
    for i, (feat, exp) in enumerate(zip(rgb_features, expected_rgb)):
        actual = tuple(feat.shape)
        check(actual == exp, f"rgb_features[{i}] shape {actual} == {exp}")
        print(f"        feat[{i}]: {feat.shape}")

    # Also test batched extractor
    with extractor:
        with torch.no_grad():
            _ = pipe.unet(latent_input, timesteps,
                          encoder_hidden_states=img_embs,
                          class_labels=cam_embs)
    rgb_batched = extractor.get_rgb_features_batched(B, use_cfg=False)
    for i, feat in enumerate(rgb_batched):
        check(feat.shape[0] == BV,
              f"batched rgb_features[{i}] batch dim == {BV} (got {feat.shape[0]})")

    # ------------------------------------------------------------------
    print("\n[4/6] Testing AggregationNetwork ...")
    from readout_heads import AggregationNetwork, HOOK_CHANNELS, PROJECTION_DIM, COMMON_SPATIAL

    agg_net = AggregationNetwork(
        feature_dims=HOOK_CHANNELS,
        projection_dim=PROJECTION_DIM,
        common_spatial=COMMON_SPATIAL,
    ).to(device=device, dtype=dtype)

    # detach features so we can run without grad
    feats_for_agg = [f.detach().to(dtype) for f in rgb_features]
    with torch.no_grad():
        agg_out = agg_net(feats_for_agg)

    exp_agg = (V, PROJECTION_DIM, COMMON_SPATIAL, COMMON_SPATIAL)
    check(tuple(agg_out.shape) == exp_agg,
          f"AggregationNetwork output shape {tuple(agg_out.shape)} == {exp_agg}")
    print(f"        AggregationNetwork: {agg_out.shape}")

    # ------------------------------------------------------------------
    print("\n[5/6] Testing SemanticHead and DepthHead ...")
    from readout_heads import build_semantic_head, build_depth_head, SemanticHead, DepthHead

    sem_head   = build_semantic_head(num_views=V, device=device)
    depth_head = build_depth_head(target_size=256, device=device)
    sem_head   = sem_head.to(dtype)
    depth_head = depth_head.to(dtype)

    clip_target  = F.normalize(torch.randn(B, 768, dtype=dtype, device=device), dim=-1)
    depth_target = torch.rand(BV, 1, 256, 256, dtype=dtype, device=device)

    with torch.no_grad():
        sem_pred   = sem_head(feats_for_agg, batch_size=B)
        depth_pred = depth_head(feats_for_agg)

    check(tuple(sem_pred.shape) == (B, 768),
          f"SemanticHead output shape {tuple(sem_pred.shape)} == ({B}, 768)")
    check(tuple(depth_pred.shape) == (BV, 1, 256, 256),
          f"DepthHead output shape {tuple(depth_pred.shape)} == ({BV}, 1, 256, 256)")
    print(f"        SemanticHead: {sem_pred.shape}")
    print(f"        DepthHead:    {depth_pred.shape}")

    sem_loss   = SemanticHead.loss(sem_pred, clip_target)
    depth_loss = DepthHead.loss(depth_pred, depth_target)
    check(torch.isfinite(sem_loss),   f"SemanticHead loss finite: {sem_loss.item():.4f}")
    check(torch.isfinite(depth_loss), f"DepthHead loss finite: {depth_loss.item():.4f}")
    print(f"        SemanticHead loss: {sem_loss.item():.4f}")
    print(f"        DepthHead    loss: {depth_loss.item():.4f}")

    # ------------------------------------------------------------------
    print("\n[6/6] Testing gradient flow through head to latents ...")
    from guidance_inference import GuidedFeatureExtractor

    # Use float32 for gradient check regardless of dtype setting
    pipe_f32 = pipe  # already float32 from load
    guided_ext = GuidedFeatureExtractor(pipe_f32.unet, num_views=V)

    latents_grad = torch.randn(2*V, 4, 32, 32, requires_grad=True,
                               dtype=torch.float32, device=device)
    # Build 8-channel input (noisy + image-cond channels)
    cond_latent = torch.zeros(2*V, 4, 32, 32, dtype=torch.float32, device=device)
    lmi = torch.cat([latents_grad, cond_latent], dim=1)   # [2V, 8, 32, 32]

    img_embs_f32 = img_embs.float()
    cam_embs_f32 = cam_embs.float()
    t_zero       = torch.zeros(2*V, dtype=torch.long, device=device)

    sem_head_f32 = build_semantic_head(num_views=V, device=device)
    clip_tgt_f32 = F.normalize(torch.randn(B, 768, device=device), dim=-1)

    with guided_ext:
        noise_pred = pipe_f32.unet(
            lmi, t_zero,
            encoder_hidden_states=img_embs_f32,
            class_labels=cam_embs_f32,
        ).sample

    rgb_feats_grad = guided_ext.get_rgb_features_batched(B, use_cfg=False)

    pred_grad = sem_head_f32(rgb_feats_grad, batch_size=B)
    loss_grad = SemanticHead.loss(pred_grad, clip_tgt_f32)

    grad = torch.autograd.grad(loss_grad, latents_grad)[0]
    check(grad is not None,         "gradient exists")
    check(torch.isfinite(grad).all(), "gradient is finite")
    check(grad.shape == latents_grad.shape,
          f"gradient shape {grad.shape} == {latents_grad.shape}")
    print(f"        grad shape: {grad.shape}  norm: {grad.norm().item():.6f}")

    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("ALL CHECKS PASSED — pipeline and heads are wired correctly.")
    print("Safe to submit training jobs.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
