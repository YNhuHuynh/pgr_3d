"""
PGR-3D Guidance Inference
--------------------------
Wraps the Wonder3D MVDiffusionImagePipeline with readout-guided sampling.

At each denoising step the standard noise prediction is augmented by:

    ε_guided(z_t, t) = ε_θ(z_t, t)
                      - σ_t · η · ∇_{z_t} L_readout(z_t, t, target)

where:
  ε_θ       = frozen Wonder3D UNet noise prediction
  η         = guidance strength (sweep: 0.1, 0.5, 1.0, 2.0)
  L_readout = loss from the trained readout head (SemanticHead or DepthHead)
  target    = CLIP embedding of input front view (for SemanticHead)
              or MiDaS depth of front view (for DepthHead)

The gradient ∇_{z_t} L is computed via torch.autograd.grad, which requires
a forward pass through the UNet WITHOUT torch.no_grad().  The UNet parameters
have requires_grad=False so no param gradients are stored.

Key design choices
------------------
1. Gradient flows through UNet features to z_t (non-detaching hook in
   GuidedWonder3DExtractor).
2. The head network (AggregationNetwork + SemanticHead / DepthHead) must have
   requires_grad=True for its parameters, but we only use autograd.grad
   w.r.t. z_t, so no head optimizer step happens.
3. Timestep gating: guidance is only applied for t in [t_min, t_max] to
   avoid noisy gradients at very high t.
4. CFG and PGR guidance are applied sequentially (CFG first, then PGR).

Usage
-----
    from guidance_inference import build_guided_pipeline, run_guided_inference
    from readout_heads import build_semantic_head, load_head

    pipe = build_guided_pipeline()
    head = build_semantic_head()
    load_head("/path/to/checkpoint.pt", head)

    images = run_guided_inference(
        pipe, head, front_pil_image,
        eta=1.0,
        guidance_scale=3.0,
        t_guidance_min=0,
        t_guidance_max=500,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
NUM_VIEWS     = 6


# ---------------------------------------------------------------------------
# Non-detaching feature extractor (for gradient computation)
# ---------------------------------------------------------------------------

HOOK_BLOCK_INDICES = [1, 2, 3]


class GuidedFeatureExtractor:
    """
    Like Wonder3DFeatureExtractor but does NOT detach hook outputs,
    so autograd.grad can backprop through the features to z_t.

    Only used during guided inference steps; non-guided steps still use
    the cheaper detaching version from feature_extractor.py.
    """

    def __init__(self, unet: torch.nn.Module, num_views: int = 6):
        self.unet      = unet
        self.num_views = num_views
        self._hooks: List = []
        self._raw: Dict[int, torch.Tensor] = {}

    def __enter__(self):
        self._raw.clear()
        self._hooks.clear()
        for idx in HOOK_BLOCK_INDICES:
            def make_hook(i):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._raw[i] = hidden   # NO detach
                return hook
            h = self.unet.up_blocks[idx].register_forward_hook(make_hook(idx))
            self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_rgb_features_batched(
        self,
        batch_size: int,
        use_cfg: bool = True,
    ) -> List[torch.Tensor]:
        """
        Returns [3] tensors of shape [B*V, C, H, W] (RGB conditional features).
        use_cfg=True: CFG layout [norm_uc, norm_c, rgb_uc, rgb_c] × B*V
        """
        V = self.num_views
        B = batch_size
        feats = []
        for idx in HOOK_BLOCK_INDICES:
            f = self._raw[idx]
            if use_cfg:
                feats.append(f[3 * B * V : 4 * B * V])
            else:
                feats.append(f[B * V : 2 * B * V])
        return feats


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_guided_pipeline(device: str = "cuda"):
    """Load frozen Wonder3D pipeline."""
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image_joint import MVDiffusionImagePipeline
    pipe = MVDiffusionImagePipeline.from_pretrained(
        WONDER3D_CKPT,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.image_encoder.eval()
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


# ---------------------------------------------------------------------------
# Guidance gradient computation
# ---------------------------------------------------------------------------

def compute_guidance_grad(
    z_t: torch.Tensor,            # [V, 4, h, w] REQUIRES GRAD
    latent_model_input: torch.Tensor,  # already-built input (with image latent concat)
    t_scalar: torch.Tensor,
    image_embeddings: torch.Tensor,
    camera_embeddings: torch.Tensor,
    pipe,
    extractor: GuidedFeatureExtractor,
    head: torch.nn.Module,
    target: torch.Tensor,         # [1, 768] for semantic or [V, 1, H, W] for depth
    head_type: str = "semantic",
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Compute ∇_{z_t} L_readout.

    Returns gradient tensor of same shape as z_t.
    """
    from readout_heads import SemanticHead, DepthHead

    with extractor:
        # UNet forward — no torch.no_grad() so computation graph is retained
        noise_pred_full = pipe.unet(
            latent_model_input, t_scalar,
            encoder_hidden_states=image_embeddings,
            class_labels=camera_embeddings,
        ).sample

    # Extract RGB conditional features [BV, C, H, W]
    rgb_features = extractor.get_rgb_features_batched(batch_size, use_cfg=True)

    # Head forward + loss
    if head_type == "semantic":
        pred = head(rgb_features, batch_size=batch_size)   # [B, 768]
        loss = SemanticHead.loss(pred, target.to(z_t.device))
    elif head_type == "depth":
        pred = head(rgb_features)                          # [BV, 1, H, W]
        loss = DepthHead.loss(pred, target.to(z_t.device))
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    # Gradient of loss w.r.t. z_t only (not UNet params — they have no grad)
    grad = torch.autograd.grad(loss, z_t, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(z_t)
    return grad.detach()


# ---------------------------------------------------------------------------
# Guided denoising loop
# ---------------------------------------------------------------------------

def run_guided_inference(
    pipe,
    head: torch.nn.Module,
    front_pil: Image.Image,
    target: Optional[torch.Tensor] = None,    # auto-computed from front_pil if None
    head_type: str = "semantic",
    eta: float = 1.0,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 50,
    t_guidance_min: int = 0,
    t_guidance_max: int = 800,
    device: str = "cuda",
    seed: Optional[int] = 42,
) -> Dict:
    """
    Run Wonder3D with PGR-3D readout guidance.

    Parameters
    ----------
    pipe         : loaded Wonder3D pipeline
    head         : trained SemanticHead or DepthHead
    front_pil    : input front-view PIL image (RGB, 256×256)
    target       : precomputed target tensor; computed from front_pil if None
      - semantic: CLIP embedding [1, 768]
      - depth: MiDaS depth [1, 1, H, W] (of front view)
    eta          : guidance strength
    guidance_scale : Wonder3D CFG scale
    t_guidance_min/max : timestep range in which to apply guidance
    device       : "cuda" or "cpu"
    seed         : random seed for reproducibility

    Returns
    -------
    dict with keys:
      "images"        : list of 6 PIL images (RGB views)
      "normal_images" : list of 6 PIL images (normal views)
      "latents"       : final latent tensor
      "guidance_applied_steps": number of steps where guidance was applied
    """
    from einops import repeat
    from data_pipeline import compute_midas_depth

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # ---- Prepare target ----
    if target is None:
        if head_type == "semantic":
            import clip as clip_lib
            import torchvision.transforms as transforms
            clip_model, clip_preproc = clip_lib.load("ViT-L/14", device=device)
            clip_model.eval()
            with torch.no_grad():
                t_in = clip_preproc(front_pil).unsqueeze(0).to(device)
                emb  = clip_model.encode_image(t_in).float()
                target = F.normalize(emb, dim=-1)   # [1, 768]
        elif head_type == "depth":
            import torchvision.transforms as T
            img_t = T.ToTensor()(front_pil).unsqueeze(0).to(device)  # [1, 3, H, W]
            with torch.no_grad():
                target = compute_midas_depth(img_t, device=device)    # [1, 1, H, W]
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    target = target.to(device)
    head   = head.to(device)
    head.eval()

    # ---- Setup scheduler ----
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    do_cfg = guidance_scale != 2.0
    image_pil_6 = [front_pil.convert("RGB")] * NUM_VIEWS * 2

    image_embeddings, image_latents = pipe._encode_image(
        image_pil_6, device, 1, do_cfg
    )

    dtype = pipe.vae.dtype
    cam_emb = pipe.camera_embedding.to(dtype)
    cam_emb = repeat(cam_emb, "Nv Nce -> (B Nv) Nce", B=1)
    camera_embeddings = pipe.prepare_camera_embedding(cam_emb, do_classifier_free_guidance=do_cfg)

    batch_size = NUM_VIEWS * 2
    latents = pipe.prepare_latents(
        batch_size, pipe.unet.config.out_channels,
        256, 256, image_embeddings.dtype, device, generator
    )

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)

    if do_cfg:
        image_embeddings  = pipe.reshape_to_cd_input(image_embeddings)
        camera_embeddings = pipe.reshape_to_cd_input(camera_embeddings)
        image_latents     = pipe.reshape_to_cd_input(image_latents)

    guided_extractor = GuidedFeatureExtractor(pipe.unet, num_views=NUM_VIEWS)
    guidance_applied = 0

    # ---- Denoising loop ----
    for i, t in enumerate(pipe.scheduler.timesteps):
        t_val = int(t.item())
        apply_guidance = (eta > 0.0) and (t_guidance_min <= t_val <= t_guidance_max)

        # ---- Build latent model input ----
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        if do_cfg:
            latent_model_input = pipe.reshape_to_cd_input(latent_model_input)
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        if apply_guidance:
            # We need gradient w.r.t. the RGB latents only.
            # The 'latents' tensor holds [V, 4, h, w] (just the RGB domain in no-CFG,
            # but with CFG it's also just the RGB half after reshape).
            # We enable grad on latents temporarily.
            latents_for_grad = latents.detach().requires_grad_(True)

            # Rebuild latent_model_input with grad-enabled latents
            lmi_grad = torch.cat([latents_for_grad] * 2) if do_cfg else latents_for_grad
            if do_cfg:
                lmi_grad = pipe.reshape_to_cd_input(lmi_grad)
            lmi_grad = torch.cat([lmi_grad, image_latents], dim=1)
            lmi_grad = pipe.scheduler.scale_model_input(lmi_grad, t)

            # ---- Standard UNet forward (no_grad) for noise pred ----
            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input, t,
                    encoder_hidden_states=image_embeddings,
                    class_labels=camera_embeddings,
                ).sample

            # ---- CFG guidance ----
            if do_cfg:
                noise_pred = pipe.reshape_to_cfg_output(noise_pred)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # ---- Readout guidance gradient ----
            grad = compute_guidance_grad(
                z_t=latents_for_grad,
                latent_model_input=lmi_grad,
                t_scalar=t,
                image_embeddings=image_embeddings,
                camera_embeddings=camera_embeddings,
                pipe=pipe,
                extractor=guided_extractor,
                head=head,
                target=target,
                head_type=head_type,
                batch_size=1,
            )

            # σ_t from scheduler (std of noise at timestep t)
            # diffusers stores alphas_cumprod; σ_t = sqrt(1 - ᾱ_t)
            alpha_cumprod_t = pipe.scheduler.alphas_cumprod[t_val].to(device)
            sigma_t = (1.0 - alpha_cumprod_t).sqrt()

            # Apply guidance to the RGB latents' noise prediction
            # noise_pred shape: [V, 4, h, w] (after CFG merging)
            # grad shape: [V, 4, h, w]
            noise_pred = noise_pred - sigma_t * eta * grad

            guidance_applied += 1
        else:
            # Standard path — no gradient tracking needed
            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input, t,
                    encoder_hidden_states=image_embeddings,
                    class_labels=camera_embeddings,
                ).sample

            if do_cfg:
                noise_pred = pipe.reshape_to_cfg_output(noise_pred)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        with torch.no_grad():
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    # ---- Decode final latents ----
    with torch.no_grad():
        # latents: [V*2, 4, h, w] → split domains
        # In joint generation, the pipeline decodes both domains.
        # The RGB views are the last V; normal are first V.
        latents_rgb    = latents[NUM_VIEWS:]
        latents_normal = latents[:NUM_VIEWS]

        def decode_views(lats):
            lats_scaled = lats / pipe.vae.config.scaling_factor
            decoded = pipe.vae.decode(lats_scaled).sample
            decoded = ((decoded + 1.0) / 2.0).clamp(0, 1).cpu()
            return [
                Image.fromarray((decoded[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                for i in range(decoded.shape[0])
            ]

        rgb_images    = decode_views(latents_rgb)
        normal_images = decode_views(latents_normal)

    return {
        "images":         rgb_images,
        "normal_images":  normal_images,
        "latents":        latents.detach().cpu(),
        "guidance_applied_steps": guidance_applied,
    }


# ---------------------------------------------------------------------------
# Sweep helper: run multiple η values
# ---------------------------------------------------------------------------

def run_eta_sweep(
    pipe,
    head: torch.nn.Module,
    front_pil: Image.Image,
    target: torch.Tensor,
    head_type: str = "semantic",
    eta_values: List[float] = (0.0, 0.1, 0.5, 1.0, 2.0),
    guidance_scale: float = 3.0,
    num_inference_steps: int = 50,
    t_guidance_min: int = 0,
    t_guidance_max: int = 800,
    device: str = "cuda",
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[float, Dict]:
    """
    Run guided inference for multiple η values.
    Returns dict: {eta: result_dict}.
    Optionally saves generated images to output_dir/{eta_X.X}/.
    """
    results = {}
    for eta in eta_values:
        print(f"  η = {eta:.2f} ...")
        res = run_guided_inference(
            pipe, head, front_pil,
            target=target,
            head_type=head_type,
            eta=eta,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            t_guidance_min=t_guidance_min,
            t_guidance_max=t_guidance_max,
            device=device,
            seed=seed,
        )
        results[eta] = res

        if output_dir is not None:
            import os
            eta_dir = Path(output_dir) / f"eta_{eta:.1f}"
            eta_dir.mkdir(parents=True, exist_ok=True)
            for v, img in enumerate(res["images"]):
                img.save(eta_dir / f"rgb_{v:02d}.png")
            for v, img in enumerate(res["normal_images"]):
                img.save(eta_dir / f"normal_{v:02d}.png")
            print(f"    Saved to {eta_dir}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PGR-3D Guided Inference")
    parser.add_argument("--front_image",  required=True,  help="Path to front view PNG")
    parser.add_argument("--head_ckpt",    required=True,  help="Path to trained head checkpoint")
    parser.add_argument("--head_type",    default="semantic", choices=["semantic", "depth"])
    parser.add_argument("--eta",          type=float, default=1.0)
    parser.add_argument("--eta_sweep",    action="store_true", help="Run η ∈ {0,0.1,0.5,1,2}")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_steps",    type=int, default=50)
    parser.add_argument("--t_min",        type=int, default=0)
    parser.add_argument("--t_max",        type=int, default=800)
    parser.add_argument("--output_dir",   required=True, help="Output directory")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    from readout_heads import build_semantic_head, build_depth_head, load_head

    print("Loading Wonder3D pipeline ...")
    pipe = build_guided_pipeline(args.device)

    print(f"Loading {args.head_type} head from {args.head_ckpt} ...")
    if args.head_type == "semantic":
        head = build_semantic_head(device=args.device)
    else:
        head = build_depth_head(device=args.device)
    load_head(args.head_ckpt, head, device=args.device)

    front_pil = Image.open(args.front_image).convert("RGB").resize((256, 256), Image.BILINEAR)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eta_sweep:
        results = run_eta_sweep(
            pipe, head, front_pil,
            target=None,
            head_type=args.head_type,
            eta_values=[0.0, 0.1, 0.5, 1.0, 2.0],
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            t_guidance_min=args.t_min,
            t_guidance_max=args.t_max,
            device=args.device,
            seed=args.seed,
            output_dir=str(out_dir),
        )
        print(f"\nEta sweep complete. {len(results)} runs saved to {out_dir}")
    else:
        res = run_guided_inference(
            pipe, head, front_pil,
            head_type=args.head_type,
            eta=args.eta,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            t_guidance_min=args.t_min,
            t_guidance_max=args.t_max,
            device=args.device,
            seed=args.seed,
        )
        for v, img in enumerate(res["images"]):
            img.save(out_dir / f"rgb_{v:02d}.png")
        for v, img in enumerate(res["normal_images"]):
            img.save(out_dir / f"normal_{v:02d}.png")
        print(f"\nGenerated images saved to {out_dir}")
        print(f"Guidance applied at {res['guidance_applied_steps']}/{args.num_steps} steps")
