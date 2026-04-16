"""
PGR-3D Motivation Experiment — Day 3 Gate
-----------------------------------------
Tests the core hypothesis: does CLIP-semantic content drift during
Wonder3D's denoising process?

Method
------
For each test image we:
  1. Compute CLIP ViT-L/14 embedding of the input front view → clip_input.
  2. Run Wonder3D's DDIM denoising loop (50 steps) with a callback that, at
     every step t, computes the *predicted clean sample* x̂_0 from the current
     noisy latents and the UNet noise prediction:

         x̂_0 = (z_t - σ_t · ε_θ) / α_t          [DDIM-style]

     This is exactly `scheduler.step(...).pred_original_sample`.
  3. Decode x̂_0 with the VAE → estimated clean RGB views (all 6).
  4. Embed each decoded view with CLIP → clip_step_t.
  5. Record cosine_sim(clip_input, clip_step_t) for each view at each step.

Expected outcome (if hypothesis is correct):
  - Cosine similarity should decrease monotonically as t decreases from 1000→0.
  - Or: it should fluctuate significantly, showing the CLIP signal is not
    preserved through denoising.

Outputs
-------
  outputs/motivation/
    clip_drift_data.csv           per-step cosine-sim for each object×view
    clip_drift_{object}.png       per-object line plot (views as separate lines)
    clip_drift_summary.png        mean ± std across all objects

Usage
-----
    python src/motivation_experiment.py \
        --objects mario alarm chicken elephant turtle \
        --output_dir outputs/motivation \
        --num_steps 50 \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
GSO_FRONTS    = "/scratch/s224696943/pgr_3d/data/gso_fronts"
NUM_VIEWS     = 6


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(device: str = "cuda"):
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image_joint import MVDiffusionImagePipeline
    pipe = MVDiffusionImagePipeline.from_pretrained(
        WONDER3D_CKPT,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.image_encoder.eval()
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


# ---------------------------------------------------------------------------
# CLIP helper (separate from data_pipeline to avoid loading MiDaS)
# ---------------------------------------------------------------------------
_clip_model = None
_clip_preproc = None


def get_clip(device: str = "cuda"):
    global _clip_model, _clip_preproc
    if _clip_model is None:
        import clip as clip_lib
        _clip_model, _clip_preproc = clip_lib.load("ViT-L/14", device=device)
        _clip_model = _clip_model.eval()
    return _clip_model, _clip_preproc


@torch.no_grad()
def clip_embed_pil(image: Image.Image, device: str = "cuda") -> torch.Tensor:
    """Return L2-normalised CLIP ViT-L/14 embedding [1, 768]."""
    model, preproc = get_clip(device)
    t = preproc(image).unsqueeze(0).to(device)
    emb = model.encode_image(t).float()
    return F.normalize(emb, dim=-1)


@torch.no_grad()
def clip_embed_tensor(rgb: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    rgb: [B, 3, H, W] float in [0, 1].
    Returns [B, 768] L2-normalised embeddings.
    """
    model, preproc = get_clip(device)
    # Resize to 224×224 using CLIP's expected size and normalise.
    # CLIP's preproc expects PIL; we replicate it manually for batch efficiency.
    from torchvision import transforms
    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std= (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    imgs = clip_transform(rgb.to(device))
    emb = model.encode_image(imgs).float()
    return F.normalize(emb, dim=-1)


# ---------------------------------------------------------------------------
# Wonder3D inference with per-step callback
# ---------------------------------------------------------------------------

def run_denoising_with_probe(
    pipe,
    front_pil: Image.Image,
    clip_input: torch.Tensor,   # [1, 768]
    num_steps: int = 50,
    guidance_scale: float = 3.0,
    device: str = "cuda",
) -> Dict:
    """
    Run Wonder3D denoising on a single front view, probing CLIP similarity at
    every denoising step.

    Returns dict:
      "timesteps":   list of int timestep values (length = num_steps)
      "step_indices": list of int step indices [0 .. num_steps-1]
      "cosine_sims": np.ndarray [num_steps, 6]  — per-view CLIP cosine sims
      "final_images": list of 6 PIL images (final generation)
    """
    from einops import repeat

    pipe.scheduler.set_timesteps(num_steps, device=device)
    timestep_list  = pipe.scheduler.timesteps.tolist()

    # ---- Encode input image ----
    image_pil_6 = [front_pil.convert("RGB")] * NUM_VIEWS * 2  # normal + rgb
    do_cfg = guidance_scale != 2.0  # pipeline logic: CFG if != 2.0

    image_embeddings, image_latents = pipe._encode_image(
        image_pil_6, device, 1, do_cfg
    )

    # Camera embeddings (same as pipeline __call__)
    dtype = pipe.vae.dtype
    cam_emb = pipe.camera_embedding.to(dtype)
    cam_emb = repeat(cam_emb, "Nv Nce -> (B Nv) Nce", B=1)
    camera_embeddings = pipe.prepare_camera_embedding(cam_emb, do_classifier_free_guidance=do_cfg)

    # Initial latents
    batch_size = NUM_VIEWS * 2
    latents = pipe.prepare_latents(
        batch_size, pipe.unet.config.out_channels,
        256, 256, image_embeddings.dtype, device, None
    )

    # Reshape for CD attention
    if do_cfg:
        image_embeddings = pipe.reshape_to_cd_input(image_embeddings)
        camera_embeddings = pipe.reshape_to_cd_input(camera_embeddings)
        image_latents = pipe.reshape_to_cd_input(image_latents)

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)

    # ---- Denoising loop with probe ----
    cosine_sims  = np.zeros((num_steps, NUM_VIEWS), dtype=np.float32)
    probe_step   = 0

    with torch.no_grad():
        for i, t in enumerate(pipe.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            if do_cfg:
                latent_model_input = pipe.reshape_to_cd_input(latent_model_input)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=image_embeddings,
                class_labels=camera_embeddings,
            ).sample

            if do_cfg:
                noise_pred = pipe.reshape_to_cfg_output(noise_pred)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step — also gives us pred_original_sample (x̂_0)
            step_output = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
            latents = step_output.prev_sample

            # ---- Probe: decode x̂_0 RGB domain and measure CLIP similarity ----
            # pred_original_sample has shape [2*V, 4, h, w] (both domains).
            # RGB domain = last V rows: [V:2V].
            x0_pred = step_output.pred_original_sample   # [2*V, 4, 32, 32]
            x0_rgb  = x0_pred[NUM_VIEWS:]                # [V, 4, 32, 32] — rgb domain
            x0_latents = x0_rgb / pipe.vae.config.scaling_factor
            decoded = pipe.vae.decode(x0_latents).sample   # [V, 3, 256, 256]
            decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)  # [0,1]

            clip_embs = clip_embed_tensor(decoded, device=device)   # [V, 768]
            sims = (clip_embs * clip_input.to(device)).sum(dim=-1)  # [V]
            cosine_sims[i] = sims.cpu().numpy()

            probe_step += 1

    # Decode final RGB-domain latents to PIL images
    final_latents = latents[NUM_VIEWS:] / pipe.vae.config.scaling_factor  # [V, 4, h, w]
    final_decoded = pipe.vae.decode(final_latents).sample
    final_decoded = ((final_decoded + 1.0) / 2.0).clamp(0, 1).cpu()
    final_images = [
        Image.fromarray((final_decoded[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        for i in range(NUM_VIEWS)
    ]

    return {
        "timesteps":    timestep_list,
        "step_indices": list(range(num_steps)),
        "cosine_sims":  cosine_sims,   # [steps, views]
        "final_images": final_images,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_object_drift(
    result: Dict,
    obj_name: str,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps   = result["step_indices"]            # 0..49
    t_vals  = result["timesteps"]               # 999..0
    sims    = result["cosine_sims"]             # [50, 6]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, NUM_VIEWS))

    for v in range(NUM_VIEWS):
        ax.plot(t_vals[::-1], sims[:, v], color=colors[v],
                alpha=0.8, linewidth=1.5, label=f"view {v}")

    ax.set_xlabel("Denoising timestep (0 = clean)", fontsize=12)
    ax.set_ylabel("CLIP cosine similarity to input", fontsize=12)
    ax.set_title(f"CLIP drift during Wonder3D denoising — {obj_name}", fontsize=13)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / f"clip_drift_{obj_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def plot_summary(
    all_results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    all_sims = []   # [N_objects, N_steps, N_views]
    t_vals   = None

    for obj, res in all_results.items():
        all_sims.append(res["cosine_sims"])
        if t_vals is None:
            t_vals = res["timesteps"]

    all_sims = np.stack(all_sims, axis=0)    # [N, steps, views]
    mean_sims = all_sims.mean(axis=(0, 2))   # [steps]
    std_sims  = all_sims.std(axis=(0, 2))    # [steps]

    t_arr = np.array(t_vals[::-1])
    ax.plot(t_arr, mean_sims[::-1], color="steelblue", linewidth=2.5, label="mean over objects & views")
    ax.fill_between(t_arr, (mean_sims - std_sims)[::-1], (mean_sims + std_sims)[::-1],
                    alpha=0.25, color="steelblue", label="±1 std")

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.6, label="similarity=0.5")
    ax.set_xlabel("Denoising timestep (0 = clean)", fontsize=12)
    ax.set_ylabel("CLIP cosine similarity to input", fontsize=12)
    ax.set_title(
        f"CLIP Drift Summary — {len(all_results)} GSO objects, Wonder3D denoising\n"
        "(low similarity = perception drift = PGR-3D is motivated)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / "clip_drift_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot saved: {out_path}")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def save_csv(all_results: Dict[str, Dict], output_dir: Path) -> None:
    csv_path = output_dir / "clip_drift_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object", "step_idx", "timestep",
                         "view0", "view1", "view2", "view3", "view4", "view5",
                         "mean_sim"])
        for obj, res in all_results.items():
            sims    = res["cosine_sims"]   # [steps, 6]
            t_vals  = res["timesteps"]
            for step_i, (t_val, sim_row) in enumerate(zip(t_vals, sims)):
                writer.writerow([
                    obj, step_i, int(t_val),
                    *[f"{s:.6f}" for s in sim_row],
                    f"{sim_row.mean():.6f}",
                ])
    print(f"CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PGR-3D Motivation Experiment")
    parser.add_argument("--objects",    nargs="+",
                        default=["mario", "alarm", "chicken", "elephant", "turtle"],
                        help="GSO object names to test")
    parser.add_argument("--gso_fronts", default=GSO_FRONTS,
                        help="Directory with {object}.png front view images")
    parser.add_argument("--output_dir", default="/scratch/s224696943/pgr_3d/outputs/motivation")
    parser.add_argument("--num_steps",  type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Wonder3D pipeline from {WONDER3D_CKPT} ...")
    pipe = load_pipeline(args.device)
    print("Pipeline loaded.\n")

    all_results: Dict[str, Dict] = {}

    for obj in args.objects:
        # Look for image under multiple possible names
        front_path = None
        for name in [f"{obj}.png", f"{obj}.jpg"]:
            p = Path(args.gso_fronts) / name
            if p.exists():
                front_path = p
                break
        if front_path is None:
            print(f"[WARN] Front image not found for {obj} in {args.gso_fronts} — skipping")
            continue

        print(f"Processing {obj} ({front_path}) ...")
        front_pil = Image.open(front_path).convert("RGB").resize((256, 256), Image.BILINEAR)

        # CLIP embedding of the input front view
        clip_input = clip_embed_pil(front_pil, device=args.device)   # [1, 768]

        result = run_denoising_with_probe(
            pipe,
            front_pil=front_pil,
            clip_input=clip_input,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            device=args.device,
        )

        all_results[obj] = result

        # Save per-object plot
        plot_object_drift(result, obj, output_dir)

        # Print summary statistics
        sims = result["cosine_sims"]   # [steps, 6]
        mean_at_start = sims[0].mean()
        mean_at_end   = sims[-1].mean()
        min_sim       = sims.min()
        print(f"  t=999 (noisy)  mean CLIP sim: {mean_at_start:.4f}")
        print(f"  t=  0 (clean)  mean CLIP sim: {mean_at_end:.4f}")
        print(f"  min sim across all steps/views: {min_sim:.4f}")
        drift = mean_at_start - mean_at_end
        print(f"  Drift (start→end): {drift:+.4f}  {'← DRIFT DETECTED' if drift > 0.05 else ''}\n")

    if not all_results:
        print("[ERROR] No objects processed. Check --gso_fronts path.")
        return

    # Summary plot and CSV
    plot_summary(all_results, output_dir)
    save_csv(all_results, output_dir)

    # Gate decision
    all_sims = np.stack([r["cosine_sims"] for r in all_results.values()], axis=0)
    mean_sim_end   = all_sims[:, -1, :].mean()   # mean at last step (t≈0)
    mean_sim_start = all_sims[:, 0,  :].mean()   # mean at first step (t≈999)
    total_drift    = mean_sim_start - mean_sim_end

    print("=" * 60)
    print("DAY-3 GATE RESULT")
    print(f"  Mean CLIP sim at t≈999 (noisy):  {mean_sim_start:.4f}")
    print(f"  Mean CLIP sim at t≈0   (clean):  {mean_sim_end:.4f}")
    print(f"  Total drift:                      {total_drift:+.4f}")
    if total_drift > 0.05:
        print("  VERDICT: DRIFT DETECTED ✓ — hypothesis supported, proceed with PGR-3D")
    elif total_drift > 0.01:
        print("  VERDICT: MARGINAL DRIFT — escalate to user before proceeding")
    else:
        print("  VERDICT: NO SIGNIFICANT DRIFT — ESCALATE TO USER IMMEDIATELY")
    print("=" * 60)
    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
