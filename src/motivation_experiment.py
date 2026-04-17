"""
PGR-3D Motivation Experiment — Day 3 Gate  (v2: pre-CD vs post-CD CKA)
-----------------------------------------------------------------------
Simultaneously probes TWO locations in the Wonder3D UNet at every
denoising step:

  (A) POST-cross-domain-attention (post-CD):
      Output of up_blocks[1,2,3] — the standard RGB generation stream
      after the JointAttnProcessor has mixed normal-domain information in.
      This is what our readout heads ultimately train on.

  (B) PRE-cross-domain-attention (pre-CD):
      Input to norm_joint_mid in the LAST BasicTransformerBlock of
      each up_block's final Transformer2DModel.  This is the RGB feature
      BEFORE cross-domain mixing — pure RGB self-attention + CLIP
      cross-attention output.

For each location and each timestep t we compute:

  1. CLIP cosine similarity (image-level):
       Decode x̂_0 (pred_original_sample) with the VAE → RGB images,
       embed with CLIP ViT-L/14, compute cosine_sim vs input image.
       This is the original Day-3 gate metric.

  2. Linear CKA (feature-level):
       For each up_block, mean-pool the feature tensor to one vector
       per view, accumulate over all objects, and compute linear CKA
       between the [N_obj×V, C] feature matrix and the
       [N_obj×V, 768] CLIP step-embedding matrix.
       CKA ∈ [0,1]; higher = features are more linearly predictive
       of CLIP content.

Three scientific scenarios (documented in docs/motivation_evidence.md):
  * Both drift similarly → cross-domain attention doesn't rescue CLIP signal
  * Post-CD drifts MORE   → cross-domain attention introduces additional drift
  * Pre-CD drifts MORE    → cross-domain attention rescues CLIP signal
                            (would require paper reframe)

Outputs
-------
  outputs/motivation/
    clip_drift_data.csv           per-step cosine-sim (objects × views × steps)
    clip_drift_{object}.png       per-object CLIP cosine-sim line plots
    clip_drift_summary.png        mean ± std CLIP cosine-sim across objects
    cka_pre_vs_post.png           SIDE-BY-SIDE: CKA(pre-CD, CLIP) and CKA(post-CD, CLIP)
                                  3 lines per panel (one per up_block)
    cka_data.csv                  CKA values per block per timestep

Usage
-----
    python src/motivation_experiment.py \\
        --objects mario alarm chicken elephant ... \\   # 25 GSO objects
        --output_dir outputs/motivation \\
        --num_steps 50 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch/s224696943/3DRAV")
sys.path.insert(0, "/scratch/s224696943/Wonder3D")

from data_pipeline import render_object   # for Objaverse GLB rendering

WONDER3D_CKPT = "/scratch/s224696943/wonder3d-v1.0"
GSO_FRONTS    = "/scratch/s224696943/pgr_3d/data/gso_fronts"
NUM_VIEWS     = 6
HOOK_BLOCK_INDICES = [1, 2, 3]


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(device: str = "cuda"):
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image_joint import MVDiffusionImagePipeline
    pipe = MVDiffusionImagePipeline.from_pretrained(
        WONDER3D_CKPT, torch_dtype=torch.float32, local_files_only=True,
    ).to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.image_encoder.eval()
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.requires_grad_(False)
    # xformers requires CUDA; do not attempt on CPU
    if device != "cpu":
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


# ---------------------------------------------------------------------------
# CLIP helper
# ---------------------------------------------------------------------------
_clip_model  = None
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
    """L2-normalised CLIP ViT-L/14 embedding [1, 768]."""
    model, preproc = get_clip(device)
    t = preproc(image).unsqueeze(0).to(device)
    return F.normalize(model.encode_image(t).float(), dim=-1)


@torch.no_grad()
def clip_embed_tensor(rgb: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    rgb: [B, 3, H, W] float in [0, 1].
    Returns [B, 768] L2-normalised embeddings.
    """
    from torchvision import transforms
    model, _ = get_clip(device)
    clip_tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std= (0.26862954, 0.26130258, 0.27577711)),
    ])
    imgs = clip_tf(rgb.to(device))
    return F.normalize(model.encode_image(imgs).float(), dim=-1)


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Unbiased linear CKA between feature matrices X [n, p] and Y [n, q].
    Both are mean-centred before computing gram matrices.

    Returns scalar in [0, 1]: 0 = no linear relationship, 1 = perfect.
    """
    X = X.double() - X.double().mean(0, keepdim=True)
    Y = Y.double() - Y.double().mean(0, keepdim=True)
    XtY = X.T @ Y           # [p, q]
    XtX = X.T @ X           # [p, p]
    YtY = Y.T @ Y           # [q, q]
    numer = (XtY ** 2).sum()
    denom = ((XtX ** 2).sum() * (YtY ** 2).sum()).sqrt()
    if denom < 1e-12:
        return 0.0
    return (numer / denom).item()


# ---------------------------------------------------------------------------
# Dual-location feature extractor
# ---------------------------------------------------------------------------

class _DualExtractor:
    """
    Registers two sets of hooks on the frozen Wonder3D UNet:

    PRE-CD  — forward_pre_hook on norm_joint_mid of the LAST BasicTransformerBlock
              in up_blocks[i].attentions[-1].  Fires BEFORE JointAttnProcessor
              mixes normal and RGB domains.  hidden_states: [batch, seq_len, C].

    POST-CD — forward_hook on up_blocks[i].  Fires AFTER all transformer blocks
              and the optional upsample.  hidden_states: [batch, C, H, W].

    For training layout (no CFG): batch = 2*V, RGB domain = rows [V : 2V].
    For CFG inference:            batch = 4*V, RGB cond  = rows [3V : 4V].
    """

    def __init__(self, unet: torch.nn.Module, num_views: int = NUM_VIEWS):
        self.unet      = unet
        self.num_views = num_views
        self._hooks: List = []
        self._pre_raw:  Dict[int, torch.Tensor] = {}   # block_idx → token-form
        self._post_raw: Dict[int, torch.Tensor] = {}   # block_idx → spatial-form

    # ---- context manager ----

    def __enter__(self):
        self._pre_raw.clear()
        self._post_raw.clear()
        self._register()
        return self

    def __exit__(self, *_):
        self._unregister()

    def _register(self):
        for bi in HOOK_BLOCK_INDICES:
            tb = self.unet.up_blocks[bi].attentions[-1].transformer_blocks[-1]

            # PRE-CD: forward_pre_hook — args[0] = hidden_states [batch, seq_len, C]
            def _make_pre(idx):
                def _hook(module, args):
                    self._pre_raw[idx] = args[0].detach()
                return _hook
            self._hooks.append(
                tb.norm_joint_mid.register_forward_pre_hook(_make_pre(bi))
            )

            # POST-CD: forward_hook — output = tensor [batch, C, H, W]
            def _make_post(idx):
                def _hook(module, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    self._post_raw[idx] = h.detach()
                return _hook
            self._hooks.append(
                self.unet.up_blocks[bi].register_forward_hook(_make_post(bi))
            )

    def _unregister(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ---- feature accessors ----

    def get_pre_rgb(self, use_cfg: bool) -> Dict[int, torch.Tensor]:
        """
        Returns {block_idx: [V, C]} — mean-pooled over token sequence.
        Slices RGB domain from [batch, seq_len, C].
        """
        V = self.num_views
        out = {}
        for bi, feat in self._pre_raw.items():
            rgb = feat[3*V:4*V] if use_cfg else feat[V:2*V]   # [V, seq_len, C]
            out[bi] = rgb.mean(dim=1)                           # [V, C]
        return out

    def get_post_rgb(self, use_cfg: bool) -> Dict[int, torch.Tensor]:
        """
        Returns {block_idx: [V, C]} — mean-pooled over spatial dims.
        Slices RGB domain from [batch, C, H, W].
        """
        V = self.num_views
        out = {}
        for bi, feat in self._post_raw.items():
            rgb = feat[3*V:4*V] if use_cfg else feat[V:2*V]   # [V, C, H, W]
            out[bi] = rgb.mean(dim=(2, 3))                      # [V, C]
        return out


# ---------------------------------------------------------------------------
# Denoising loop with dual-location probe
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
    Run one Wonder3D denoising trajectory and record at every step:
      - CLIP cosine similarity of decoded x̂_0 vs input image
      - Pre-CD and post-CD mean-pooled feature vectors for all up_blocks

    Returns dict with keys:
      timesteps        list[int]                    (length = num_steps)
      cosine_sims      np.ndarray [steps, V]
      pre_cd_feats     {block_idx: Tensor [steps, V, C]}
      post_cd_feats    {block_idx: Tensor [steps, V, C]}
      clip_step_embs   Tensor [steps, V, 768]
      final_images     list[PIL.Image]
    """
    from einops import repeat

    pipe.scheduler.set_timesteps(num_steps, device=device)
    timestep_list = pipe.scheduler.timesteps.tolist()
    do_cfg        = (guidance_scale != 2.0)

    # ---- Encode input ----
    image_pil_6 = [front_pil.convert("RGB")] * NUM_VIEWS * 2
    image_embeddings, image_latents = pipe._encode_image(
        image_pil_6, device, 1, do_cfg
    )
    cam_emb = pipe.camera_embedding.to(pipe.vae.dtype)
    cam_emb = repeat(cam_emb, "Nv Nce -> (B Nv) Nce", B=1)
    camera_embeddings = pipe.prepare_camera_embedding(cam_emb, do_classifier_free_guidance=do_cfg)

    latents = pipe.prepare_latents(
        NUM_VIEWS * 2, pipe.unet.config.out_channels,
        256, 256, image_embeddings.dtype, device, None,
    )
    if do_cfg:
        image_embeddings  = pipe.reshape_to_cd_input(image_embeddings)
        camera_embeddings = pipe.reshape_to_cd_input(camera_embeddings)
        image_latents     = pipe.reshape_to_cd_input(image_latents)

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)
    V = NUM_VIEWS

    # ---- Per-step storage ----
    cosine_sims     = np.zeros((num_steps, V), dtype=np.float32)
    clip_step_list  = []               # [V, 768] per step
    pre_list        = {bi: [] for bi in HOOK_BLOCK_INDICES}
    post_list       = {bi: [] for bi in HOOK_BLOCK_INDICES}

    extractor = _DualExtractor(pipe.unet, num_views=V)
    extractor._register()

    try:
        with torch.no_grad():
            for i, t in enumerate(pipe.scheduler.timesteps):
                lmi = torch.cat([latents] * 2) if do_cfg else latents
                if do_cfg:
                    lmi = pipe.reshape_to_cd_input(lmi)
                lmi = torch.cat([lmi, image_latents], dim=1)
                lmi = pipe.scheduler.scale_model_input(lmi, t)

                noise_pred = pipe.unet(
                    lmi, t,
                    encoder_hidden_states=image_embeddings,
                    class_labels=camera_embeddings,
                ).sample

                # Capture features BEFORE any CFG manipulation
                pre_feats  = extractor.get_pre_rgb(use_cfg=do_cfg)
                post_feats = extractor.get_post_rgb(use_cfg=do_cfg)

                if do_cfg:
                    noise_pred = pipe.reshape_to_cfg_output(noise_pred)
                    uc, cond   = noise_pred.chunk(2)
                    noise_pred = uc + guidance_scale * (cond - uc)

                step_out = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents  = step_out.prev_sample

                # ---- CLIP cosine similarity from decoded x̂_0 ----
                x0_rgb     = step_out.pred_original_sample[V:]   # RGB domain [V, 4, h, w]
                x0_latents = x0_rgb / pipe.vae.config.scaling_factor
                decoded    = pipe.vae.decode(x0_latents).sample  # [V, 3, 256, 256]
                decoded    = ((decoded + 1.0) / 2.0).clamp(0, 1)

                clip_step  = clip_embed_tensor(decoded, device=device)   # [V, 768]
                sims       = (clip_step * clip_input.to(device)).sum(-1) # [V]
                cosine_sims[i] = sims.cpu().numpy()
                clip_step_list.append(clip_step.cpu())

                # ---- Store feature vectors ----
                for bi in HOOK_BLOCK_INDICES:
                    pre_list[bi].append(pre_feats[bi].cpu())
                    post_list[bi].append(post_feats[bi].cpu())

    finally:
        extractor._unregister()

    # ---- Decode final images ----
    with torch.no_grad():
        final_lat = latents[V:] / pipe.vae.config.scaling_factor
        final_dec = pipe.vae.decode(final_lat).sample
        final_dec = ((final_dec + 1.0) / 2.0).clamp(0, 1).cpu()
    final_images = [
        Image.fromarray((final_dec[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        for i in range(V)
    ]

    return {
        "timesteps":      timestep_list,
        "step_indices":   list(range(num_steps)),
        "cosine_sims":    cosine_sims,
        "pre_cd_feats":   {bi: torch.stack(pre_list[bi],  dim=0) for bi in HOOK_BLOCK_INDICES},
        "post_cd_feats":  {bi: torch.stack(post_list[bi], dim=0) for bi in HOOK_BLOCK_INDICES},
        "clip_step_embs": torch.stack(clip_step_list, dim=0),   # [steps, V, 768]
        # clip_input stored for input-front CKA (constant across steps; [1, 768])
        "clip_input":     clip_input.cpu(),
        "final_images":   final_images,
    }


# ---------------------------------------------------------------------------
# CKA computation (aggregated over objects)
# ---------------------------------------------------------------------------

def compute_cka_curves(
    all_results: Dict[str, Dict],
    num_steps: int,
) -> Dict:
    """
    Three CKA variants (see docs/VIEW_HANDLING_DECISIONS.md Q1):

    1. decoded-all  [n = N_obj*V = 150]:
         X = features all views;  Y = decoded CLIP of same view at same step.
         Measures: do features predict their own decoded output?  Most statistical
         power but confounded at high-noise timesteps.

    2. decoded-per-view  [n = N_obj = 25, per view]:
         Same as (1) but split by view index.  Reveals whether front view (v=0)
         drifts differently from novel views (v=1..5).

    3. input-front  [n = N_obj = 25]  ← PRIMARY paper metric:
         X = features of view 0 only;  Y = INPUT CLIP of each object (constant
         across timesteps, varies across objects).
         Measures: do front-view features predict the input image's CLIP embedding?
         Decreasing trend = CLIP signal drifts in the front-view feature stream.
         Y is NOT rank-1 because each object has a different input CLIP embedding.

    Returns dict with keys:
      cka_pre_decoded_all   [steps, 3]   — variants 1, pre-CD
      cka_post_decoded_all  [steps, 3]
      cka_pre_per_view      [steps, 3, V]  — variant 2, pre-CD
      cka_post_per_view     [steps, 3, V]
      cka_pre_input_front   [steps, 3]   — variant 3, pre-CD
      cka_post_input_front  [steps, 3]
    """
    V = NUM_VIEWS

    # Accumulators: per step, per block
    pre_accum  = {bi: [[] for _ in range(num_steps)] for bi in HOOK_BLOCK_INDICES}
    post_accum = {bi: [[] for _ in range(num_steps)] for bi in HOOK_BLOCK_INDICES}
    clip_decoded_accum = [[] for _ in range(num_steps)]   # [V, 768] per object per step
    clip_input_accum   = []                               # [1, 768] per object (step-invariant)

    for result in all_results.values():
        clip_input_accum.append(result["clip_input"])   # [1, 768]
        for s in range(num_steps):
            clip_decoded_accum[s].append(result["clip_step_embs"][s])   # [V, 768]
            for bi in HOOK_BLOCK_INDICES:
                pre_accum[bi][s].append(result["pre_cd_feats"][bi][s])    # [V, C]
                post_accum[bi][s].append(result["post_cd_feats"][bi][s])  # [V, C]

    # [N_obj, 768] — one input CLIP per object, same across all steps
    Y_input = torch.cat(clip_input_accum, dim=0)   # [N, 768]

    n_blocks = len(HOOK_BLOCK_INDICES)
    cka_pre_decoded_all   = np.zeros((num_steps, n_blocks), dtype=np.float64)
    cka_post_decoded_all  = np.zeros((num_steps, n_blocks), dtype=np.float64)
    cka_pre_per_view      = np.zeros((num_steps, n_blocks, V), dtype=np.float64)
    cka_post_per_view     = np.zeros((num_steps, n_blocks, V), dtype=np.float64)
    cka_pre_input_front   = np.zeros((num_steps, n_blocks), dtype=np.float64)
    cka_post_input_front  = np.zeros((num_steps, n_blocks), dtype=np.float64)

    print(f"  Computing CKA curves (3 variants, {len(all_results)} objects) ...", end="", flush=True)
    for s in range(num_steps):
        # Stack across objects: all views
        Y_dec_all = torch.cat(clip_decoded_accum[s], dim=0)   # [N*V, 768]

        for bi_idx, bi in enumerate(HOOK_BLOCK_INDICES):
            pre_stacked  = torch.cat(pre_accum[bi][s],  dim=0)   # [N*V, C]
            post_stacked = torch.cat(post_accum[bi][s], dim=0)   # [N*V, C]

            # --- Variant 1: decoded-all ---
            cka_pre_decoded_all[s,  bi_idx] = linear_cka(pre_stacked,  Y_dec_all)
            cka_post_decoded_all[s, bi_idx] = linear_cka(post_stacked, Y_dec_all)

            # --- Variant 2: decoded-per-view ---
            for v in range(V):
                # Every V-th row belongs to view v (objects are stacked in order)
                X_pre_v  = pre_stacked[v::V]    # [N, C]
                X_post_v = post_stacked[v::V]   # [N, C]
                Y_dec_v  = Y_dec_all[v::V]      # [N, 768]
                cka_pre_per_view[s,  bi_idx, v] = linear_cka(X_pre_v,  Y_dec_v)
                cka_post_per_view[s, bi_idx, v] = linear_cka(X_post_v, Y_dec_v)

            # --- Variant 3: input-front (PRIMARY) ---
            X_pre_front  = pre_stacked[0::V]    # [N, C]  — view 0 of each object
            X_post_front = post_stacked[0::V]   # [N, C]
            # Y_input: [N, 768] — input CLIP per object, constant across steps
            cka_pre_input_front[s,  bi_idx] = linear_cka(X_pre_front,  Y_input)
            cka_post_input_front[s, bi_idx] = linear_cka(X_post_front, Y_input)

    print(" done.")

    return {
        "cka_pre_decoded_all":  cka_pre_decoded_all,
        "cka_post_decoded_all": cka_post_decoded_all,
        "cka_pre_per_view":     cka_pre_per_view,
        "cka_post_per_view":    cka_post_per_view,
        "cka_pre_input_front":  cka_pre_input_front,
        "cka_post_input_front": cka_post_input_front,
    }


# ---------------------------------------------------------------------------
# Bootstrap 95% CIs on the primary (input-front) CKA metric
# ---------------------------------------------------------------------------

def bootstrap_cka_ci(
    all_results: Dict[str, Dict],
    num_steps: int,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> Dict:
    """
    Non-parametric bootstrap 95% CIs on the primary (input-front) CKA metric.

    Resamples N objects with replacement n_bootstrap times.  For each resample
    computes input-front CKA (view-0 features vs input CLIP) at every step and
    block index.  Returns 2.5/97.5 percentile bands.

    With n_bootstrap=200, N=30, 50 steps, 3 blocks the computation takes
    roughly 15-30 s on CUDA (linear algebra on [30, C] matrices).

    Returns dict keys:
      ci_pre_lo   [steps, n_blocks]  — 2.5th  pct, pre-CD
      ci_pre_hi   [steps, n_blocks]  — 97.5th pct, pre-CD
      ci_post_lo  [steps, n_blocks]  — 2.5th  pct, post-CD
      ci_post_hi  [steps, n_blocks]  — 97.5th pct, post-CD
    """
    rng      = np.random.default_rng(seed)
    objects  = list(all_results.values())
    N        = len(objects)
    n_blocks = len(HOOK_BLOCK_INDICES)

    # Pre-index: [obj_idx][bi_idx] → Tensor [steps, C]  (view-0 only)
    pre_vecs   = [[obj["pre_cd_feats"][bi][:, 0, :]  for bi in HOOK_BLOCK_INDICES]
                  for obj in objects]
    post_vecs  = [[obj["post_cd_feats"][bi][:, 0, :] for bi in HOOK_BLOCK_INDICES]
                  for obj in objects]
    clip_inputs = [obj["clip_input"] for obj in objects]   # list of [1, 768]

    boot_pre  = np.zeros((n_bootstrap, num_steps, n_blocks), dtype=np.float64)
    boot_post = np.zeros((n_bootstrap, num_steps, n_blocks), dtype=np.float64)

    print(f"  Bootstrap CI: {n_bootstrap} resamples × {num_steps} steps × {n_blocks} blocks …",
          end="", flush=True)
    for b in range(n_bootstrap):
        idx     = rng.integers(0, N, size=N)
        Y_input = torch.cat([clip_inputs[i] for i in idx], dim=0)   # [N, 768]
        for s in range(num_steps):
            for bi_idx in range(n_blocks):
                X_pre  = torch.stack([pre_vecs[i][bi_idx][s]  for i in idx], dim=0)
                X_post = torch.stack([post_vecs[i][bi_idx][s] for i in idx], dim=0)
                boot_pre[b,  s, bi_idx] = linear_cka(X_pre,  Y_input)
                boot_post[b, s, bi_idx] = linear_cka(X_post, Y_input)
    print(" done.")

    return {
        "ci_pre_lo":  np.percentile(boot_pre,   2.5, axis=0),   # [steps, n_blocks]
        "ci_pre_hi":  np.percentile(boot_pre,  97.5, axis=0),
        "ci_post_lo": np.percentile(boot_post,  2.5, axis=0),
        "ci_post_hi": np.percentile(boot_post, 97.5, axis=0),
    }


# ---------------------------------------------------------------------------
# Objaverse sampling (OOD robustness pass)
# ---------------------------------------------------------------------------

def sample_objaverse_objects(
    glb_dir: str = "/scratch/s224696943/objaverse_30k/objaverse/hf-objaverse-v1/glbs",
    n: int = 30,
    seed: int = 42,
) -> List[tuple]:
    """
    Sample n Objaverse GLB files at random (fixed seed for reproducibility).

    Searches recursively under glb_dir for *.glb files.
    Returns list of (uid, glb_path_str) tuples where uid = Path(p).stem.

    Raises FileNotFoundError if no GLBs are found.
    """
    import glob as _glob
    pattern  = str(Path(glb_dir) / "**" / "*.glb")
    all_glbs = _glob.glob(pattern, recursive=True)
    if not all_glbs:
        raise FileNotFoundError(f"No *.glb files found under {glb_dir}")
    rng    = np.random.default_rng(seed)
    chosen = rng.choice(all_glbs, size=min(n, len(all_glbs)), replace=False)
    return [(Path(p).stem, str(p)) for p in sorted(chosen)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_object_drift(result: Dict, obj_name: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_vals = result["timesteps"]
    sims   = result["cosine_sims"]   # [steps, 6]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, NUM_VIEWS))
    for v in range(NUM_VIEWS):
        ax.plot(t_vals[::-1], sims[:, v], color=colors[v],
                alpha=0.8, linewidth=1.5, label=f"view {v}")
    ax.set_xlabel("Denoising timestep (0 = clean)", fontsize=12)
    ax.set_ylabel("CLIP cosine similarity to input", fontsize=12)
    ax.set_title(f"CLIP drift — {obj_name}", fontsize=13)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"clip_drift_{obj_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(all_results: Dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_sims = np.stack([r["cosine_sims"] for r in all_results.values()])
    mean_sims = all_sims.mean(axis=(0, 2))
    std_sims  = all_sims.std(axis=(0, 2))
    t_arr = np.array(list(all_results.values())[0]["timesteps"][::-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_arr, mean_sims[::-1], color="steelblue", linewidth=2.5, label="mean (objects × views)")
    ax.fill_between(t_arr, (mean_sims - std_sims)[::-1], (mean_sims + std_sims)[::-1],
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="sim=0.5")
    ax.set_xlabel("Denoising timestep (0 = clean)", fontsize=12)
    ax.set_ylabel("CLIP cosine similarity to input", fontsize=12)
    ax.set_title(
        f"CLIP Drift — {len(all_results)} GSO objects · Wonder3D · (PGR-3D motivation)\n"
        "Low similarity ↓ = CLIP signal drifts = our readout guidance is motivated",
        fontsize=11,
    )
    ax.legend(fontsize=10); ax.set_xlim(0, 1000); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "clip_drift_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/clip_drift_summary.png")


def plot_cka_comparison(
    cka_curves: Dict,
    t_vals: list,
    output_dir: Path,
    bootstrap_ci: Dict = None,
) -> None:
    """
    Three-panel figure (per CKA variant) × 2 rows (pre-CD / post-CD):
      Row 1: input-front CKA — PRIMARY paper metric (front-view features vs input CLIP)
      Row 2: decoded-all CKA — secondary (all views vs decoded CLIP, n=N*V)
      Row 3: decoded-per-view — shows front vs novel view drift separately

    bootstrap_ci: optional dict from bootstrap_cka_ci() — if provided, shades
                  95% CI bands on the PRIMARY input-front figure only.

    Saved as:
      cka_input_front.png  — panel pair for the primary metric (paper figure)
      cka_decoded_all.png  — secondary sanity-check panel
      cka_per_view.png     — per-view breakdown (up_blocks[2] only, clearest)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_arr        = np.array(t_vals[::-1])    # 0 → 999
    block_labels = ["up_blocks[1] (1280ch)", "up_blocks[2] (640ch)", "up_blocks[3] (320ch)"]
    block_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    view_colors  = plt.cm.tab10(np.linspace(0, 1, NUM_VIEWS))

    def _plot_panel(ax, cka_vals, title, ylabel=True, ci_lo=None, ci_hi=None):
        for bi_idx in range(3):
            c = cka_vals[:, bi_idx][::-1]
            ax.plot(t_arr, c, color=block_colors[bi_idx], linewidth=2,
                    label=block_labels[bi_idx])
            if ci_lo is not None and ci_hi is not None:
                lo = ci_lo[:, bi_idx][::-1]
                hi = ci_hi[:, bi_idx][::-1]
                ax.fill_between(t_arr, lo, hi, alpha=0.18, color=block_colors[bi_idx])
        ax.set_xlabel("Denoising timestep (0 = clean)", fontsize=10)
        if ylabel:
            ax.set_ylabel("Linear CKA", fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1000); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ci_suffix = " (shaded: bootstrap 95% CI)" if bootstrap_ci is not None else ""

    # --- Figure 1: input-front CKA (PRIMARY) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    _plot_panel(
        axes[0], cka_curves["cka_pre_input_front"],
        f"PRE-CD · front-view features vs INPUT CLIP\n(before JointAttnProcessor){ci_suffix}",
        ci_lo=bootstrap_ci["ci_pre_lo"]  if bootstrap_ci else None,
        ci_hi=bootstrap_ci["ci_pre_hi"]  if bootstrap_ci else None,
    )
    _plot_panel(
        axes[1], cka_curves["cka_post_input_front"],
        f"POST-CD · front-view features vs INPUT CLIP\n(after domain mixing, our readout location){ci_suffix}",
        ylabel=False,
        ci_lo=bootstrap_ci["ci_post_lo"] if bootstrap_ci else None,
        ci_hi=bootstrap_ci["ci_post_hi"] if bootstrap_ci else None,
    )
    fig.suptitle(
        "PRIMARY metric: CKA(front-view UNet features, input CLIP) vs timestep\n"
        "Decreasing = front-view features lose predictive power over input semantics = DRIFT",
        fontsize=11,
    )
    fig.tight_layout()
    p1 = output_dir / "cka_input_front.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p1}")

    # --- Figure 2: decoded-all CKA (secondary) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    _plot_panel(axes[0], cka_curves["cka_pre_decoded_all"],
                "PRE-CD · all views vs decoded CLIP (n=N×6)")
    _plot_panel(axes[1], cka_curves["cka_post_decoded_all"],
                "POST-CD · all views vs decoded CLIP (n=N×6)", ylabel=False)
    fig.suptitle("Secondary: CKA(features all views, decoded CLIP) — highest statistical power",
                 fontsize=11)
    fig.tight_layout()
    p2 = output_dir / "cka_decoded_all.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p2}")

    # --- Figure 3: per-view decoded CKA for up_blocks[2] (middle block, clearest) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, which, title in [
        (axes[0], "cka_pre_per_view",  "PRE-CD · up_blocks[2] · per view"),
        (axes[1], "cka_post_per_view", "POST-CD · up_blocks[2] · per view"),
    ]:
        bi_idx = 1   # up_blocks[2] is index 1 in the 3-element array
        for v in range(NUM_VIEWS):
            c = cka_curves[which][:, bi_idx, v][::-1]
            lw = 2.5 if v == 0 else 1.2
            ls = "-" if v == 0 else "--"
            ax.plot(t_arr, c, color=view_colors[v], linewidth=lw, linestyle=ls,
                    label=f"view {v}" + (" (front)" if v == 0 else ""))
        ax.set_xlabel("Denoising timestep (0 = clean)", fontsize=10)
        ax.set_ylabel("Linear CKA", fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1000); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Per-view decoded CKA — does front view (v=0) drift differently from novel views?",
        fontsize=11,
    )
    fig.tight_layout()
    p3 = output_dir / "cka_per_view.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p3}")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def save_clip_csv(all_results: Dict, output_dir: Path) -> None:
    csv_path = output_dir / "clip_drift_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "step_idx", "timestep",
                    "view0","view1","view2","view3","view4","view5","mean_sim"])
        for obj, res in all_results.items():
            for si, (t_val, row) in enumerate(zip(res["timesteps"], res["cosine_sims"])):
                w.writerow([obj, si, int(t_val),
                             *[f"{s:.6f}" for s in row], f"{row.mean():.6f}"])
    print(f"  CSV: {csv_path}")


def save_cka_csv(cka_curves: Dict, t_vals: list, output_dir: Path) -> None:
    csv_path = output_dir / "cka_data.csv"
    V = NUM_VIEWS
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["step_idx", "timestep"]
        for tag in ["pre_inputfront", "post_inputfront",
                    "pre_decoded_all", "post_decoded_all"]:
            for bi in range(3):
                header.append(f"{tag}_block{bi+1}")
        for v in range(V):
            for tag in ["pre_decoded", "post_decoded"]:
                for bi in range(3):
                    header.append(f"{tag}_view{v}_block{bi+1}")
        w.writerow(header)
        for si, t_val in enumerate(t_vals):
            row = [si, int(t_val)]
            for arr in [cka_curves["cka_pre_input_front"],
                        cka_curves["cka_post_input_front"],
                        cka_curves["cka_pre_decoded_all"],
                        cka_curves["cka_post_decoded_all"]]:
                row.extend([f"{arr[si,bi]:.6f}" for bi in range(3)])
            for v in range(V):
                for arr in [cka_curves["cka_pre_per_view"],
                            cka_curves["cka_post_per_view"]]:
                    row.extend([f"{arr[si,bi,v]:.6f}" for bi in range(3)])
            w.writerow(row)
    print(f"  CKA CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Complete GSO-30 evaluation set — identical to gen_gso_configs.py:GSO_OBJECTS.
# Use all 30 for the motivation experiment so motivation and evaluation share
# the same object distribution (no confound).
# Canonical GSO-30, sourced from /scratch/s224696943/GSO/gso/ (official Google Scanned Objects).
# Real-world captured objects (1st–15th) removed — no GT mesh, not reproducible for reviewers.
# See configs/realworld_inthewild.txt for the separated qualitative set.
GSO_OBJECTS_30 = [
    "alarm", "backpack", "bell", "blocks", "chicken",
    "cream", "elephant", "grandfather", "grandmother", "hat",
    "leather", "lion", "lunch_bag", "mario", "oil",
    "school_bus1", "school_bus2", "shoe", "shoe1", "shoe2",
    "shoe3", "soap", "sofa", "sorter", "sorting_board",
    "stucking_cups", "teapot", "toaster", "train", "turtle",
]


def main():
    parser = argparse.ArgumentParser(description="PGR-3D Motivation Experiment v2")
    parser.add_argument("--objects",       nargs="+", default=GSO_OBJECTS_30)
    parser.add_argument("--gso_fronts",    default=GSO_FRONTS)
    parser.add_argument("--output_dir",    default="/scratch/s224696943/pgr_3d/outputs/motivation")
    parser.add_argument("--num_steps",     type=int,   default=50)
    parser.add_argument("--guidance_scale",type=float, default=3.0)
    parser.add_argument("--device",        default="cuda")
    # ---- Objaverse OOD pass ----
    parser.add_argument("--skip_objaverse", action="store_true",
                        help="Skip Objaverse-30 OOD pass (useful for quick tests)")
    parser.add_argument("--objaverse_glb_dir", default=(
        "/scratch/s224696943/objaverse_30k/objaverse/hf-objaverse-v1/glbs"))
    parser.add_argument("--objaverse_render_cache", default=(
        "/scratch/s224696943/pgr_3d/data/objaverse_renders"))
    parser.add_argument("--n_objaverse", type=int, default=30,
                        help="Number of Objaverse objects to sample for OOD check")
    # ---- Bootstrap CIs ----
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for 95%% CI on input-front CKA (0 = skip)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gso_dir = output_dir / "gso30"
    gso_dir.mkdir(exist_ok=True)

    print("Loading Wonder3D pipeline ...")
    pipe = load_pipeline(args.device)
    print(f"Pipeline loaded.  Device: {args.device}\n")

    # -----------------------------------------------------------------------
    # Pass 1 — GSO-30  (primary evaluation set)
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("PASS 1 — GSO-30  (primary evaluation set)")
    print("=" * 65)
    all_results_gso: Dict[str, Dict] = {}
    skipped_gso: List[str] = []

    for obj in args.objects:
        front_path = None
        for ext in [".png", ".jpg"]:
            p = Path(args.gso_fronts) / f"{obj}{ext}"
            if p.exists():
                front_path = p; break
        if front_path is None:
            print(f"[WARN] No front image for '{obj}' in {args.gso_fronts} — skipping")
            skipped_gso.append(obj); continue

        print(f"[{len(all_results_gso)+1}/{len(args.objects)}] {obj} ...")
        front_pil  = Image.open(front_path).convert("RGB").resize((256, 256), Image.BILINEAR)
        clip_input = clip_embed_pil(front_pil, device=args.device)

        result = run_denoising_with_probe(
            pipe, front_pil, clip_input,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            device=args.device,
        )
        all_results_gso[obj] = result
        plot_object_drift(result, obj, gso_dir)

        sims  = result["cosine_sims"]
        drift = sims[0].mean() - sims[-1].mean()
        print(f"    CLIP drift (noisy→clean): {drift:+.4f}  "
              f"(t=999: {sims[0].mean():.3f}, t=0: {sims[-1].mean():.3f})")

    if not all_results_gso:
        print("[ERROR] No GSO objects processed."); return

    plot_summary(all_results_gso, gso_dir)
    save_clip_csv(all_results_gso, gso_dir)
    cka_curves_gso = compute_cka_curves(all_results_gso, num_steps=args.num_steps)

    # Bootstrap CIs on the primary (input-front) CKA metric
    ci_gso = None
    if args.n_bootstrap > 0:
        ci_gso = bootstrap_cka_ci(
            all_results_gso, args.num_steps, n_bootstrap=args.n_bootstrap
        )

    t_vals_gso = list(all_results_gso.values())[0]["timesteps"]
    plot_cka_comparison(cka_curves_gso, t_vals=t_vals_gso,
                        output_dir=gso_dir, bootstrap_ci=ci_gso)
    save_cka_csv(cka_curves_gso, t_vals=t_vals_gso, output_dir=gso_dir)

    # -----------------------------------------------------------------------
    # Pass 2 — Objaverse-30  (OOD robustness check)
    # -----------------------------------------------------------------------
    all_results_objaverse: Dict[str, Dict] = {}
    cka_curves_objaverse = None
    ci_objaverse = None

    if not args.skip_objaverse:
        print("\n" + "=" * 65)
        print("PASS 2 — Objaverse-30  (OOD robustness check)")
        print("=" * 65)
        objaverse_dir = output_dir / "objaverse30"
        objaverse_dir.mkdir(exist_ok=True)
        render_cache  = Path(args.objaverse_render_cache)
        render_cache.mkdir(parents=True, exist_ok=True)

        try:
            objaverse_objects = sample_objaverse_objects(
                glb_dir=args.objaverse_glb_dir, n=args.n_objaverse, seed=42,
            )
            print(f"  Sampled {len(objaverse_objects)} Objaverse objects (seed=42).")
        except FileNotFoundError as e:
            print(f"[WARN] {e} — skipping OOD pass")
            objaverse_objects = []

        # Save sampled UIDs so the exact 30 objects are reproducible
        if objaverse_objects:
            manifest_path = Path(args.output_dir).parent.parent / "configs" / "objaverse30_motivation.txt"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as _mf:
                _mf.write("# Objaverse-30 motivation experiment — sampled UIDs (seed=42)\n")
                for _uid, _glb in objaverse_objects:
                    _mf.write(f"{_uid}\t{_glb}\n")
            print(f"  UID manifest saved: {manifest_path}")

        skipped_objaverse: List[str] = []
        for uid, glb_path in objaverse_objects:
            print(f"[OOD {len(all_results_objaverse)+1}/{len(objaverse_objects)}] {uid} ...")
            ok = render_object(glb_path, str(render_cache), uid)
            if not ok:
                print("    Render failed — skipping")
                skipped_objaverse.append(uid); continue
            front_png = render_cache / uid / "rgb_0.png"
            if not front_png.exists():
                print("    rgb_0.png missing after render — skipping")
                skipped_objaverse.append(uid); continue

            front_pil  = Image.open(front_png).convert("RGB").resize((256, 256), Image.BILINEAR)
            clip_input = clip_embed_pil(front_pil, device=args.device)
            result = run_denoising_with_probe(
                pipe, front_pil, clip_input,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                device=args.device,
            )
            all_results_objaverse[uid] = result
            plot_object_drift(result, uid, objaverse_dir)

            sims  = result["cosine_sims"]
            drift = sims[0].mean() - sims[-1].mean()
            print(f"    CLIP drift (noisy→clean): {drift:+.4f}  "
                  f"(t=999: {sims[0].mean():.3f}, t=0: {sims[-1].mean():.3f})")

        if all_results_objaverse:
            plot_summary(all_results_objaverse, objaverse_dir)
            save_clip_csv(all_results_objaverse, objaverse_dir)
            cka_curves_objaverse = compute_cka_curves(
                all_results_objaverse, num_steps=args.num_steps
            )
            if args.n_bootstrap > 0:
                ci_objaverse = bootstrap_cka_ci(
                    all_results_objaverse, args.num_steps, n_bootstrap=args.n_bootstrap
                )
            t_vals_ood = list(all_results_objaverse.values())[0]["timesteps"]
            plot_cka_comparison(cka_curves_objaverse, t_vals=t_vals_ood,
                                output_dir=objaverse_dir, bootstrap_ci=ci_objaverse)
            save_cka_csv(cka_curves_objaverse, t_vals=t_vals_ood,
                         output_dir=objaverse_dir)
            print(f"  Objaverse-30 complete: {len(all_results_objaverse)} processed, "
                  f"{len(skipped_objaverse)} skipped")

    # -----------------------------------------------------------------------
    # Gate decision  (primary: GSO-30 input-front CKA)
    # -----------------------------------------------------------------------
    all_sims_gso    = np.stack([r["cosine_sims"] for r in all_results_gso.values()])
    mean_sim_start  = all_sims_gso[:, 0,  :].mean()
    mean_sim_end    = all_sims_gso[:, -1, :].mean()
    total_drift     = mean_sim_start - mean_sim_end

    mean_cka_pre_start  = cka_curves_gso["cka_pre_input_front"][0].mean()
    mean_cka_pre_end    = cka_curves_gso["cka_pre_input_front"][-1].mean()
    mean_cka_post_start = cka_curves_gso["cka_post_input_front"][0].mean()
    mean_cka_post_end   = cka_curves_gso["cka_post_input_front"][-1].mean()
    pre_cka_drift  = mean_cka_pre_start  - mean_cka_pre_end
    post_cka_drift = mean_cka_post_start - mean_cka_post_end

    print("\n" + "=" * 65)
    print("DAY-3 GATE RESULT  (primary: GSO-30)")
    print(f"  Objects processed: {len(all_results_gso)}  (skipped: {skipped_gso})")
    print(f"\n  CLIP cosine sim (image-level):")
    print(f"    t≈999 (noisy): {mean_sim_start:.4f}")
    print(f"    t≈0   (clean): {mean_sim_end:.4f}")
    print(f"    drift:         {total_drift:+.4f}")
    print(f"\n  Linear CKA — input-front (primary metric):")
    print(f"    PRE-CD  — t≈999: {mean_cka_pre_start:.4f}  t≈0: {mean_cka_pre_end:.4f}  Δ={pre_cka_drift:+.4f}")
    print(f"    POST-CD — t≈999: {mean_cka_post_start:.4f}  t≈0: {mean_cka_post_end:.4f}  Δ={post_cka_drift:+.4f}")

    if ci_gso is not None:
        pre_w0  = (ci_gso["ci_pre_hi"][0]  - ci_gso["ci_pre_lo"][0]).mean()
        pre_wT  = (ci_gso["ci_pre_hi"][-1] - ci_gso["ci_pre_lo"][-1]).mean()
        post_w0 = (ci_gso["ci_post_hi"][0]  - ci_gso["ci_post_lo"][0]).mean()
        post_wT = (ci_gso["ci_post_hi"][-1] - ci_gso["ci_post_lo"][-1]).mean()
        print(f"    Bootstrap 95% CI widths: "
              f"PRE  t≈999={pre_w0:.4f}  t≈0={pre_wT:.4f} | "
              f"POST t≈999={post_w0:.4f}  t≈0={post_wT:.4f}")

    if all_results_objaverse:
        ood_sims  = np.stack([r["cosine_sims"] for r in all_results_objaverse.values()])
        ood_drift = ood_sims[:, 0, :].mean() - ood_sims[:, -1, :].mean()
        ood_consistent = abs(ood_drift - total_drift) < 0.05
        print(f"\n  OOD check (Objaverse-30): CLIP drift = {ood_drift:+.4f}  "
              f"({'consistent with GSO-30 ✓' if ood_consistent else 'DIFFERS from GSO-30 — inspect plots'})")
        if cka_curves_objaverse is not None:
            ood_pre_drift  = (cka_curves_objaverse["cka_pre_input_front"][0].mean()
                              - cka_curves_objaverse["cka_pre_input_front"][-1].mean())
            ood_post_drift = (cka_curves_objaverse["cka_post_input_front"][0].mean()
                              - cka_curves_objaverse["cka_post_input_front"][-1].mean())
            print(f"    OOD CKA drift: PRE Δ={ood_pre_drift:+.4f}  POST Δ={ood_post_drift:+.4f}")
            if ci_objaverse is not None:
                ood_pre_w0  = (ci_objaverse["ci_pre_hi"][0]  - ci_objaverse["ci_pre_lo"][0]).mean()
                ood_pre_wT  = (ci_objaverse["ci_pre_hi"][-1] - ci_objaverse["ci_pre_lo"][-1]).mean()
                ood_post_w0 = (ci_objaverse["ci_post_hi"][0]  - ci_objaverse["ci_post_lo"][0]).mean()
                ood_post_wT = (ci_objaverse["ci_post_hi"][-1] - ci_objaverse["ci_post_lo"][-1]).mean()
                print(f"    OOD bootstrap 95% CI widths: "
                      f"PRE  t≈999={ood_pre_w0:.4f}  t≈0={ood_pre_wT:.4f} | "
                      f"POST t≈999={ood_post_w0:.4f}  t≈0={ood_post_wT:.4f}")

    if total_drift > 0.05:
        verdict = "DRIFT DETECTED ✓ — proceed with PGR-3D"
    elif total_drift > 0.01:
        verdict = "MARGINAL DRIFT — escalate to user before proceeding"
    else:
        verdict = "NO SIGNIFICANT DRIFT — ESCALATE TO USER IMMEDIATELY"

    print(f"\n  VERDICT: {verdict}")

    if pre_cka_drift > 0.01 and post_cka_drift > 0.01:
        scenario = "SCENARIO A: Both pre-CD and post-CD drift → cross-domain attention doesn't rescue CLIP signal"
    elif post_cka_drift - pre_cka_drift > 0.05:
        scenario = "SCENARIO B: Post-CD drifts more → cross-domain attention INTRODUCES additional drift"
    elif pre_cka_drift - post_cka_drift > 0.05:
        scenario = "SCENARIO C: Pre-CD drifts more → cross-domain attention RESCUES CLIP signal — REFRAME NEEDED"
    else:
        scenario = "SCENARIO A/unclear: pre and post drift similarly"
    print(f"  CKA SCENARIO: {scenario}")
    print("=" * 65)
    print(f"\nGSO-30 outputs in:       {gso_dir}")
    if not args.skip_objaverse and all_results_objaverse:
        print(f"Objaverse-30 outputs in: {output_dir / 'objaverse30'}")


if __name__ == "__main__":
    main()
