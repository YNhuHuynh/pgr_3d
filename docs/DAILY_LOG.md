# Daily Log — PGR-3D

Format: at end of each work session, append a dated entry.

---

## Entries

### Day 0 — 2026-04-16

**Setup phase before Day 1.**

**Done:**
- Project plan finalized
- Claude Code handoff document prepared
- Idea notes preserved
- Folder structure created at `/scratch/s224696943/pgr_3d/`
- Existing resources confirmed:
  - Wonder3D checkpoint: `/scratch/s224696943/wonder3d-v1.0` (MVDiffusionImagePipeline, CLIP image-conditioned v1.0)
  - Wonder3D codebase: `/scratch/s224696943/Wonder3D`
  - Example run script: `/scratch/s224696943/3DRAV/000.run_won3d.sh`
  - GSO eval objects: `/scratch/s224696943/3DRAV/evaluation/gso/` (42 objects)
  - GPU: H100 via `salloc --gpus=h100:1 -p gpu-large --cpus-per-task=4 --mem=100GB --constraint=cuda-12.6`
  - Base conda env: `3drav_a100` (torch 2.1.0+cu121, diffusers 0.19.3, transformers 4.30.2, pytorch3d, clip, lpips, open3d, trimesh — needs wandb)
  - Objaverse 30k: rsync from `rds-storage.deakin.edu.au:/RDS/RDS76808-Vision3D-Project/vlm-3d-objects/objaverse_30k`

**Key decisions:**
- Use Wonder3D v1.0 (image-conditioned) — checkpoint at `/scratch/s224696943/wonder3d-v1.0`
- New conda env `pgr3d` (cloned from `3drav_a100` + wandb)
- SemanticHead targets CLIP ViT-L/14 image embeddings (same as Wonder3D's own conditioning)

**Open questions:**
- wandb credentials: unknown — user needs to run `wandb login` once in the env
- Objaverse 30k not yet synced to scratch

**Next session (Day 1, Apr 17):**
1. Create `pgr3d` conda env
2. Implement `src/feature_extractor.py`
3. Implement `src/metrics.py`
4. Run Wonder3D baseline on 5 GSO objects, verify numbers

---

### Day 2 — 2026-04-17 (session 2 continuation)

**Planned:** motivation experiment code, guidance inference, eval loop.

**Done:**
- Implemented `src/motivation_experiment.py`:
  - Runs Wonder3D denoising loop with a per-step callback
  - At each step computes predicted x̂_0 (scheduler.pred_original_sample), decodes with VAE,
    embeds all 6 views with CLIP ViT-L/14
  - Records cosine_sim(clip_input, clip_step) for each view × each timestep
  - Outputs: per-object line plots, summary plot (mean ± std), clip_drift_data.csv
  - Prints Day-3 gate verdict (DRIFT DETECTED / MARGINAL / NO DRIFT) with numeric thresholds
- Implemented `src/guidance_inference.py`:
  - `GuidedFeatureExtractor`: non-detaching hook variant to allow autograd through UNet features
  - `build_guided_pipeline()`: loads frozen Wonder3D pipeline
  - `compute_guidance_grad()`: computes ∇_{z_t} L_readout via torch.autograd.grad
  - `run_guided_inference()`: full denoising loop with PGR guidance:
      ε_guided = ε_θ - σ_t · η · ∇_{z_t} L_readout
    CFG and PGR are applied sequentially; timestep gating via [t_min, t_max]
  - `run_eta_sweep()`: convenience wrapper for η ∈ {0, 0.1, 0.5, 1.0, 2.0}
  - CLI: `python src/guidance_inference.py --front_image ... --head_ckpt ... --eta_sweep`
- Implemented `src/eval_gso.py`:
  - Three modes: `metrics_only` (existing meshes), `reconstruct`, `full` (generate+recon+eval)
  - Per-object: geometry (CD-L1/L2, IoU) + NVS (PSNR/SSIM/LPIPS) metrics
  - Mesh alignment: pre-rotation from 000.run_won3d.sh conventions, unit-diagonal normalisation
  - Appends to mesh/nvs CSV files via metrics.py helpers
  - Aggregate summary printed at end
- Wrote `scripts/slurm_motivation.sh`: SLURM job for Day-3 motivation experiment (2h, H100)

**Also done (session 3 continuation):**
- `scripts/slurm_eval_gso.sh`: SLURM eval job; supports metrics_only/reconstruct/full modes;
  environment vars (MODE/SETUP/ETA/HEAD_CKPT) overridable at sbatch time
- `scripts/slurm_guidance_sweep.sh`: runs eta ∈ {0.0,0.1,0.5,1.0,2.0} for all 5 test objects;
  generates multi-view images into `outputs/eta_sweep/{obj}/eta_X/`
- `src/test_feature_shapes.py`: 6-step sanity check (pipeline load, hook shapes, aggregation
  shapes, head forward/loss, gradient flow); run before submitting training jobs
- `scripts/gen_gso_configs.py`: added WEKA→local fallback for front images

**Not done / deferred:**
- Actually running any jobs (requires GPU allocation)
- `src/eval_gso.py` NVS section assumes GT renders exist at `gso/{obj}/renders/` — may need
  to verify actual path on cluster once baseline runs

**Known issues / notes:**
- `guidance_inference.py` gradient computation runs TWO UNet forward passes per guided step
  (one no_grad for CFG, one with grad for ∇_{z_t}). Memory use on H100 should be fine for
  V=6 views at 32×32 latents. If OOM, reduce to single-pass (get eps + grad in one call).
- `eval_gso.py` uses `transforms3d` for mesh rotation parsing; confirm it's in pgr3d env
  (it's in 3drav_a100 base — `pip install transforms3d` if missing).

**Next session (Day 3, Apr 18-19):**
1. Submit baseline: `cd /scratch/s224696943/pgr_3d && sbatch scripts/slurm_baseline_gso.sh`
2. Submit motivation experiment: `sbatch scripts/slurm_motivation.sh`
3. Review motivation results → make Day-3 go/no-go decision
4. Start Objaverse render job: `sbatch scripts/slurm_render_objaverse.sh`

**Notes for user:**
- **URGENT before Day 3 gate (Apr 19):** Submit motivation job after getting GPU allocation:
    `cd /scratch/s224696943/pgr_3d && sbatch scripts/slurm_motivation.sh`
  Output in `outputs/motivation/clip_drift_summary.png` — if drift > 0.05, green light.
- Parallel: also submit baseline job to get Wonder3D numbers while waiting for motivation result.
- Objaverse rsync likely still running — check with `jobs` or `rsync` process status.

---

### Day 1 — 2026-04-17

**Planned:** Env setup, feature extractor, metrics, baseline run on 5 GSO objects.

**Done:**
- Created project folder `/scratch/s224696943/pgr_3d/{docs,src,scripts,configs,outputs}`
- Wrote all handoff docs to `docs/`
- Created `pgr3d` conda env (cloned from `3drav_a100`, added wandb 0.26.0). All imports verified: torch 2.1.0+cu121, diffusers 0.19.3, clip, lpips, open3d, wandb.
- Read Wonder3D UNet architecture thoroughly:
  - Checkpoint: `/scratch/s224696943/wonder3d-v1.0` (MVDiffusionImagePipeline, CLIP image-conditioned)
  - `up_blocks[1,2,3]` = CrossAttnUpBlockMV2D at channels 1280, 640, 320 — these are our hook targets
  - Batch layout during inference (with CFG): `[norm_uc×6, norm_cond×6, rgb_uc×6, rgb_cond×6]` = 24 total; RGB-cond = last 6
  - Training layout (no CFG): `[normal×6, rgb×6]` = 12; RGB = last 6
- Implemented `src/feature_extractor.py`:
  - `Wonder3DFeatureExtractor`: context-manager hook wrapper for up_blocks[1,2,3]
  - `AggregationNetwork`: Diffusion-Hyperfeatures-style multi-scale aggregation (bottleneck conv + resize + softmax-weighted sum + timestep MLP)
  - Methods: `get_rgb_features()`, `get_rgb_features_batched()` handle both training and CFG inference layouts
- Implemented `src/metrics.py`:
  - Geometry: `chamfer_l1`, `chamfer_l2`, `volume_iou` (open3d RaycastingScene)
  - Image: `psnr`, `ssim`, `lpips_score` (skimage + lpips AlexNet)
  - Convenience: `mesh_metrics()`, `image_metrics()`, CSV append helpers
- Wrote `scripts/gen_gso_configs.py`: generates per-object Wonder3D YAML configs for GSO objects pointing to front view images at `/weka/s224696943/3DRAV/data/expB/{obj}/B/front.png`
- Wrote `scripts/slurm_baseline_gso.sh`: end-to-end SLURM job (inference → reconstruction → metrics) for 5 GSO objects: mario, alarm, chicken, elephant, turtle

**Not done / deferred:**
- Actually running the baseline job (requires GPU allocation) — ready to submit
- `src/feature_extractor.py` not yet tested with an actual forward pass (no GPU today) — Day 2 task

**Issues encountered:**
- `pgr3d` env clone completed with minor protobuf conflict (open-clip-torch vs wandb) — not affecting our usage since we use `clip` (openai) not `open-clip-torch`
- Wonder3D pipeline Python files are in `/scratch/s224696943/3DRAV/mvdiffusion/pipelines/` not in the Wonder3D codebase directly — SLURM script adds both to PYTHONPATH

**Decisions made:**
- Wonder3D v1.0 (image-conditioned): CLIP image embedding of input is the conditioning signal AND our readout target — cleaner story
- Hook on `up_blocks[1,2,3]` (CrossAttn blocks, channels 1280/640/320), not `up_blocks[0]` (UpBlock2D, no cross-attention)
- Training batch layout: no CFG during readout head training (simpler, less memory)
- Aggregation: resize all features to 32×32 (= latent resolution), 128-dim bottleneck

**Day 1 addendum (Apr 17, session 2):**
- Copied all 41 GSO front view images to `data/gso_fronts/`
- Cloned Readout Guidance repo to `readout_guidance_ref/` for architecture reference
- Started Objaverse 30k rsync (2,041 GLBs available so far in 160 subfolders)
- Implemented `src/readout_heads.py`: full RG-style BottleneckBlock + AggregationNetwork + SemanticHead + DepthHead, closely mirroring RG's dhf/aggregation_network.py
- Implemented `src/data_pipeline.py`: Objaverse GLB rendering (via blenderproc+blender-3.3.0) + ObjaverseRenderDataset + MiDaS depth + CLIP embedding with disk caching
- Implemented `src/train_readout.py`: full RG-style training loop — frozen Wonder3D UNet, both-domain batch, feature extraction, head training, wandb logging, checkpointing every 500 steps
- Wrote `scripts/blender_render_6views.py`: blenderproc script for 6-view orthographic rendering
- Wrote `scripts/slurm_render_objaverse.sh`, `slurm_train_semantic.sh`, `slurm_train_depth.sh`

**Next session (Day 2, Apr 18):**
1. Submit `slurm_baseline_gso.sh` and monitor (sbatch scripts/slurm_baseline_gso.sh from `/scratch/s224696943/pgr_3d`)
2. Verify feature shapes with a quick test forward pass once GPU is allocated
3. Begin `src/data_pipeline.py` (Objaverse loader) — also start Objaverse rsync

**Notes for user:**
- **ACTION NEEDED before running baseline:** Run `wandb login` inside `pgr3d` env once to set up wandb credentials. Or set `WANDB_MODE=offline` in the SLURM script if you want to skip wandb for now.
- Submit baseline: `cd /scratch/s224696943/pgr_3d && sbatch scripts/slurm_baseline_gso.sh`
- Objaverse rsync command (run in background or as a job): `rsync -aH --info=progress2 rds-storage.deakin.edu.au:/RDS/RDS76808-Vision3D-Project/vlm-3d-objects/objaverse_30k /scratch/s224696943/`
- CRITICAL check for Day 3: the motivation experiment needs GPU. Gate: does CLIP-cosine-similarity drop during denoising? This is your go/no-go decision.

