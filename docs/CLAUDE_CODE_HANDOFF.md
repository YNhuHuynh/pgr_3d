# Claude Code Handoff — PGR-3D Project

**Start here if you are Claude Code taking over this project.** Read this document fully before any action.

---

## 0. Who you are, who the user is

You (Claude Code) are a coding agent working on a research paper targeting **NeurIPS 2026 main track** (deadline May 6, 2026). The user is a researcher with access to an H100 GPU cluster using SLURM. The user has already run Wonder3D successfully. The user will delegate coding tasks to you and to Codex; the user handles research decisions, debugging intuition, and writing.

## 1. Project in one paragraph

We want to test the hypothesis: *in multi-view 3D diffusion models (Wonder3D), perception information present at the input conditioning (CLIP embedding) drifts during the denoising process; enforcing perception consistency at intermediate UNet features via a readout-style loss improves 3D generation quality without retraining the backbone.* We adapt Readout Guidance (Luo et al., CVPR 2024) to Wonder3D, train lightweight readout heads on frozen Wonder3D features, and use classifier-guidance-style sampling at inference. We evaluate on Google Scanned Objects (GSO, 30 objects, same subset as SyncDreamer/Wonder3D) using Chamfer Distance, Volume IoU, PSNR, SSIM, LPIPS.

## 2. Key references

- **Wonder3D paper:** arxiv 2310.15008 — *Wonder3D: Single Image to 3D using Cross-Domain Diffusion*, CVPR 2024.
- **Wonder3D checkpoint:** `/scratch/s224696943/wonder3d-v1.0` — MVDiffusionImagePipeline with CLIPVisionModelWithProjection (image-conditioned, v1.0).
- **Readout Guidance paper:** arxiv 2312.02150 — *Readout Guidance: Learning Control from Diffusion Features*, CVPR 2024.
- **Wonder3D repo:** `/scratch/s224696943/Wonder3D`
- **Example run script:** `/scratch/s224696943/3DRAV/000.run_won3d.sh`

## 3. Environment details

- **Cluster base path:** `/scratch/s224696943/`
- **Wonder3D codebase:** `/scratch/s224696943/Wonder3D` (read-only reference)
- **Wonder3D checkpoint:** `/scratch/s224696943/wonder3d-v1.0`
- **Project folder:** `/scratch/s224696943/pgr_3d`
- **GSO evaluation objects:** `/scratch/s224696943/3DRAV/evaluation/gso/`
- **Objaverse 30k:** sync from `rds-storage.deakin.edu.au:/RDS/RDS76808-Vision3D-Project/vlm-3d-objects/objaverse_30k` to `/scratch/s224696943/objaverse_30k`
- **GPU allocation:** `salloc --gpus=h100:1 -p gpu-large --cpus-per-task=4 --mem=100GB --constraint=cuda-12.6`
- **Conda env:** `pgr3d` (cloned from `3drav_a100`, adds wandb)
- **CUDA 12.6** — torch 2.1.0+cu121 is compatible

## 4. Architecture design

### 4.1 Wonder3D version decision

Using **Wonder3D v1.0** checkpoint (`MVDiffusionImagePipeline` with `CLIPVisionModelWithProjection`). This is image-conditioned — CLIP image embeddings of the input front view are the conditioning signal. Our SemanticHead targets these same embeddings. The drift story: the CLIP image embedding used to condition the UNet should remain decodable from intermediate features, but we hypothesize it doesn't.

### 4.2 Feature extraction

Hook into the **last 3 decoder blocks** of the Wonder3D UNet (`up_blocks[1]`, `up_blocks[2]`, `up_blocks[3]` or equivalent — verify by reading `unet_mv2d_condition.py`).

Feature tensor shape at each hook: `[B * 2 * V, C, H, W]` where B=batch, 2=RGB+normal domains, V=6 views. Extract RGB-domain features only (first half of the batch dim after demultiplexing).

### 4.3 Aggregation network

For each of 3 feature maps:
1. Per-layer bottleneck conv: `C_l → 128`
2. Bilinear resize to `64×64`
3. Learnable scalar weights (softmax-normalized) for weighted sum
4. Optional timestep MLP conditioning

Output: `[B*V, 128, 64, 64]`

### 4.4 Readout heads

**SemanticHead:**
- Input: `[B*V, 128, 64, 64]`
- Global avg pool → `[B*V, 128]`
- Mean over V views → `[B, 128]`
- MLP (128→512→768) → `[B, 768]`  ← CLIP ViT-L/14 image embedding dim
- Loss: `1 - cosine_similarity(pred, target_clip_image_emb)`

**DepthHead:**
- Input: `[B*V, 128, 64, 64]`
- 3× Conv(3,ReLU) → `[B*V, 1, 64, 64]`
- Upsample to `256×256`
- Loss: scale-invariant L1 vs MiDaS depth on clean rendered RGB

### 4.5 Training

- Freeze Wonder3D UNet entirely
- Train: AggregationNetwork + SemanticHead + DepthHead
- Data: Objaverse 30k subset (start 200 objects, scale to 1k/5k)
- Optimizer: AdamW lr=1e-4, weight_decay=0.01
- Steps: 5k, batch 4 objects (=24 view-wise samples)
- Log to wandb, format: `pgr_{head}_{data_size}_{date}`

### 4.6 Inference guidance

```
ε_guided(z_t, t) = ε_θ(z_t, t) - σ_t · η · ∇_{z_t} L_readout(z_t, t, target)
```

- η sweep: {0.1, 0.5, 1.0, 2.0}
- Timestep range sweep: {all, t<250, t<500, t<750}

## 5. Non-negotiable rules

1. Do NOT retrain Wonder3D UNet. Frozen always.
2. Always evaluate on GSO subset at `/scratch/s224696943/3DRAV/evaluation/gso/`.
3. Log every experiment to wandb: `pgr_{head}_{data_size}_{date}`.
4. Save checkpoints every 500 steps.
5. Never push to Wonder3D repo. All code in `/scratch/s224696943/pgr_3d`.
6. Stop and escalate if Day 3 motivation experiment fails.
7. Stop and escalate if Day 10 pivot shows no improvement.

## 6. Code deliverables

| File | Day |
|------|-----|
| `src/feature_extractor.py` | Day 2 |
| `src/metrics.py` | Day 2 |
| `src/data_pipeline.py` | Day 4 |
| `src/readout_heads.py` | Day 4 |
| `src/train_readout.py` | Day 4 |
| `src/guidance_inference.py` | Day 7 |
| `src/eval_gso.py` | Day 8 |
