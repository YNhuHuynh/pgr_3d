# PGR-3D Conda Environment Setup

Two conda environments are used in this project.  They are **not
interchangeable** due to a PyTorch ABI incompatibility.

---

## Environments

### 1. `pgr3d` — Wonder3D training + evaluation pipeline

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.0+cu121 |
| pytorch3d | ABI-matched to torch 2.1.0 |
| BlenderProc | 2.6.1 (targets Blender 3.3, uses python3.10) |
| Transformers | **DO NOT UPGRADE** — pinned by torch 2.1 constraint |

**Used for:**
- `train_readout_caption.py` — CaptionHead training
- `train_readout.py` — DepthHead / SemanticHead training
- `motivation_experiment.py` — CKA experiment
- `eval_gso.py` — GSO evaluation (PSNR/SSIM/LPIPS)
- `render_gso30_eval_novelviews.py` — GT novel-view rendering (Blender)
- `render_objaverse_chunk.py` — Objaverse rendering (BlenderProc)
- Any script in `src/` or `scripts/` **except** `caption_pipeline.py`

**Why torch 2.1 is pinned:**
pytorch3d requires a specific CUDA ABI that matches the torch version it was
compiled against.  Upgrading torch would break pytorch3d imports and all
geometry-dependent code.  Do not upgrade transformers either — the current
pinned version works and a newer version would require torch ≥ 2.4.

---

### 2. `unique3d` — Qwen2.5-VL caption generation only

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.0 |
| qwen-vl-utils | installed |

**Used for:**
- `caption_pipeline.py` — Qwen2.5-VL-72B-Instruct inference **only**

**Why separate:**
Qwen2.5-VL requires `transformers >= 4.49`, which in turn requires
`torch >= 2.4`.  The `pgr3d` env has torch 2.1.0 (pinned for pytorch3d
ABI), so these requirements are irreconcilable.  `unique3d` was originally
created for the UniqueSD/3D project and happens to have a compatible stack.

---

## Usage convention

| Script | Env |
|---|---|
| `caption_pipeline.py` | `unique3d` |
| `text_encoder.py` (T5) | either — prefer `unique3d` after caption gen |
| `train_readout_caption.py` | `pgr3d` |
| `train_readout.py` | `pgr3d` |
| `motivation_experiment.py` | `pgr3d` |
| `eval_gso.py` | `pgr3d` |
| `render_gso30_eval_novelviews.py` | `pgr3d` |
| `render_objaverse_chunk.py` | `pgr3d` |

**T5 note:** `text_encoder.py` (google-t5/t5-base, 768-dim) is standard
enough to run in either env.  Run it in `unique3d` right after caption
generation to avoid an extra env switch when batch-encoding captions.

---

## SLURM script env audit (verified 2026-04-17)

| Script | Env | Status |
|---|---|---|
| `slurm_caption_sanity.sh` | `unique3d` | correct |
| `slurm_motivation.sh` | `pgr3d` | correct |
| `slurm_render_objaverse.sh` | `pgr3d` | correct |
| `slurm_render_objaverse_200.sh` | `pgr3d` | correct |
| `slurm_render_gso30_eval.sh` | `pgr3d` | correct (fixed 2026-04-17) |
| `slurm_train_depth.sh` | `pgr3d` | correct |
| `slurm_train_semantic.sh` | `pgr3d` | correct |
| `slurm_eval_gso.sh` | `pgr3d` | correct |
| `slurm_guidance_sweep.sh` | `pgr3d` | correct |
| `slurm_baseline_gso.sh` | `pgr3d` | correct |
| `slurm_benchmark_midas.sh` | `pgr3d` | correct |

Any new SLURM script must include:
```bash
source /sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh
conda activate {pgr3d|unique3d}
```

---

## Known issues

- **`pgr3d` transformers upgraded accidentally** (2026-04-17): During
  debugging of caption job 71803, `pip install transformers>=4.49` was run
  in `pgr3d`, upgrading it to 5.5.4.  This version requires torch ≥ 2.4,
  so it disables all model loading on pgr3d's torch 2.1.0.  This has no
  practical impact (caption pipeline uses `unique3d`) but is noted here.
  If any `pgr3d` code breaks on transformers import, downgrade with:
  ```bash
  conda activate pgr3d
  pip install transformers==4.30.2 --no-cache-dir
  ```
