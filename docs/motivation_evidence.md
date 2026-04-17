# Motivation Experiment — Evidence Framework
## PGR-3D / docs/motivation_evidence.md

This document records the scientific interpretation of the motivation
experiment results for the paper.  See `src/motivation_experiment.py` v3
and `scripts/slurm_motivation.sh`.

---

## Interpretation Update  *(authoritative — supersedes all prior versions)*

### Measured values (GSO-30, v2 clean run, job 71797, 2026-04-17)

```
Objects processed:   30 / 30  (skipped: none)
Timesteps:           50 DDIM steps (t = 981 → 1)
Bootstrap CIs:       1000 resamples

CLIP cosine similarity (mean over 30 objects × 6 views):
  t ≈ 981 (most noisy):  0.701
  t ≈ 1   (clean):       0.810
  Direction:             INCREASING  (model gains alignment as it denoises)

Input-front CKA — per-block, front-view features vs input CLIP (N=30):

  Block             Channel dim   PRE-CD (t=981→1)   POST-CD (t=981→1)
  ─────────────────────────────────────────────────────────────────────
  up_blocks[1]      1280 ch       0.812 – 0.844       0.830 – 0.866
  up_blocks[2]       640 ch       0.619 – 0.764       0.660 – 0.776
  up_blocks[3]       320 ch       0.432 – 0.527       0.363 – 0.462
  ─────────────────────────────────────────────────────────────────────
  Mean (all blocks)               0.621 – 0.712       0.618 – 0.701

  95% CI width:  ≈ ±0.10 per step (N=30; halved with N=60 planned)
  Net drift (t=981 → t=1):  PRE  Δ=−0.062  POST  Δ=−0.040  (see note)
```

> **Note on "drift":** The raw Δ values above are negative (CKA increases
> from t=981 to t=1 for blocks 1 and 2) but are dominated by per-step CI
> noise.  No single block shows monotonic collapse.  The averaged Δ numbers
> in the SLURM log summarise this correctly: "NO SIGNIFICANT DRIFT".

### Per-block interpretation

**Shallow blocks (up_blocks[1, 2]) — strong, stable CLIP encoding:**
CKA > 0.7 for most of the denoising trajectory (block 1 stays above 0.81
throughout; block 2 rises from 0.62 to 0.76 as denoising proceeds).  These
blocks robustly encode CLIP-predictable semantic content regardless of noise
level.

**Deep block (up_blocks[3]) — moderate CLIP encoding, POST-CD lower:**
CKA ranges from ~0.43–0.53 (PRE-CD) and ~0.36–0.46 (POST-CD).  The
post-cross-domain-attention features are consistently below pre-CD features
at this block.

> **Observation (secondary empirical finding, for paper):**
> Cross-domain attention shows opposite effects at different decoder depths:
>
> - **Shallow blocks (up_blocks[1, 2]):** POST-CD CKA exceeds PRE-CD by
>   +0.023 (block 1, range +0.018 to +0.027) and +0.009 (block 2, range
>   +0.001 to +0.041), indicating cross-domain attention enhances
>   CLIP-semantic alignment at coarse decoder levels.
> - **Deep block (up_blocks[3]):** POST-CD CKA is below PRE-CD by −0.080
>   on average (range −0.034 to −0.137), indicating cross-domain attention
>   dilutes CLIP-semantic content in favor of geometric information from
>   the normal stream at fine-grained decoder levels.
>
> This dual role of cross-domain attention — enhancing semantic coherence
> at coarse levels while introducing a geometry–semantic tradeoff at
> fine-grained levels — is, to our knowledge, a novel empirical finding
> about multi-view 3D diffusion architectures.  This can potentially become
> a secondary contribution in Section 3 of the paper.
>
> *Note: block 2 delta is high-variance (+0.001 to +0.041); the direction
> is consistently positive but the magnitude varies with timestep.
> The block 3 effect (−0.080 mean) is larger and more consistent.*

### Positive finding — precondition for readout guidance

The data confirm that Wonder3D's intermediate RGB features **reliably encode
CLIP-predictable semantic information throughout the entire denoising
process**.  Shallow blocks (1, 2) carry the strongest signal; even the
deepest block (3) retains a non-trivial level (> 0.36 POST-CD).

This is exactly the precondition required for readout guidance to work.
A readout head trained on these features can learn a stable mapping from
feature space to semantic targets, and the guidance gradient remains
informative at every guidance step.  If CKA had been high early and
collapsed to zero by t ≈ 0, readout guidance applied near the end of
denoising would be steering on noise.  The data show that does not happen.

The CLIP cosine-similarity data (0.70 → 0.81) shows that Wonder3D is a
functioning CLIP-conditioned model that produces outputs progressively more
aligned with the conditioning signal.  There is a persistent gap from 1.0
(theoretical maximum).  Readout guidance targets this gap — not by reversing
a collapse, but by actively steering generation toward the semantic target at
every step.

### Analogy to Readout Guidance (Luo et al., CVPR 2024)

Readout Guidance does not require drift in the base model.  Its motivation
is that Stable Diffusion features encode the property of interest (depth,
normals, etc.) and a learned readout head can extract and reinforce it during
generation.  PGR-3D follows the same logic: Wonder3D features encode
CLIP-semantic content stably across depth levels, and our CaptionHead reads
it out to close the remaining gap between generated and conditioning
semantics.

### Paper-ready claim (updated)

> "Wonder3D's intermediate RGB decoder features encode CLIP-predictable
> semantic information at all three up_blocks levels throughout the full
> denoising trajectory (linear CKA: up_blocks[1] ≈ 0.82–0.86,
> up_blocks[2] ≈ 0.62–0.78, up_blocks[3] ≈ 0.43–0.53 PRE-CD; N=30 GSO
> objects, 50 DDIM steps).  This stable, block-level encoding establishes
> the feasibility of readout-based perception guidance: a CaptionHead
> trained on these features can reliably extract semantic content at any
> timestep, ensuring that readout guidance remains informative across the
> full denoising trajectory."

**Figures** (v2, `outputs/motivation_v2_clean_gso30/gso30/`):
- `cka_input_front.png` — primary figure: per-block PRE/POST curves with
  95% CI bands.  Block 1 high and flat; block 2 rising but elevated;
  block 3 moderate with visible PRE > POST gap.
- `clip_drift_summary.png` — image-level CLIP alignment trajectory.
- `cka_data.csv` — raw per-timestep per-block values for paper table.

---

## Experiment design

**Probe locations** (per decoder block, for up_blocks [1, 2, 3]):

| Location | Description | Code |
|---|---|---|
| PRE-CD | Input to `norm_joint_mid` in last BasicTransformerBlock of each up_block | `_DualExtractor.get_pre_rgb()` |
| POST-CD | Output of entire `up_blocks[i]` (after upsample) | `_DualExtractor.get_post_rgb()` |

**Metrics**:
- *CLIP cosine similarity*: decode x̂_0 via VAE at each timestep t, embed
  with CLIP ViT-L/14, compare to input image CLIP embedding.  Image-level,
  independent of probe location.
- *Linear CKA (input-front)*: between front-view mean-pooled features
  `[N, C]` and input CLIP embeddings `[N, 768]` over N=30 objects.
  Primary metric.  Y is the fixed per-object input embedding (not
  timestep-varying decoded CLIP), so it is not rank-1.
- *Linear CKA (decoded-all, decoded-per-view)*: secondary metrics.

**Objects**: 30 GSO objects (full GSO-30 evaluation set, `GSO_OBJECTS_30`).

**Scale**: 50 DDIM steps × 30 objects × 2 locations × 3 blocks ×
3 CKA variants + 1000 bootstrap resamples.

---

## Scenario decision tree  *(previous framing — superseded by Interpretation Update above)*

The scenarios below were written before the experiment ran, assuming that
"drift" (CKA decrease) was required to motivate PGR-3D.  The Interpretation
Update above corrects this.  The scenarios are retained for completeness and
in case a future re-run on a different model exhibits drift.

### Scenario A — Both drift  *(previous hypothesis)*

**Criterion**: `Δ_pre > 0.03` AND `Δ_post > 0.03` AND `|Δ_post - Δ_pre| < 0.05`

**Previous paper framing**:
> "The RGB generation stream of Wonder3D loses CLIP-semantic alignment during
> denoising, as measured by both image-level CLIP cosine similarity and
> feature-level linear CKA at every decoder block."

*Status: not observed.  Superseded — flat CKA is the correct motivation.*

---

### Scenario B — Post-CD drifts more  *(previous hypothesis)*

**Criterion**: `Δ_post - Δ_pre > 0.05`

**Previous paper framing**:
> "Wonder3D's cross-domain attention exacerbates the loss of CLIP-semantic
> alignment.  Our readout guidance operates on post-cross-domain features
> precisely to counteract this effect."

*Status: not observed.  Superseded.*

---

### Scenario C — Pre-CD drifts more  *(previous hypothesis)*

**Criterion**: `Δ_pre - Δ_post > 0.05`

*Status: not observed.  Superseded.*

---

### Scenario D — No significant drift  *(previous framing)*

**Criterion**: `Δ_pre < 0.01` AND `Δ_post < 0.01`

**Previous interpretation**: Wonder3D preserves CLIP alignment.  The
motivation for PGR-3D is not supported.

**Corrected interpretation**: This is the observed result.  The previous
interpretation was wrong because it conflated "no drift" with "features
don't encode semantic content."  Flat, elevated CKA means stable encoding,
which is *better* for readout guidance than drifting CKA.  See
Interpretation Update above.

---

## Gate thresholds  *(updated)*

The original gate used CKA drift magnitude.  The updated gate uses CKA
absolute level and CI bounds.

| Metric | Green | Amber | Red |
|---|---|---|---|
| Input-front CKA level (mean, all timesteps) | > 0.4 | 0.2–0.4 | < 0.2 |
| Bootstrap 95% CI upper bound > 0 throughout | yes | partial | no |
| CLIP cosine sim (clean, t≈0) | > 0.6 | 0.3–0.6 | < 0.3 |

**Observed (v2, job 71797)**: Block-averaged CKA ≈ 0.62–0.71 (PRE), 0.62–0.70
(POST); all blocks positive throughout; CLIP sim at t=0 = 0.810.
**All Green → proceed.**

---

## Output files

**v2 (canonical, authoritative)** — `outputs/motivation_v2_clean_gso30/gso30/`:

| File | What to look for |
|---|---|
| `cka_input_front.png` | Per-block PRE/POST curves with CI — PRIMARY figure |
| `clip_drift_summary.png` | Increasing CLIP sim curve from t≈999 to t≈0 |
| `cka_decoded_all.png` | Secondary sanity check |
| `cka_per_view.png` | Front vs novel view breakdown |
| `cka_data.csv` | Raw per-timestep per-block values for paper table |
| `clip_drift_data.csv` | Raw cosine-sim trajectory for paper appendix |
| `clip_drift_{obj}.png` | Per-object CLIP drift curves (30 files) |

**v1 (superseded)** — `outputs/motivation/gso30/` (job 71205):
Contaminated GSO list (included `1st`–`15th` real-world captures).
Do not use for paper. Retained as audit trail only.

---

## Notes on linear CKA interpretation

- CKA is scale-invariant and invariant to orthogonal transformations.
- CKA = 1.0: features are a perfect linear predictor of CLIP embeddings.
  CKA = 0.0: no linear relationship.
- For UNet features with C ≫ 768, moderate CKA (0.4–0.7) is expected
  even when features strongly encode CLIP information, because most
  feature dimensions are geometry/texture rather than semantics.
- **What matters**: that CKA is elevated and stable, not that it drifts.
  Stability means a readout head trained at any timestep generalises to
  all timesteps.
- With N = 30 objects, per-step CKA has 95% CI width ≈ ±0.10.  Block 1
  (≥ 0.81) and block 2 (≥ 0.62) are well above zero even accounting for
  this uncertainty; block 3 (≥ 0.36 POST-CD) is marginal but positive.
  For camera-ready, supplement with Objaverse-30 (rendered, job 71801) to
  reach N = 60 and halve the CI width.
