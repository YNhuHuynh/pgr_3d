# View Handling — Precise Design Decisions
## PGR-3D / docs/VIEW_HANDLING_DECISIONS.md

Authoritative answers to four questions about view handling.
All claims carry exact code line references.

---

## Q1 — CKA computation: what is computed, and what should be computed

### What the current code does (before this session's fix)

`compute_cka_curves()` in `motivation_experiment.py` accumulates:

```python
# line ~400-415 (compute_cka_curves)
clip_accum[s].append(result["clip_step_embs"][s])   # [V=6, 768]
pre_accum[bi][s].append(result["pre_cd_feats"][bi][s])   # [V=6, C]

Y = torch.cat(clip_accum[s], dim=0)        # [N_obj*6, 768]
X = torch.cat(pre_accum[bi][s], dim=0)     # [N_obj*6, C]
cka = linear_cka(X, Y)                     # ONE scalar per block per timestep
```

This is **option (c)**: stack `[N_objects × V, C]`, compute one CKA.

The CLIP target `Y[j]` = CLIP embedding of the **decoded x̂_0 of view j** at step s.
That is, each feature vector's paired CLIP target is that same view's predicted
clean image embedding — not the input CLIP. This measures:
> "Do UNet features of view i predict what view i will look like at this step?"

### Why this is not quite the right scientific question

The motivation claim is: "CLIP-relevant semantic content DRIFTS during denoising
of the **RGB generation stream**." To establish that, we need to measure how
well features predict the INPUT CLIP embedding — which is fixed and represents
what the model *should* be generating. Using decoded CLIP as the target partially
confounds the metric: if the model is generating something random at t=999, the
features might still have high CKA with decoded CLIP because they literally
computed that prediction.

### What the code now computes (after this session's fix)

Three CKA variants, each answering a distinct question:

| CKA variant | X | Y | n per step | Scientific question |
|---|---|---|---|---|
| **decoded-all** (existing) | `[N*6, C]` features all views | `[N*6, 768]` decoded CLIP all views | 180 | Do features predict their own decoded view? |
| **decoded-per-view** (new) | `[N, C]` features of view i | `[N, 768]` decoded CLIP of view i | 30 | Same, but per view — front vs novel |
| **input-front** (new, **primary**) | `[N, C]` features of view 0 only | `[N, 768]` INPUT CLIP per object | 30 | Do front-view features predict the input image? |

The **input-front CKA is the primary scientific metric** for the paper claim.
- `X` = `pre_cd_feats[bi][s][0]` or `post_cd_feats[bi][s][0]`  (view 0 of each object)
- `Y` = `clip_input` per object (constant across timesteps, varies across objects — NOT rank-1)
- A decreasing `input-front CKA` as t → 0 confirms: "front-view features lose
  predictive power over the input CLIP as denoising progresses" = DRIFT.

The decoded-per-view CKA answers the per-view concern:
does drift happen differently for view 0 (front, conditioning) vs views 1-5 (novel)?

### Statistical power caveat

With N=30 objects (full GSO-30), per-view CKA uses n=30 samples, decoded-all
uses n=180. The per-view n=30 is marginal but workable for a trend metric;
if the curves are noisy in the camera-ready, supplement with Objaverse objects
to reach n=50+. The decoded-all CKA (n=180) has solid statistical power.

**Object set**: all 30 objects are from the GSO-30 eval set used in the
geometry and NVS evaluation — same distribution, no motivation/eval confound.

---

## Q2 — SemanticHead architecture: exact mean-pool location

The forward pass is at `readout_heads.py:222-230`:

```python
agg    = self.aggregation_network(features, emb)   # line 222: [BV, 384, 32, 32]
pooled = agg.mean(dim=[-2, -1])                    # line 224: [BV, 384]  ← GAP
BV     = pooled.shape[0]
V      = self.num_views
B      = BV // V if batch_size is None else batch_size
pooled = pooled.view(B, V, -1).mean(dim=1)         # line 229: [B, 384]   ← mean over V views
return self.mlp(pooled)                             # line 230: [B, 768]
```

This is **option (b)**: GAP each view (line 224), then mean across all V=6 views
(line 229), then MLP.

**NOT option (a)**: the mean over views (line 229) operates on GAP'd vectors
`[B, V, 384]`, NOT on the spatial feature maps `[BV, 384, 32, 32]`. The
order matters: GAP first (spatial → channel), then mean over views (views → batch).
Functionally equivalent to option (a) for the mean itself, but the gradient
behaviour differs because GAP is applied PER VIEW before combining.

**Gradient path through the mean:**
The `mean(dim=1)` at line 229 divides gradients uniformly by V=6.  
Any gradient w.r.t. the output `[B, 768]` produces identical gradient magnitude
at each `pooled[b, v, :]` for all v.  This is the correct inductive bias for
training: all 6 views' features are equally responsible for predicting the
front-view CLIP target.

---

## Q3 — Guidance gradient: does it flow equally to all 6 views?

### Trace from loss to z_t

```
L = 1 - cosine_sim(mlp(mean_V(GAP(agg))), clip_target)   [scalar]
      ↑ SemanticHead.loss(), readout_heads.py:238-240

∂L/∂mlp_out    [B=1, 768]
    → ∂L/∂pooled_mean  [B=1, 384]  via MLP backward (readout_heads.py:230)
    → ∂L/∂pooled_v     [B=1, 384]  for each v:  = (1/V) * ∂L/∂pooled_mean
       ↑ mean(dim=1) at line 229 — factor of 1/V per view
    → ∂L/∂agg_v        [1, 384, H, W]  via GAP backward: = (1/(H*W)) * expand
       ↑ GAP at line 224 — factor of 1/(H*W)
    → ... backprop through AggregationNetwork (bottleneck + mixing weights)
    → ... back to raw features feat[V+v] via GuidedFeatureExtractor (no detach)
    → ... back through frozen UNet up_blocks (UNet params have requires_grad=False)
    → ∂L/∂z_t[V+v]     [1, 4, 32, 32]  for each v = 0..5
```

**Answer: YES, gradient flows equally to ALL 6 RGB-domain latent views.**

The 1/V factor from the view-mean (line 229) means each view receives 1/6 of
the gradient that a single-view head would deliver. This is real but benign:
- η (guidance strength) is a free hyperparameter and absorbs this factor.
  η=1.0 with 6-view mean has the same effect as η=1/6 with front-view-only.
- All 6 RGB-domain latents `z_t[6:12]` are steered simultaneously, which is
  desirable: the full multi-view generation moves toward the CLIP target, not
  just the front view.
- The 5 normal-domain latents `z_t[0:6]` receive ZERO gradient because
  GuidedFeatureExtractor only hooks up_blocks (post-UNet computation) and the
  normal domain is the first half — it has a gradient path through cross-domain
  attention, but the AggregationNetwork only processes RGB features
  (`feat[V:2V]`, guidance_inference.py:125-127), so the backward graph does
  not extend to the normal-domain rows.

**Concern raised:** does 1/V per-view gradient weaken guidance?

**Conclusion:** Not a bug. The η sweep (η ∈ {0.1, 0.5, 1.0, 2.0}) covers the
effective range. The 6-view gradient is preferable to front-only because it
steers the multi-view manifold rather than a single projection.

---

## Q4 — Why we did NOT use only the front-view feature as head input

**Alternative design rejected**: `feat[V:V+1]` (batch row 6 = front view only)
as the only input to SemanticHead, so the head sees one view instead of six.

**Reasons for rejection (in paper-ready language):**

1. **MVAttnProcessor couples views before the hook fires.** The multi-view
   self-attention inside each `BasicTransformerBlock` (`attn1`, which uses
   `MVAttnProcessor` with `multiview_attention=True`) already attends over
   all V views jointly before any of our hooks capture features. The feature
   at row `V+0` (front view) is already an aggregate of information from all
   6 views via cross-view keys/values (`transformer_mv2d.py:784-788`).
   "Front-view-only" is a misnomer — there is no view in the post-attention
   feature space that is unaware of the others.

2. **Training signal density.** Using all V=6 views provides 6× the feature-CLIP
   pairs per object per training step (BV=24 aggregation ops vs BV=4 with front
   only at B=4). With a small Objaverse subset, this matters for sample efficiency.
   The SemanticHead's goal is to decode "what 3D object is being generated" —
   all views encode this, not just the front.

3. **Guidance steers the full multi-view manifold.** When we apply
   `∇_{z_t} L_semantic`, we want to improve the semantic alignment of the
   complete generated set of views, not just the front view reprojection.
   If we used only view 0's features, the gradient would predominantly flow
   to `z_t[6]` (front-view RGB latent) and weakly to others through cross-domain
   attention paths — creating an inconsistency where the front view improves
   but novel views are steered indirectly and unpredictably.

**Tradeoff acknowledged:** Using all views reduces the per-view gradient magnitude
by 1/V (see Q3). This is compensated by the η hyperparameter. If per-view
gradient strength becomes critical (e.g., at small η), a front-view-weighted
pooling (`weights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]`) is a valid ablation but
not the default design.
