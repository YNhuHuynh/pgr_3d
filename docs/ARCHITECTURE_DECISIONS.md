# Architecture Decisions & Scientific Correctness Reference
## PGR-3D — authoritative answers for paper writing

This document records precise, code-verified answers to four design questions.
All line references are to files as of the Day 2 commit.

---

## Q1 — Domain ordering: which half is RGB?

**Answer: rows `[V : 2*V]` (i.e. the SECOND half) are RGB.**

### Training layout (no CFG, 2×V batch)

```
latent_model_input = torch.cat([input_normal, input_rgb], dim=0)
                                ^^^^^^^^^^^    ^^^^^^^^^
                                rows 0..5      rows 6..11   (V=6)
```

Code: `train_readout.py:160`

Feature extraction, training:
```python
# feature_extractor.py:233
rgb_feats.append(feat[V : 2 * V])   # ← rows 6..11 = RGB domain
```

### Inference layout (CFG, 4×V batch after reshape_to_cd_input)

`reshape_to_cd_input` (pipeline_mvdiffusion_image_joint.py:298-307) takes the
scheduler's input `[norm_uc, rgb_uc, norm_cond, rgb_cond]` and RE-ORDERS it to
`[norm_uc, norm_cond, rgb_uc, rgb_cond]` inside the UNet forward pass so that
normal and RGB positions are adjacent for cross-domain attention. After the
forward pass, `reshape_to_cfg_output` reverses the order.

Inside the UNet (where hooks fire):
```
rows  0.. 5 = norm_uc    (V=6)
rows  6..11 = norm_cond
rows 12..17 = rgb_uc
rows 18..23 = rgb_cond   ← we want this
```

Code: `feature_extractor.py:228`
```python
rgb_feats.append(feat[3 * V : 4 * V])   # rgb_cond, rows 18..23
```

### Source confirmation

`pipeline_mvdiffusion_image_joint.py:299-307`:
```python
input_norm_uc, input_rgb_uc, input_norm_cond, input_rgb_cond = torch.chunk(input, 4)
return torch.cat([input_norm_uc, input_norm_cond, input_rgb_uc, input_rgb_cond], dim=0)
```
Normal is always FIRST; RGB is always SECOND (training) or THIRD+FOURTH (CFG).

---

## Q2 — Camera embedding row ordering; which view is "front"?

### pipe.camera_embedding — raw values (shape [12, 5])

Format: `[elevation, delta_elevation, delta_azimuth, domain_normal_ind, domain_rgb_ind]`

```
row  0: d_az=0.0000  dom=[1,0]  ← normal, VIEW 0 = FRONT  (delta_az = 0)
row  1: d_az=0.8125  dom=[1,0]  ← normal, view 1
row  2: d_az=1.6934  dom=[1,0]  ← normal, view 2
row  3: d_az=3.1406  dom=[1,0]  ← normal, view 3
row  4: d_az=4.8359  dom=[1,0]  ← normal, view 4
row  5: d_az=5.5859  dom=[1,0]  ← normal, view 5
row  6: d_az=0.0000  dom=[0,1]  ← RGB,    VIEW 0 = FRONT  (delta_az = 0)
row  7: d_az=0.8125  dom=[0,1]  ← RGB,    view 1
...
row 11: d_az=5.5859  dom=[0,1]  ← RGB,    view 5
```

Source: `pipeline_mvdiffusion_image_joint.py:136-149`.

### Is row i aligned with feature row i?

Yes — by construction. In `train_readout.py:179-181`:
```python
cam_norm_bv = cam_norm.unsqueeze(0).expand(B, V, 10).reshape(BV, 10)  # [BV, 10]
cam_rgb_bv  = cam_rgb.unsqueeze(0).expand(B, V, 10).reshape(BV, 10)   # [BV, 10]
camera_embeddings = torch.cat([cam_norm_bv, cam_rgb_bv], dim=0)        # [2*BV, 10]
```

The feature tensor has layout `[normal(BV), rgb(BV)]`.  
The camera_embeddings tensor has the identical layout `[cam_norm_bv, cam_rgb_bv]`.  
Within each half, view index i maps to row i.  
Therefore `camera_embeddings[V + i]` and `rgb_features[:, i, ...]` are the same view. ✓

### Which view is the CLIP target?

**View 0 (front)** — `delta_azimuth = 0.0000`, meaning no azimuth offset from the
input image. This is the view whose CLIP embedding matches the conditioning
image that Wonder3D takes as input.

#### PAPER DESIGN DECISION — SemanticHead targets front-view CLIP, NOT multi-view average

`SemanticHead` internally mean-pools UNet features over all V views before
predicting a single CLIP embedding vector. But the **training target** is the
CLIP embedding of **view 0 (front) only**:

```python
# data_pipeline.py:274-285 — _load_clip_emb():
#   "Returns CLIP embedding of view-0 (front face)."
clip_emb = CLIP_encoder(rgb[:, 0, ...])   # [B, 768]  — front view only
```

**Rationale (paper framing):**  The Wonder3D input IS the front view image.
The conditioning signal IS its CLIP embedding. Our SemanticHead is trained to
predict "what does the network 'know' about the original input?" — which is
naturally measured against the CLIP embedding of that input. Averaging CLIP
embeddings across all 6 views would mix the front-view semantic signal with
novel-view signals that have no direct correspondence to the conditioning
image, diluting the motivation.

**What this means in practice:**  View 0's CLIP embedding is both the UNet's
conditioning signal AND our readout target.  The head's job is to decode
whether that conditioning information is preserved or has drifted.  This is
the scientific claim.

---

## Q3 — Cross-domain attention contamination: what is the scientific object?

### What `cd_attention_mid=True` does

Every `BasicTransformerBlock` in `up_blocks[1,2,3]` has `cd_attention_mid=True`
(verified at runtime; see below). Inside each block's forward pass
(`transformer_mv2d.py:574-576`):

```
attn1 (multi-view self-attention, within each domain)
    ↓
attn_joint_mid  ← cd_attention_mid: JointAttnProcessor
    ↓
attn2 (cross-attention to CLIP image embedding)
    ↓
feed-forward
```

`JointAttnProcessor` (`transformer_mv2d.py:910-965`):
- Splits batch into `[chunk_0=normal(BV), chunk_1=rgb(BV)]`
- Concatenates K and V across both domains: `key = cat([key_normal, key_rgb], dim=1)`
- Each domain attends to BOTH domains' keys/values simultaneously
- Result: each normal token and each RGB token has seen the full information
  from both domains before exiting the transformer block

Our hook fires on the output of the **entire up_block**, after ALL transformer
blocks complete — i.e. AFTER cross-domain mixing.

Runtime check (wonder3d-v1.0):
```
up_blocks[1]: cd_attention_mid=True,  cd_attention_last=False, mvcd_attention=False
up_blocks[2]: cd_attention_mid=True,  cd_attention_last=False, mvcd_attention=False
up_blocks[3]: cd_attention_mid=True,  cd_attention_last=False, mvcd_attention=False
```

### The scientific claim we are making — PAPER FRAMING DECISION

We are probing the **cross-attended RGB stream** — features AFTER
`JointAttnProcessor` has mixed normal-domain information into the RGB stream.

#### Why this is the right object (not pre-cross-domain features)

1. **Causal link to output**: The post-CD RGB stream is precisely what drives
   the final decoded RGB image. Probing it connects our readout directly to
   the model output.

2. **Motivating claim is stronger**: "The RGB generation stream (which has
   access to cross-domain 3D geometry information) loses CLIP-semantic content
   during denoising" is a stronger and more surprising claim than "isolated RGB
   self-attention features drift" — because one might expect cross-domain
   attention to rescue semantic content.

3. **Framing**: Use **"the RGB generation stream"** or **"the cross-domain
   attended RGB features"** in the paper — NOT "pure RGB-only features".
   The cross-domain mixing is design-intentional and happens at every
   decoder block.

#### Pre-CD vs Post-CD — empirical validation

The `motivation_experiment.py` v2 now simultaneously probes BOTH locations
and computes Linear CKA vs timestep for each. The three scenarios and their
implications are documented in `docs/motivation_evidence.md`.

If the results show SCENARIO C (pre-CD drifts more than post-CD), the paper
framing must be revised before submission. See `motivation_evidence.md`.

**Canonical paper language:** *"We probe the RGB generation stream of Wonder3D's
UNet decoder — specifically the cross-domain attended features at up_blocks
[1, 2, 3] — and measure their CLIP alignment as a function of denoising
timestep using linear CKA."*

---

## Q4 — Hardcoded 6-vs-12 audit

Grep of all `src/` files for potential confusion sites (excluding known-correct
constants and architectural dimensions):

```
train_readout.py:59:    NUM_VIEWS = 6                ← module constant, correct
guidance_inference.py:67:    NUM_VIEWS = 6           ← module constant, correct
guidance_inference.py:286:  [front_pil] * NUM_VIEWS * 2   ← 2 domains × 6 = 12, correct
guidance_inference.py:297:  batch_size = NUM_VIEWS * 2    ← 12, correct
guidance_inference.py:401:  latents_rgb = latents[NUM_VIEWS:]   ← slice(6,12), correct
motivation_experiment.py:66:  NUM_VIEWS = 6          ← module constant, correct
motivation_experiment.py:169: [front_pil] * NUM_VIEWS * 2   ← 12, correct
motivation_experiment.py:183: batch_size = NUM_VIEWS * 2    ← 12, correct
motivation_experiment.py:198: cosine_sims = np.zeros((num_steps, NUM_VIEWS))  ← [steps,6], correct
motivation_experiment.py:240: final_latents = latents[NUM_VIEWS:]  ← RGB slice, correct
test_feature_shapes.py:124:   pipe.camera_embedding  # [12, 5]    ← comment, correct
```

**No raw integer literals `6` or `12` in array indexing or shape assertions** —
all occurrences use `NUM_VIEWS`, `V`, `BV`, or derived expressions (`2*V`,
`V : 2*V`, `NUM_VIEWS:`). The only places `6` appears as a literal are in
`num_views: int = 6` default arguments in function signatures, which is correct
default plumbing.

**Confirmed clean.** No 6-vs-12 confusion remaining in src/.

---

## Q5 — Image Conditioning: CLIP not T5

**Status resolved (critical bug fix, 2026-04-17).**

`prepare_unet_inputs()` in `train_readout_caption.py` originally passed `t5_emb`
as `encoder_hidden_states` to the UNet. This was incorrect: Wonder3D expects
CLIP image embeddings as conditioning, not T5 text embeddings.

**Incorrect (removed):**
```python
clip_bv = t5_emb.to(device).unsqueeze(1).expand(B, V, 768)...
image_embeddings = clip_bv.repeat(2, 1, 1)
```

**Correct (current):**
```python
view0_pil = [Image.fromarray(...) for i in range(B)]
clip_pixel = pipe.feature_extractor(images=view0_pil, ...).pixel_values
clip_embed = pipe.image_encoder(clip_pixel).image_embeds.float()  # [B, 768]
clip_bv = clip_embed.unsqueeze(1).expand(B, V, 768)...
image_embeddings = clip_bv.repeat(2, 1, 1)   # [2BV, 1, 768]
```

This matches `_encode_image()` in `pipeline_mvdiffusion_image_joint.py:155-156`:
```python
image_embeddings = self.image_encoder(image_pt).image_embeds   # .image_embeds NOT .last_hidden_state
image_embeddings = image_embeddings.unsqueeze(1)               # [B, 1, 768]
```

**`t5_emb` role:** Loss target only (`CaptionHead.loss(pred, t5_emb)`). Not UNet input.

---

## Q6 — AggregationNetwork Timestep Conditioning

**Status: implemented, not used.**

`BottleneckBlock` contains `emb_proj: nn.Linear(1280, bottleneck_channels)` for
timestep injection. `AggregationNetwork.forward(features, emb=None)` and
`CaptionHead.forward(features, emb=None)` both accept the optional argument.

**Current training:** `emb=None` passed at all call sites. Timestep conditioning
is inactive. This is a deliberate simplification for the initial experiment —
hooking `unet.time_embedding` output to feed `emb` is left for future work.

**Impact:** AggregationNetwork is timestep-agnostic. Features from different
noise levels `t ~ U(0, 1000)` are treated identically. The head must be
timestep-robust implicitly.

---

## Summary table

| Question | Answer |
|---|---|
| Which half is RGB? | Second half: `feat[V:2V]` (training) or `feat[3V:4V]` (CFG cond) |
| Front view index? | View 0, `delta_azimuth = 0`. CLIP target = `CLIP(view_0)` |
| Cross-domain contamination? | Yes — intentional. We probe the cross-attended RGB stream. This IS the correct scientific object; it drives the final decoded image. |
| Hardcoded 6/12 bugs? | None. All indexing uses `NUM_VIEWS`, `V`, or derived expressions. |
| Image conditioning? | CLIP `.image_embeds` from `pipe.image_encoder(feature_extractor(view0))`. T5 is loss target only. |
| Timestep conditioning? | Built in to AggregationNetwork but inactive (emb=None). Simplification for v1. |
