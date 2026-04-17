# Asymmetric Domain Guidance
## PGR-3D / docs/asymmetric_domain_guidance.md

This document records the design decision that readout guidance steers **only
the RGB-domain latents** during inference, and explains why this is both correct
and desirable.

---

## What "asymmetric" means here

Wonder3D jointly denoises two domains in the same UNet forward pass:

```
z_t  [12×V, 4, 32, 32]  layout at inference (CFG off for clarity):
  rows  0.. 5  — normal domain  (geometry / surface-normal generation)
  rows  6..11  — RGB domain     (photometric appearance generation)
```

Our readout loss `L_semantic` produces a gradient `∂L/∂z_t`.
**That gradient is non-zero only for `z_t[6:12]` (RGB domain).**
The normal-domain rows `z_t[0:6]` receive zero guidance gradient.

---

## Why the gradient is zero for the normal domain

The causal chain from `z_t` to `L_semantic`:

```
z_t  →  UNet forward pass  →  UNet features  →  SemanticHead  →  L_semantic
```

Inside the UNet, `GuidedFeatureExtractor.get_rgb_features_batched()` slices:

```python
# guidance_inference.py:125–127
feat = hook_output              # [4*B*V, C, H, W]  (CFG batch)
rgb_feats = feat[3*B*V:4*B*V]  # [B*V, C, H, W]   — rgb_cond rows only
```

This explicit slice at `3*B*V : 4*B*V` cuts the backward graph from the normal
domain rows (`0 : 3*B*V`) to `L_semantic`.  After this slice, no gradient can
flow back to those rows.

The AggregationNetwork and SemanticHead downstream of this slice only see RGB
features, so the backward graph for `L_semantic` only reaches `z_t[6:12]`
(the RGB domain latents, corresponding to `rgb_cond` inside the CFG batch).

**Normal-domain latents `z_t[0:6]` are steered exclusively by the original
Wonder3D score function — readout guidance is invisible to them.**

---

## Why this is the correct design

### 1. Scientific consistency

Our motivation claim (see `docs/motivation_evidence.md`) is that the **RGB
generation stream** loses CLIP-semantic alignment during denoising.  The fix
targets precisely that stream.  Steering the normal domain with an RGB-derived
CLIP loss would be scientifically incoherent: normal-domain features encode
surface geometry, not photometric semantics.

### 2. Geometry stability

The normal domain is Wonder3D's geometry backbone.  Introducing a CLIP-based
gradient into normal-domain denoising would corrupt the geometry prediction,
potentially causing shape collapse even when the semantic guidance is working
correctly.  Leaving `z_t[0:6]` untouched preserves Wonder3D's geometry
consistency guarantees.

### 3. Cross-domain attention propagates the semantic correction

Even though normal-domain latents receive no direct guidance gradient, they
are not unaffected at the image level.  At each denoising step *t-1*, the
updated `z_{t-1}[6:12]` (RGB, corrected) feeds into the next forward pass via
`JointAttnProcessor`'s cross-domain keys/values:

```
attn_joint_mid keys  = cat([K_normal, K_rgb], dim=1)
                                          ↑ updated by guidance
```

The normal-domain transformer tokens attend to the corrected RGB keys, so the
geometry coherently tracks the semantically steered RGB generation — without
receiving a direct gradient from our loss.

### 4. No gradient leakage through cross-domain attention backward pass

One might ask: does the gradient from `L_semantic` propagate backward through
`JointAttnProcessor` into the normal domain via the cross-attention path?

**No.**  The feature hook is registered on the `up_blocks[i]` *output* — after
the JointAttnProcessor has already completed.  At inference, the hook captures
the forward-pass output tensor with `detach()`.  The AggregationNetwork and
SemanticHead operate on these detached features.  The guidance gradient is then
applied directly to `z_t` via the explicit update:

```python
# guidance_inference.py (simplified)
grad = torch.autograd.grad(L_semantic, z_t_rgb)[0]
z_t[6:12] = z_t[6:12] - eta * grad
```

There is no backward pass through the UNet for guidance — guidance is a
gradient-in-latent-space update, not a training step.  So no backward graph
exists through `JointAttnProcessor` at inference time.

---

## Summary table

| Domain | Rows in z_t | Receives guidance gradient | Rationale |
|---|---|---|---|
| Normal (geometry) | 0..5 | **No** | Geometry backbone; no CLIP-semantic supervision needed; cross-domain attention propagates semantic correction indirectly |
| RGB (appearance) | 6..11 | **Yes** | Direct target of our motivation claim; CLIP-predictive features live here |

---

## Paper framing

> "Our readout guidance applies to the RGB generation stream exclusively.
> The normal-domain latents are not directly steered; their semantic alignment
> is maintained implicitly through cross-domain attention, which attends over
> the semantically corrected RGB features at each denoising step."

---

## Code references

| What | Where |
|---|---|
| RGB slice cutting backward graph | `guidance_inference.py:125–127` |
| Guidance update (latent-space, no UNet backward) | `guidance_inference.py:~380–410` |
| Cross-domain attention (JointAttnProcessor) | `transformer_mv2d.py:910–965` |
| Domain ordering confirmation | `docs/ARCHITECTURE_DECISIONS.md Q1` |
