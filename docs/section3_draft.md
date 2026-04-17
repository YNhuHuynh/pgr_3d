# Section 3: Empirical Study of Perception Representations in Multi-View 3D Diffusion

Before introducing our perception-guided readout framework, we first investigate how perception-relevant information is represented in the intermediate features of a pre-trained multi-view 3D diffusion model. This study serves two purposes: (1) it establishes whether readout-based perception extraction is feasible on such models, and (2) it reveals a previously unreported structural property of cross-domain attention — a central architectural component of modern multi-view 3D diffusion — which motivates our design choices.

## 3.1 Experimental Setup

We analyze Wonder3D~\cite{long2024wonder3d}, a representative multi-view 3D diffusion model that generates six consistent novel views (three RGB and three normal) from a single input image, conditioned on a CLIP image embedding. Wonder3D's UNet architecture extends Stable Diffusion with multi-view attention for cross-view consistency and cross-domain attention (CD) for information exchange between the RGB and normal streams.

**Probe targets.** For each object, we compute the CLIP ViT-L/14 image embedding of the input front view and use it as the semantic anchor. Our goal is to measure whether — and where — this semantic signal is accessible from the UNet's intermediate features during denoising.

**Probe locations.** We place forward hooks at three decoder blocks of increasing depth: up\_blocks[1] (channel dimension 1280), up\_blocks[2] (640), and up\_blocks[3] (320). At each block, we extract features at two architecturally meaningful points: immediately before the cross-domain attention module (PRE-CD) and immediately after (POST-CD). This gives us six probe streams per object per timestep.

**Metric.** We use linear Centered Kernel Alignment (CKA)~\cite{kornblith2019similarity} between the mean-pooled RGB-domain feature at the front view and the input CLIP embedding, computed across 30 objects per timestep. CKA is dimension-invariant and measures the extent to which a linear function can predict one representation from the other. A stable, high CKA indicates that the feature reliably encodes the CLIP signal.

**Data.** We use the canonical 30-object GSO test subset~\cite{downs2022gso} used by prior work~\cite{liu2023syncdreamer,long2024wonder3d}. For each object, we run Wonder3D inference for 50 DDIM timesteps and extract features at every step.

**Uncertainty quantification.** We report mean CKA with bootstrap 95% confidence intervals (n=1000 resamples over objects), rendered as shaded bands in Figure~\ref{fig:cka_motivation}.

## 3.2 Findings

### Finding 1: CLIP-semantic information is preserved across the entire denoising trajectory.

Across all three decoder blocks and both probe locations, CKA values are remarkably stable as a function of denoising timestep. We observe no systematic drift between high-noise (t ≈ 1000) and clean (t ≈ 0) regimes. Per-block mean CKA values are:

| Block | PRE-CD | POST-CD |
|-------|--------|---------|
| up\_blocks[1] (1280 ch) | 0.812 - 0.844 | 0.830 - 0.866 |
| up\_blocks[2] (640 ch) | 0.619 - 0.764 | 0.660 - 0.776 |
| up\_blocks[3] (320 ch) | 0.432 - 0.527 | 0.363 - 0.462 |

This observation has a direct implication for our approach: because the CLIP-semantic signal does not decay through denoising, readout heads trained to extract perception-relevant content from these features can be applied at any timestep without requiring timestep-specific adaptation beyond conditioning on t.

### Finding 2: Semantic encoding strength decreases with decoder depth.

Shallow decoder blocks (up\_blocks[1,2]) encode CLIP information substantially more strongly than the deepest block (up\_blocks[3]), with a gap of roughly 0.3 in absolute CKA between the shallowest (~0.83 at up\_blocks[1]) and deepest (~0.45 at up\_blocks[3]) levels. This is consistent with the general observation in representation learning that shallow features in decoder hierarchies carry more abstract semantic content, while deeper features focus on local spatial and textural information. For our framework, this suggests that readout heads targeting semantic guidance should draw features from shallow blocks, while heads targeting geometric properties may benefit from deeper blocks.

### Finding 3 (novel): Cross-domain attention plays a dual, depth-dependent role.

To our knowledge, this is the first systematic analysis of how cross-domain attention modulates perception representations in multi-view 3D diffusion. Comparing POST-CD to PRE-CD features within each block, we observe a striking pattern:

| Block | Mean (POST-CD − PRE-CD) | Direction |
|-------|--------------------------|-----------|
| up\_blocks[1] (coarse) | +0.023 (range +0.018 to +0.027) | enhances CLIP alignment |
| up\_blocks[2] (intermediate) | +0.009 (range +0.001 to +0.041) | small and noisy |
| up\_blocks[3] (fine) | −0.080 (range −0.034 to −0.137) | dilutes CLIP alignment |

Cross-domain attention *enhances* semantic coherence at coarse decoder levels and *dilutes* it at fine-grained levels. A plausible interpretation is that cross-domain attention performs two complementary functions: at coarse scales, it propagates semantic context across the RGB and normal streams, reinforcing identity-level information; at fine scales, it injects geometric information from the normal stream into the RGB stream, which necessarily trades off against semantic content that does not encode geometry. This is compatible with Wonder3D's design intent of using cross-domain attention as a mechanism for enforcing geometric consistency between the two generated modalities.

The dual role of cross-domain attention has practical implications. In our framework (Section 4), we train readout heads on POST-CD features because these are the features that directly produce the final decoded RGB images, and we guide the denoising trajectory based on their predictions. Finding 3 indicates that such guidance will operate on features whose semantic content has already been subtly reshaped by cross-domain attention — an effect that is modest for coarse readouts but more pronounced at fine-grained levels.

## 3.3 Summary and Implications for Our Approach

Three empirical findings guide our method design:

1. **Readout-based extraction is feasible.** CLIP-semantic information is present and stable throughout denoising, meaning lightweight readout heads can learn to extract it without needing to track drift.

2. **Layer selection matters.** Shallow decoder features are the natural targets for semantic readout heads; deeper features may be more appropriate for geometric readouts. In Section 4 we build readouts over an aggregation of all three decoder blocks, letting the aggregation network learn appropriate per-layer weighting.

3. **Cross-domain attention is not semantic-neutral.** Guidance applied at POST-CD features will interact with cross-domain attention differently at different scales, an effect we characterize empirically in our ablations (Section 5).

These findings are specific to Wonder3D's architecture but we expect them to apply to related cross-domain multi-view diffusion models (e.g., Wonder3D++, Era3D); verifying this is left for future work.
