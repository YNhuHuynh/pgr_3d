"""
Wonder3D Feature Extractor
--------------------------
Hooks into the last 3 decoder up-blocks of the frozen Wonder3D UNet and
exposes aggregated multi-scale features for readout head training and
inference-time guidance.

Architecture recap (from wonder3d-v1.0/unet/config.json):
  up_blocks[0] : UpBlock2D,            1280 ch, spatial ~8×8   (not hooked)
  up_blocks[1] : CrossAttnUpBlockMV2D, 1280 ch, spatial ~16×16  ← hooked
  up_blocks[2] : CrossAttnUpBlockMV2D,  640 ch, spatial ~32×32  ← hooked
  up_blocks[3] : CrossAttnUpBlockMV2D,  320 ch, spatial ~32×32  ← hooked

Batch layout inside the UNet forward pass (CD-attention / joint mode):
  With 2 domains (normal + RGB) and V=6 views, no CFG:
    full batch dim = 2 * V = 12
    layout: [normal_0..5, rgb_0..5]
    RGB features = slice(V, 2*V) = slice(6, 12)

  With CFG (inference), after reshape_to_cd_input:
    full batch dim = 4 * V = 24
    layout: [norm_uc, norm_cond, rgb_uc, rgb_cond]  each 6 views
    RGB conditional features = slice(3*V, 4*V) = slice(18, 24)

Usage
-----
    extractor = Wonder3DFeatureExtractor(unet, num_views=6)
    with extractor:                       # registers / removes hooks
        noise_pred = unet(...)            # normal forward pass
    features = extractor.get_features()  # dict with raw hook outputs
    agg      = extractor.aggregate(features, timestep_emb)  # [B*V, C, H, W]
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hook indices: last 3 decoder blocks (indices 1, 2, 3 of up_blocks)
# ---------------------------------------------------------------------------
HOOK_BLOCK_INDICES = [1, 2, 3]
HOOK_CHANNELS      = [1280, 640, 320]   # output channels per hooked block
AGGREGATED_CHANNELS = 128               # common channel dim after bottleneck


class AggregationNetwork(nn.Module):
    """
    Diffusion-Hyperfeatures-style aggregation of multi-scale UNet features.

    For each of the 3 hooked decoder blocks:
      1. Per-layer bottleneck conv:  C_l → AGGREGATED_CHANNELS
      2. Bilinear resize to target_size
    Then weighted sum (softmax-normalised learnable scalars) across layers.

    Optional timestep MLP modulates the per-layer weights.
    """

    def __init__(
        self,
        in_channels: List[int] = HOOK_CHANNELS,
        out_channels: int = AGGREGATED_CHANNELS,
        target_size: int = 32,
        use_timestep_conditioning: bool = True,
        timestep_emb_dim: int = 1280,   # matches Wonder3D time_embed_dim
    ):
        super().__init__()
        self.target_size = target_size
        self.use_timestep_conditioning = use_timestep_conditioning
        n_layers = len(in_channels)

        # Per-layer bottleneck: GroupNorm + Conv 1×1
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(min(32, c // 4), c),
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                nn.GELU(),
            )
            for c in in_channels
        ])

        # Learnable per-layer aggregation weights (raw logits → softmax)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

        # Optional timestep MLP that predicts per-layer weight offsets
        if use_timestep_conditioning:
            self.timestep_mlp = nn.Sequential(
                nn.Linear(timestep_emb_dim, 256),
                nn.SiLU(),
                nn.Linear(256, n_layers),
            )
        else:
            self.timestep_mlp = None

    def forward(
        self,
        features: List[torch.Tensor],          # list of [BV, C_l, H_l, W_l]
        timestep_emb: Optional[torch.Tensor] = None,  # [BV, timestep_emb_dim]
    ) -> torch.Tensor:
        """
        Returns:
            aggregated: [BV, out_channels, target_size, target_size]
        """
        assert len(features) == len(self.bottlenecks), \
            f"Expected {len(self.bottlenecks)} feature maps, got {len(features)}"

        # Bottleneck + resize each layer
        processed = []
        for feat, bottleneck in zip(features, self.bottlenecks):
            x = bottleneck(feat)                          # [BV, C_out, H, W]
            x = F.interpolate(
                x, size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False
            )
            processed.append(x)                           # [BV, C_out, T, T]

        # Compute per-layer weights
        weights = self.layer_logits.unsqueeze(0)          # [1, n_layers]
        if self.timestep_mlp is not None and timestep_emb is not None:
            # timestep_emb: [BV, emb_dim] → [BV, n_layers]
            t_offsets = self.timestep_mlp(timestep_emb)
            weights = weights + t_offsets                 # [BV, n_layers]
        weights = torch.softmax(weights, dim=-1)          # [BV, n_layers]

        # Weighted sum
        stacked = torch.stack(processed, dim=1)           # [BV, n_layers, C, T, T]
        if weights.dim() == 1:
            weights = weights.view(1, -1, 1, 1, 1)
        else:
            weights = weights.view(weights.shape[0], -1, 1, 1, 1)  # [BV, n, 1, 1, 1]
        aggregated = (stacked * weights).sum(dim=1)       # [BV, C, T, T]

        return aggregated


class Wonder3DFeatureExtractor:
    """
    Context manager that attaches forward hooks to the last 3 decoder
    up-blocks of a frozen Wonder3D UNet.

    Example
    -------
        extractor = Wonder3DFeatureExtractor(pipeline.unet, num_views=6)
        with extractor:
            out = pipeline.unet(latent_model_input, t, ...)
        rgb_feats = extractor.get_rgb_features()   # list of 3 tensors

    The hooks fire on each up_block's final output (after the optional
    upsample layer inside the block).
    """

    def __init__(self, unet: nn.Module, num_views: int = 6):
        self.unet = unet
        self.num_views = num_views
        self._hooks: List = []
        self._raw_features: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------
    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------
    def _make_hook(self, block_idx: int):
        def hook(module, input, output):
            # output may be a tuple (hidden_states, ...) for cross-attn blocks
            hidden = output[0] if isinstance(output, tuple) else output
            self._raw_features[block_idx] = hidden.detach()
        return hook

    def _register_hooks(self):
        self._raw_features.clear()
        self._hooks.clear()
        for idx in HOOK_BLOCK_INDICES:
            block = self.unet.up_blocks[idx]
            h = block.register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Feature access
    # ------------------------------------------------------------------
    def get_features(self) -> Dict[int, torch.Tensor]:
        """Raw hook outputs keyed by block index (1, 2, 3)."""
        return dict(self._raw_features)

    def get_rgb_features(
        self,
        use_cfg: bool = False,
        cond_only: bool = True,
    ) -> List[torch.Tensor]:
        """
        Extract the RGB-domain conditional feature maps as a list ordered
        from block 1 → 2 → 3 (coarsest → finest).

        Parameters
        ----------
        use_cfg : bool
            True when CFG is active (batch = 4 × num_views inside UNet);
            False for training (batch = 2 × num_views).
        cond_only : bool
            If True, return only the conditional (not unconditional) RGB slice.
            Only relevant when use_cfg=True.
        """
        V = self.num_views
        rgb_feats = []
        for idx in HOOK_BLOCK_INDICES:
            feat = self._raw_features[idx]   # [B_full, C, H, W]
            B_full = feat.shape[0]

            if use_cfg:
                # Layout (after reshape_to_cd_input):
                # [norm_uc(V), norm_cond(V), rgb_uc(V), rgb_cond(V)]
                if cond_only:
                    rgb_feats.append(feat[3 * V : 4 * V])   # rgb_cond
                else:
                    rgb_feats.append(feat[2 * V : 3 * V])   # rgb_uc
            else:
                # Training layout: [normal(V), rgb(V)]
                rgb_feats.append(feat[V : 2 * V])
        return rgb_feats                                     # each [V, C, H, W]

    def get_rgb_features_batched(
        self,
        batch_size: int,
        use_cfg: bool = False,
    ) -> List[torch.Tensor]:
        """
        Same as get_rgb_features but handles B > 1 objects in the batch.

        Returns list of tensors with shape [B*V, C, H, W].
        """
        V = self.num_views
        B = batch_size
        rgb_feats = []
        for idx in HOOK_BLOCK_INDICES:
            feat = self._raw_features[idx]   # [B*n_domains*V, C, H, W]
            if use_cfg:
                # 4 * B * V total, interleaved: [norm_uc(B*V), norm_c(B*V), rgb_uc(B*V), rgb_c(B*V)]
                rgb_feats.append(feat[3 * B * V : 4 * B * V])
            else:
                # 2 * B * V total: [norm(B*V), rgb(B*V)]
                rgb_feats.append(feat[B * V : 2 * B * V])
        return rgb_feats   # each [B*V, C, H, W]
