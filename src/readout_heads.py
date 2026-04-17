"""
PGR-3D Readout Heads
--------------------
Closely follows the Readout Guidance architecture (Luo et al., CVPR 2024)
adapted for Wonder3D's multi-view UNet.

Two heads:
  SemanticHead  -- predicts CLIP ViT-L/14 image embedding from aggregated
                   multi-view features. Loss: 1 - cosine_similarity.
  DepthHead     -- predicts per-view depth map from aggregated features.
                   Loss: scale-invariant MSE vs MiDaS depth.

Both share a single AggregationNetwork (RG-style BottleneckBlock + mixing
weights) that converts multi-scale UNet features [up_blocks 1,2,3] into a
common 384-dim feature map.

Reference: dhf/aggregation_network.py in google-research/readout_guidance
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Feature-level constants (from wonder3d-v1.0/unet/config.json)
# ---------------------------------------------------------------------------
HOOK_BLOCK_INDICES = [1, 2, 3]
HOOK_CHANNELS      = [1280, 640, 320]   # channel dim of each hooked block
COMMON_SPATIAL     = 32                  # resize all to 32×32 (latent size)
PROJECTION_DIM     = 384                 # aggregated feature channels (matches RG default)
CLIP_DIM           = 768                 # CLIP ViT-L/14 image embedding dim


# ---------------------------------------------------------------------------
# Primitives (from RG's BottleneckBlock)
# ---------------------------------------------------------------------------
class _GroupNormConv2d(nn.Conv2d):
    """Conv2d with GroupNorm baked in, for weight-init compatibility."""
    def __init__(self, in_channels: int, out_channels: int, *args,
                 num_norm_groups: int = 32, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.norm = nn.GroupNorm(num_norm_groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(super().forward(x))


class BottleneckBlock(nn.Module):
    """
    RG's residual bottleneck block with optional timestep conditioning.
    Operates on [B, C, H, W] tensors.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        num_norm_groups: int = 32,
        emb_channels: int = 1280,   # Wonder3D time_embed_dim
    ):
        super().__init__()

        self.shortcut = (
            _GroupNormConv2d(in_channels, out_channels, kernel_size=1, bias=False,
                             num_norm_groups=num_norm_groups)
            if in_channels != out_channels else None
        )
        self.conv1 = _GroupNormConv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False,
                                      num_norm_groups=num_norm_groups)
        self.conv2 = _GroupNormConv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                                      padding=1, bias=False, num_norm_groups=num_norm_groups)
        self.conv3 = _GroupNormConv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False,
                                      num_norm_groups=num_norm_groups)

        # Timestep conditioning: linear projection into bottleneck_channels
        self.emb_proj = nn.Linear(emb_channels, bottleneck_channels) if emb_channels > 0 else None

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if self.shortcut is not None:
            nn.init.kaiming_normal_(self.shortcut.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,   # [B, emb_channels]
    ) -> torch.Tensor:
        out = F.relu(self.conv1(x))

        if emb is not None and self.emb_proj is not None:
            emb_out = F.relu(self.emb_proj(emb.to(out.dtype)))   # [B, btl_ch]
            out = out + emb_out[:, :, None, None]

        out = F.relu(self.conv2(out))
        out = self.conv3(out)

        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return F.relu(out + shortcut)


# ---------------------------------------------------------------------------
# Aggregation Network
# ---------------------------------------------------------------------------
class AggregationNetwork(nn.Module):
    """
    Adapts RG's AggregationNetwork for Wonder3D's 3-layer hook output.

    Input:  list of 3 tensors from up_blocks[1,2,3], each [BV, C_l, H_l, W_l]
    Output: aggregated feature map [BV, projection_dim, COMMON_SPATIAL, COMMON_SPATIAL]

    Steps:
      1. Resize each layer to COMMON_SPATIAL
      2. Per-layer BottleneckBlock: C_l → projection_dim
      3. Softmax-normalised learnable mixing weights (per layer × per timestep bucket)
      4. Weighted sum
    """

    def __init__(
        self,
        feature_dims: List[int] = HOOK_CHANNELS,
        projection_dim: int = PROJECTION_DIM,
        num_norm_groups: int = 32,
        emb_channels: int = 1280,
        common_spatial: int = COMMON_SPATIAL,
    ):
        super().__init__()
        self.feature_dims  = feature_dims
        self.projection_dim = projection_dim
        self.common_spatial = common_spatial

        # Per-layer bottleneck (ResNet-style, from RG)
        self.bottleneck_layers = nn.ModuleList([
            BottleneckBlock(
                in_channels=c,
                out_channels=projection_dim,
                bottleneck_channels=projection_dim // 4,
                num_norm_groups=min(num_norm_groups, c // 4),
                emb_channels=emb_channels,
            )
            for c in feature_dims
        ])

        # Mixing weights (one scalar per layer, softmax-normalised like RG)
        n = len(feature_dims)
        self.mixing_weights = nn.Parameter(torch.ones(n))

        # Logit scale for CLIP contrastive loss (kept for compatibility)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        features: List[torch.Tensor],          # [BV, C_l, H_l, W_l] × n_layers
        emb: Optional[torch.Tensor] = None,    # [BV, emb_channels] timestep emb
    ) -> torch.Tensor:
        assert len(features) == len(self.bottleneck_layers)

        mixing_weights = F.softmax(self.mixing_weights, dim=0)   # [n]
        output = None

        for feat, bottleneck, w in zip(features, self.bottleneck_layers, mixing_weights):
            # Resize to common spatial resolution
            if feat.shape[-2:] != (self.common_spatial, self.common_spatial):
                feat = F.interpolate(feat, size=(self.common_spatial, self.common_spatial),
                                     mode="bilinear", align_corners=False)
            x = bottleneck(feat, emb)   # [BV, projection_dim, S, S]
            output = w * x if output is None else output + w * x

        return output   # [BV, projection_dim, S, S]


# ---------------------------------------------------------------------------
# SemanticHead
# ---------------------------------------------------------------------------
class SemanticHead(nn.Module):
    """
    Predicts CLIP ViT-L/14 image embedding from aggregated multi-view features.

    Architecture:
        [BV, projection_dim, S, S]
        → global avg pool → [BV, projection_dim]
        → mean over V views → [B, projection_dim]
        → MLP(projection_dim → 512 → clip_dim) → [B, clip_dim]

    Loss: 1 - cosine_similarity(pred, target_clip_emb)
    """

    def __init__(
        self,
        aggregation_network: AggregationNetwork,
        num_views: int = 6,
        clip_dim: int = CLIP_DIM,
        projection_dim: int = PROJECTION_DIM,
    ):
        super().__init__()
        self.aggregation_network = aggregation_network
        self.num_views = num_views

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 512),
            nn.GELU(),
            nn.Linear(512, clip_dim),
        )

    def forward(
        self,
        features: List[torch.Tensor],          # [BV, C_l, H_l, W_l] × 3
        emb: Optional[torch.Tensor] = None,    # [BV, emb_channels]
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns predicted CLIP embeddings [B, clip_dim].
        batch_size is inferred as BV // num_views if not given.
        """
        agg = self.aggregation_network(features, emb)   # [BV, proj_dim, S, S]
        # Global average pool over spatial dims
        pooled = agg.mean(dim=[-2, -1])                 # [BV, proj_dim]
        # Average over V views → [B, proj_dim]
        BV = pooled.shape[0]
        V  = self.num_views
        B  = BV // V if batch_size is None else batch_size
        pooled = pooled.view(B, V, -1).mean(dim=1)      # [B, proj_dim]
        return self.mlp(pooled)                          # [B, clip_dim]

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        1 - mean cosine similarity. Both [B, clip_dim], already L2-normalised
        by the CLIP model, but we normalise again for safety.
        """
        pred_n   = F.normalize(pred,   dim=-1)
        target_n = F.normalize(target, dim=-1)
        return (1 - (pred_n * target_n).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# DepthHead
# ---------------------------------------------------------------------------
class DepthHead(nn.Module):
    """
    Predicts per-view depth maps from aggregated multi-view features.

    Architecture (follows RG's spatial/output_head design):
        [BV, projection_dim, S, S]
        → 3× Conv(3, SiLU) → [BV, 1, S, S]
        → upsample to target_size → [BV, 1, H, W]
        → Tanh (maps to (-0.5, 0.5) after /2 shift)

    Loss: scale-invariant MSE between predicted and MiDaS depth
    """

    def __init__(
        self,
        aggregation_network: AggregationNetwork,
        target_size: int = 256,
        projection_dim: int = PROJECTION_DIM,
    ):
        super().__init__()
        self.aggregation_network = aggregation_network
        self.target_size = target_size

        # Output head: same as RG's spatial head
        self.output_head = nn.Sequential(
            nn.Conv2d(projection_dim, 128, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns predicted depth maps [BV, 1, target_size, target_size]."""
        agg  = self.aggregation_network(features, emb)   # [BV, proj, S, S]
        pred = self.output_head(agg)                      # [BV, 1, S, S] in (-1,1)
        if pred.shape[-1] != self.target_size:
            pred = F.interpolate(pred, size=(self.target_size, self.target_size),
                                 mode="bilinear", align_corners=False)
        return pred   # [BV, 1, H, W]

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Scale-invariant MSE (following RG and MiDaS convention).
        pred, target: [BV, 1, H, W] — pred is in (-1,1), target is raw MiDaS output.
        We normalise both to zero-mean, unit-variance before computing MSE.
        """
        # Resize target to match pred spatial size
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(target.float(), size=pred.shape[-2:],
                                   mode="bilinear", align_corners=False)

        def scale_shift_invariant_normalise(x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            flat = x.view(b, -1)
            mu  = flat.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
            std = flat.std(dim=1, keepdim=True).clamp(min=1e-5).view(b, 1, 1, 1)
            return (x - mu) / std

        pred_n   = scale_shift_invariant_normalise(pred)
        target_n = scale_shift_invariant_normalise(target.to(pred.dtype))
        return F.mse_loss(pred_n, target_n)


# ---------------------------------------------------------------------------
# CaptionHead
# ---------------------------------------------------------------------------
T5_DIM = 768   # T5-base encoder output dim (same as CLIP_DIM, kept separate for clarity)


class CaptionHead(nn.Module):
    """
    Predicts mean-pooled T5-base embedding from aggregated multi-view features.

    Architecture:
        [BV, projection_dim, S, S]
        → global avg pool → [BV, projection_dim]
        → mean over V views → [B, projection_dim]
        → MLP(projection_dim → t5_dim → t5_dim) → [B, t5_dim]

    Loss: 1 - cosine_similarity(pred, target_t5_emb)
    Target: mean-pooled, L2-normalised T5-base embedding [B, 768]
    """

    def __init__(
        self,
        aggregation_network: AggregationNetwork,
        num_views: int = 6,
        t5_dim: int = T5_DIM,
        projection_dim: int = PROJECTION_DIM,
    ):
        super().__init__()
        self.aggregation_network = aggregation_network
        self.num_views = num_views

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, t5_dim),
            nn.GELU(),
            nn.Linear(t5_dim, t5_dim),
        )

    def forward(
        self,
        features: List[torch.Tensor],          # [BV, C_l, H_l, W_l] × 3
        emb: Optional[torch.Tensor] = None,    # [BV, emb_channels]
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Returns predicted T5 embeddings [B, t5_dim]."""
        agg    = self.aggregation_network(features, emb)   # [BV, proj_dim, S, S]
        pooled = agg.mean(dim=[-2, -1])                    # [BV, proj_dim]
        BV = pooled.shape[0]
        V  = self.num_views
        B  = BV // V if batch_size is None else batch_size
        pooled = pooled.view(B, V, -1).mean(dim=1)         # [B, proj_dim]
        return self.mlp(pooled)                             # [B, t5_dim]

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - mean cosine similarity. Both [B, t5_dim]."""
        pred_n   = F.normalize(pred,   dim=-1)
        target_n = F.normalize(target, dim=-1)
        return (1 - (pred_n * target_n).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Factory: build a matched pair (aggregation_net, head)
# ---------------------------------------------------------------------------

def build_semantic_head(num_views: int = 6, device: str = "cuda") -> SemanticHead:
    agg = AggregationNetwork(
        feature_dims=HOOK_CHANNELS,
        projection_dim=PROJECTION_DIM,
    )
    head = SemanticHead(agg, num_views=num_views)
    return head.to(device)


def build_depth_head(target_size: int = 256, device: str = "cuda") -> DepthHead:
    agg = AggregationNetwork(
        feature_dims=HOOK_CHANNELS,
        projection_dim=PROJECTION_DIM,
    )
    head = DepthHead(agg, target_size=target_size)
    return head.to(device)


def build_caption_head(num_views: int = 6, device: str = "cuda") -> CaptionHead:
    agg = AggregationNetwork(
        feature_dims=HOOK_CHANNELS,
        projection_dim=PROJECTION_DIM,
    )
    head = CaptionHead(agg, num_views=num_views)
    return head.to(device)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_head(path: str, head: nn.Module, optimizer, step: int, config: dict):
    torch.save({
        "step":           step,
        "config":         config,
        "head_state":     head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_head(path: str, head: nn.Module, optimizer=None, device: str = "cuda"):
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt["head_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt.get("config", {})
