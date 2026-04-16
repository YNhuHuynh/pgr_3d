"""
PGR-3D Metrics
--------------
Geometry metrics  : Chamfer Distance (L1, L2), Volume IoU
Image quality     : PSNR, SSIM, LPIPS

All functions accept either numpy arrays or torch tensors where sensible.
CSV logging helpers are included for eval_gso.py.

Dependencies (already in 3drav_a100 / pgr3d env):
    pytorch3d, trimesh, open3d, lpips, scikit-image, torch, torchvision
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import open3d as o3d
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
import lpips as lpips_lib


# ---------------------------------------------------------------------------
# Geometry — Chamfer Distance
# ---------------------------------------------------------------------------

def sample_mesh_points(mesh: trimesh.Trimesh, n: int = 100_000) -> np.ndarray:
    """Sample surface points from a trimesh mesh."""
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float32)


def chamfer_l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Symmetric Chamfer L2 distance between two point clouds.
    Points should already be in consistent scale/units.
    """
    a_t = torch.from_numpy(a).float().unsqueeze(0)   # [1, N, 3]
    b_t = torch.from_numpy(b).float().unsqueeze(0)   # [1, M, 3]

    # pairwise squared distances
    diff_ab = a_t.unsqueeze(2) - b_t.unsqueeze(1)    # [1, N, M, 3]
    dist2_ab = (diff_ab ** 2).sum(-1)                 # [1, N, M]

    min_ab = dist2_ab.min(dim=2).values.mean()        # mean over N
    min_ba = dist2_ab.min(dim=1).values.mean()        # mean over M
    return float((min_ab + min_ba) / 2)


def chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer L1 distance."""
    a_t = torch.from_numpy(a).float().unsqueeze(0)
    b_t = torch.from_numpy(b).float().unsqueeze(0)

    diff_ab = a_t.unsqueeze(2) - b_t.unsqueeze(1)    # [1, N, M, 3]
    dist_ab = diff_ab.abs().sum(-1)                   # [1, N, M]

    min_ab = dist_ab.min(dim=2).values.mean()
    min_ba = dist_ab.min(dim=1).values.mean()
    return float((min_ab + min_ba) / 2)


# ---------------------------------------------------------------------------
# Geometry — Volume IoU
# ---------------------------------------------------------------------------

def _build_raycast_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    verts = o3d.core.Tensor(np.array(mesh.vertices, dtype=np.float32))
    faces = o3d.core.Tensor(np.array(mesh.faces, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(verts, faces)
    return scene


def _occupancy(scene: o3d.t.geometry.RaycastingScene, pts: np.ndarray) -> np.ndarray:
    q = o3d.core.Tensor(pts.astype(np.float32))
    sdf = scene.compute_signed_distance(q).numpy()
    return sdf <= 0   # inside ↔ SDF ≤ 0


def volume_iou(
    pred: trimesh.Trimesh,
    gt: trimesh.Trimesh,
    vox_res: int = 64,
    pad: float = 0.05,
) -> float:
    """
    Volumetric IoU computed on a uniform grid fitted to the union of both
    meshes' bounding boxes.

    Parameters
    ----------
    pred, gt : trimesh.Trimesh — meshes in the *same* coordinate frame
    vox_res  : grid resolution along each axis
    pad      : fractional padding around the bounding box
    """
    # Build bounding grid over union of both meshes
    all_verts = np.concatenate([np.array(pred.vertices), np.array(gt.vertices)], axis=0)
    lo = all_verts.min(axis=0)
    hi = all_verts.max(axis=0)
    span = hi - lo
    lo -= span * pad
    hi += span * pad

    xs = np.linspace(lo[0], hi[0], vox_res)
    ys = np.linspace(lo[1], hi[1], vox_res)
    zs = np.linspace(lo[2], hi[2], vox_res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    pred_scene = _build_raycast_scene(pred)
    gt_scene   = _build_raycast_scene(gt)

    pred_occ = _occupancy(pred_scene, pts)
    gt_occ   = _occupancy(gt_scene, pts)

    inter = float((pred_occ & gt_occ).sum())
    union = float((pred_occ | gt_occ).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Image quality — PSNR, SSIM, LPIPS
# ---------------------------------------------------------------------------

def _to_uint8_np(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a tensor/array in [0,1] float or [0,255] uint8 to uint8 HWC."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)   # CHW → HWC
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def psnr(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
) -> float:
    """PSNR in dB. Inputs can be [H,W,C] or [C,H,W] float in [0,1] or uint8."""
    p = _to_uint8_np(pred)
    g = _to_uint8_np(gt)
    return float(skimage_psnr(g, p, data_range=255))


def ssim(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
) -> float:
    """SSIM in [0,1]. Inputs can be [H,W,C] or [C,H,W] float in [0,1] or uint8."""
    p = _to_uint8_np(pred)
    g = _to_uint8_np(gt)
    channel_axis = 2 if p.ndim == 3 else None
    return float(skimage_ssim(g, p, data_range=255, channel_axis=channel_axis))


# Lazily initialised LPIPS model (AlexNet, same as common benchmarks)
_lpips_fn: Optional[lpips_lib.LPIPS] = None


def _get_lpips_fn() -> lpips_lib.LPIPS:
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips_lib.LPIPS(net="alex").eval()
        if torch.cuda.is_available():
            _lpips_fn = _lpips_fn.cuda()
    return _lpips_fn


def lpips_score(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    LPIPS in [0, ~1] (lower is better). Inputs can be [H,W,C] or [C,H,W]
    float in [0,1] or uint8.
    """
    fn = _get_lpips_fn()

    def to_tensor_norm(x):
        if isinstance(x, np.ndarray):
            if x.dtype == np.uint8:
                x = x.astype(np.float32) / 255.0
            if x.ndim == 3 and x.shape[2] in (1, 3):
                x = x.transpose(2, 0, 1)
            x = torch.from_numpy(x)
        # x is now CHW float [0,1]; LPIPS wants [-1,1]
        x = x.float() * 2.0 - 1.0
        if x.ndim == 3:
            x = x.unsqueeze(0)   # [1,C,H,W]
        if torch.cuda.is_available():
            x = x.cuda()
        return x

    with torch.no_grad():
        score = fn(to_tensor_norm(pred), to_tensor_norm(gt))
    return float(score.mean().cpu())


# ---------------------------------------------------------------------------
# Convenience: compute all metrics for a single mesh pair
# ---------------------------------------------------------------------------

def mesh_metrics(
    pred: trimesh.Trimesh,
    gt: trimesh.Trimesh,
    n_surface_pts: int = 100_000,
    vox_res: int = 64,
) -> Dict[str, float]:
    """
    Returns dict with keys: cd_l1, cd_l2, iou
    Both meshes must be in the same coordinate frame and scale.
    """
    pred_pts = sample_mesh_points(pred, n_surface_pts)
    gt_pts   = sample_mesh_points(gt,   n_surface_pts)
    return {
        "cd_l1": chamfer_l1(pred_pts, gt_pts),
        "cd_l2": chamfer_l2(pred_pts, gt_pts),
        "iou":   volume_iou(pred, gt, vox_res=vox_res),
    }


def image_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Returns dict with keys: psnr, ssim, lpips."""
    return {
        "psnr":  psnr(pred, gt),
        "ssim":  ssim(pred, gt),
        "lpips": lpips_score(pred, gt),
    }


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

MESH_CSV_HEADER = [
    "timestamp", "object", "setup", "name",
    "cd_l1", "cd_l2", "iou",
    "pred", "gt",
]

NVS_CSV_HEADER = [
    "timestamp", "object", "setup", "name",
    "psnr", "ssim", "lpips",
    "view_idx",
]


def append_mesh_csv(
    csv_path: Union[str, Path],
    row: Dict,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MESH_CSV_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
        writer.writerow(row)


def append_nvs_csv(
    csv_path: Union[str, Path],
    row: Dict,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NVS_CSV_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
        writer.writerow(row)
