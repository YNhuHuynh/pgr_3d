"""Metrics for PGR-3D image-to-3D diffusion evaluation.

Example
-------
>>> evaluate_object("gt.obj", "pred.obj", gt_view_paths, pred_view_paths)
{'cd': 0.02, 'iou': 0.62, 'psnr': 24.1, 'ssim': 0.86, 'lpips': 0.13}
"""

from __future__ import annotations

import csv
import datetime
import logging
import tempfile
from pathlib import Path
from typing import Dict, Union

import numpy as np

LOGGER = logging.getLogger(__name__)
_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def _warn(msg: str, exc: Exception | None = None) -> None:
    LOGGER.warning("%s%s", msg, f": {exc}" if exc else "")


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_mesh(path: str):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    return mesh


def _unit_sphere(mesh):
    mesh = mesh.copy()
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) / 2.0
    verts = np.asarray(mesh.vertices, dtype=np.float64) - center
    radius = float(np.linalg.norm(verts, axis=1).max())
    if radius <= 0:
        raise ValueError("degenerate mesh")
    mesh.vertices = verts / radius
    return mesh


def _unit_cube(mesh):
    mesh = mesh.copy()
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) / 2.0
    scale = float((bounds[1] - bounds[0]).max())
    if scale <= 0:
        raise ValueError("degenerate mesh")
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * (2.0 / scale)
    return mesh


def _unit_cube_pair(mesh_a, mesh_b):
    """Normalize two meshes with one shared transform into [-1, 1]^3."""
    mesh_a, mesh_b = mesh_a.copy(), mesh_b.copy()
    bounds = np.stack([mesh_a.bounds, mesh_b.bounds], axis=0)
    lo, hi = bounds[:, 0, :].min(axis=0), bounds[:, 1, :].max(axis=0)
    center = (lo + hi) / 2.0
    scale = float((hi - lo).max())
    if scale <= 0:
        raise ValueError("degenerate mesh pair")
    for mesh in (mesh_a, mesh_b):
        mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * (2.0 / scale)
    return mesh_a, mesh_b


def _to_pytorch3d(mesh, device):
    import torch
    from pytorch3d.structures import Meshes

    verts = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])


def chamfer_distance(
    gt_mesh_path: str,
    pred_mesh_path: str,
    num_samples: int = 10000,
    normalize: bool = True,
) -> float:
    """Compute symmetric Chamfer Distance (L2) between two mesh files.

    Example: ``chamfer_distance("gt.obj", "pred.obj", num_samples=10000)``.
    Meshes are loaded with trimesh, optionally centered/scaled to a unit
    bounding sphere using ``mesh.bounds``, converted to PyTorch3D Meshes, and
    sampled uniformly on their surfaces. CUDA is used when available.
    """
    try:
        import torch
        from pytorch3d.loss import chamfer_distance as p3d_chamfer
        from pytorch3d.ops import sample_points_from_meshes

        gt, pred = _load_mesh(gt_mesh_path), _load_mesh(pred_mesh_path)
        if normalize:
            gt, pred = _unit_sphere(gt), _unit_sphere(pred)

        device = _device()
        with torch.no_grad():
            gt_pts = sample_points_from_meshes(_to_pytorch3d(gt, device), num_samples)
            pr_pts = sample_points_from_meshes(_to_pytorch3d(pred, device), num_samples)
            cd, _ = p3d_chamfer(
                pr_pts, gt_pts, norm=2, point_reduction="mean", batch_reduction="mean"
            )
        return float((cd * 0.5).detach().cpu().item())
    except Exception as exc:
        _warn(f"Chamfer Distance failed for {pred_mesh_path} vs {gt_mesh_path}", exc)
        return float(np.nan)


def _voxel_occupancy(mesh, grid_resolution: int) -> np.ndarray:
    pitch = 2.0 / float(grid_resolution)
    vox = mesh.voxelized(pitch=pitch)
    try:
        vox = vox.fill()
    except Exception as exc:
        _warn("voxel fill failed; using surface voxels", exc)

    occ = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=bool)
    points = np.asarray(vox.points, dtype=np.float64)
    if points.size == 0:
        return occ

    idx = np.floor((points + 1.0) / pitch).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < grid_resolution), axis=1)
    idx = idx[valid]
    if len(idx):
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def volume_iou(
    gt_mesh_path: str,
    pred_mesh_path: str,
    grid_resolution: int = 64,
) -> float:
    """Compute Volume IoU after normalizing both meshes into ``[-1, 1]^3``.

    Example: ``volume_iou("gt.obj", "pred.obj", grid_resolution=64)``.
    Voxelization uses trimesh on a shared ``grid_resolution ** 3`` grid. The
    pair uses one shared normalization transform so relative offsets are kept.
    """
    try:
        gt_mesh, pred_mesh = _unit_cube_pair(_load_mesh(gt_mesh_path), _load_mesh(pred_mesh_path))
        gt = _voxel_occupancy(gt_mesh, grid_resolution)
        pred = _voxel_occupancy(pred_mesh, grid_resolution)
        inter = np.logical_and(gt, pred).sum(dtype=np.float64)
        union = np.logical_or(gt, pred).sum(dtype=np.float64)
        return float(inter / union) if union > 0 else 0.0
    except Exception as exc:
        _warn(f"Volume IoU failed for {pred_mesh_path} vs {gt_mesh_path}", exc)
        return float(np.nan)


def _read_image(path: str) -> np.ndarray:
    from PIL import Image
    from torchvision.transforms import functional as TF

    tensor = TF.to_tensor(Image.open(path).convert("RGB"))
    return (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _image_batch(images: list[str] | np.ndarray) -> np.ndarray:
    if isinstance(images, np.ndarray):
        arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None]
    else:
        arr = np.stack([_read_image(str(p)) for p in images], axis=0)

    if arr.ndim != 4 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError("images must be [N, H, W, C]")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr * 255.0 if float(np.nanmax(arr)) <= 1.0 else arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _lpips_model():
    global _LPIPS_DEVICE, _LPIPS_MODEL
    import lpips

    device = _device()
    if _LPIPS_MODEL is None or _LPIPS_DEVICE != device:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
        _LPIPS_DEVICE = device
    return _LPIPS_MODEL, device


def _lpips_tensor(images: np.ndarray, device):
    import torch

    arr = images.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


def psnr_ssim_lpips(
    gt_images: list[str] | np.ndarray,
    pred_images: list[str] | np.ndarray,
) -> dict:
    """Compute average PSNR, SSIM, and LPIPS for paired rendered views.

    Example: ``psnr_ssim_lpips(["gt.png"], ["pred.png"])``. Inputs may be
    path lists or ``[N, H, W, 3]`` uint8 ``[0, 255]`` / float ``[0, 1]`` arrays.
    LPIPS uses a cached ``lpips.LPIPS(net="alex")`` singleton on CUDA if present.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        gt, pred = _image_batch(gt_images), _image_batch(pred_images)
        if gt.shape != pred.shape or gt.shape[0] == 0:
            raise ValueError(f"invalid paired image batches: {gt.shape} vs {pred.shape}")
        psnr = [peak_signal_noise_ratio(g, p, data_range=255) for g, p in zip(gt, pred)]
        ssim = [
            structural_similarity(g, p, data_range=255, channel_axis=-1)
            for g, p in zip(gt, pred)
        ]
    except Exception as exc:
        _warn("PSNR/SSIM computation failed", exc)
        return {"psnr": float(np.nan), "ssim": float(np.nan), "lpips": float(np.nan)}

    try:
        import torch

        model, device = _lpips_model()
        with torch.no_grad():
            lpips_value = model(_lpips_tensor(gt, device), _lpips_tensor(pred, device))
        lpips_value = float(lpips_value.mean().detach().cpu().item())
    except Exception as exc:
        _warn("LPIPS computation failed", exc)
        lpips_value = float(np.nan)

    return {"psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim)), "lpips": lpips_value}


def evaluate_object(
    gt_mesh_path: str,
    pred_mesh_path: str,
    gt_images: list[str],
    pred_images: list[str],
) -> dict:
    """Run all metrics for one object and return ``cd/iou/psnr/ssim/lpips``.

    Example: ``evaluate_object("gt.obj", "pred.obj", gt_paths, pred_paths)``.
    Failures are logged as warnings and represented as ``np.nan`` values.
    """
    images = psnr_ssim_lpips(gt_images, pred_images)
    return {
        "cd": chamfer_distance(gt_mesh_path, pred_mesh_path),
        "iou": volume_iou(gt_mesh_path, pred_mesh_path),
        **images,
    }


# Backwards-compatible helpers used by src/eval_gso.py.


def mesh_metrics(pred, gt, n_surface_pts: int = 100000, vox_res: int = 64) -> Dict[str, float]:
    """Legacy wrapper returning ``cd_l1``, ``cd_l2``, and ``iou``."""
    with tempfile.TemporaryDirectory() as tmp:
        pred_path, gt_path = Path(tmp) / "pred.obj", Path(tmp) / "gt.obj"
        pred.export(pred_path)
        gt.export(gt_path)
        return {
            "cd_l1": float(np.nan),
            "cd_l2": chamfer_distance(str(gt_path), str(pred_path), n_surface_pts, normalize=False),
            "iou": volume_iou(str(gt_path), str(pred_path), vox_res),
        }


def image_metrics(pred: Union[np.ndarray, object], gt: Union[np.ndarray, object]) -> Dict[str, float]:
    """Legacy wrapper for one prediction/GT image pair."""
    return psnr_ssim_lpips(np.asarray(gt)[None, ...], np.asarray(pred)[None, ...])


MESH_CSV_HEADER = ["timestamp", "object", "setup", "name", "cd_l1", "cd_l2", "iou", "pred", "gt"]
NVS_CSV_HEADER = ["timestamp", "object", "setup", "name", "psnr", "ssim", "lpips", "view_idx"]


def append_mesh_csv(csv_path: Union[str, Path], row: Dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MESH_CSV_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
        writer.writerow(row)


def append_nvs_csv(csv_path: Union[str, Path], row: Dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NVS_CSV_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
        writer.writerow(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    from PIL import Image, ImageDraw, ImageFilter
    import trimesh

    def mark(ok: bool) -> str:
        return "✓" if ok else "✗"

    def finite(x: float) -> bool:
        return bool(np.isfinite(x))

    def print_metric(name: str, value: float, expected: str, ok: bool) -> None:
        if np.isposinf(value):
            shown = "inf"
        elif np.isnan(value):
            shown = "nan"
        else:
            shown = f"{value:.4f}"
        print(f"  {name}: {shown} (expected: {expected}) {mark(ok)}")

    def export_sphere(path: Path, center: tuple[float, float, float], radius: float = 0.5) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        mesh.apply_translation(center)
        mesh.export(path)

    def make_reference_image(size: int = 128) -> np.ndarray:
        img = Image.new("RGB", (size, size), (24, 36, 48))
        draw = ImageDraw.Draw(img)
        for y in range(size):
            color = (30 + y, 70 + y // 3, 140 + y // 2)
            draw.line([(0, y), (size, y)], fill=color)
        draw.rectangle((18, 22, 76, 86), fill=(220, 72, 66))
        draw.ellipse((54, 42, 116, 104), fill=(68, 180, 126))
        draw.polygon([(12, 116), (62, 72), (116, 116)], fill=(235, 205, 70))
        return np.asarray(img, dtype=np.uint8)

    def blur_image(image: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=2.0)))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_path = tmp / "sphere_gt.obj"
        same_path = tmp / "sphere_same.obj"
        shifted_path = tmp / "sphere_shifted_0p3.obj"
        disjoint_path = tmp / "sphere_disjoint_3p0.obj"
        export_sphere(gt_path, (0.0, 0.0, 0.0))
        export_sphere(same_path, (0.0, 0.0, 0.0))
        export_sphere(shifted_path, (0.3, 0.0, 0.0))
        export_sphere(disjoint_path, (3.0, 0.0, 0.0))

        print("\n[Geometry Smoke Test] Chamfer Distance")
        cd = chamfer_distance(str(gt_path), str(shifted_path), num_samples=2048)
        print_metric("CD", cd, "finite > 0 when PyTorch3D ABI is valid", finite(cd) and cd > 0)

        print("\n[Test Case A] Shifted spheres, displacement = 0.3")
        iou_shifted = volume_iou(str(gt_path), str(shifted_path), grid_resolution=64)
        print_metric("IoU", iou_shifted, "0.0 < IoU < 1.0", finite(iou_shifted) and 0.0 < iou_shifted < 1.0)

        print("\n[Test Case B] Identical spheres")
        iou_same = volume_iou(str(gt_path), str(same_path), grid_resolution=64)
        print_metric("IoU", iou_same, "1.0", finite(iou_same) and np.isclose(iou_same, 1.0))

        print("\n[Test Case C] Disjoint spheres, displacement = 3.0")
        iou_disjoint = volume_iou(str(gt_path), str(disjoint_path), grid_resolution=64)
        print_metric("IoU", iou_disjoint, "0.0", finite(iou_disjoint) and np.isclose(iou_disjoint, 0.0))

        rng = np.random.default_rng(7)
        ref = make_reference_image()
        identical = ref.copy()
        noise = rng.integers(0, 256, size=ref.shape, dtype=np.uint8)
        blurred = blur_image(ref)

        print("\n[Test Case D] Identical images")
        scores = psnr_ssim_lpips(ref[None], identical[None])
        print_metric("PSNR", scores["psnr"], "inf or > 100", np.isposinf(scores["psnr"]) or scores["psnr"] > 100)
        print_metric("SSIM", scores["ssim"], "1.0", finite(scores["ssim"]) and np.isclose(scores["ssim"], 1.0))
        print_metric("LPIPS", scores["lpips"], "≈ 0.0", finite(scores["lpips"]) and scores["lpips"] < 0.01)

        print("\n[Test Case E] Random noise vs reference image")
        scores = psnr_ssim_lpips(ref[None], noise[None])
        print_metric("PSNR", scores["psnr"], "< 15", finite(scores["psnr"]) and scores["psnr"] < 15)
        print_metric("SSIM", scores["ssim"], "< 0.3", finite(scores["ssim"]) and scores["ssim"] < 0.3)
        print_metric("LPIPS", scores["lpips"], "> 0.5", finite(scores["lpips"]) and scores["lpips"] > 0.5)

        print("\n[Test Case F] Slightly blurred reference image")
        scores = psnr_ssim_lpips(ref[None], blurred[None])
        print_metric("PSNR", scores["psnr"], "15 to 45", finite(scores["psnr"]) and 15 <= scores["psnr"] <= 45)
        print_metric("SSIM", scores["ssim"], "0.3 to 1.0", finite(scores["ssim"]) and 0.3 < scores["ssim"] < 1.0)
        print_metric("LPIPS", scores["lpips"], "0.01 to 0.5", finite(scores["lpips"]) and 0.01 < scores["lpips"] < 0.5)

        print("\n[Wrapper Smoke Test] evaluate_object")
        print(evaluate_object(str(gt_path), str(shifted_path), ref[None], blurred[None]))
