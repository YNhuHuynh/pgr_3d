"""
BlenderProc script: render 6 orthographic views of an object.
Matches Wonder3D's 6-view camera config (elevation=0, azimuths 0/45/90/135/180/225 deg).

Usage (via blenderproc):
    blenderproc run --blender-install-path /path/to/blender \
        scripts/blender_render_6views.py \
        --object_path /path/to/object.glb \
        --output_dir /path/to/output/{uid}/ \
        --uid {uid} \
        --resolution 256 \
        --ortho_scale 1.35
"""

import argparse
import os
import sys
import numpy as np

import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject
import bpy

# Wonder3D 6-view camera azimuths (degrees)
AZIMUTHS   = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0]
ELEVATIONS = [0.0,  0.0,  0.0,   0.0,   0.0,   0.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--uid",         required=True)
    parser.add_argument("--resolution",  type=int, default=256)
    parser.add_argument("--ortho_scale", type=float, default=1.35)
    # blenderproc passes extra args; ignore them
    args, _ = parser.parse_known_args()
    return args


def az_el_to_xyz(azimuth_deg: float, elevation_deg: float, dist: float = 1.5):
    """Convert azimuth/elevation angles to camera XYZ position."""
    az  = np.radians(azimuth_deg)
    el  = np.radians(elevation_deg)
    x   = dist * np.cos(el) * np.cos(az)
    y   = dist * np.cos(el) * np.sin(az)
    z   = dist * np.sin(el)
    return [x, y, z]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bproc.init()
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.set_noise_threshold(0.01)
    bproc.renderer.set_max_amount_of_samples(128)

    # Set white background
    bpy.context.scene.world.color = (1.0, 1.0, 1.0)

    # Load the object
    objs = bproc.loader.load_obj(args.object_path) if args.object_path.endswith(".obj") \
        else bproc.loader.load_blend(args.object_path) if args.object_path.endswith(".blend") \
        else _load_glb(args.object_path)

    # Normalise to unit sphere
    bbox_min = np.inf * np.ones(3)
    bbox_max = -np.inf * np.ones(3)
    for obj in objs:
        bb = obj.get_bound_box()
        bbox_min = np.minimum(bbox_min, bb.min(axis=0))
        bbox_max = np.maximum(bbox_max, bb.max(axis=0))
    center = (bbox_min + bbox_max) / 2.0
    scale  = 1.0 / max(bbox_max - bbox_min)
    for obj in objs:
        obj.set_location(obj.get_location() - center)
        obj.set_scale(obj.get_scale() * scale)

    # Add light
    bproc.lighting.light_surface(
        objs, power=200.0, emission_strength=4.0
    ) if hasattr(bproc.lighting, "light_surface") else _add_default_light()

    # Set resolution
    bproc.camera.set_resolution(args.resolution, args.resolution)

    # Orthographic projection
    bpy.context.scene.camera.data.type = "ORTHO"
    bpy.context.scene.camera.data.ortho_scale = args.ortho_scale

    for i, (az, el) in enumerate(zip(AZIMUTHS, ELEVATIONS)):
        cam_pos = az_el_to_xyz(az, el, dist=1.5)
        # Camera looks at origin
        rotation = bproc.camera.rotation_from_forward_vec(
            np.array([0, 0, 0]) - np.array(cam_pos),
            inplane_rot=0.0
        )
        cam2world = bproc.math.build_transformation_mat(cam_pos, rotation)
        bproc.camera.add_camera_pose(cam2world)

    # Render
    data = bproc.renderer.render()   # returns dict with "colors" key

    # Save RGB
    colors = data["colors"]   # list of N [H, W, 4] RGBA uint8
    for i, rgba in enumerate(colors):
        rgb   = rgba[:, :, :3]
        alpha = rgba[:, :, 3:4] / 255.0
        # Composite over white background
        rgb   = (rgb / 255.0 * alpha + (1.0 - alpha)).clip(0, 1)
        rgb   = (rgb * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb).save(os.path.join(args.output_dir, f"rgb_{i}.png"))

    print(f"Saved {len(colors)} views to {args.output_dir}")


def _load_glb(path: str):
    """Load GLB using Blender's built-in GLTF importer."""
    bpy.ops.import_scene.gltf(filepath=path)
    objs = []
    for bpy_obj in bpy.context.selected_objects:
        if bpy_obj.type == "MESH":
            objs.append(MeshObject(bpy_obj))
    return objs


def _add_default_light():
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_energy(5.0)
    light.set_location([0, 0, 5])


if __name__ == "__main__":
    main()
