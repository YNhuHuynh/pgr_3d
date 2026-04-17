import blenderproc as bproc   # must be absolute first line (BlenderProc requirement)
"""
BlenderProc script: render 6 orthographic views of an object.
Matches Wonder3D's 6-view camera config (elevation=0, azimuths 0/45/90/135/180/225 deg).

Usage (via blenderproc):
    blenderproc run --custom-blender-path /path/to/blender-3.3.0-linux-x64 \
        scripts/blender_render_6views.py \
        --object_path /path/to/object.glb \
        --output_dir /path/to/output/{uid}/ \
        --uid {uid} \
        --resolution 256 \
        --ortho_scale 1.35
"""

import argparse
import math
import os
import sys
import numpy as np

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
    parser.add_argument("--rotate_x",   type=float, default=0.0,
                        help="Extra X-rotation (degrees) applied after normalisation. "
                             "Use -90 to correct Y-up OBJ meshes (e.g. GSO).")
    parser.add_argument("--rotate_z",   type=float, default=0.0,
                        help="Extra Z-rotation (degrees) applied after normalisation. "
                             "Per-object front-face alignment (e.g. from IPS3D angles).")
    parser.add_argument("--num_views",  type=int, default=6,
                        help="How many views to render (uses first N azimuths from the "
                             "Wonder3D 6-view list: 0,45,90,135,180,225 deg). "
                             "Use 1 to render only the front view (azimuth=0).")
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
    is_glb = not (args.object_path.endswith(".obj") or args.object_path.endswith(".blend"))
    if args.object_path.endswith(".obj"):
        objs = bproc.loader.load_obj(args.object_path)
    elif args.object_path.endswith(".blend"):
        objs = bproc.loader.load_blend(args.object_path)
    else:
        objs = _load_glb(args.object_path)  # returns root-level objects

    # Normalise to unit sphere centred at world origin.
    #
    # For OBJ: `objs` are the mesh objects themselves (flat hierarchy); use them
    # for both bounds and transform.
    # For GLB (Objaverse/Sketchfab): `objs` are the root-level Empties.
    #   - Bounds must be computed from mesh CHILDREN (get_bound_box on an Empty
    #     returns the tiny default Empty bounds, not its children's bounds).
    #   - Transforms must be applied to the ROOT Empties so that the world
    #     positions of all children move correctly.
    # Normalization formula: new_loc = scale * (old_loc - center)
    #   This correctly places the world-space bounding box center at the origin
    #   regardless of the initial object scale.
    if is_glb:
        mesh_objs = [MeshObject(o) for o in bpy.context.scene.objects if o.type == "MESH"]
        norm_objs = objs  # root Empties
    else:
        mesh_objs = norm_objs = objs

    bbox_min = np.inf  * np.ones(3)
    bbox_max = -np.inf * np.ones(3)
    for obj in mesh_objs:
        bb = obj.get_bound_box()
        bbox_min = np.minimum(bbox_min, bb.min(axis=0))
        bbox_max = np.maximum(bbox_max, bb.max(axis=0))
    center = (bbox_min + bbox_max) / 2.0
    scale  = 1.0 / max(bbox_max - bbox_min)
    for obj in norm_objs:
        obj.set_location(scale * (np.array(obj.get_location()) - center))
        obj.set_scale(np.array(obj.get_scale()) * scale)
    # Flush the dependency graph so matrix_world is recalculated for all children
    # before the renderer reads transforms.  Required when modifying Empty parents.
    bpy.context.view_layer.update()

    # Optional extra rotation (degrees → radians) applied AFTER normalisation.
    # For GSO OBJ meshes: rotate_x=-90 fixes Y-up→Z-up; rotate_z aligns front face
    # with the Wonder3D +X camera axis (value = 90 - IPS3D base_angle).
    if args.rotate_x != 0.0 or args.rotate_z != 0.0:
        # Bake current location/scale into vertex data (transform_apply) so that
        # the local origin resets to (0,0,0) = world-space geometric centre.
        # Without this, rotation_euler pivots around the local origin which is at
        # -scale*bbox_centre after normalisation, causing off-centre renders.
        bpy.ops.object.select_all(action='DESELECT')
        for obj in norm_objs:
            obj.blender_obj.select_set(True)
        if norm_objs:
            bpy.context.view_layer.objects.active = norm_objs[0].blender_obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        rx = math.radians(args.rotate_x)
        rz = math.radians(args.rotate_z)
        for obj in norm_objs:
            obj.blender_obj.rotation_euler[0] += rx
            obj.blender_obj.rotation_euler[2] += rz
        bpy.context.view_layer.update()

    # Add light
    # light_surface() requires BsdfPrincipled nodes which many GLBs lack; use
    # a simple SUN light which is reliable across all object types.
    _add_default_light()

    # Set resolution
    bproc.camera.set_resolution(args.resolution, args.resolution)

    # Orthographic projection
    bpy.context.scene.camera.data.type = "ORTHO"
    bpy.context.scene.camera.data.ortho_scale = args.ortho_scale

    n_views = min(args.num_views, len(AZIMUTHS))
    for i, (az, el) in enumerate(zip(AZIMUTHS[:n_views], ELEVATIONS[:n_views])):
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
    """Load GLB using Blender's built-in GLTF importer.

    Returns the ROOT-LEVEL objects (typically Empties) from the imported scene.
    Moving these roots in world space correctly repositions all mesh children.

    Objaverse/Sketchfab GLBs commonly have:
        Sketchfab_model (Empty, parent=None)
          └── *.obj.cleaner... (Empty)
                └── Object_2 (Mesh)

    Applying set_location/set_scale to a leaf Mesh does NOT move it in world
    space because the parent Empty carries the real world offset.  Applying to
    the root Empty does, because matrix_world propagates down the hierarchy.
    """
    bpy.ops.import_scene.gltf(filepath=path)
    imported = list(bpy.context.selected_objects)
    # Root objects: no parent among the imported set
    roots = [MeshObject(o) for o in imported if o.parent is None]
    if not roots:
        # Fallback: GLB had no Empty wrapper — leaf meshes are roots
        roots = [MeshObject(o) for o in imported if o.type == "MESH"]
    return roots


def _add_default_light():
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_energy(5.0)
    light.set_location([0, 0, 5])


if __name__ == "__main__":
    main()
