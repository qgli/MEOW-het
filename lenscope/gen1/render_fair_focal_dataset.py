"""
Render first-generation scenes into multi-camera training packs (run inside Blender).

Each scene is rendered with Cycles at --num_poses camera poses (default 8) for each of
the eight base cameras in render_unicol_dataset.BASE_CAMERAS: five pinholes with 35 mm
equivalent focal lengths of 14, 24, 40, 70 and 135 mm (1920x1080), two fisheyes with
180 and 220 degree fields of view (1080x1080) and one equirectangular panorama
(2160x1080), at --samples samples per pixel (default 64).

Helpers reused from render_unicol_dataset:
  - BASE_CAMERAS (5 pinhole + 2 fisheye + 1 equirectangular)
  - setup_render_engine(): Cycles on the GPU with OptiX
  - setup_camera_pinhole/fisheye/erp(): camera setup
  - generate_rays_pinhole/fisheye/erp(): ground-truth ray directions (closed form)
  - extract_z_pass / z_pass_to_radial_distance / render_and_grab_pixels
  - save_pack_npz(): NPZ pack writer
  - extract_extrinsics(): Blender camera frame -> dataset camera frame (X right, Y up, Z forward)
  - auto_detect_orbit_params() / generate_orbit_trajectory(): orbit poses

Output layout:
  <out>/<blend name>/
    metadata.json                  # pose source, camera convention, per-frame intrinsics/extrinsics
    {base}_pose{NN}_pack.npz       # rgb, rays, radial depth (float16) and mask (uint8)
    {base}_pose{NN}_debug.png      # visualization PNG (optional)
  <out>/render_summary.json

Usage (through the wrapper that makes render_unicol_dataset importable):
  blender -b --python lenscope/gen1/render_fair_focal.py -- \\
    --pick scene_pick.json \\
    --out  debug_data \\
    --num_poses 8 \\
    --samples 64 \\
    --debug_png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import bpy  # type: ignore
import numpy as np
from mathutils import Vector, Euler  # type: ignore

# Camera, ray and pack helpers of the first-generation scene generator (made importable by
# lenscope/gen1/render_fair_focal.py)
import render_unicol_dataset as runi  # type: ignore  # noqa: E402


# Native 1080p resolutions (BASE_CAMERAS defaults). MapAnything's load_images() will internally
# resize to fixed_mapping(518) at inference time; analytical rays can be recomputed at any res.
# (W, H) per camera type
RESOLUTION_OVERRIDE = {
    "pin_14mm":  (1920, 1080),  # 16:9
    "pin_24mm":  (1920, 1080),
    "pin_40mm":  (1920, 1080),
    "pin_70mm":  (1920, 1080),
    "pin_135mm": (1920, 1080),
    "fish_180":  (1080, 1080),  # 1:1
    "fish_220":  (1080, 1080),
    "erp":       (2160, 1080),  # 2:1
}

# Raycast validation parameters for fallback orbit poses
RAYCAST_MIN_CLEAR_M = 0.30   # camera must be at least this far from any wall
RAYCAST_OUTDOOR_M   = 50.0   # if all 6 rays go beyond this, camera is outdoors
RAYCAST_MIN_HITS    = 4      # need at least this many of 6 directional rays to hit


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--pick", type=str, required=True)
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for the per-scene render packs (the root directory of the training dataset).",
    )
    p.add_argument("--num_poses", type=int, default=8)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--debug_png", action="store_true")
    p.add_argument("--scene_indices", type=str, default="",
                   help="comma-separated indices into pick['top']; empty=all")
    return p.parse_args(argv)


def setup_camera_for_base(cam_data, base):
    """Apply intrinsic settings for a given BASE_CAMERAS entry. Returns (W, H, intr_dict)."""
    name = base["name"]
    W, H = RESOLUTION_OVERRIDE[name]
    btype = base["type"]
    if btype == "pinhole":
        intr = runi.setup_camera_pinhole(cam_data, W, H, base["fov_deg"])
    elif btype == "fisheye":
        intr = runi.setup_camera_fisheye(cam_data, W, H, base["fov_deg"])
    elif btype == "erp":
        intr = runi.setup_camera_erp(cam_data, W, H)
    else:
        raise ValueError(f"unknown base type: {btype}")
    return W, H, intr


def gen_rays_for_base(base, H, W):
    btype = base["type"]
    if btype == "pinhole":
        return runi.generate_rays_pinhole(H, W, base["fov_deg"])
    elif btype == "fisheye":
        return runi.generate_rays_fisheye(H, W, base["fov_deg"])
    elif btype == "erp":
        return runi.generate_rays_erp(H, W)
    else:
        raise ValueError(f"unknown base type: {btype}")


def render_one_pose(scene, cam_obj, base, pose, scene_out_dir, debug_png):
    """Render a single (base, pose) frame; save NPZ pack. Returns metadata dict."""
    name = base["name"]
    pose_idx = pose["index"]
    tag = f"{name}_pose{pose_idx:02d}"

    # set camera intrinsics + resolution
    cam_data = cam_obj.data
    W, H, intr = setup_camera_for_base(cam_data, base)
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    # set camera pose
    cam_obj.location = Vector(pose["location"])
    cam_obj.rotation_euler = Euler(pose["rotation_euler"], "XYZ")
    bpy.context.view_layer.update()

    # set up the Z-pass file output before rendering
    z_path_dir = scene_out_dir / "_zpass_tmp"
    z_path_dir.mkdir(parents=True, exist_ok=True)
    runi._setup_z_file_output(scene, str(z_path_dir), tag)

    # render RGB (linear → sRGB float [0,1]); also writes Z-pass EXR via compositor
    t0 = time.time()
    rgb_srgb = runi.render_and_grab_pixels(scene, W, H)  # [H, W, 3] float in [0,1]

    # extract Z-pass → radial depth (meters)
    try:
        z_buffer = runi.extract_z_pass(scene, str(z_path_dir), tag)  # [H, W] float32
        if z_buffer is None:
            z_buffer = np.zeros((H, W), dtype=np.float32)
    except Exception as e:
        print(f"    [warn] extract_z_pass failed for {tag}: {e}; using zeros")
        z_buffer = np.zeros((H, W), dtype=np.float32)

    # GT rays + valid mask
    rays, valid = gen_rays_for_base(base, H, W)
    rays = rays.astype(np.float32)
    mask = valid.astype(np.uint8)

    # Z-buffer → radial distance along ray
    try:
        depth_radial = runi.z_pass_to_radial_distance(z_buffer, rays, valid, base["type"])
    except Exception as e:
        print(f"    [warn] z_pass_to_radial_distance failed for {tag}: {e}")
        depth_radial = z_buffer

    # save pack
    pack_path = runi.save_pack_npz(
        str(scene_out_dir),
        tag,
        rgb_srgb.astype(np.float16),
        rays.astype(np.float16),
        depth_radial.astype(np.float16),
        mask,
        debug_png=debug_png,
    )
    t_total = time.time() - t0

    # extrinsics in the dataset camera convention (X right, Y up, Z forward)
    c2w, pos = runi.extract_extrinsics(cam_obj)

    return {
        "tag": tag,
        "base_name": name,
        "base_type": base["type"],
        "fov_deg": base["fov_deg"],
        "eq_focal_mm": base.get("eq_focal_mm"),
        "resolution": [W, H],
        "intrinsics": intr,
        "pose_index": pose_idx,
        "theta_deg": pose.get("theta_deg", 0.0),
        "camera_location": list(pose["location"]),
        "camera_rotation_euler": list(pose["rotation_euler"]),
        "camera_to_world_unicol_4x4": c2w.tolist(),
        "camera_position_world": pos.tolist(),
        "depth_stats": {
            "min": float(np.min(depth_radial[valid])) if valid.any() else 0.0,
            "max": float(np.max(depth_radial[valid])) if valid.any() else 0.0,
            "mean": float(np.mean(depth_radial[valid])) if valid.any() else 0.0,
        },
        "render_time_s": round(t_total, 3),
        "pack_file": Path(pack_path).name,
    }


def _is_pose_inside_room(scene, loc):
    """Cast 6 axis-aligned rays from loc; return True iff pose is plausibly inside a room.
    Reject if any ray hits within RAYCAST_MIN_CLEAR_M (camera clipping wall),
    or if fewer than RAYCAST_MIN_HITS out of 6 directions hit anything within RAYCAST_OUTDOOR_M."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = Vector(loc)
    directions = [Vector(d) for d in [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    ]]
    n_hits = 0
    min_dist = float("inf")
    for d in directions:
        hit, location, _normal, _idx, _obj, _mat = scene.ray_cast(depsgraph, origin, d)
        if not hit:
            continue
        dist = (location - origin).length
        if dist > RAYCAST_OUTDOOR_M:
            continue
        n_hits += 1
        if dist < min_dist:
            min_dist = dist
    if min_dist < RAYCAST_MIN_CLEAR_M:
        return False, f"too close to wall (min={min_dist:.2f}m)"
    if n_hits < RAYCAST_MIN_HITS:
        return False, f"only {n_hits}/6 rays hit (likely outdoors)"
    return True, f"OK (hits={n_hits}/6, min_clear={min_dist:.2f}m)"


def _resolve_poses(scene, num_poses):
    """Strategy: 1) use the Camera_NN objects placed by the scene generator (target-to-camera,
    T2C); 2) fall back to a bounding-box orbit filtered by ray casting.
    Returns (poses_list, source_tag)."""
    # --- Strategy 1: T2C pre-placed cameras ---
    poses = runi.detect_scene_cameras(num_poses)
    if poses is not None:
        # validate each pose with raycast (T2C placement is usually good but verify)
        valid_poses = []
        for p in poses:
            ok, msg = _is_pose_inside_room(scene, p["location"])
            if ok:
                valid_poses.append(p)
            else:
                print(f"    [skip Camera_NN pose {p['index']}] {msg}")
        if len(valid_poses) >= num_poses:
            for i, p in enumerate(valid_poses[:num_poses]):
                p["index"] = i
            return valid_poses[:num_poses], "t2c_camera_nn"
        print(f"    [t2c] only {len(valid_poses)}/{num_poses} valid; trying additional Camera_NN...")
        # try more Camera_NN beyond the first num_poses
        all_poses = runi.detect_scene_cameras(50) or []
        for p in all_poses[num_poses:]:
            ok, msg = _is_pose_inside_room(scene, p["location"])
            if ok:
                valid_poses.append(p)
            if len(valid_poses) >= num_poses:
                break
        if len(valid_poses) >= num_poses:
            for i, p in enumerate(valid_poses[:num_poses]):
                p["index"] = i
            return valid_poses[:num_poses], "t2c_camera_nn_extended"

    # --- Strategy 2: bbox orbit + raycast filter ---
    print("    [fallback] bbox orbit + raycast validation")
    center, radius, height, look_at = runi.auto_detect_orbit_params()
    # oversample then filter
    candidates = runi.generate_orbit_trajectory(num_poses * 4, radius, height, center, look_at)
    valid_poses = []
    for p in candidates:
        ok, msg = _is_pose_inside_room(scene, p["location"])
        if ok:
            valid_poses.append(p)
        if len(valid_poses) >= num_poses:
            break
    if not valid_poses:
        # last resort: shrink radius
        for shrink in [0.7, 0.5, 0.3]:
            candidates = runi.generate_orbit_trajectory(num_poses * 4, radius * shrink, height, center, look_at)
            for p in candidates:
                ok, _ = _is_pose_inside_room(scene, p["location"])
                if ok:
                    valid_poses.append(p)
                if len(valid_poses) >= num_poses:
                    break
            if len(valid_poses) >= num_poses:
                break
    for i, p in enumerate(valid_poses[:num_poses]):
        p["index"] = i
    return valid_poses[:num_poses], "bbox_orbit_raycast"


def render_scene(blend_path, out_dir: Path, num_poses: int, samples: int, debug_png: bool):
    print(f"\n{'='*70}\n[scene] {blend_path}\n{'='*70}")
    bpy.ops.wm.read_factory_settings(use_empty=False)
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    # Override the default sample count of render_unicol_dataset
    runi.RENDER_SAMPLES = samples
    runi.setup_render_engine()
    scene = bpy.context.scene
    scene.cycles.samples = samples
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass

    # resolve poses before deleting any cameras
    poses, pose_source = _resolve_poses(scene, num_poses)
    print(f"    pose_source = {pose_source}, n_poses = {len(poses)}")
    if len(poses) == 0:
        raise RuntimeError(f"no valid poses for {blend_path}")

    # now remove pre-existing cameras (their poses have been copied)
    for o in list(scene.objects):
        if o.type == "CAMERA":
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

    # create one camera object that is reused for all frames
    cam_data = bpy.data.cameras.new("FairCam")
    cam_obj = bpy.data.objects.new("FairCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    n_total = len(runi.BASE_CAMERAS) * len(poses)
    done = 0
    t_start = time.time()

    for base in runi.BASE_CAMERAS:
        for pose in poses:
            done += 1
            print(f"\n[{done}/{n_total}] {base['name']}  pose{pose['index']:02d}")
            try:
                meta = render_one_pose(scene, cam_obj, base, pose, out_dir, debug_png)
                all_frames.append(meta)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [error] {e}")

    # write per-scene metadata
    scene_meta = {
        "blend_path": str(blend_path),
        "blend_name": Path(blend_path).stem,
        "n_frames": len(all_frames),
        "num_poses": num_poses,
        "pose_source": pose_source,
        "coordinate_system": {
            "name": "unicol_z_forward",
            "X": "right",
            "Y": "up",
            "Z": "forward (optical axis)",
            "blender_conversion": "z_unicol = -z_blender",
        },
        "render_settings": {
            "engine": "CYCLES",
            "samples": samples,
            "resolution_mapping_basis": "mapanything fixed_mapping(518)",
        },
        "frames": all_frames,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(scene_meta, f, indent=2)
    print(f"\n[scene done] {len(all_frames)}/{n_total} frames in {(time.time()-t_start)/60:.1f} min, "
          f"metadata -> {out_dir/'metadata.json'}")
    return scene_meta


def main():
    args = parse_args()
    pick = json.load(open(args.pick))
    top = pick["top"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(top)))
    if args.scene_indices.strip():
        indices = [int(x) for x in args.scene_indices.split(",")]

    print(f"[main] {len(indices)} scenes -> {out_root} | num_poses={args.num_poses} samples={args.samples}")
    print(f"[main] render resolutions: {RESOLUTION_OVERRIDE}")

    summary = []
    t0 = time.time()
    for i in indices:
        r = top[i]
        bp = r["blend_path"]
        name = Path(bp).stem
        scene_dir = out_root / name
        if (scene_dir / "metadata.json").exists():
            print(f"[main] skip {name} (metadata.json already exists)")
            continue
        try:
            sm = render_scene(bp, scene_dir, args.num_poses, args.samples, args.debug_png)
            summary.append({"scene_index": i, "name": name, "n_frames": sm["n_frames"]})
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[main] FAIL scene {i} {name}: {e}")

    with open(out_root / "render_summary.json", "w") as f:
        json.dump({
            "n_scenes": len(summary),
            "num_poses": args.num_poses,
            "samples": args.samples,
            "elapsed_min": round((time.time() - t0) / 60, 2),
            "scenes": summary,
        }, f, indent=2)
    print(f"\n[done] all scenes in {(time.time()-t0)/60:.1f} min, summary -> {out_root/'render_summary.json'}")


if __name__ == "__main__":
    main()
