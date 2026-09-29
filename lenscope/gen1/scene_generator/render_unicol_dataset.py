"""
render_unicol_dataset.py — training-data renderer for first-generation scenes (Blender Python script).

For each scene and camera pose, renders 8 base camera models and packs RGB + rays + depth
directly into a compressed Float16 .npz.

Base cameras (8 bases for full frequency-band coverage without Nyquist aliasing):
  name       eq. focal  horizontal FoV  Blender setting
  pin_14mm   14mm       ~104°           camera.type = 'PERSP'
  pin_24mm   24mm       ~73°            camera.type = 'PERSP'
  pin_40mm   40mm       ~48°            camera.type = 'PERSP'
  pin_70mm   70mm       ~28°            camera.type = 'PERSP'
  pin_135mm  135mm      ~15°            camera.type = 'PERSP'
  fish_180   —          180°            panorama_type = 'FISHEYE_EQUIDISTANT'
  fish_220   —          220°            panorama_type = 'FISHEYE_EQUIDISTANT'
  erp        —          360°×180°       panorama_type = 'EQUIRECTANGULAR'

I/O: direct Float16 NPZ storage
  - No PNG output (avoids sRGB quantization and the zlib CPU decoding bottleneck)
  - Linear RGB pixels are grabbed directly from the render
  - Linear RGB → sRGB conversion (keeps the color space correct)
  - Values clamped to [0, 1]
  - RGB + rays + depth + mask packed into a single Float16 .npz

  Color space: Blender Cycles outputs linear RGB (scene-linear). The sRGB OETF must be
  applied explicitly before storage; otherwise downstream models see wrong colors.

Output (per base × pose):
  output/{base}_{pose}_pack.npz  — pack: rgb[H,W,3] rays[H,W,3]
                                         depth[H,W] mask[H,W] (float16)
  output/metadata.json           — parameters of all 8 base cameras + extrinsics

Coordinate conventions:
  Output camera frame ('unicol_z_forward'): +Z forward (optical axis), +X right, +Y up
  Blender camera frame:                     -Z forward, +X right, +Y up
  Conversion: x_out = x_bl, y_out = y_bl, z_out = -z_bl

  Pixels: i=0 is the top row, j=0 the left column, pixel center = (i+0.5, j+0.5)

Usage:
  blender --background scene.blend --python render_unicol_dataset.py

Requires: Blender 3.x / 4.x (Cycles), NumPy (bundled with Blender)
"""

import bpy
import numpy as np
import os
import json
import math
import time
from mathutils import Vector, Euler, Matrix


# ---- Configuration ----

# --- Output path ---
# Priority: environment variable > <script dir>/output > <.blend dir>/output
def _resolve_output_dir():
    """Resolve the output directory reliably across run environments."""
    env = os.environ.get('UNICOL_OUTPUT')
    if env:
        return env
    # __file__ is usually available in --python mode
    try:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    except NameError:
        pass
    # Fall back to the directory of the .blend file
    if bpy.data.filepath:
        return os.path.join(os.path.dirname(bpy.data.filepath), 'output')
    # Last resort
    return os.path.join(os.getcwd(), 'unicol_output')


OUTPUT_DIR = _resolve_output_dir()

# ---- Base cameras (8 bases for full frequency-band coverage without Nyquist aliasing) ----
#
# Focal length to FoV: FOV_h = 2·arctan(36 / (2·f_eq))  (36mm = full-frame sensor width)
#
# Safe crop limit of each base: f_max = f_base / 0.6:
#   14mm  → 23.3mm  | 24mm → 40mm  | 40mm → 66.6mm
#   70mm  → 116.6mm | 135mm → 225mm

def _eq_focal_to_hfov(f_eq_mm: float) -> float:
    """35mm-equivalent focal length → horizontal FoV (degrees)."""
    return 2.0 * math.degrees(math.atan(36.0 / (2.0 * f_eq_mm)))

# 8 base cameras: (name, type, equivalent focal length in mm, FoV in degrees, resolution WxH)
BASE_CAMERAS = [
    # --- 5 Pinhole bases ---
    {'name': 'pin_14mm',  'type': 'pinhole',  'eq_focal_mm': 14.0,
     'fov_deg': _eq_focal_to_hfov(14.0),    'res': (1920, 1080)},
    {'name': 'pin_24mm',  'type': 'pinhole',  'eq_focal_mm': 24.0,
     'fov_deg': _eq_focal_to_hfov(24.0),    'res': (1920, 1080)},
    {'name': 'pin_40mm',  'type': 'pinhole',  'eq_focal_mm': 40.0,
     'fov_deg': _eq_focal_to_hfov(40.0),    'res': (1920, 1080)},
    {'name': 'pin_70mm',  'type': 'pinhole',  'eq_focal_mm': 70.0,
     'fov_deg': _eq_focal_to_hfov(70.0),    'res': (1920, 1080)},
    {'name': 'pin_135mm', 'type': 'pinhole',  'eq_focal_mm': 135.0,
     'fov_deg': _eq_focal_to_hfov(135.0),   'res': (1920, 1080)},
    # --- 2 Fisheye bases ---
    {'name': 'fish_180',  'type': 'fisheye',  'eq_focal_mm': None,
     'fov_deg': 180.0,                       'res': (1080, 1080)},
    {'name': 'fish_220',  'type': 'fisheye',  'eq_focal_mm': None,
     'fov_deg': 220.0,                       'res': (1080, 1080)},
    # --- 1 equirectangular base ---
    {'name': 'erp',       'type': 'erp',      'eq_focal_mm': None,
     'fov_deg': 360.0,                       'res': (2160, 1080)},
]

# Resolution lookup table (backward compatibility)
RESOLUTIONS = {b['name']: b['res'] for b in BASE_CAMERAS}

# --- Render quality ---
RENDER_SAMPLES   = 32          # standalone default; render_fair_focal_dataset.py sets --samples (64 for the dataset)
USE_DENOISER     = True        # OptiX/OIDN denoising
GPU_BACKEND      = 'OPTIX'     # 'OPTIX' | 'CUDA' | 'HIP' | 'METAL'

# --- Debug ---
# UNICOL_DEBUG_PNG=1  →  after saving the NPZ, also export its sRGB RGB as an 8-bit PNG
#                         for quick visual inspection, not used for training (file name: *_debug.png)
DEBUG_PNG = (os.environ.get('UNICOL_DEBUG_PNG', '0') == '1')

# --- Multi-view trajectory parameters ---
NUM_POSES        = 5           # number of viewpoints sampled on the orbit
ORBIT_RADIUS     = 2.0         # orbit radius (m)
ORBIT_HEIGHT     = 1.0         # camera height above the ground (m)
ORBIT_CENTER     = (0.0, 0.0, 0.0)  # orbit center (scene origin)
LOOK_AT          = (0.0, 0.0, 0.0)  # camera look-at point (scene origin)


# ---- Render engine setup ----

def setup_render_engine():
    """
    Configure Cycles with OptiX GPU acceleration and denoising.

    OptiX setup:
      1. Set compute_device_type = 'OPTIX'
      2. Call get_devices() to refresh the device list
      3. Explicitly enable all OPTIX/CUDA GPU devices (multi-GPU rendering)
      4. Set the GPU tile size so that multiple GPUs render tiles in parallel
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = RENDER_SAMPLES

    # --- Denoising: OptiX AI denoiser (uses the RTX Tensor Cores) ---
    if USE_DENOISER:
        scene.cycles.use_denoising = True
        try:
            scene.cycles.denoiser = 'OPTIX'
            scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
        except TypeError:
            try:
                scene.cycles.denoiser = 'OPENIMAGEDENOISE'
            except TypeError:
                pass

    # --- Adaptive sampling ---
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 16

    # --- Force-enable OptiX GPU rendering ---
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences

    # Prefer OptiX; fall back to CUDA if it is unavailable
    backend_order = ['OPTIX', 'CUDA']
    gpu_activated = False

    for backend in backend_order:
        try:
            cycles_prefs.compute_device_type = backend
            cycles_prefs.get_devices()

            # Enable all GPU devices (OPTIX and CUDA types) and disable the CPU
            gpu_devs = []
            for dev in cycles_prefs.devices:
                if dev.type in (backend, 'CUDA', 'OPTIX'):
                    dev.use = True
                    gpu_devs.append(dev.name)
                elif dev.type == 'CPU':
                    dev.use = False

            if gpu_devs:
                gpu_activated = True
                print(f"  GPU backend: {backend}")
                for name in gpu_devs:
                    print(f"     → {name}")
                break

        except Exception as e:
            print(f"  Warning: {backend} unavailable: {e}")
            continue

    if not gpu_activated:
        print(f"  Warning: no GPU available, falling back to CPU")
        scene.cycles.device = 'CPU'
    else:
        # Tile size: with several GPUs there must be enough tiles to render in parallel.
        # tile_size=4096 on a 4096×4096 image yields a single tile and leaves the second GPU idle;
        # 256-512 keeps several GPUs busy (Blender Cycles distributes the tiles automatically).
        n_gpus = len(gpu_devs)
        optimal_tile = 512 if n_gpus >= 2 else 2048
        try:
            scene.cycles.tile_size = optimal_tile
            print(f"  Tile size: {optimal_tile} (GPUs: {n_gpus}, multi-GPU tile parallelism)")
        except AttributeError:
            pass
        # Persistent data: keep BVH/textures in GPU memory
        scene.render.use_persistent_data = True

    # --- Output format (no PNG is saved; pixels are grabbed directly) ---
    # Basic settings are kept so that the Z-pass File Output node works
    scene.render.image_settings.file_format = 'PNG'  # not actually used
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.color_mode = 'RGB'

    # --- Enable the Z pass (depth) ---
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    scene.use_nodes = True

    print(f"  Cycles: {RENDER_SAMPLES} samples, denoise={'on' if USE_DENOISER else 'off'}")
    print(f"  Z-Pass: enabled (for radial distance extraction)")


# ---- Camera types ----

def setup_camera_pinhole(cam_data, W, H, fov_deg):
    """
    Configure the Blender camera as a pinhole camera with the given FoV.

    Used for the 5 pinhole bases (14mm to 135mm).
    """
    cam_data.type = 'PERSP'
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.angle = math.radians(fov_deg)

    f_px = W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    print(f"    Pinhole: FOV={fov_deg:.1f}°, f={f_px:.2f}px, res={W}×{H}")
    return {'fov_deg': fov_deg, 'f_px': f_px}


def setup_camera_fisheye(cam_data, W, H, fov_deg):
    """
    Configure the Blender camera as an equidistant fisheye with the given FoV.

    Used for the 180° and 220° fisheye bases.
    """
    cam_data.type = 'PANO'
    cam_data.panorama_type = 'FISHEYE_EQUIDISTANT'
    cam_data.fisheye_fov = math.radians(fov_deg)

    half_fov = math.radians(fov_deg) / 2.0
    r_max = min(W, H) / 2.0
    f_theta = r_max / half_fov

    print(f"    Fisheye: FOV={fov_deg:.1f}°, f_θ={f_theta:.2f}px, "
          f"r_max={r_max:.0f}px, valid≈{math.pi/4*100:.1f}%")
    return {'fov_deg': fov_deg, 'f_theta': f_theta, 'r_max': r_max}


def setup_camera_erp(cam_data, W, H):
    """
    Configure the Blender camera as an equirectangular panorama (360°×180°).

    Blender Cycles:
      camera.type = 'PANO'
      cycles.panorama_type = 'EQUIRECTANGULAR'
      latitude/longitude range: ±90° / ±180° by default (full sphere)
    """
    cam_data.type = 'PANO'
    # Blender 4.x: the panorama properties moved from cam_data.cycles.* to cam_data.*
    cam_data.panorama_type = 'EQUIRECTANGULAR'

    # Cover the full sphere
    cam_data.latitude_min  = math.radians(-90)
    cam_data.latitude_max  = math.radians(90)
    cam_data.longitude_min = math.radians(-180)
    cam_data.longitude_max = math.radians(180)

    print(f"    ERP: 360°×180°, res={W}×{H}")
    return {'fov_deg': 360.0}


# ---- Ground-truth ray fields (analytic) ----
# The camera ray field r(i,j) ∈ S² is fully determined by the projection model and the
# intrinsics and does not depend on the scene; it is computed with closed-form formulas
# vectorized in NumPy. All rays are in the output camera frame (Z-forward, X-right, Y-up).

def generate_rays_pinhole(H, W, fov_deg):
    """
    Ground-truth pinhole ray field (Z-forward camera frame).

    Derivation:
      Focal length:  f = W / (2·tan(FOV/2))

      Pixel (i, j) → image-plane coordinates:
        u = j + 0.5 - W/2         (X: positive to the right)
        v = H/2 - i - 0.5         (Y: positive up, row index flipped)

      Ray direction:
        ray = normalize(u, v, f)   (Z-forward = focal direction)

      Checks:
        center pixel (H/2, W/2) → u≈0, v≈0 → ray≈(0,0,1) = +Z
        angle field cos(θ) = f / √(u²+v²+f²)
        at u=±W/2, θ = FOV/2

    Args:
        H, W: image resolution (rows, columns)
        fov_deg: horizontal field of view (degrees)

    Returns:
        rays:  [H, W, 3] float64 unit ray directions
        valid: [H, W] bool validity mask (all valid for a pinhole camera)
    """
    fov_rad = np.deg2rad(fov_deg)
    f = W / (2.0 * np.tan(fov_rad / 2.0))

    # Pixel grid: i = row index (0 = top), j = column index (0 = left)
    j_grid, i_grid = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
    )

    u = (j_grid + 0.5) - W / 2.0          # image-plane X
    v = (H / 2.0) - (i_grid + 0.5)        # image-plane Y (Y-up)
    w = np.full_like(u, f)                 # Z = focal length (forward)

    rays = np.stack([u, v, w], axis=-1)    # [H, W, 3]
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    valid = np.ones((H, W), dtype=bool)
    return rays, valid


def generate_rays_fisheye(H, W, fov_deg):
    """
    Ground-truth equidistant fisheye ray field (Z-forward camera frame).

    Derivation (equidistant projection: r = f_θ · θ):
      Parameters:
        r_max = min(H, W) / 2       (image-circle radius, pixels)
        f_θ   = r_max / (FOV/2)     (equidistant focal length)

      Pixel (i, j) → offset from the image center:
        dx = j + 0.5 - W/2
        dy = H/2 - i - 0.5          (Y-up)
        r  = √(dx² + dy²)           (distance to the center, pixels)

      Spherical angles:
        θ = r / f_θ                  (incidence angle, 0 = optical axis)
        ψ = atan2(dy, dx)            (azimuth in the image plane)

      Cartesian ray (Z-forward):
        x = sin(θ)·cos(ψ)
        y = sin(θ)·sin(ψ)
        z = cos(θ)

      Valid region: r ≤ r_max (circle, θ ≤ FOV/2),
        π/4 ≈ 78.5% of the square image

      Checks:
        center (H/2, W/2) → r=0, θ=0 → ray=(0,0,1) = +Z
        edge r=r_max → θ=90° for a 180° FoV → ray in the equatorial plane

    Args:
        H, W: image resolution (rows, columns)
        fov_deg: full field of view (degrees)

    Returns:
        rays:  [H, W, 3] float64 unit ray directions
        valid: [H, W] bool validity mask (circular FoV region)
    """
    fov_rad = np.deg2rad(fov_deg)
    half_fov = fov_rad / 2.0
    r_max = min(H, W) / 2.0
    f_theta = r_max / half_fov

    j_grid, i_grid = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
    )

    dx = (j_grid + 0.5) - W / 2.0
    dy = (H / 2.0) - (i_grid + 0.5)

    r_pix = np.sqrt(dx**2 + dy**2)

    # Valid pixels: inside the circular FoV boundary
    valid = r_pix <= r_max

    # Incidence angle and azimuth
    theta = r_pix / f_theta
    psi   = np.arctan2(dy, dx)

    # Clamp to avoid out-of-range NaN (invalid pixels are overwritten below)
    theta_safe = np.clip(theta, 0.0, half_fov + 0.001)

    # Spherical → Cartesian (Z-forward)
    rx = np.sin(theta_safe) * np.cos(psi)
    ry = np.sin(theta_safe) * np.sin(psi)
    rz = np.cos(theta_safe)

    rays = np.stack([rx, ry, rz], axis=-1)

    # Invalid pixels: set to the optical axis (avoids NaN; masked out during training)
    rays[~valid] = np.array([0.0, 0.0, 1.0])

    return rays, valid


def generate_rays_erp(H, W):
    """
    Ground-truth equirectangular (ERP) ray field (Z-forward camera frame).

    Derivation:
      Longitude/latitude mapping:
        lon = (j + 0.5) / W × 2π − π    ∈ [−π, π)      longitude
        lat = π/2 − (i + 0.5) / H × π   ∈ (−π/2, π/2)  latitude

      Cartesian ray (Z-forward):
        x = cos(lat) · sin(lon)          X-right
        y = sin(lat)                     Y-up
        z = cos(lat) · cos(lon)          Z-forward

      Checks:
        center (H/2, W/2) → lon≈0, lat≈0 → ray≈(0,0,1) = +Z
        left edge j=0 → lon≈-π → ray≈(0,0,-1)  (behind the camera)
        top row i=0 → lat≈π/2 → ray≈(0,1,0)   (north pole)

      The center of the Blender EQUIRECTANGULAR image is the camera -Z direction,
      i.e. +Z of the output frame. Longitude increases to the right.

    Args:
        H, W: image resolution (rows, columns)

    Returns:
        rays:  [H, W, 3] float64 unit ray directions
        valid: [H, W] bool validity mask (all pixels valid)
    """
    j_grid, i_grid = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
    )

    lon = (j_grid + 0.5) / W * 2.0 * np.pi - np.pi     # longitude
    lat = np.pi / 2.0 - (i_grid + 0.5) / H * np.pi     # latitude

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    rays = np.stack([x, y, z], axis=-1)

    valid = np.ones((H, W), dtype=bool)
    return rays, valid


# ---- Extrinsics and metadata ----

def extract_extrinsics(cam_obj):
    """
    Extract the extrinsics of a Blender camera object in the output camera convention.

    Blender camera local frame:
      X → right,  Y → up,  Z → towards the viewer (behind the camera)
      ∴ optical axis = -Z_blender

    Output camera frame:
      X → right,  Y → up,  Z → forward (optical axis)

    camera-to-world transform:
      the columns of C2W_blender are [X_cam | Y_cam | Z_cam | pos] in world coordinates;
      the conversion only flips the third column (Z):
        C2W_out = C2W_blender @ diag(1, 1, -1, 1)

    Returns:
        c2w:      [4, 4] float64, camera-to-world (output convention)
        position: [3]    float64, camera position in world coordinates
    """
    c2w_bl = np.array(cam_obj.matrix_world, dtype=np.float64)

    # Flip the Z axis: Blender Z-back → output Z-forward
    flip = np.diag([1.0, 1.0, -1.0, 1.0])
    c2w = c2w_bl @ flip

    position = c2w_bl[:3, 3].copy()
    return c2w, position


def build_metadata(cam_obj, camera_results):
    """
    Build the complete JSON metadata dictionary.

    Contents:
      - description of the output coordinate system
      - camera pose in the world (extrinsics)
      - intrinsics and file paths of each camera type
      - render settings
    """
    c2w, pos = extract_extrinsics(cam_obj)

    meta = {
        '_comment': 'Training-data metadata — generated automatically by render_unicol_dataset.py',
        'version': '1.0',
        'coordinate_system': {
            'name': 'unicol_z_forward',
            'X': 'right',
            'Y': 'up',
            'Z': 'forward (optical axis)',
            'handedness': 'right-handed',
            'blender_conversion': 'z_unicol = -z_blender',
        },
        'camera_pose': {
            'position_world': pos.tolist(),
            'camera_to_world_4x4': c2w.tolist(),
            'rotation_euler_xyz_rad': list(cam_obj.rotation_euler),
        },
        'cameras': camera_results,
        'render_settings': {
            'engine': 'CYCLES',
            'samples': RENDER_SAMPLES,
            'denoiser': USE_DENOISER,
            'gpu_backend': GPU_BACKEND,
        },
        'blender_version': '.'.join(str(v) for v in bpy.app.version),
    }
    return meta


# ---- Multi-view trajectory ----

def generate_orbit_trajectory(num_poses, radius, height, center, look_at):
    """
    Generate camera poses evenly spaced on a circular orbit in Blender world coordinates.

    Math:
      position of camera_i = center + (R·cos(θ_i), R·sin(θ_i), height)
      θ_i = 2π · i / N,  i = 0, 1, ..., N-1

      Camera orientation: towards look_at
      Up: world +Z (Blender convention)

    Returns:
        list of dict: [{
            'location': (x, y, z),
            'rotation_euler': (rx, ry, rz),  # Blender XYZ, radians
        }, ...]
    """
    poses = []
    cx, cy, cz = center
    lx, ly, lz = look_at

    for i in range(num_poses):
        theta = 2.0 * math.pi * i / num_poses

        # Camera position (world coordinates)
        px = cx + radius * math.cos(theta)
        py = cy + radius * math.sin(theta)
        pz = cz + height

        # Build the look-at rotation matrix
        # Blender camera: -Z = optical axis pointing forward, +Y = camera up
        cam_pos = Vector((px, py, pz))
        target  = Vector((lx, ly, lz))
        forward = (target - cam_pos).normalized()   # world-space forward = target - pos

        # Blender camera convention: -Z is the optical axis,
        # so the rotation must align camera -Z with forward,
        # i.e. camera +Z with -forward
        up_world = Vector((0.0, 0.0, 1.0))

        # Right-handed frame: right = forward × up
        right = forward.cross(up_world).normalized()
        if right.length < 1e-6:
            # forward ≈ ±Z: degenerate case
            right = Vector((1.0, 0.0, 0.0))
        # Recompute up to keep the frame orthogonal
        up = right.cross(forward).normalized()

        # Rotation matrix whose columns are the camera X, Y, Z axes in world coordinates
        # Blender camera: X=right, Y=up, Z=-forward (the optical axis is -Z)
        # Matrix() takes rows, so the columns are [right, up, -forward]
        rot_mat = Matrix((
            (right.x,    up.x,    -forward.x),
            (right.y,    up.y,    -forward.y),
            (right.z,    up.z,    -forward.z),
        ))  # no transpose needed: Matrix takes rows, so the columns line up

        euler = rot_mat.to_euler('XYZ')

        poses.append({
            'index': i,
            'location': (px, py, pz),
            'rotation_euler': (euler.x, euler.y, euler.z),
            'theta_deg': math.degrees(theta),
        })

        print(f"    Pose {i}: θ={math.degrees(theta):6.1f}°  "
              f"pos=({px:+.3f}, {py:+.3f}, {pz:+.3f})  "
              f"rot=({math.degrees(euler.x):+.1f}°, "
              f"{math.degrees(euler.y):+.1f}°, "
              f"{math.degrees(euler.z):+.1f}°)")

    return poses


def detect_scene_cameras(num_poses):
    """
    Detect the cameras placed in the scene by the procedural scene generator.

    The procedural scene places max(5, 5 * floor(area / 15 m^2)) cameras inside the room, each with its
    own position and orientation. These cameras are reused directly as render poses instead of a fixed orbit
    around the origin.

    Returns
    -------
    list | None : list of poses if enough cameras are found, otherwise None
    """
    import re
    cam_objs = []
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' and re.match(r'Camera_\d+', obj.name):
            cam_objs.append(obj)

    cam_objs.sort(key=lambda c: c.name)

    if len(cam_objs) < num_poses:
        return None

    poses = []
    for i, cam in enumerate(cam_objs[:num_poses]):
        euler = cam.rotation_euler
        loc = cam.location
        poses.append({
            'index': i,
            'location': (loc.x, loc.y, loc.z),
            'rotation_euler': (euler.x, euler.y, euler.z),
            'theta_deg': 0.0,
            'source': f'scene_camera:{cam.name}',
        })
        print(f"    Pose {i} (scene camera {cam.name}): "
              f"pos=({loc.x:+.3f}, {loc.y:+.3f}, {loc.z:+.3f})  "
              f"rot=({math.degrees(euler.x):+.1f}°, "
              f"{math.degrees(euler.y):+.1f}°, "
              f"{math.degrees(euler.z):+.1f}°)")

    return poses


def auto_detect_orbit_params():
    """
    Infer orbit parameters from the scene meshes (fallback).

    When the scene has no built-in cameras, the room center and a reasonable orbit radius are
    inferred from the bounding boxes of all MESH objects instead of a fixed (0,0,0).
    """
    all_x, all_y, all_z = [], [], []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for corner in obj.bound_box:
                wc = obj.matrix_world @ Vector(corner)
                all_x.append(wc.x)
                all_y.append(wc.y)
                all_z.append(wc.z)

    if not all_x:
        return ORBIT_CENTER, ORBIT_RADIUS, ORBIT_HEIGHT, LOOK_AT

    cx = (min(all_x) + max(all_x)) / 2
    cy = (min(all_y) + max(all_y)) / 2
    z_floor = min(all_z)
    z_ceil = max(all_z)

    # Orbit radius: a third of the smaller room dimension, at least 1m and at most 4m
    rx = (max(all_x) - min(all_x)) / 3
    ry = (max(all_y) - min(all_y)) / 3
    radius = max(1.0, min(min(rx, ry), 4.0))

    height = z_floor + (z_ceil - z_floor) * 0.4
    center = (cx, cy, 0.0)
    look_at = (cx, cy, height * 0.8)

    print(f"    Auto-detected orbit: center=({cx:.1f},{cy:.1f})  R={radius:.1f}m  H={height:.1f}m")
    return center, radius, height, look_at


# ---- Validation ----

def validate_rays(rays, valid, cam_name):
    """
    Run self-consistency checks on a generated ray field.

    Checks:
      1. Valid pixels have unit norm (normalized)
      2. The center pixel points along +Z (the optical axis, for every camera)
      3. No NaN / Inf
    """
    H, W = valid.shape
    errors = []

    # --- Check 1: normalization ---
    norms = np.linalg.norm(rays[valid], axis=-1)
    norm_err = np.max(np.abs(norms - 1.0))
    if norm_err > 1e-10:
        errors.append(f"normalization error too large: {norm_err:.2e}")

    # --- Check 2: center pixel → +Z ---
    ci, cj = H // 2, W // 2
    center = rays[ci, cj]
    # Expect center ≈ (0, 0, 1); allow for pixel quantization
    z_component = center[2]
    if z_component < 0.99:
        # With odd resolutions the center pixel can be slightly off: relax to 0.95
        if z_component < 0.95:
            errors.append(f"center ray deviates from the optical axis: z={z_component:.6f}")

    # --- Check 3: NaN / Inf ---
    if np.any(np.isnan(rays)) or np.any(np.isinf(rays)):
        errors.append("ray field contains NaN or Inf")

    # --- Print a report ---
    n_valid = valid.sum()
    print(f"    Validation [{cam_name}]:")
    print(f"      valid pixels: {n_valid}/{H*W} ({n_valid/(H*W)*100:.1f}%)")
    print(f"      normalization max|err|: {norm_err:.2e}")
    print(f"      center ray: ({center[0]:+.8f}, {center[1]:+.8f}, {center[2]:+.8f})")

    if errors:
        for e in errors:
            print(f"      {e}")
        return False
    else:
        print(f"      all checks passed")
        return True


def validate_cross_consistency(all_rays, cam_obj):
    """
    Cross-camera consistency check: rays of all camera types must agree on the optical axis.

    The center-pixel ray direction (camera frame) must be (0, 0, 1) for every camera type,
    which checks that the three projection formulas converge at the image center.
    """
    print("\n  Cross-camera consistency check:")
    centers = {}
    for name, (rays, valid) in all_rays.items():
        H, W = valid.shape
        centers[name] = rays[H // 2, W // 2]

    # Compare the center rays pairwise
    names = list(centers.keys())
    all_close = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = centers[names[i]], centers[names[j]]
            cos_angle = np.dot(a, b)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            ok = angle_deg < 1.0  # center-ray error < 1° (limited by resolution quantization)
            status = "ok" if ok else "mismatch"
            print(f"    {names[i]:8s} vs {names[j]:8s}: "
                  f"angle = {angle_deg:.4f}°  {status}")
            if not ok:
                all_close = False

    return all_close


# ---- Rendering and export ----

def render_rgb(scene, filepath):
    """Render an RGB image to filepath (deprecated; kept only as a debugging fallback)."""
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


# ---- sRGB conversion and Float16 NPZ packing ----

def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB to sRGB (IEC 61966-2-1 OETF).

    Formula:
      if C_linear ≤ 0.0031308:
          C_srgb = 12.92 × C_linear
      else:
          C_srgb = 1.055 × C_linear^(1 / 2.4) − 0.055

    Blender Cycles outputs linear RGB, so the conversion must happen here. Skipping it stores
    the Float16 RGB with the wrong tone curve and distorts the color distribution seen by the
    DINOv2 feature extractor.

    Input and output range: [0, 1]
    """
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0031308), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def render_and_grab_pixels(scene, W, H):
    """
    Render and grab linear RGB float pixels (headless-safe).

    The frame is rendered to a temporary OpenEXR file that is read back into NumPy
    (reliable, works with --background).

    No PNG is saved, which avoids:
      - sRGB quantization (0-255 truncation)
      - the zlib compression/decompression CPU bottleneck
      - loss of HDR information

    Returns
    -------
    rgb_srgb : np.ndarray [H, W, 3] float32, sRGB color space, range [0, 1]
    """
    import tempfile

    # Render to a temporary OpenEXR file and read it back.
    # In --background mode the Viewer Node is unreliable (it may keep an old resolution);
    # an OpenEXR file output is the most reliable option.

    # Save the original output settings
    orig_filepath = scene.render.filepath
    orig_format = scene.render.image_settings.file_format
    orig_color_mode = scene.render.image_settings.color_mode
    orig_color_depth = scene.render.image_settings.color_depth
    orig_exr_codec = scene.render.image_settings.exr_codec

    try:
        # Configure a temporary EXR output
        tmp_dir = tempfile.mkdtemp(prefix='unicol_render_')
        tmp_path = os.path.join(tmp_dir, 'render_tmp')
        scene.render.filepath = tmp_path
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '32'
        scene.render.image_settings.exr_codec = 'ZIP'

        # Render (writes the temporary EXR)
        bpy.ops.render.render(write_still=True)

        # Read the EXR into numpy
        exr_path = tmp_path + '.exr'
        if not os.path.exists(exr_path):
            # Blender may not add the extension
            candidates = [tmp_path, exr_path,
                          tmp_path + '0001.exr', tmp_path + '.exr']
            exr_path = None
            for c in candidates:
                if os.path.exists(c):
                    exr_path = c
                    break
            if exr_path is None:
                raise RuntimeError(
                    f"rendered EXR not found: {tmp_path}.*  (directory contents: "
                    f"{os.listdir(tmp_dir)})")

        # Load the EXR through Blender (keeps linear float values)
        img = bpy.data.images.load(exr_path)
        pixels_flat = np.array(img.pixels[:], dtype=np.float32)
        bpy.data.images.remove(img)

        n_pixels = W * H
        n_channels = len(pixels_flat) // n_pixels
        if n_channels < 3 or len(pixels_flat) < n_pixels * n_channels:
            raise RuntimeError(
                f"EXR pixel count mismatch: total={len(pixels_flat)}, "
                f"expected {n_pixels}×{n_channels}")

        pixels = pixels_flat[:n_pixels * n_channels].reshape(H, W, n_channels)
        rgb_linear = pixels[::-1, :, :3].copy()  # flip from bottom-left to top-left origin

    finally:
        # Restore the original settings
        scene.render.filepath = orig_filepath
        scene.render.image_settings.file_format = orig_format
        scene.render.image_settings.color_mode = orig_color_mode
        scene.render.image_settings.color_depth = orig_color_depth
        scene.render.image_settings.exr_codec = orig_exr_codec

        # Remove the temporary files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Color space conversion: linear RGB → sRGB, clamped to [0, 1]
    rgb_srgb = linear_to_srgb(rgb_linear)

    return rgb_srgb


def save_pack_npz(output_dir, tag, rgb, rays, depth, mask, debug_png=False):
    """
    Pack RGB + rays + depth + mask into a single Float16 .npz.

    Storage format:
      rgb:   [H, W, 3] float16, sRGB color space, range [0, 1]
      rays:  [H, W, 3] float16, unit vectors
      depth: [H, W]    float16, Euclidean radial distance (meters)
      mask:  [H, W]    uint8,   valid-pixel mask (0 or 1)

    Size of one pack (1920×1080):
      RGB:   1920×1080×3×2 ≈ 11.9 MB
      Rays:  1920×1080×3×2 ≈ 11.9 MB
      Depth: 1920×1080×2   ≈  4.0 MB
      Mask:  1920×1080×1   ≈  2.0 MB
      Total: ~30 MB (~15-20 MB after npz compression)

    Parameters
    ----------
    debug_png : bool
        If True, also export an 8-bit PNG from rgb (sRGB float [0,1]) for visual debugging.
        The PNG is not used for training; it is only for visual inspection.

    Returns: path of the saved file
    """
    pack_path = os.path.join(output_dir, f'{tag}_pack.npz')
    np.savez_compressed(
        pack_path,
        rgb=rgb.astype(np.float16),
        rays=rays.astype(np.float16),
        depth=depth.astype(np.float16) if depth is not None
              else np.zeros(rays.shape[:2], dtype=np.float16),
        mask=mask.astype(np.uint8),
    )
    fsize_mb = os.path.getsize(pack_path) / (1024.0 * 1024.0)
    print(f"    Pack: {os.path.basename(pack_path)} ({fsize_mb:.1f} MB)")

    # Debug PNG: converted directly from the sRGB RGB in the NPZ, not re-rendered by Blender
    if debug_png:
        try:
            # Save the PNG with the built-in Blender Image API (no PIL needed)
            png_path = os.path.join(output_dir, f'{tag}_debug.png')
            H_img, W_img = rgb.shape[:2]
            img = bpy.data.images.new(f'debug_{tag}', W_img, H_img, alpha=False)
            # Blender expects a flat RGBA array (bottom-left origin)
            rgba = np.ones((H_img, W_img, 4), dtype=np.float32)
            rgba[:, :, :3] = rgb[::-1]  # flip vertically back to a bottom-left origin
            img.pixels.foreach_set(rgba.ravel())
            img.filepath_raw = png_path
            img.file_format = 'PNG'
            img.save()
            bpy.data.images.remove(img)
            png_kb = os.path.getsize(png_path) / 1024.0
            print(f"     Debug PNG: {os.path.basename(png_path)} ({png_kb:.0f} KB)")
        except Exception as e:
            print(f"    Warning: failed to save the debug PNG: {e}")

    return pack_path


def extract_z_pass(scene, output_dir, cam_name):
    """
    Extract the Z-pass depth buffer from a Blender render.
    A File Output node writes a temporary EXR file, which is then read back into a NumPy array.

    Blender Z-pass semantics:
      - PERSP (pinhole): z-depth (distance along the optical axis)
      - PANO (fisheye/equirectangular): Euclidean radial distance

    Returns:
        z_buffer: [H, W] float32 ndarray, or None if the extraction fails
    """
    W = scene.render.resolution_x
    H = scene.render.resolution_y

    try:
        # Read the EXR file written by the File Output node,
        # which appends the frame number: depth_out0001.exr
        exr_path = os.path.join(output_dir, f'depth_out_{cam_name}0001.exr')
        if not os.path.exists(exr_path):
            # Try the name without a frame number
            exr_path = os.path.join(output_dir, f'depth_out_{cam_name}.exr')
        if not os.path.exists(exr_path):
            # Search the directory for a matching file
            import glob
            candidates = glob.glob(os.path.join(output_dir, f'depth_out_{cam_name}*.exr'))
            if candidates:
                exr_path = candidates[0]
            else:
                print(f"    Warning: depth EXR file not found: depth_out_{cam_name}*.exr")
                print(f"         directory contents: {os.listdir(output_dir)}")
                return None

        print(f"    Reading depth: {os.path.basename(exr_path)}")

        # Read the EXR with Blender itself
        img = bpy.data.images.load(exr_path)
        pixels = np.array(img.pixels[:], dtype=np.float32)
        iW, iH = img.size[0], img.size[1]
        channels = img.channels
        bpy.data.images.remove(img)

        print(f"    EXR: {iW}×{iH}, {channels}ch, "
              f"pixels={pixels.size}")

        if iW != W or iH != H:
            print(f"    Warning: EXR size ({iW}×{iH}) ≠ render resolution ({W}×{H})")
            return None

        # Parse the pixels (1, 3 or 4 channels)
        if channels >= 1:
            pixels = pixels.reshape(iH, iW, channels)
            z_buffer = pixels[:, :, 0]
        else:
            z_buffer = pixels.reshape(iH, iW)

        # Blender's pixel origin is bottom-left; flip to top-left
        z_buffer = z_buffer[::-1, :].copy()

        # Remove the temporary file
        os.remove(exr_path)

        return z_buffer
    except Exception as e:
        print(f"    Warning: Z-Pass extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _setup_z_file_output(scene, output_dir, cam_name):
    """
    Set up the compositor node tree:
      Render Layers → Image → Composite (RGB output)
      Render Layers → Depth → File Output (EXR depth file)

    The File Output node writes the Z pass to an EXR file during rendering, which is much
    more reliable than the Viewer Node (not affected by the image cache size).
    """
    scene.use_nodes = True
    tree = scene.node_tree

    # Remove existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Render Layers node
    rl_node = tree.nodes.new('CompositorNodeRLayers')
    rl_node.location = (0, 0)

    # Composite node (RGB output)
    comp_node = tree.nodes.new('CompositorNodeComposite')
    comp_node.location = (400, 0)
    tree.links.new(rl_node.outputs['Image'], comp_node.inputs['Image'])

    # File Output node (depth → EXR)
    fo_node = tree.nodes.new('CompositorNodeOutputFile')
    fo_node.location = (400, -200)
    fo_node.base_path = output_dir
    fo_node.format.file_format = 'OPEN_EXR'
    fo_node.format.color_depth = '32'
    fo_node.format.color_mode = 'BW'

    # Output file name prefix (Blender appends the frame number)
    fo_node.file_slots[0].path = f'depth_out_{cam_name}'

    # Connect Depth → File Output
    tree.links.new(rl_node.outputs['Depth'], fo_node.inputs[0])


def z_pass_to_radial_distance(z_buffer, rays, valid, cam_type):
    """
    Convert a Blender Z pass to Euclidean radial distance.

    Pinhole:
      Z-Pass = depth along the optical axis = r · cos(θ)
      ∴ r = Z-Pass / cos(θ) = Z-Pass / ray_z
      where ray_z is the Z component of the unit ray

    Fisheye / equirectangular:
      Z-Pass = Euclidean radial distance r (Blender's behavior for panoramic cameras)
      ∴ r = Z-Pass (no conversion needed)

    Special values:
      Z-Pass = 1e10 → sky / infinity, set to 0
      Z-Pass ≤ 0 → invalid, set to 0
    """
    H, W = z_buffer.shape
    depth = z_buffer.astype(np.float64).copy()

    # Sky pixels (Blender uses a very large value for infinity)
    sky_mask = (depth > 1e9) | (depth <= 0)

    if cam_type == 'pinhole':
        # Pinhole: r = z_depth / cos(θ) = z_depth / ray_z
        ray_z = rays[:, :, 2]  # Z component = cos(θ)
        # Avoid division by zero
        ray_z_safe = np.where(np.abs(ray_z) > 1e-8, ray_z, 1e-8)
        depth = depth / ray_z_safe
    # Fisheye / equirectangular: the Z pass is already the Euclidean distance, no conversion needed

    # Set sky and invalid pixels to 0
    depth[sky_mask] = 0.0
    depth[~valid] = 0.0

    return depth.astype(np.float32)


def save_rays_npy(rays, filepath):
    """Save a ray field as a compressed .npz (float16)."""
    # float16 is precise enough for unit vectors (error < 0.001°)
    fp = filepath.replace('.npy', '.npz')
    np.savez_compressed(fp, rays.astype(np.float16))
    return fp


def save_mask_npy(mask, filepath):
    """Save the validity mask as a compressed .npz (bool)."""
    fp = filepath.replace('.npy', '.npz')
    np.savez_compressed(fp, mask)
    return fp


# ---- Main ----

def main():
    """
    Main entry point: multi-view data generation pipeline.

    Steps:
      1. Initialize the render engine
      2. Build the camera trajectory (N poses)
      3. For each pose:
         a. Set the camera pose (position + orientation)
         b. For each base camera in turn:
            - configure the Blender camera intrinsics
            - render RGB + Z-Pass
            - compute the ground-truth ray field analytically
            - validate + save
         c. Cross-camera consistency check
      4. Export metadata.json (with the extrinsics of all poses)
    """
    t_start = time.time()

    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_render_engine()

    scene = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None:
        print("Error: the scene has no active camera. Add a camera to the scene.")
        return

    cam_data = cam_obj.data
    print(f"\n{'═'*64}")
    print(f"  First-generation scene renderer: 8 base cameras, direct Float16 NPZ output")
    print(f"{'═'*64}")
    print(f"  Camera: {cam_obj.name}")
    print(f"  Trajectory: {NUM_POSES} poses × {len(BASE_CAMERAS)} bases")
    print(f"  Output: {OUTPUT_DIR}")

    # Trajectory selection
    # UNICOL_FORCE_ORBIT=1 forces the outside-in orbit (the scene cameras are ignored)
    force_orbit = os.environ.get('UNICOL_FORCE_ORBIT', '0') == '1'
    print(f"\n  Detecting scene cameras... (force_orbit={force_orbit})")
    poses = None if force_orbit else detect_scene_cameras(NUM_POSES)
    if poses is not None:
        print(f"  Using {len(poses)} built-in scene cameras (multi-view inside the room)")
    else:
        if force_orbit:
            print(f"  Forced orbit mode (UNICOL_FORCE_ORBIT=1)")
        else:
            print(f"  Not enough scene cameras, falling back to an automatic orbit...")
        center, radius, height, look_at = auto_detect_orbit_params()
        poses = generate_orbit_trajectory(
            NUM_POSES, radius, height, center, look_at
        )

    # Pipeline of the 8 hard-coded base cameras
    def _make_setup_fn(cam_type, fov_deg):
        """Closure: build the parameterized setup function of one base."""
        if cam_type == 'pinhole':
            return lambda cd, w, h: setup_camera_pinhole(cd, w, h, fov_deg)
        elif cam_type == 'fisheye':
            return lambda cd, w, h: setup_camera_fisheye(cd, w, h, fov_deg)
        elif cam_type == 'erp':
            return lambda cd, w, h: setup_camera_erp(cd, w, h)
        else:
            raise ValueError(f"unknown camera type: {cam_type}")

    def _make_gen_fn(cam_type, fov_deg):
        """Closure: build the parameterized ray generator of one base."""
        if cam_type == 'pinhole':
            return lambda H, W: generate_rays_pinhole(H, W, fov_deg)
        elif cam_type == 'fisheye':
            return lambda H, W: generate_rays_fisheye(H, W, fov_deg)
        elif cam_type == 'erp':
            return lambda H, W: generate_rays_erp(H, W)
        else:
            raise ValueError(f"unknown camera type: {cam_type}")

    pipeline = []
    for base in BASE_CAMERAS:
        pipeline.append({
            'name':       base['name'],
            'cam_type':   base['type'],
            'eq_focal':   base['eq_focal_mm'],
            'fov_deg':    base['fov_deg'],
            'res':        base['res'],
            'setup':      _make_setup_fn(base['type'], base['fov_deg']),
            'gen':        _make_gen_fn(base['type'], base['fov_deg']),
        })
    print(f"\n  Pipeline: {len(pipeline)} bases:")
    for p in pipeline:
        eq = f"{p['eq_focal']:.0f}mm" if p['eq_focal'] else "—"
        print(f"    {p['name']:12s}  type={p['cam_type']:8s}  "
              f"eq={eq:>6s}  FOV={p['fov_deg']:.1f}°  "
              f"res={p['res'][0]}×{p['res'][1]}")

    # Global metadata collection (per pose)
    all_pose_meta = []

    # ---- Pose loop (outer) ----
    for pose in poses:
        pose_idx = pose['index']
        pose_tag = f"pose{pose_idx:02d}"

        print(f"\n{'━'*64}")
        print(f"  POSE {pose_idx}  θ={pose['theta_deg']:.1f}°  "
              f"pos=({pose['location'][0]:+.3f}, "
              f"{pose['location'][1]:+.3f}, "
              f"{pose['location'][2]:+.3f})")
        print(f"{'━'*64}")

        # Set the camera pose
        cam_obj.location = Vector(pose['location'])
        cam_obj.rotation_euler = Euler(pose['rotation_euler'], 'XYZ')

        # Random EV offset per pose (mild, [-0.5, 1.0], to avoid overexposure),
        # applied on top of the scene film_exposure set by the scene generator's auto-exposure
        import random as _rng_module
        base_exposure = scene.cycles.film_exposure
        ev_offset = _rng_module.uniform(-0.5, 1.0)
        scene.cycles.film_exposure = base_exposure * (2.0 ** ev_offset)
        print(f"    EV offset: {ev_offset:+.2f}  "
              f"film_exposure: {base_exposure:.3f} → {scene.cycles.film_exposure:.3f}")

        # Force a scene update (refreshes the internal Blender matrices)
        bpy.context.view_layer.update()

        # Base extrinsics before jitter (reference and base_matrix_world)
        base_matrix_world = cam_obj.matrix_world.copy()
        c2w_unicol, pos_world = extract_extrinsics(cam_obj)

        camera_results = {}
        all_rays = {}

        # ---- Camera loop (inner) ----
        for cam_cfg in pipeline:
            name = cam_cfg['name']
            W, H = cam_cfg['res']
            cam_type = cam_cfg['cam_type']
            fov_deg = cam_cfg['fov_deg']

            print(f"\n{'─'*64}")
            print(f"  [{name.upper()}] {pose_tag}  {W}×{H}  FOV={fov_deg:.1f}°")
            print(f"{'─'*64}")

            # Reset to the base pose, then apply FoV-aware pose jitter:
            # every inner iteration restores the base pose before applying the camera-specific jitter
            cam_obj.location = Vector(pose['location'])
            cam_obj.rotation_euler = Euler(pose['rotation_euler'], 'XYZ')
            bpy.context.view_layer.update()
            base_matrix_world = cam_obj.matrix_world.copy()

            if cam_type == 'erp':
                # Equirectangular: fully random 3D rotation to remove the "equator prior"
                cam_obj.rotation_euler = Euler(
                    (_rng_module.uniform(0, 2 * math.pi),
                     _rng_module.uniform(0, 2 * math.pi),
                     _rng_module.uniform(0, 2 * math.pi)),
                    'XYZ',
                )
                print(f"    ERP random rotation: "
                      f"({math.degrees(cam_obj.rotation_euler.x):.1f}°, "
                      f"{math.degrees(cam_obj.rotation_euler.y):.1f}°, "
                      f"{math.degrees(cam_obj.rotation_euler.z):.1f}°)")
            else:
                # Pinhole / fisheye: jitter proportional to the FoV (15% of FoV)
                max_jitter_rad = math.radians(fov_deg) * 0.15
                jx = _rng_module.uniform(-max_jitter_rad, max_jitter_rad)
                jy = _rng_module.uniform(-max_jitter_rad, max_jitter_rad)
                # Jitter applied as a local matrix (avoids gimbal lock)
                jitter_mat = Euler((jx, jy, 0.0), 'XYZ').to_matrix().to_4x4()
                cam_obj.matrix_world = base_matrix_world @ jitter_mat
                print(f"    Pose jitter: jx={math.degrees(jx):+.2f}°  "
                      f"jy={math.degrees(jy):+.2f}°  "
                      f"(max={math.degrees(max_jitter_rad):.1f}° = 15%×FOV)")

            bpy.context.view_layer.update()

            # Extrinsics of this camera (after jitter)
            cam_c2w_unicol, cam_pos_world = extract_extrinsics(cam_obj)

            # 1. Set the resolution
            scene.render.resolution_x = W
            scene.render.resolution_y = H
            scene.render.resolution_percentage = 100

            # 2. Configure the camera intrinsics
            cam_info = cam_cfg['setup'](cam_data, W, H)

            # 3. Set up the compositor nodes (Z-Pass → File Output EXR)
            file_tag = f"{name}_{pose_tag}"
            _setup_z_file_output(scene, OUTPUT_DIR, file_tag)

            # 4. Render and grab float pixels directly (no PNG)
            print(f"    Rendering and grabbing float pixels (no PNG)...")
            t_render = time.time()
            rgb_srgb = render_and_grab_pixels(scene, W, H)
            dt_render = time.time() - t_render
            print(f"    RGB (sRGB float16): {rgb_srgb.shape}  ({dt_render:.1f}s)")

            # 5. Ground-truth ray field (analytic, pure NumPy)
            print(f"    Computing GT ray field...")
            t_ray = time.time()
            rays, valid = cam_cfg['gen'](H, W)
            dt_ray = time.time() - t_ray
            print(f"    Ray generation: {dt_ray*1000:.1f}ms")

            # 6. Extract the Z pass → Euclidean radial distance
            print(f"    Extracting Z-Pass → radial distance...")
            z_buffer = extract_z_pass(scene, OUTPUT_DIR, file_tag)
            depth = None
            if z_buffer is not None:
                depth = z_pass_to_radial_distance(z_buffer, rays, valid, cam_type)
                valid_depth = depth[valid & (depth > 0)]
                if len(valid_depth) > 0:
                    print(f"    Depth stats: min={valid_depth.min():.3f}m, "
                          f"median={np.median(valid_depth):.3f}m, "
                          f"max={valid_depth.max():.3f}m, "
                          f"sky pixels={((depth == 0) & valid).sum()}")
                else:
                    print(f"    Warning: no valid depth values")
            else:
                print(f"    Warning: Z-Pass extraction failed, skipping depth")

            # 7. Validate the rays
            validate_rays(rays, valid, f"{name}_{pose_tag}")

            # 8. Pack RGB + rays + depth + mask into a Float16 .npz
            save_pack_npz(
                OUTPUT_DIR, file_tag,
                rgb=rgb_srgb,
                rays=rays.astype(np.float32),
                depth=depth,
                mask=valid,
                debug_png=DEBUG_PNG,
            )

            # 9. Collect metadata
            result = {
                'type': cam_type,
                'base_name': name,
                'eq_focal_mm': cam_cfg['eq_focal'],
                'resolution_wh': [W, H],
                'pack_file':  f'{file_tag}_pack.npz',
                'valid_pixels': int(valid.sum()),
                'total_pixels': H * W,
            }
            result.update(cam_info)

            if cam_type == 'pinhole':
                f_px = cam_info['f_px']
                result['intrinsics_K'] = [
                    [f_px, 0.0, W / 2.0],
                    [0.0,  f_px, H / 2.0],
                    [0.0,  0.0,  1.0],
                ]
            if cam_type == 'fisheye':
                result['projection_model'] = \
                    'equidistant: r = f_theta * theta'
            if cam_type == 'erp':
                result['projection_model'] = \
                    'equirectangular: lon ∈ [-π,π), lat ∈ (-π/2,π/2)'

            # Extrinsics of this camera (after jitter)
            result['camera_to_world_4x4'] = cam_c2w_unicol.tolist()

            camera_results[name] = result
            all_rays[name] = (rays, valid)

        # Cross-camera consistency check (per pose)
        # After jitter the cameras have different extrinsics, but the ray fields must stay consistent
        validate_cross_consistency(all_rays, cam_obj)

        # Collect the metadata of this pose
        pose_meta = {
            'pose_index': pose_idx,
            'theta_deg': pose['theta_deg'],
            'position_world': pos_world.tolist(),
            'camera_to_world_4x4': c2w_unicol.tolist(),  # base pose (no jitter)
            'rotation_euler_xyz_rad': list(pose['rotation_euler']),
            'ev_offset': ev_offset,
            'film_exposure': scene.cycles.film_exposure,
            'cameras': camera_results,  # each camera has its own camera_to_world_4x4
        }
        all_pose_meta.append(pose_meta)

    # ---- Export metadata ----
    metadata = {
        '_comment': 'Multi-view training-data metadata — generated '
                    'automatically by render_unicol_dataset.py',
        'version': '3.0',
        'format': 'float16_npz_pack',
        'coordinate_system': {
            'name': 'unicol_z_forward',
            'X': 'right',
            'Y': 'up',
            'Z': 'forward (optical axis)',
            'handedness': 'right-handed',
            'blender_conversion': 'z_unicol = -z_blender',
        },
        'trajectory': {
            'type': 'orbit',
            'num_poses': NUM_POSES,
            'radius_m': ORBIT_RADIUS,
            'height_m': ORBIT_HEIGHT,
            'center': list(ORBIT_CENTER),
            'look_at': list(LOOK_AT),
        },
        'poses': all_pose_meta,
        'render_settings': {
            'engine': 'CYCLES',
            'samples': RENDER_SAMPLES,
            'denoiser': USE_DENOISER,
            'gpu_backend': GPU_BACKEND,
        },
        'blender_version': '.'.join(str(v) for v in bpy.app.version),
    }

    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n  Metadata: {os.path.basename(meta_path)}")

    # ---- Summary ----
    dt_total = time.time() - t_start
    n_total = NUM_POSES * len(pipeline)
    print(f"\n{'═'*64}")
    print(f"  Multi-view data generation complete")
    print(f"     {NUM_POSES} poses × {len(pipeline)} cameras "
          f"= {n_total} renders  ({dt_total:.1f}s)")
    print(f"{'═'*64}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            fsize = os.path.getsize(fpath)
            unit = 'KB'
            size_val = fsize / 1024.0
            if size_val > 1024:
                unit = 'MB'
                size_val /= 1024.0
            print(f"    {fname:45s} {size_val:>8.1f} {unit}")
    print(f"{'═'*64}")
