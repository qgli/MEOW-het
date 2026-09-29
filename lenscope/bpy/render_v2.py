#!/usr/bin/env python3
"""Panorama renderer for second-generation scenes: RGB only (ground truth comes
from ray casting, never from the renderer).

Run inside Blender:
  blender -b <scene.blend> --python lenscope/bpy/render_v2.py -- \
      --poses <scene_out>/poses.json --out <scene_out>/renders \
      [--erp-w 3200] [--samples 32] [--pins 24,14]

Per pose: one equirectangular panorama (Cycles PANO equirectangular) + optional
pinhole probes. AGEN jsonl stage logs make failures diagnosable from the logs
alone. Depth/semantic passes are deliberately absent (light and geometry are
decoupled).

Pose convention (must match core/sampler.pose_to_R): world z-up; camera OpenCV
(x right, y down, z forward); yaw about +z from +x; pitch positive up.
Blender cameras look along -Z with +Y up -> R_blender = R_cv @ diag(1,-1,-1).
"""
import json
import os
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix


def log(event, **kw):
    print("AGEN " + json.dumps({"event": event, "ts": round(time.time(), 2), **kw}), flush=True)


def _save_hdr_sidecar(scene, png_filepath, erp_w, m):
    """Save the current Render Result as a wrap-blended scene-linear EXR half."""
    import math

    import numpy as np

    hdr_path = png_filepath.rsplit(".", 1)[0] + "_hdr.exr"
    ims = scene.render.image_settings
    prev = (ims.file_format, ims.color_depth, ims.color_mode)
    try:
        ims.file_format = "OPEN_EXR"
        ims.color_depth = "16"
        ims.color_mode = "RGB"
        ims.exr_codec = "ZIP"
        bpy.data.images["Render Result"].save_render(hdr_path, scene=scene)
    finally:
        ims.file_format, ims.color_depth, ims.color_mode = prev
    img = bpy.data.images.load(hdr_path)
    img.colorspace_settings.name = "Non-Color"
    wt, h = img.size
    a = np.empty(h * wt * 4, dtype=np.float32)
    img.pixels.foreach_get(a)          # numpy fast path (pixels[:] is ~100x slower)
    a = a.reshape(h, wt, 4)
    bpy.data.images.remove(img)
    out = a[:, m:m + erp_w].copy()
    ramp = 0.5 * (1.0 + np.cos(math.pi * (m - np.arange(m)) / m))
    ramp = ramp[None, :, None].astype(np.float32)
    out[:, :m] = ramp * out[:, :m] + (1 - ramp) * a[:, m + erp_w:m + erp_w + m]
    res = bpy.data.images.new("erp_hdr_out", width=erp_w, height=h,
                              alpha=False, float_buffer=True)
    res.colorspace_settings.name = "Non-Color"
    res.pixels.foreach_set(np.ascontiguousarray(out, dtype=np.float32).ravel())
    # Image.save() ignores scene image_settings and writes float32 (56MB probe);
    # save_render() honours them -> half + ZIP (~8-15MB).
    try:
        ims.file_format = "OPEN_EXR"
        ims.color_depth = "16"
        ims.color_mode = "RGB"
        ims.exr_codec = "ZIP"
        res.save_render(hdr_path, scene=scene)
    finally:
        ims.file_format, ims.color_depth, ims.color_mode = prev
    bpy.data.images.remove(res)
    log("hdr_sidecar", path=hdr_path.rsplit("/", 1)[-1], erp_w=erp_w)


def render_erp_wrapped(scene, cam_data, erp_w, filepath, margin_deg=2.0):
    """Equirectangular render with a true longitude wrap.

    Image-space stages that don't wrap (pixel filter, OIDN denoise) leave a
    visible discontinuity at the 0/360 seam (~x2.4 the interior median column
    jump in a panorama viewer), and a training image carries that seam into
    the model. Render [-180-d, +180+d] at the same angular pixel pitch (margin
    m px, d = 360*m/W), cosine-feather the duplicated overlap, crop to exact
    360. Ground truth is ray cast (never through the renderer) and unaffected.
    Cost: +2m/W pixels (~1.1% at W=2048)."""
    import math

    import numpy as np

    # margin: angular request with a pixel floor; the OIDN/filter edge
    # influence radius is ~8-16px regardless of resolution (m=6px leaves a
    # 1.31x seam residual; a 16px floor brings the seam to interior level)
    m = max(16, int(round(erp_w * margin_deg / 360.0)))
    lon = math.pi * (1.0 + 2.0 * m / erp_w)
    try:
        cam_data.longitude_min, cam_data.longitude_max = -lon, lon
        cam_data.latitude_min, cam_data.latitude_max = -math.pi / 2, math.pi / 2
    except Exception:                                    # legacy cycles path
        cam_data.cycles.longitude_min = -lon
        cam_data.cycles.longitude_max = lon
    scene.render.resolution_x = erp_w + 2 * m
    scene.render.resolution_y = erp_w // 2
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    # optional HDR output (GENESIS_HDR_SIDECAR=1) alongside the unchanged Standard
    # PNG: the same Render Result is saved once more as scene-linear EXR half
    # (EXR bypasses the view transform), wrap-blended in linear space
    # (physically the correct domain), cropped, and stored next to the PNG as
    # <stem>_hdr.exr. Packs and the loader consume the PNG unchanged; the EXR
    # is a sidecar for color-domain augmentation.
    if os.environ.get("GENESIS_HDR_SIDECAR", "0") == "1":
        _save_hdr_sidecar(scene, filepath, erp_w, m)

    # blend the wrap overlap in file space (Non-Color: no double EOTF trip)
    img = bpy.data.images.load(filepath)
    img.colorspace_settings.name = "Non-Color"
    wt, h = img.size
    a = np.empty(h * wt * 4, dtype=np.float32)
    img.pixels.foreach_get(a)          # numpy fast path
    a = a.reshape(h, wt, 4)
    bpy.data.images.remove(img)
    out = a[:, m:m + erp_w].copy()
    # one-sided transition: out[W-1] comes from wide col W+m-1; its duplicate
    # continuation (cols W+m..W+2m-1 = wrap copies of j=0..m-1) is spatially
    # contiguous with it in the render, so easing j=0..m-1 from the duplicate
    # back to the primary makes the seam exactly as smooth as the interior.
    # (A right-side blend would pull in the render's left-edge columns, which
    # have the worst filter/denoise quality, so it is deliberately absent.)
    ramp = 0.5 * (1.0 + np.cos(math.pi * (m - np.arange(m)) / m))  # 0 -> 1
    ramp = ramp[None, :, None].astype(np.float32)
    out[:, :m] = ramp * out[:, :m] + (1 - ramp) * a[:, m + erp_w:m + erp_w + m]
    res = bpy.data.images.new("erp_wrap_out", width=erp_w, height=h,
                              alpha=False, float_buffer=False)
    res.colorspace_settings.name = "Non-Color"
    res.pixels.foreach_set(np.ascontiguousarray(out, dtype=np.float32).ravel())
    res.filepath_raw = filepath
    res.file_format = "PNG"
    res.save()
    bpy.data.images.remove(res)
    # restore full-360 for any later plain render on this camera
    try:
        cam_data.longitude_min, cam_data.longitude_max = -math.pi, math.pi
    except Exception:
        cam_data.cycles.longitude_min = -math.pi
        cam_data.cycles.longitude_max = math.pi


def pose_to_R_cv(yaw, pitch):
    """Must stay formula-identical to core/sampler.pose_to_R (right = fwd x
    world_up; the opposite sign turns the panorama upside down)."""
    import math
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = (cy * cp, sy * cp, sp)
    right = (sy, -cy, 0.0)
    down = (fwd[1] * right[2] - fwd[2] * right[1],
            fwd[2] * right[0] - fwd[0] * right[2],
            fwd[0] * right[1] - fwd[1] * right[0])
    # columns = cam axes in world (cv): right, down, fwd
    return [[right[0], down[0], fwd[0]],
            [right[1], down[1], fwd[1]],
            [right[2], down[2], fwd[2]]]


def cam_matrix(pos, yaw, pitch):
    Rcv = pose_to_R_cv(yaw, pitch)
    # blender cam: x right, y up, z backward = cv @ diag(1,-1,-1)
    Rb = [[Rcv[r][0], -Rcv[r][1], -Rcv[r][2]] for r in range(3)]
    m = Matrix(((Rb[0][0], Rb[0][1], Rb[0][2], pos[0]),
                (Rb[1][0], Rb[1][1], Rb[1][2], pos[1]),
                (Rb[2][0], Rb[2][1], Rb[2][2], pos[2]),
                (0, 0, 0, 1)))
    return m


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    poses_file = Path(arg("--poses"))
    out = Path(arg("--out"))
    erp_w = int(arg("--erp-w", 3200))
    samples = int(arg("--samples", 32))
    pins = [int(x) for x in arg("--pins", "24,14").split(",") if x]
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(poses_file.read_text())
    poses = data["poses"]
    log("render_start", scene=data.get("scene"), n_poses=len(poses), erp_w=erp_w,
        samples=samples, blender=bpy.app.version_string)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    # CPU thread quota: parallel shards that each run all-core OIDN/BVH
    # oversubscribe the CPU (throughput can fall ~30x). GENESIS_RENDER_THREADS pins
    # Blender's pool; OIDN_NUM_THREADS (env, read by the OIDN lib itself) must
    # be set alongside by the launcher.
    _thr = int(os.environ.get("GENESIS_RENDER_THREADS", "0"))
    if _thr > 0:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = _thr
        log("thread_quota", threads=_thr)
    # Explicit color management (the Blender 4.x default, AgX, renders darker
    # and desaturated). Standard = linear render through the sRGB OETF, which
    # matches the pack rgb semantics (sRGB [0,1]); exposure is adjustable
    # (--exposure) and logged.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(arg("--exposure", 0.0))
    scene.view_settings.gamma = 1.0
    log("color_mgmt", view_transform="Standard", exposure=scene.view_settings.exposure)
    scene.cycles.use_denoising = True   # denoising affects RGB only (ground truth is ray cast)
    # Quality profile: Blender's default light-path settings trade physical
    # accuracy for noise (clamp_indirect=10 crushes indirect highlights,
    # filter_glossy=1.0 blurs sharp reflections, 12 total / 4 diffuse /
    # 4 glossy bounces starve multi-bounce GI in closed interiors).
    # GENESIS_RENDER_QUALITY=1 enables the physically fuller profile; unset keeps
    # Blender's defaults unchanged.
    if os.environ.get("GENESIS_RENDER_QUALITY", "0") == "1":
        c = scene.cycles
        c.max_bounces = 16
        c.diffuse_bounces = 6
        c.glossy_bounces = 6
        c.transmission_bounces = 12     # glass-heavy interiors
        c.transparent_max_bounces = 12
        c.sample_clamp_direct = 0.0
        # 10 (default) crushes GI highlights; 0 = fully physical (fireflies
        # handled by SPP+OIDN). Tunable via env.
        c.sample_clamp_indirect = float(os.environ.get("GENESIS_CLAMP_INDIRECT", "30"))
        c.blur_glossy = 0.5             # filter_glossy: sharper reflections, OIDN absorbs the noise
        c.caustics_reflective = True
        c.caustics_refractive = True
        log("quality_profile", max_bounces=16, diffuse=6, glossy=6,
            clamp_indirect=c.sample_clamp_indirect, filter_glossy=0.5, caustics=True)
    # view transform override (to compare AgX/Filmic film-like highlight
    # rolloff with Standard's hard sRGB clip, the classic "CG look" source).
    # The default keeps Standard, which the training data uses.
    _vt = os.environ.get("GENESIS_VIEW_TRANSFORM")
    if _vt:
        scene.view_settings.view_transform = _vt
        log("view_transform_override", view_transform=_vt)
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        prefs.preferences.compute_device_type = "OPTIX"
        for d in prefs.preferences.get_devices_for_type("OPTIX") or []:
            d.use = True
        scene.cycles.device = "GPU"
    # OIDN (CPU) denoiser: OptiX denoiser creation can fail on some GPU and
    # Blender combinations ("Failed to create OptiX denoiser" while OptiX ray
    # tracing works), and it fails at render time, losing whole scenes. The
    # extra CPU time of OIDN is an accepted trade-off.
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    cam_data = bpy.data.cameras.new("agen_cam")
    cam = bpy.data.objects.new("agen_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # (the longitude-wrap helper is defined at module level: render_erp_wrapped)

    t0 = time.time()
    for p in poses:
        # pose-level resume: poses with finished outputs are skipped, so a
        # restart does not re-render partially rendered scenes from pose 0
        _png = out / f"pose{p['id']:03d}_erp.png"
        _hdr = out / f"pose{p['id']:03d}_erp_hdr.exr"
        _need_hdr = os.environ.get("GENESIS_HDR_SIDECAR", "0") == "1"
        if _png.exists() and (not _need_hdr or _hdr.exists()):
            log("erp_skip_existing", pose=p["id"])
            continue
        cam.matrix_world = cam_matrix(p["pos"], p["yaw"], p["pitch"])
        # equirectangular panorama
        cam_data.type = "PANO"
        try:
            cam_data.panorama_type = "EQUIRECTANGULAR"          # camera data (Blender 4.x)
        except Exception:
            cam_data.cycles.panorama_type = "EQUIRECTANGULAR"   # older Blender: Cycles camera settings
        render_erp_wrapped(scene, cam_data, erp_w,
                           str(out / f"pose{p['id']:03d}_erp.png"))
        log("erp_done", pose=p["id"], secs=round(time.time() - t0, 1))
        # pinhole probes (sharpness anchors)
        for mm in pins:
            cam_data.type = "PERSP"
            cam_data.lens = mm
            cam_data.sensor_width = 36.0
            scene.render.resolution_x = 1920
            scene.render.resolution_y = 1080
            scene.render.filepath = str(out / f"pose{p['id']:03d}_pin{mm}.png")
            bpy.ops.render.render(write_still=True)
        log("pose_done", pose=p["id"], secs=round(time.time() - t0, 1))
    log("render_all_done", secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
