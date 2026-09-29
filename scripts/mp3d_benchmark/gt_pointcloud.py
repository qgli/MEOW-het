#!/usr/bin/env python3
"""Build exact ground-truth scene point clouds for Matterport3D from raw
undistorted perspective depth + .conf extrinsics (no skybox, no yaw recovery).

Each perspective depth map is back-projected through its intrinsics K and
camera-to-world c2w (all from undistorted_camera_parameters/*.conf). The GT of the
MP3D point-map evaluation (eval_mp3d_panoramas.py) is built from these clouds by
gt_erp_pointmap.py; predictions are aligned to it (pointmap_eval.py), so the GT
only needs to be a geometrically exact world point cloud.

Depth: undistorted_depth_images/*.png, 16-bit, 0.25 mm units => metres = v/4000
(the up-tilted camera 2 measures a median 1.32 m to the ceiling, consistent with a
1.58 m rig height and a ~2.9 m ceiling). Poses are in metres, so depth must be
metric too.
"""
from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np

from parse_conf import parse_conf_from_zip, scan_dir

DEPTH_SCALE = 1.0 / 4000.0  # 0.25 mm units -> metres


def build_name_index(zf: zipfile.ZipFile) -> dict:
    """basename -> full archive name (one namelist() pass per zip)."""
    return {os.path.basename(n): n for n in zf.namelist()}


def _read_png_u16(zf: zipfile.ZipFile, endswith: str, name_index: dict | None = None):
    import cv2
    if name_index is not None:
        full = name_index.get(endswith)
        if full is None:
            return None
        data = zf.read(full)
    else:
        cand = [n for n in zf.namelist() if n.endswith(endswith)]
        if not cand:
            return None
        data = zf.read(cand[0])
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def backproject_view(depth_m, K, c2w, stride=4, dmin=0.1, dmax=20.0):
    """Perspective depth (metres) -> world XYZ points. stride subsamples pixels."""
    H, W = depth_m.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[ys, xs]
    m = (z > dmin) & (z < dmax)
    xs, ys, z = xs[m], ys[m], z[m]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # Pixel ray in camera frame. Matterport .conf c2w uses the OpenGL/Habitat
    # convention (+X right, +Y up, -Z forward), not OpenCV: with OpenGL the
    # multi-view overlap consistency improves from 8.83 cm to 1.28 cm (median NN
    # distance) and the fraction below 2 cm from 25% to 57%. Depth is planar
    # (z along -Z); the Euclidean interpretation makes the consistency worse.
    Xc = (xs - cx) / fx * z
    Yc = -(ys - cy) / fy * z   # OpenGL: image +y is down -> world +Y up
    Zc = -z                    # OpenGL: camera looks along -Z
    pts_c = np.stack([Xc, Yc, Zc, np.ones_like(Zc)], 0)  # 4xN
    pts_w = (c2w @ pts_c)[:3].T  # Nx3
    return pts_w


def scene_pointcloud(scan_path, uuids=None, stride=4, max_panos=None,
                     panos=None, zf=None, name_index=None):
    """Build a world point cloud from the given panorama uuids (or all).

    panos / zf / name_index may be passed by callers that iterate many panos of
    one scan, so the .conf parse, the zip open and the namelist scan all happen
    once per scan instead of once per pano/view."""
    import cv2  # noqa
    if panos is None:
        panos = parse_conf_from_zip(scan_dir(scan_path))
    if uuids is None:
        uuids = list(panos.keys())
    if max_panos:
        uuids = uuids[:max_panos]

    own_zip = zf is None
    if own_zip:
        zf = zipfile.ZipFile(os.path.join(scan_path, "undistorted_depth_images.zip"))
    if name_index is None:
        name_index = build_name_index(zf)
    allpts = []
    try:
        for u in uuids:
            if u not in panos:
                continue
            for v in panos[u].views:
                dname = os.path.basename(v.depth)
                d = _read_png_u16(zf, dname, name_index)
                if d is None:
                    continue
                pts = backproject_view(d.astype(np.float64) * DEPTH_SCALE, v.K, v.c2w, stride)
                allpts.append(pts)
    finally:
        if own_zip:
            zf.close()
    return np.concatenate(allpts, 0) if allpts else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True,
                    help="Matterport3D scan directory (contains undistorted_camera_parameters.zip "
                         "and undistorted_depth_images.zip)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-panos", type=int, default=3)
    ap.add_argument("--out", default="scripts/mp3d_benchmark/gt_out/scene_gt.ply")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    panos = parse_conf_from_zip(scan_dir(args.scan))
    uuids = list(panos.keys())[: args.max_panos]
    centers = np.array([panos[u].center for u in uuids])
    pts = scene_pointcloud(args.scan, uuids, args.stride)
    print(f"[gt] panos={len(uuids)}  points={len(pts):,}")
    print(f"[gt] cloud extent (m): {np.round(pts.min(0),2)} .. {np.round(pts.max(0),2)}")
    print(f"[gt] cloud span={np.round(pts.max(0)-pts.min(0),2)}  diag={np.linalg.norm(pts.max(0)-pts.min(0)):.2f} m")
    # cross-panorama consistency: camera centres should lie inside the cloud bbox
    print(f"[gt] pano centres:\n{np.round(centers,2)}")
    inside = np.all((centers >= pts.min(0) - 0.5) & (centers <= pts.max(0) + 0.5), 1)
    print(f"[gt] centres inside cloud bbox: {inside.sum()}/{len(centers)} "
          f"(expect all; confirms depth scale 1/4000 consistent with metric poses)")
    # write a simple PLY for visual check
    sub = pts[:: max(1, len(pts) // 200000)]  # subsample for viewing
    with open(args.out, "w") as f:
        # header vertex count must equal the number of points actually written
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(sub)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in sub:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    print(f"[gt] wrote {args.out} ({len(sub):,} pts, subsampled from {len(pts):,})")


if __name__ == "__main__":
    main()
