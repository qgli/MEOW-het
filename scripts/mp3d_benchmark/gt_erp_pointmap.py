#!/usr/bin/env python3
"""Build a per-ERP-pixel GT point map for a Matterport panorama, matching Wid3R's
GT representation (one 3D point per ERP pixel, ERP coverage, depth clipped to 10m).

Method (deterministic): take the exact world GT point cloud for one pano
(perspective depth back-projected with the OpenGL convention of gt_pointcloud.py),
express it in that pano's ERP frame (centred at the pano centre, world axes), and
z-buffer it into an HxW equirectangular grid. Each ERP pixel (lon,lat) keeps the
nearest world point along its viewing ray => a dense ERP point map P[h,w] in 3D
(world coords) with a validity mask.

This gives: (a) ERP coverage like Wid3R (not full perspective FoV), (b) max-depth
clipping (10m). ERP convention matches cube2equirect / BiFuse SphereGrid:
lon = atan2(x, y) from +Y, lat = asin(z) (world Z up after pano frame).

Performance:
- source stride default 4: the GT only lands on a 1024x512 ERP grid (~0.5M px);
  full resolution would back-project ~24M points per pano, while stride 4 still
  gives ~1.5M candidates (~3x the pixel count).
- z-buffer vectorized (sort far->near + fancy assign; last write wins = nearest).
- erp_point_maps_for_scan(): one .conf parse + one zip open + one namelist scan
  per scan, with an optional npz cache (precompute_erp_gt.py fills it; the
  evaluation then loads in milliseconds).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parse_conf import parse_conf_from_zip, scan_dir
from gt_pointcloud import build_name_index, scene_pointcloud

MAX_DEPTH = 10.0
GT_STRIDE = 4  # source perspective-depth subsample; see the performance notes above


def _zbuffer_erp(pts, centre, erp_w, erp_h, max_depth):
    """Bin world points into an ERP grid around `centre`, keep nearest per pixel."""
    d = pts - centre
    rng = np.linalg.norm(d, axis=1)
    keep = (rng > 1e-3) & (rng < max_depth)
    pts, d, rng = pts[keep], d[keep], rng[keep]
    dn = d / rng[:, None]
    # world directions -> (lon,lat): lon from +Y toward +X, lat from world +Z (up)
    lon = np.arctan2(dn[:, 0], dn[:, 1])
    lat = np.arcsin(np.clip(dn[:, 2], -1, 1))
    # ERP pixel (match cube2equirect mapping)
    u = ((lon + np.pi) / (2 * np.pi) * erp_w).astype(int) % erp_w
    v = ((np.pi / 2 - lat) / np.pi * erp_h).astype(int)
    v = np.clip(v, 0, erp_h - 1)
    P = np.zeros((erp_h, erp_w, 3), np.float64)
    valid = np.zeros((erp_h, erp_w), bool)
    # z-buffer: sort far -> near; NumPy fancy assignment writes in index order,
    # so for duplicate pixels the last (= nearest) point wins.
    order = np.argsort(-rng)
    vv, uu = v[order], u[order]
    P[vv, uu] = pts[order]
    valid[vv, uu] = True
    return P, valid


def erp_point_map(scan_path, uuid, erp_w=1024, erp_h=512, max_depth=MAX_DEPTH,
                  stride=GT_STRIDE, panos=None, zf=None, name_index=None):
    """Return (P[H,W,3] world points, valid[H,W] bool) for one pano's ERP GT.

    The pano's ERP frame is centred at its camera centre with world axes (the ERP
    az/el directions are world directions; a constant pano yaw is irrelevant because
    point-map eval is gauge-free via Umeyama). We bin each GT world point into the ERP
    pixel of its direction-from-centre and keep the nearest (z-buffer)."""
    if panos is None:
        panos = parse_conf_from_zip(scan_dir(scan_path))
    centre = panos[uuid].center
    pts = scene_pointcloud(scan_path, [uuid], stride=stride, max_panos=1,
                           panos=panos, zf=zf, name_index=name_index)
    return _zbuffer_erp(pts, centre, erp_w, erp_h, max_depth)


def _cache_path(cache_dir, scan_id, uuid, erp_w, erp_h, stride):
    return os.path.join(cache_dir, scan_id,
                        f"{uuid}_erp{erp_w}x{erp_h}_s{stride}.npz")


def erp_point_maps_for_scan(scan_path, uuids, erp_w=1024, erp_h=512,
                            max_depth=MAX_DEPTH, stride=GT_STRIDE,
                            cache_dir=None):
    """Yield (uuid, P, valid) for many panos of one scan efficiently.

    Opens the .conf and the depth zip once for the whole scan. When cache_dir
    is set, per-pano npz files are loaded when present and written when not
    (so the first run fills the cache; precompute_erp_gt.py does it offline)."""
    import zipfile

    scan_id = os.path.basename(os.path.normpath(scan_path))
    todo = list(uuids)
    if cache_dir:
        remaining = []
        for u in todo:
            cp = _cache_path(cache_dir, scan_id, u, erp_w, erp_h, stride)
            if os.path.exists(cp):
                z = np.load(cp)
                yield u, z["P"].astype(np.float64), z["valid"]
            else:
                remaining.append(u)
        todo = remaining
    if not todo:
        return

    panos = parse_conf_from_zip(scan_dir(scan_path))
    dzip = os.path.join(scan_path, "undistorted_depth_images.zip")
    with zipfile.ZipFile(dzip) as zf:
        name_index = build_name_index(zf)
        for u in todo:
            P, valid = erp_point_map(scan_path, u, erp_w, erp_h, max_depth,
                                     stride=stride, panos=panos, zf=zf,
                                     name_index=name_index)
            if cache_dir:
                cp = _cache_path(cache_dir, scan_id, u, erp_w, erp_h, stride)
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                np.savez_compressed(cp, P=P.astype(np.float32), valid=valid)
            yield u, P, valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: gt_erp_pointmap.py <scan_dir>  (a Matterport3D scan directory)")
    scan = sys.argv[1]
    panos = parse_conf_from_zip(scan_dir(scan))
    uuid = list(panos.keys())[0]
    P, valid = erp_point_map(scan, uuid)
    pts = P[valid]
    print(f"ERP point map: {valid.shape} coverage={valid.mean()*100:.1f}% "
          f"pts={valid.sum():,} extent(m)={np.round(pts.min(0),2)}..{np.round(pts.max(0),2)}")
    print(f"range from centre: max={np.linalg.norm(pts-panos[uuid].center,axis=1).max():.2f}m (should be <= 10)")
