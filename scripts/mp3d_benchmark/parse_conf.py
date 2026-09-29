#!/usr/bin/env python3
"""Parse Matterport3D undistorted_camera_parameters/*.conf into per-panorama
camera intrinsics + camera-to-world extrinsics, and derive each panorama's
centre (the mean of its camera centres).

.conf format (one block per camera 0, 1 and 2, each followed by its 6 yaw views):
  intrinsics_matrix fx 0 cx  0 fy cy  0 0 1
  scan <depth>.png <color>.jpg  <16 floats row-major camera-to-world 4x4>
  ... (6 yaw rows) ...

Image naming: {uuid}_i{cam}_{yaw}.jpg  (cam in 0..2, yaw in 0..5) => 18 views
per panorama. All 18 share (approximately) the same optical centre = the
panorama centre.

ERP convention (matches cube2equirect / PanoContext im2Sphere):
  ERP local frame: +Y = longitude 0 (front), +X = longitude +90, +Z = up (pole).
  No panorama rotation is estimated here; the ERP GT point maps
  (gt_erp_pointmap.py) use world-aligned axes centred at the panorama centre, and
  a global constant rotation is absorbed by the Sim3 alignment of the evaluation.
"""
from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np


@dataclass
class View:
    cam: int
    yaw: int
    color: str
    depth: str
    K: np.ndarray          # 3x3 intrinsics
    c2w: np.ndarray        # 4x4 camera-to-world


@dataclass
class Panorama:
    uuid: str
    views: list = field(default_factory=list)

    @property
    def center(self):
        """Panorama centre = mean of the 18 camera centres (world)."""
        return np.mean([v.c2w[:3, 3] for v in self.views], axis=0)

    def level_views(self):
        return sorted([v for v in self.views if v.cam == 1], key=lambda v: v.yaw)


_NAME_RE = re.compile(r"([0-9a-f]+)_i(\d)_(\d)\.jpg")


def parse_conf_text(text: str) -> dict:
    """Return {uuid: Panorama}. Robust to blank lines and repeated intrinsics."""
    panos: dict[str, Panorama] = {}
    K = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "intrinsics_matrix":
            vals = list(map(float, parts[1:10]))
            K = np.array(vals, dtype=np.float64).reshape(3, 3)
        elif parts[0] == "scan":
            depth, color = parts[1], parts[2]
            m = _NAME_RE.match(color)
            if m is None:
                continue
            uuid, cam, yaw = m.group(1), int(m.group(2)), int(m.group(3))
            c2w = np.array(list(map(float, parts[3:19])), dtype=np.float64).reshape(4, 4)
            panos.setdefault(uuid, Panorama(uuid)).views.append(
                View(cam, yaw, color, depth, K.copy(), c2w)
            )
    return panos


def parse_conf_from_zip(zip_path: str) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".conf"))
        text = z.read(name).decode("utf-8", errors="replace")
    return parse_conf_text(text)


def scan_dir(scan_dir: str) -> str:
    """Path to a scan's undistorted_camera_parameters zip."""
    return os.path.join(scan_dir, "undistorted_camera_parameters.zip")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True,
                    help="Matterport3D scan directory (contains undistorted_camera_parameters.zip)")
    args = ap.parse_args()

    panos = parse_conf_from_zip(scan_dir(args.scan))
    print(f"[parse] scan={os.path.basename(args.scan)}  n_panoramas={len(panos)}")
    nv = [len(p.views) for p in panos.values()]
    print(f"[parse] views/pano: min={min(nv)} max={max(nv)} (expect 18)")
    u0 = next(iter(panos))
    p0 = panos[u0]
    print(f"[parse] sample pano {u0}: centre={np.round(p0.center, 3)}")
    lv = p0.level_views()
    print(f"[parse] level(cam=1) yaws={[v.yaw for v in lv]}")
    # camera +Z axis in world and centre of each level (cam=1) view; the .conf poses
    # follow the OpenGL convention, so the camera looks along -Z
    for v in lv:
        fwd_negz = -v.c2w[:3, 2]
        fwd_posz = v.c2w[:3, 2]
        print(f"   yaw{v.yaw}: +Zworld={np.round(fwd_posz,3)}  centre={np.round(v.c2w[:3,3],3)}")
