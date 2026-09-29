#!/usr/bin/env python3
"""Heterogeneous camera synthesis from an ERP panorama (CAM3R-style benchmark).

CAM3R (arXiv 2603.22631, Sec. D) builds its heterogeneous 2D3DS/360Loc benchmark by
synthesising perspective and fisheye views from the original equirectangular
panoramas, then mixes camera models within a multi-view group and measures relative
pose (RRA/RTA/mAA/ATE). This module reproduces that synthesis so that our models,
MapAnything, VGGT and pi3 can be compared with CAM3R's published numbers.

Each synthesized view is a crop of the panorama looking along (yaw, pitch):
  * perspective : pinhole, square FoV, returns intrinsics K.
  * fisheye     : equidistant (r = f*theta) model, the base term of the Kannala-Brandt
                  model used by CAM3R; returns None for K (it is not a pinhole) and is
                  black outside the image circle.

Both also return R_pano_from_cam (3x3): the rotation mapping the synthesized camera's
local frame (+X right, +Y up, +Z forward) into the panorama's own camera frame, both in
this module's +Y-up convention. For a panorama pose in the OpenCV convention (+Y down),
the ground-truth c2w of the synthesized view is:
        synth_c2w = pano_c2w @ blkdiag(FLIP_Y @ R_pano_from_cam @ FLIP_Y, 1),  FLIP_Y = diag(1,-1,1)
(the panorama centre is shared; only the orientation changes), which gives the
heterogeneous relative-pose GT.

ERP convention (as in the 2D3DS panoramas):
  phi   = atan2(rx, rz)   azimuth  in (-pi, pi]   (left edge -pi, right +pi)
  theta = asin(ry)        elevation in [-pi/2,pi/2] (+Y up, top +pi/2)
"""
from __future__ import annotations

import math

import numpy as np


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def _pano_from_cam(yaw, pitch):
    """R s.t. ray_pano = R @ ray_cam. Pitch about X then yaw about Y (erp convention)."""
    return _rot_y(yaw) @ _rot_x(pitch)


def _sample_erp(erp, rx, ry, rz):
    """Bilinear-sample an ERP image given pano-frame ray dirs (H,W arrays). Horizontal wrap."""
    H, W = erp.shape[:2]
    phi = np.arctan2(rx, rz)
    theta = np.arcsin(np.clip(ry, -1.0, 1.0))
    u = (phi / (2 * math.pi) + 0.5) * W
    v = (0.5 - theta / math.pi) * H
    u = np.clip(u, 0, W - 1 - 1e-5)
    v = np.clip(v, 0, H - 1 - 1e-5)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = (u0 + 1) % W
    v1 = np.minimum(v0 + 1, H - 1)
    du = (u - u0).astype(np.float32)[..., None]
    dv = (v - v0).astype(np.float32)[..., None]
    c00 = erp[v0, u0].astype(np.float32)
    c10 = erp[v0, u1].astype(np.float32)
    c01 = erp[v1, u0].astype(np.float32)
    c11 = erp[v1, u1].astype(np.float32)
    out = (c00 * (1 - du) * (1 - dv) + c10 * du * (1 - dv)
           + c01 * (1 - du) * dv + c11 * du * dv)
    return out


def perspective_view(erp, yaw, pitch, fov_deg=90.0, size=512):
    """ERP -> pinhole crop. Returns (img uint8 HxWx3, R_pano_from_cam 3x3, K 3x3)."""
    fov = math.radians(fov_deg)
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin[::-1])           # +x right, +y up
    f = 1.0 / math.tan(fov / 2.0)                  # focal in normalised units
    rx, ry, rz = xx, yy, np.full_like(xx, f)
    n = np.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / n, ry / n, rz / n
    R = _pano_from_cam(yaw, pitch)
    px = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] * rz
    py = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] * rz
    pz = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] * rz
    img = np.clip(_sample_erp(erp, px, py, pz), 0, 255).astype(np.uint8)
    fpx = f * (size / 2.0)                          # focal in pixels
    K = np.array([[fpx, 0, size / 2.0], [0, fpx, size / 2.0], [0, 0, 1]], np.float64)
    return img, R, K


def fisheye_view(erp, yaw, pitch, fov_deg=180.0, size=512):
    """ERP -> equidistant fisheye crop. Returns (img uint8, R_pano_from_cam, None).

    Equidistant: a pixel at normalised radius rho in [0,1] maps to incidence angle
    theta = rho * (fov/2). Outside the unit circle is black (circular fisheye)."""
    half = math.radians(fov_deg) / 2.0
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin[::-1])
    rho = np.sqrt(xx * xx + yy * yy)
    valid = rho <= 1.0
    theta = rho * half                              # incidence from optical axis
    safe = np.where(rho < 1e-8, 1.0, rho)
    st = np.sin(theta)
    rx = st * (xx / safe)
    ry = st * (yy / safe)
    rz = np.cos(theta)
    R = _pano_from_cam(yaw, pitch)
    px = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] * rz
    py = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] * rz
    pz = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] * rz
    img = np.clip(_sample_erp(erp, px, py, pz), 0, 255).astype(np.uint8)
    img[~valid] = 0
    return img, R, None


if __name__ == "__main__":
    import argparse
    import os
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--erp", default="", help="path to an ERP RGB image (2:1)")
    ap.add_argument("--out", default="scripts/mp3d_benchmark/het_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.erp and os.path.exists(args.erp):
        erp = cv2.imread(args.erp)
    else:
        # synthetic ERP test pattern: colored lat/long grid to verify geometry
        H, W = 512, 1024
        erp = np.zeros((H, W, 3), np.uint8)
        for j in range(0, W, 64):
            erp[:, j:j + 2] = (0, 255, 255)
        for i in range(0, H, 64):
            erp[i:i + 2, :] = (255, 0, 255)
        erp[:, :2] = (0, 0, 255)          # phi=-pi seam red
        print("[selftest] using synthetic grid ERP")

    tiles = []
    for kind, yaw in [("ERP", 0)] + [(k, y) for k in ("persp", "fish")
                                     for y in (0, math.pi / 2)]:
        if kind == "ERP":
            t = cv2.resize(erp, (512, 256))
            lbl = "ERP"
        elif kind == "persp":
            img, R, K = perspective_view(erp, yaw, 0.0, 90.0, 256)
            t = img
            lbl = f"persp yaw={int(math.degrees(yaw))} detR={np.linalg.det(R):.2f}"
        else:
            img, R, _ = fisheye_view(erp, yaw, 0.0, 180.0, 256)
            t = img
            lbl = f"fish yaw={int(math.degrees(yaw))} detR={np.linalg.det(R):.2f}"
        t = cv2.copyMakeBorder(t, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(t, lbl, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        h = 280
        t = cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
        tiles.append(t)
    sheet = np.hstack([cv2.resize(t, (300, 280)) for t in tiles])
    p = os.path.join(args.out, "het_synth_selftest.png")
    cv2.imwrite(p, sheet)
    print(f"[selftest] det(R) should be +1 for all (proper rotation, no mirror).")
    print(f"[selftest] saved {p}")
