"""Per-pose omnidirectional ground truth by ray casting.

Ground truth never touches the renderer. Depth = first solid hit (glass is
penetrated and flagged); normal/sem/inst/flags = hit-face attributes. The
equirectangular pixel-ray convention must match the renderer's panorama camera
(cross-checked against the rendered panoramic depth); it also matches the
camera resampler convention (image centre = az 0 = camera forward).
"""
from __future__ import annotations

import numpy as np

from .mesh import TriSoup, raycast_solid
from .sampler import pose_to_R


def erp_dirs(W, H):
    """Unit directions per pixel in the camera frame (x right, y down, z fwd);
    az in [-pi,pi) over width (centre col = 0), el in [-pi/2,pi/2], row0 = +el (up)."""
    az = ((np.arange(W) + 0.5) / W - 0.5) * 2 * np.pi
    el = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    azg, elg = np.meshgrid(az, el)
    d = np.stack([np.cos(elg) * np.sin(azg), -np.sin(elg), np.cos(elg) * np.cos(azg)], -1)
    return d.reshape(-1, 3)


def cast_pose(soup: TriSoup, pos, yaw, pitch, W=512, H=256, far=50.0, use_embree=False):
    """Returns dict of (H,W) arrays: depth f32 (0 = miss/far), normal f32 (H,W,3),
    sem u8, inst u16, flags u8 (incl. glass-crossed accumulation)."""
    R = pose_to_R(yaw, pitch)            # cam->world (world z-up)
    d_cam = erp_dirs(W, H)
    d_world = d_cam @ R.T
    r = raycast_solid(soup, np.asarray(pos, np.float64), d_world, use_embree=use_embree)
    t = r["t"]
    face = r["face"]
    hit = np.isfinite(t) & (t <= far)
    depth = np.where(hit, t, 0.0).astype(np.float32).reshape(H, W)
    sem = np.zeros(len(t), np.uint8)
    inst = np.zeros(len(t), np.uint16)
    flags = r["flags_accum"].copy()
    f_ok = face >= 0
    idx = np.where(hit & f_ok)[0]
    sem[idx] = soup.sem[face[idx]]
    inst[idx] = soup.inst[face[idx]]
    flags[idx] |= soup.flags[face[idx]]
    # geometric normals of hit faces
    tri = soup.tri
    nrm_f = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm_f /= np.maximum(np.linalg.norm(nrm_f, axis=1, keepdims=True), 1e-12)
    normal = np.zeros((len(t), 3), np.float32)
    normal[idx] = nrm_f[face[idx]]
    # orient towards the camera
    flip = (normal[idx] * d_world[idx]).sum(1) > 0
    normal[idx[flip]] *= -1
    return {"depth": depth, "normal": normal.reshape(H, W, 3),
            "sem": sem.reshape(H, W), "inst": inst.reshape(H, W),
            "flags": flags.reshape(H, W).astype(np.uint8),
            "mask": hit.reshape(H, W)}
