#!/usr/bin/env python3
# Copyright (c) 2026 The MEOW Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Pairwise covisibility precomputation for first-generation scenes.

For each scene directory ``<root>/<scene_name>/`` containing ``metadata.json`` and
``<tag>_pack.npz`` files (``<tag>`` = ``<base_name>_pose<NN>``), computes an asymmetric
N x N covisibility matrix over the frames whose pack exists and writes it to
``<scene_dir>/covisibility/v0/``.

Design:

1. **Train-time crop.** Only the centre square of each pack is counted. It is the region
   the dataset loader keeps for a square target (``ProcThorUnicol._resize_and_centre_crop``
   in ``mapanything/datasets/procthor_unicol.py`` resizes the shorter side to the target
   size and centre-crops). Pixels outside that square are masked out before any
   reprojection.

2. **Ray-map based reprojection.** Packs store unit ray directions in the renderer camera
   frame plus the distance along each ray, so reprojection needs no analytic camera model
   (pinhole intrinsics, fisheye angle mapping, equirectangular atan2): source points are
   transformed into the target camera frame, normalised, and matched to the closest unit
   ray of the target's ray map by cosine similarity. The same lookup serves pinhole,
   fisheye and equirectangular panorama cameras; only points in front of the target
   camera (Z > 0) are matched.

3. **Asymmetric covisibility.** ``cov[i, j] = (# of i's points that project
   into j's train-visible region with consistent depth) / (# of i's
   train-visible valid points)``. Diagonal entries are 1.0. Storing both
   directions keeps field-of-view asymmetries (e.g. a panorama contains a
   pinhole view almost fully, so ``cov[pinhole, erp] ≈ 1`` while
   ``cov[erp, pinhole] ≪ 1``).

4. **Downsampling.** Each pack is sampled on a ``--downsample`` x
   ``--downsample`` grid (default 128) to bound GPU memory.

Outputs
-------
``<scene_dir>/covisibility/v0/covisibility.npy`` -- float32, shape (N, N).
``<scene_dir>/covisibility/v0/frame_meta.json``  -- the parameters (``n_frames``,
    ``downsample``, ``cos_thres``, ``depth_abs``, ``depth_rel``, ...) and ``frames``, a list
    of dicts with keys ``index``, ``tag``, ``base_name``, ``base_type``, ``pose_index``,
    ``n_valid_src_points``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch


# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------

# Renderer camera convention (X right, Y up, Z forward): the forward axis is +Z.
# Rays and poses are used in this convention without conversion.
DEFAULT_DOWNSAMPLE = 128
DEFAULT_OUT_SUBDIR = "covisibility/v0"
DEFAULT_OUT_NAME = "covisibility.npy"

# Cosine similarity threshold: how close (in angle) a reprojected ray must lie
# to the nearest ray in the target ray map to count as a hit. 0.998 is about
# 3.6 deg (cos(2.5 deg) ~= 0.99905), which tolerates the angular quantisation
# of the downsampled grid.
DEFAULT_COS_THRES = 0.998

# Depth consistency: absolute + relative error budget (in metres / fraction).
DEFAULT_DEPTH_ABS = 0.10  # 10 cm
DEFAULT_DEPTH_REL = 0.05  # 5 %


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _centre_square_mask(H: int, W: int) -> np.ndarray:
    """Boolean mask of the centre-square crop used at train time."""
    short = min(H, W)
    top = (H - short) // 2
    left = (W - short) // 2
    m = np.zeros((H, W), dtype=bool)
    m[top : top + short, left : left + short] = True
    return m


def _load_frame(pack_path: Path, downsample: int, device: torch.device):
    """Load one pack and sample it on a D x D grid (D = ``downsample``).

    Returns a dict with keys:
        - ``rays_cam_grid`` (D*D, 3) unit ray directions in the camera frame
          (the target's queryable ray map).
        - ``depth_grid`` (D*D,) distance along the ray in metres.
        - ``valid_grid`` (D*D,) bool: native mask, centre-square crop and
          positive depth (the target lookup mask and the source-point filter).
        - ``H``, ``W`` (pack size) and ``D``.
    Tensors are on ``device``.
    """
    with np.load(pack_path) as pack:
        rays = np.asarray(pack["rays"]).astype(np.float32)  # (H, W, 3) cam frame
        depth = np.asarray(pack["depth"]).astype(np.float32)  # (H, W)
        mask = np.asarray(pack["mask"]).astype(bool)  # (H, W)
    H, W, _ = rays.shape

    # Apply train-time centre-square crop mask in addition to native mask.
    crop_mask = _centre_square_mask(H, W)
    full_valid = mask & crop_mask & (depth > 0)

    # Downsample to (D, D) at evenly spaced pixel indices
    D = downsample
    stride_h = max(1, H // D)
    stride_w = max(1, W // D)
    # Use exactly D x D regardless of remainder
    ys = np.linspace(0, H - 1, D).astype(np.int64)
    xs = np.linspace(0, W - 1, D).astype(np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (D, D)

    rays_grid = rays[yy, xx]  # (D, D, 3)
    depth_grid = depth[yy, xx]  # (D, D)
    valid_grid = full_valid[yy, xx]  # (D, D) bool

    # Re-normalise rays (float16 storage round-off)
    n = np.linalg.norm(rays_grid, axis=-1, keepdims=True)
    rays_grid = rays_grid / np.clip(n, 1e-8, None)

    # Tensors on the target device
    rays_cam_t = torch.from_numpy(rays_grid.reshape(-1, 3)).to(device)  # (D*D, 3)
    depth_t = torch.from_numpy(depth_grid.reshape(-1)).to(device)
    valid_t = torch.from_numpy(valid_grid.reshape(-1)).to(device)

    return {
        "rays_cam_grid": rays_cam_t,
        "depth_grid": depth_t,
        "valid_grid": valid_t,
        "H": H,
        "W": W,
        "D": D,
    }


def _process_scene(
    scene_dir: Path,
    downsample: int,
    cos_thres: float,
    depth_abs: float,
    depth_rel: float,
    device: torch.device,
    overwrite: bool,
):
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  [skip] no metadata.json: {scene_dir.name}")
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    out_dir = scene_dir / DEFAULT_OUT_SUBDIR
    out_npy = out_dir / DEFAULT_OUT_NAME
    if out_npy.exists() and not overwrite:
        return "exists"

    # Build frame list = every frame that actually has its pack on disk.
    frames = []
    for fmeta in meta["frames"]:
        tag = fmeta["tag"]
        pack = scene_dir / f"{tag}_pack.npz"
        if not pack.is_file():
            continue
        frames.append(
            {
                "tag": tag,
                "base_name": fmeta["base_name"],
                "base_type": fmeta["base_type"],
                "pose_index": fmeta["pose_index"],
                "c2w_unicol": np.asarray(
                    fmeta["camera_to_world_unicol_4x4"], dtype=np.float32
                ),
                "pack_path": pack,
            }
        )

    N = len(frames)
    if N == 0:
        print(f"  [skip] no packs: {scene_dir.name}")
        return None

    # ----- Load all frames + reproject to world (per-frame source points) -----
    per_frame = []
    for f in frames:
        loaded = _load_frame(f["pack_path"], downsample, device)
        c2w = torch.from_numpy(f["c2w_unicol"]).to(device)  # (4, 4)
        R = c2w[:3, :3]  # cam -> world rotation
        t = c2w[:3, 3]  # cam centre in world

        rays_cam = loaded["rays_cam_grid"]  # (D*D, 3)
        depth = loaded["depth_grid"]  # (D*D,)
        valid = loaded["valid_grid"]  # (D*D,) bool

        # World points: P_world = t + R @ (rays_cam * depth)
        pts_cam_full = rays_cam * depth.unsqueeze(-1)  # (D*D, 3)
        pts_world_full = (R @ pts_cam_full.T).T + t  # (D*D, 3)

        # Source points = train-visible valid only
        src_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        src_pts_world = pts_world_full[src_idx]  # (V, 3)
        src_depth = depth[src_idx]  # (V,) ray-depth metres

        per_frame.append(
            {
                "rays_cam_grid": rays_cam,
                "valid_grid": valid,
                "depth_grid": depth,
                "c2w": c2w,
                "R": R,
                "t": t,
                "src_pts_world": src_pts_world,
                "src_depth": src_depth,
                "V": int(src_pts_world.shape[0]),
                "base_name": f["base_name"],
                "base_type": f["base_type"],
            }
        )

    # ----- Pairwise reprojection -----
    cov = torch.zeros((N, N), dtype=torch.float32)

    for i in range(N):
        fi = per_frame[i]
        V_i = fi["V"]
        cov[i, i] = 1.0
        if V_i == 0:
            continue
        P_world = fi["src_pts_world"]  # (V_i, 3) on device

        for j in range(N):
            if i == j:
                continue
            fj = per_frame[j]
            if fj["valid_grid"].sum() == 0:
                cov[i, j] = 0.0
                continue

            # Transform i's world points into j's cam frame:
            # P_cam_j = R_j^T @ (P_world - t_j)
            P_cam_j = (P_world - fj["t"]) @ fj["R"]  # (V_i, 3)

            # Forward-facing filter (Z > 0 in the camera frame)
            fwd_mask = P_cam_j[:, 2] > 0
            n_fwd = int(fwd_mask.sum())
            if n_fwd == 0:
                cov[i, j] = 0.0
                continue
            P_fwd = P_cam_j[fwd_mask]  # (n_fwd, 3)
            dist_fwd = P_fwd.norm(dim=-1).clamp_min(1e-8)  # (n_fwd,)
            d_fwd = P_fwd / dist_fwd.unsqueeze(-1)  # (n_fwd, 3) unit dirs

            # Cosine similarity vs j's ray map (D*D, 3). Chunk over query.
            ray_map_j = fj["rays_cam_grid"]  # (D*D, 3)
            depth_map_j = fj["depth_grid"]  # (D*D,)
            valid_map_j = fj["valid_grid"]  # (D*D,) bool

            chunk = 4096
            hits = torch.zeros(n_fwd, dtype=torch.bool, device=device)
            for s in range(0, n_fwd, chunk):
                e = min(s + chunk, n_fwd)
                cos = d_fwd[s:e] @ ray_map_j.T  # (chunk, D*D)
                max_cos, max_idx = cos.max(dim=-1)
                # Lookup target depth + validity
                tgt_depth = depth_map_j[max_idx]
                tgt_valid = valid_map_j[max_idx]
                ray_ok = max_cos > cos_thres
                # Depth consistency: |dist_fwd - tgt_depth| <= abs + rel*tgt_depth
                err = (dist_fwd[s:e] - tgt_depth).abs()
                tol = depth_abs + depth_rel * tgt_depth
                depth_ok = err <= tol
                hits[s:e] = ray_ok & tgt_valid & depth_ok

            cov[i, j] = float(hits.sum()) / float(V_i)

    # ----- Write outputs -----
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, cov.cpu().numpy().astype(np.float32))

    meta_out = [
        {
            "index": k,
            "tag": frames[k]["tag"],
            "base_name": frames[k]["base_name"],
            "base_type": frames[k]["base_type"],
            "pose_index": frames[k]["pose_index"],
            "n_valid_src_points": per_frame[k]["V"],
        }
        for k in range(N)
    ]
    with open(out_dir / "frame_meta.json", "w") as f:
        json.dump(
            {
                "n_frames": N,
                "downsample": downsample,
                "cos_thres": cos_thres,
                "depth_abs": depth_abs,
                "depth_rel": depth_rel,
                "denominator": "n_valid_source_points",
                "diagonal": "1.0",
                "frame_format": "asymmetric: cov[i,j] = fraction of i's "
                "train-visible points that hit j's train-visible region "
                "with consistent depth",
                "frames": meta_out,
            },
            f,
            indent=2,
        )
    return "done"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        help="Directory containing per-scene folders",
    )
    p.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Specific scene folder names. Default: all proc_scene_* dirs.",
    )
    p.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    p.add_argument("--cos-thres", type=float, default=DEFAULT_COS_THRES)
    p.add_argument("--depth-abs", type=float, default=DEFAULT_DEPTH_ABS)
    p.add_argument("--depth-rel", type=float, default=DEFAULT_DEPTH_REL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--limit", type=int, default=0, help="Process at most N scenes (debug)."
    )
    args = p.parse_args()

    device = torch.device(args.device)
    root = Path(args.root)

    if args.scenes:
        scene_names = list(args.scenes)
    else:
        scene_names = sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and d.name.startswith("proc_scene_")
        )

    if args.limit:
        scene_names = scene_names[: args.limit]

    print(f"[covis] scenes: {len(scene_names)}  device: {device}  D={args.downsample}")

    t0 = time.time()
    done = skipped = errored = 0
    for i, name in enumerate(scene_names):
        scene_dir = root / name
        t_s = time.time()
        try:
            status = _process_scene(
                scene_dir,
                args.downsample,
                args.cos_thres,
                args.depth_abs,
                args.depth_rel,
                device,
                args.overwrite,
            )
            if status == "done":
                done += 1
            elif status == "exists":
                skipped += 1
            else:
                continue
        except Exception as e:  # pragma: no cover
            print(f"  [error] {name}: {type(e).__name__}: {e}")
            errored += 1
            continue
        dt = time.time() - t_s
        print(
            f"  [{i + 1}/{len(scene_names)}] {name}: {status} ({dt:.1f}s)"
        )

    total = time.time() - t0
    print(
        f"[covis] finished. done={done} skipped={skipped} errored={errored} "
        f"total_time={total:.1f}s"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
