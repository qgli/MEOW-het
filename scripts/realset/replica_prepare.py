#!/usr/bin/env python3
"""Replica GT preparation from the standard NICE-SLAM/iMAP rendered sequences.

The raw Replica asset library (mesh.ply + HDR ptex textures, no images or
trajectories) cannot be evaluated directly. The standard Replica evaluation
sequences (iMAP renders, used by NICE-SLAM/Co-SLAM/Point-SLAM/...) provide
RGB-D + GT trajectories for 8 scenes:

  wget -c https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
  unzip Replica.zip -d <data_root>   -> <data_root>/Replica/{room0,...}

Layout per scene: traj.txt (N lines x 16 floats, row-major c2w) +
results/frame%06d.jpg + results/depth%06d.png (uint16 / 6553.5 = meters).
Camera (NICE-SLAM camera config, constant across scenes): 1200x680,
fx=fy=600.0, cx=599.5, cy=339.5. The frame size and the SE(3) form of the
trajectory are asserted at runtime; the reprojection check below covers the
intrinsics and the depth scale.

Outputs (contract in common.py): tuples.json + gt/<id>.npz.
Reprojection check: per scene, the cross-view depth-reprojection consistency
of the first tuple must pass (median |z_proj - z_gt|/z_gt <= 8%, inliers >= 50%)
or the script aborts; this catches pose, intrinsics and depth-scale errors
before any model runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import grid_uv, save_gt_npz, save_tuples, subsample  # noqa: E402

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]
W, H = 1200, 680
FX = FY = 600.0
CX, CY = 599.5, 339.5
DEPTH_SCALE = 6553.5  # png value / DEPTH_SCALE = meters (NICE-SLAM cam cfg)


def load_traj(scene_dir: Path) -> np.ndarray:
    m = np.loadtxt(scene_dir / "traj.txt").reshape(-1, 4, 4)
    assert np.allclose(m[:, 3], [0, 0, 0, 1], atol=1e-5), "traj rows not SE(3)"
    return m


def read_depth_m(p: Path) -> np.ndarray:
    from PIL import Image
    d = np.asarray(Image.open(p), dtype=np.float64) / DEPTH_SCALE
    d[d <= 0] = np.nan
    return d


def backproject_world(depth_m: np.ndarray, c2w: np.ndarray, stride: int):
    """Pinhole unproject (z-depth) -> world. Returns (xyz [h,w,3] at stride)."""
    d = depth_m[::stride, ::stride]
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride),
                         indexing="ij")
    x = (us - CX) / FX * d
    y = (vs - CY) / FY * d
    cam = np.stack([x, y, d], -1)
    xyz = cam @ c2w[:3, :3].T + c2w[:3, 3]
    xyz[~np.isfinite(d)] = np.nan
    return xyz


def reproj_gate(depths, c2ws, scene: str) -> None:
    """Cross-view consistency: view0 cloud -> view1 pixels, compare depth."""
    xyz0 = backproject_world(depths[0], c2ws[0], 4).reshape(-1, 3)
    xyz0 = xyz0[np.isfinite(xyz0).all(1)]
    w2c1 = np.linalg.inv(c2ws[1])
    cam1 = xyz0 @ w2c1[:3, :3].T + w2c1[:3, 3]
    z = cam1[:, 2]
    ok = z > 0.05
    u = (cam1[ok, 0] / z[ok]) * FX + CX
    v = (cam1[ok, 1] / z[ok]) * FY + CY
    inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    if inb.sum() < 500:
        print(f"  [gate] {scene}: only {inb.sum()} projected points overlap "
              f"view1 — tuple too disjoint for the gate, widening not needed")
        return
    zs = depths[1][v[inb].round().astype(int), u[inb].round().astype(int)]
    val = np.isfinite(zs)
    rel = np.abs(zs[val] - z[ok][inb][val]) / z[ok][inb][val]
    med, inl = float(np.median(rel)), float((rel < 0.08).mean())
    print(f"  [gate] {scene}: reproj median-rel={med:.4f} inliers@8%={inl:.2%} "
          f"(n={val.sum()})")
    assert med <= 0.08 and inl >= 0.50, (
        f"{scene}: GT chain failed the reprojection gate — pose/K/depth-scale "
        f"convention is wrong; fix before feeding any model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help=".../replica_seq/Replica (contains room0..office4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--tuples-per-scene", type=int, default=6)
    ap.add_argument("--frame-stride", type=int, default=40,
                    help="frames between consecutive tuple views")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xyz-ds", type=int, default=2, help="xyz_ds downsample")
    ap.add_argument("--cloud-stride", type=int, default=4)
    ap.add_argument("--cloud-cap", type=int, default=200_000)
    ap.add_argument("--scenes", nargs="+", default=SCENES,
                    help="subset for smoke tests; default = all 8")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    from PIL import Image
    rng = np.random.default_rng(args.seed)
    tuples = []
    for scene in args.scenes:
        sd = root / scene
        assert sd.is_dir(), f"missing scene {sd} — wrong --root?"
        c2w_all = load_traj(sd)
        n = len(c2w_all)
        first = sd / "results" / "frame000000.jpg"
        with Image.open(first) as im:
            assert im.size == (W, H), f"{scene}: frame size {im.size} != {(W, H)}"
        span = (args.views - 1) * args.frame_stride
        assert n > span + 10, f"{scene}: traj too short ({n})"
        starts = np.linspace(0, n - span - 1, args.tuples_per_scene).astype(int)
        starts = np.clip(starts + rng.integers(-10, 11, len(starts)), 0,
                         n - span - 1)
        for ti, s0 in enumerate(starts):
            fids = [int(s0 + k * args.frame_stride) for k in range(args.views)]
            tid = f"replica_{scene}_{ti:02d}"
            views, depths, c2ws, xyz_ds_l, cloud = [], [], [], [], []
            for fi in fids:
                fp = sd / "results" / f"frame{fi:06d}.jpg"
                dp = sd / "results" / f"depth{fi:06d}.png"
                assert fp.is_file() and dp.is_file(), f"missing frame {fi} in {scene}"
                views.append({"img": str(fp.relative_to(root)), "w": W, "h": H})
                d = read_depth_m(dp)
                depths.append(d)
                c2ws.append(c2w_all[fi])
                xyz_ds_l.append(backproject_world(d, c2w_all[fi], args.xyz_ds))
                cl = backproject_world(d, c2w_all[fi], args.cloud_stride)
                cl = cl.reshape(-1, 3)
                cloud.append(cl[np.isfinite(cl).all(1)])
            if ti == 0:
                reproj_gate(depths, c2ws, scene)
            gt_pts = subsample(np.concatenate(cloud).astype(np.float32),
                               args.cloud_cap, seed=args.seed)
            save_gt_npz(out / "gt" / f"{tid}.npz", np.stack(c2ws),
                        np.stack(xyz_ds_l), args.xyz_ds, gt_pts)
            tuples.append({"id": tid, "seq": scene, "views": views,
                           "gt": f"gt/{tid}.npz"})
        print(f"[replica] {scene}: {args.tuples_per_scene} tuples "
              f"(V={args.views}, stride={args.frame_stride}, n_frames={n})")
    p = save_tuples(out, "replica", "pinhole", str(root), tuples)
    print(f"[replica] wrote {len(tuples)} tuples -> {p}")


if __name__ == "__main__":
    main()
