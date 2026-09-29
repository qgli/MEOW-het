#!/usr/bin/env python3
"""Bridge second-generation scenes into training shards in the first-generation pack layout.

Scene directories are written in the layout that the first-generation dataset
class (ProcThorUnicol) already reads, so the training path is unchanged; mixing
both generations means adding a second dataset entry that points at this root.
Packs are a superset of the first-generation format: the four keys
{rgb, rays, depth, mask} plus the extra channels {sem, inst, flags, normal}
(np.load is lazy, so the first-generation loader ignores the extras).

Conventions shared with the first-generation packs:
  - rays: y-up camera convention (image top = +y). cast.erp_dirs is OpenCV
    y-down, so rays_v1 = erp_dirs * [1, -1, 1].
  - camera_to_world_unicol_4x4: the camera frame (right, up, forward) is
    left-handed, det(R) = -1 by design ("unicol_z_forward"); world z-up is the
    engine world. c2w_unicol[:3, :3] = R_cv @ diag(1, -1, 1); [:3, 3] = pos.
  - depth: radial (metres along the unit ray); the cast distance t is radial
    (cross-checked against Cycles panoramic Z: median error <= 1e-4).
  - metadata.json: {"frames": [{tag, base_name, base_type, pose_index, fov_deg,
    resolution, camera_to_world_unicol_4x4, pack_file}], ...}; the loader uses
    only tag/base_name/base_type/pose_index/c2w.
  - covisibility/v0/{covisibility.npy (N, N) f32, frame_meta.json {frames: [...]}},
    filled from the pose-graph sampler's raycast covisibility edges (symmetric,
    diagonal 1, absent edge 0). The sampler's covisibility is the minimum of
    the two directed fractions, whereas the first-generation matrix is
    asymmetric; this is recorded in the provenance.
  - Scenes made only of equirectangular panoramas cannot satisfy the strict
    type-diversity level (L1) of the loader's tuple ladder, which therefore
    falls back to the relaxed levels by design.

Usage:
  python lenscope/bridge_v1_shard.py <fixtures_root> <samples_dir> <rgb_root> \
      <out_root> [--W 1024]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lenscope.core import cast  # noqa: E402
from lenscope.core.mesh import load_fixture  # noqa: E402
from lenscope.core.sampler import pose_to_R  # noqa: E402

FY = np.diag([1.0, -1.0, 1.0])


def _octa_channels(soup, pos, yaw, pitch, W):
    """Equal-area octahedral depth/flags for one pose (enabled by GENESIS_PACK_OCTA=1).

    Two channels (depth in mm as uint16, ambiguity flags as uint8), cast
    exactly along the texel directions; no released loader reads them. Adds
    about one raycast per pose; RGB and the other pack keys are unchanged
    (disabled = identical packs)."""
    from lenscope.core.mesh import (FLAG_EMISSIVE, FLAG_GLASS, FLAG_MIRROR,
                                     raycast_solid)
    from lenscope.core.sampler import pose_to_R
    from lenscope.octa import ea_octa as eo

    R = eo.r_for_erp(W)
    d_cam = eo.texel_dirs(R).reshape(-1, 3)
    Rm = pose_to_R(yaw, pitch)
    d_world = d_cam @ Rm.T
    r = raycast_solid(soup, np.asarray(pos, np.float64), d_world,
                      use_embree=True)
    t, face = r["t"], r["face"]
    hit = np.isfinite(t) & (t <= 50.0)
    depth_mm = np.where(hit, np.clip(t * 1000.0, 0, 65535), 0)
    amb = FLAG_GLASS | FLAG_MIRROR | FLAG_EMISSIVE
    flags = r["flags_accum"].copy()
    idx = np.where(hit & (face >= 0))[0]
    flags[idx] |= (soup.flags[face[idx]] & amb)
    meta = json.dumps(dict(mapping="ea_octa_v2", R=int(R), frame="cast_cam",
                           depth_unit="mm", erp_w=int(W)))
    return {
        "octa_depth": depth_mm.astype(np.uint16).reshape(R, R),
        "octa_flags": flags.astype(np.uint8).reshape(R, R),
        "octa_meta": np.frombuffer(meta.encode(), np.uint8),
    }


def bridge_scene(fix_dir: Path, sample_json: Path, rgb_dir: Path, out_dir: Path,
                 W: int = 1024) -> dict:
    from PIL import Image

    H = W // 2
    soup = load_fixture(fix_dir / "mesh.npz", fix_dir / "objects.json")
    res = json.loads(sample_json.read_text())
    poses = res["poses"]
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs_cv = cast.erp_dirs(W, H).reshape(H, W, 3)
    rays_v1 = (dirs_cv * np.array([1.0, -1.0, 1.0])).astype(np.float16)

    frames, tags = [], []
    n_px_bad = 0
    for p in poses:
        i = p["id"]
        # minimum width 2: the online camera sampler reconstructs tags as
        # f"erp_pose{pose:02d}" (3 digits from id 100 on)
        tag = f"erp_pose{i:02d}"
        png = rgb_dir / f"pose{i:03d}_erp.png"
        if not png.exists():
            continue
        rgb = np.asarray(Image.open(png).convert("RGB"), np.float32) / 255.0
        if rgb.shape[:2] != (H, W):
            raise ValueError(f"{png}: {rgb.shape} != ({H},{W})")
        g = cast.cast_pose(soup, p["pos"], p["yaw"], p["pitch"], W=W, H=H,
                           use_embree=True)
        n_px_bad += int((~g["mask"]).sum())
        n_valid = int(g["mask"].sum())
        R_cv = pose_to_R(p["yaw"], p["pitch"])
        c2w = np.eye(4)
        c2w[:3, :3] = R_cv @ FY
        c2w[:3, 3] = np.asarray(p["pos"], float)
        # Uncompressed by default: zlib in savez_compressed is single-threaded
        # and took about 20-40 s of the ~75 s per-pose bridge time, while plain
        # savez writes in about 2 s. Keys and dtypes are identical and np.load
        # reads both; the cost is ~48 MB per pose on disk.
        # GENESIS_PACK_COMPRESS=1 restores compression.
        _savez = (np.savez_compressed
                  if os.environ.get("GENESIS_PACK_COMPRESS", "0") == "1"
                  else np.savez)
        # GENESIS_PACK_OCTA=1 appends equal-area octahedral depth/flags keys;
        # off (default) leaves the pack key set unchanged.
        octa_extra = ({} if os.environ.get("GENESIS_PACK_OCTA", "0") != "1"
                      else _octa_channels(soup, p["pos"], p["yaw"], p["pitch"], W))
        _savez(
            out_dir / f"{tag}_pack.npz",
            rgb=rgb.astype(np.float16),
            rays=rays_v1,
            depth=g["depth"].astype(np.float16),
            mask=g["mask"].astype(np.uint8),
            # extra channels (ignored by the first-generation loader)
            sem=g["sem"], inst=g["inst"], flags=g["flags"],
            normal=g["normal"].astype(np.float16),
            **octa_extra,
        )
        frames.append({
            "tag": tag, "base_name": "erp", "base_type": "erp",
            "pose_index": int(i), "fov_deg": 360.0,
            "resolution": [W, H],
            "camera_to_world_unicol_4x4": [[round(v, 6) for v in row]
                                           for row in c2w.tolist()],
            "pack_file": f"{tag}_pack.npz",
            "supervision_only": bool(p.get("supervision_only", False)),
        })
        tags.append((tag, int(i), n_valid))

    # covisibility from the pose-graph sampler edges
    id2row = {pid: r for r, (_, pid, _) in enumerate(tags)}
    n = len(tags)
    cov = np.zeros((n, n), np.float32)
    np.fill_diagonal(cov, 1.0)
    for a, b, w, *rest in res["edges"]:
        if a in id2row and b in id2row:
            cov[id2row[a], id2row[b]] = cov[id2row[b], id2row[a]] = float(w)
    cdir = out_dir / "covisibility" / "v0"
    cdir.mkdir(parents=True, exist_ok=True)
    np.save(cdir / "covisibility.npy", cov)
    (cdir / "frame_meta.json").write_text(json.dumps({
        "n_frames": n,
        "source": "second-generation pose-graph sampler (ray-cast covisibility, minimum of the two directed fractions)",
        "frames": [{"tag": t, "base_name": "erp", "base_type": "erp", "index": r,
                    "pose_index": pid, "n_valid_src_points": nv}
                   for r, (t, pid, nv) in enumerate(tags)],
    }, indent=1))

    meta = {
        "pack_layout": "first-generation pack keys plus sem, inst, flags, normal",
        "provenance": {
            "fixtures": str(fix_dir), "sample_json": str(sample_json),
            "rgb": str(rgb_dir), "renderer": "cycles",
            "gt": "ray cast on the scene mesh (radial depth; median difference to Cycles depth <= 1e-4)",
            "bridged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "coordinate_system": {"name": "unicol_z_forward", "X": "right",
                              "Y": "up", "Z": "forward (optical axis)"},
        "num_poses": n,
        "frames": frames,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1))
    return {"n_frames": n, "n_edges": int((cov > 0).sum() - n) // 2,
            "miss_px_total": n_px_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixtures_root"); ap.add_argument("samples_dir")
    ap.add_argument("rgb_root"); ap.add_argument("out_root")
    ap.add_argument("--W", type=int, default=1024)
    # run_production calls this once per scene with --only; without it every
    # call would re-bridge all sample files (quadratic total cost).
    ap.add_argument("--only", default=None,
                    help="bridge exactly this scene name")
    ap.add_argument("--force", action="store_true",
                    help="re-bridge even if packs already complete")
    a = ap.parse_args()
    out_root = Path(a.out_root)
    summary = {}
    for sj in sorted(Path(a.samples_dir).glob("*_sample.json")):
        name = sj.name.replace("_sample.json", "")
        if a.only and name != a.only:
            continue
        fix = Path(a.fixtures_root) / name
        rgb = Path(a.rgb_root) / name
        if not fix.exists() or not rgb.exists():
            print(f"[skip] {name} (fixture or rgb missing)")
            continue
        # skip scenes whose packs are already complete (independent of --only)
        sd = out_root / "scenes" / name
        try:
            n_pose = len(json.loads(sj.read_text())["poses"])
        except Exception:
            n_pose = -1
        if (not a.force and sd.exists()
                and len(list(sd.glob("*_pack.npz"))) == n_pose
                and (sd / "metadata.json").exists()):
            print(f"[done] {name} ({n_pose} packs, skip)")
            summary[name] = {"skipped": True, "n_frames": n_pose}
            continue
        t0 = time.time()
        s = bridge_scene(fix, sj, rgb, out_root / "scenes" / name, W=a.W)
        s["secs"] = round(time.time() - t0, 1)
        summary[name] = s
        print(f"[ok] {name}: {s}")
    (out_root / "bridge_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
