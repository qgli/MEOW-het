#!/usr/bin/env python3
"""Shared data contracts for the real-capture evaluation harness.

Pipeline:
  prepare (adt_prepare / replica_prepare / tuples_from_2d3ds, mapanything env)
      -> <out>/tuples.json + <out>/gt/<tuple_id>.npz + extracted frames
  predict_* (one per model family, each in its own conda env)
      -> <out>/preds/<model>/<tuple_id>.npz   (same contract for every model)
  eval_realset (mapanything env, CPU)
      -> results_<model>.json  (pose RRA/RTA/AUC@30 + pointmap Acc/Comp/N.C.)

Every stage above imports this module, including the baseline envs
(pi3vggt / mast3r), so it depends on numpy and the standard library only
(no torch or mapanything imports at module level).

tuples.json:
  {"dataset": "adt", "track": "fisheye", "frames_root": "<abs>",
   "tuples": [{"id": "adt_fe_0000", "seq": "<name>",
               "views": [{"img": "<rel to frames_root>", "w": W, "h": H}, ...],
               "gt": "gt/<id>.npz"}, ...]}

gt npz (written by prepare; camera-model-free so eval needs no camera code):
  c2w      [V,4,4] f64   GT camera-to-world (meters)
  xyz_ds   [V,Hd,Wd,3] f16  per-view world-frame point map, downsampled by `ds`
                            from the fed image grid; NaN = invalid depth
  ds       int           downsample factor (fed-pixel uv -> xyz_ds index // ds)
  gt_pts   [N,3] f32     fused world cloud for Acc/Comp/N.C. (<=cap, seeded)

preds npz (written by every predict_*):
  c2w      [V,4,4] f64   predicted camera-to-world (any gauge/scale)
  pts      [M,3] f32     fused predicted world points (may be empty: pose-only)
  uv       [M,2] f32     pixel coords of each pts row in the original fed image
  vidx     [M]   i32     view index of each pts row
  proc     str           note: processed resolution / preprocessing used
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_tuples(out_dir: Path, dataset: str, track: str, frames_root: str,
                tuples: list) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "tuples.json"
    with open(p, "w") as f:
        json.dump({"dataset": dataset, "track": track,
                   "frames_root": str(frames_root), "tuples": tuples}, f, indent=1)
    return p


def load_tuples(tuples_json: str):
    with open(tuples_json) as f:
        spec = json.load(f)
    assert spec.get("tuples"), f"no tuples in {tuples_json}"
    root = Path(spec.get("frames_root", "."))
    if not root.is_absolute():             # relative frames_root: relative to the folder of tuples.json
        spec["frames_root"] = str(Path(tuples_json).resolve().parent / root)
    return spec


def img_paths(spec: dict, tup: dict) -> list:
    root = Path(spec["frames_root"])
    return [str(root / v["img"]) for v in tup["views"]]


def save_gt_npz(path: Path, c2w, xyz_ds, ds: int, gt_pts):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, c2w=np.asarray(c2w, np.float64),
                        xyz_ds=np.asarray(xyz_ds, np.float16),
                        ds=np.int32(ds), gt_pts=np.asarray(gt_pts, np.float32))


class TupleTimer:
    """Per-tuple inference time and peak VRAM.
    Wraps the full prediction span (load -> infer -> outputs on CPU); for
    global-alignment methods this includes the optimization, which is part of
    their per-tuple cost. torch is imported lazily (the module stays
    numpy-only for envs without it)."""

    def __init__(self, device=None):
        self.device = device if (device and str(device).startswith("cuda")) else None
        self.sec = float("nan")
        self.vram_gb = float("nan")

    def __enter__(self):
        import time
        if self.device is not None:
            try:
                import torch
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                self.device = None
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        if self.device is not None:
            try:
                import torch
                torch.cuda.synchronize(self.device)
                self.vram_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
            except Exception:
                pass
        self.sec = time.perf_counter() - self._t0
        return False


def timing_summary(tag: str, secs, vrams):
    """Print the aggregate [timing] line predictors emit at DONE (first tuple
    excluded as warm-up)."""
    import numpy as _np
    s = _np.asarray([x for x in secs[1:] if _np.isfinite(x)])
    v = _np.asarray([x for x in vrams if _np.isfinite(x)])
    if len(s):
        print(f"[timing] {tag}: n={len(s)} mean={s.mean():.2f}s p50={_np.median(s):.2f}s "
              f"max={s.max():.2f}s peak_vram={v.max():.2f}GB" if len(v) else
              f"[timing] {tag}: n={len(s)} mean={s.mean():.2f}s p50={_np.median(s):.2f}s")


def save_pred_npz(path: Path, c2w, pts=None, uv=None, vidx=None, proc="",
                  sec=None, vram_gb=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    V = len(c2w)
    c2w = np.asarray(c2w, np.float64)
    assert c2w.shape == (V, 4, 4), f"bad c2w {c2w.shape}"
    assert np.isfinite(c2w).all(), "non-finite c2w"
    if pts is None or len(pts) == 0:
        pts = np.zeros((0, 3), np.float32)
        uv = np.zeros((0, 2), np.float32)
        vidx = np.zeros((0,), np.int32)
    pts = np.asarray(pts, np.float32)
    uv = np.asarray(uv, np.float32)
    vidx = np.asarray(vidx, np.int32)
    assert len(pts) == len(uv) == len(vidx), "pts/uv/vidx length mismatch"
    keep = np.isfinite(pts).all(1) & np.isfinite(uv).all(1)
    extra = {}
    if sec is not None:
        extra["sec"] = np.float64(sec)
    if vram_gb is not None:
        extra["vram_gb"] = np.float64(vram_gb)
    np.savez_compressed(path, c2w=c2w, pts=pts[keep], uv=uv[keep],
                        vidx=vidx[keep], proc=np.str_(proc), **extra)


def subsample(arr, cap: int, seed: int = 0):
    if len(arr) <= cap:
        return arr
    idx = np.random.default_rng(seed).choice(len(arr), cap, replace=False)
    return arr[idx]


def resize_crop_uv_map(orig_wh, proc_wh):
    """Map processed-grid pixel coords -> original-image pixel coords for the
    ubiquitous 'scale-preserving resize then center crop' preprocessing
    (mapanything fixed_mapping, dust3r/mast3r 512-crop). Returns (s, ox, oy)
    with  u_orig = (u_proc + ox) / s,  v_orig = (v_proc + oy) / s.
    Also exact for pure resize (ox = oy = 0)."""
    W0, H0 = orig_wh
    Wp, Hp = proc_wh
    s = max(Wp / W0, Hp / H0)
    ox = (W0 * s - Wp) / 2.0
    oy = (H0 * s - Hp) / 2.0
    return s, ox, oy


def grid_uv(h, w, stride: int):
    """Pixel-center uv grid (u=x, v=y) at a stride, flattened [K,2] float32."""
    vs, us = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride),
                         indexing="ij")
    return np.stack([us.ravel(), vs.ravel()], 1).astype(np.float32)
