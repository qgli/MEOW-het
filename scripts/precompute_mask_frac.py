"""Precompute per-frame statistics of first-generation scene packs.

For every ``<base>_pose<NN>_pack.npz`` in each sub-directory of ``--root``, reads the
``mask``, ``depth`` and ``rgb`` arrays and records the valid-pixel fraction, the mean and
standard deviation of the depth over valid pixels, and the standard deviation of the RGB
values over valid pixels (averaged over the channels). Writes a single JSON file::

    {
      "<scene_name>": {
        "<tag>": {"valid_frac": ..., "depth_mean": ..., "depth_std": ..., "rgb_std": ...},
        ...
      },
      ...
    }

(all four values are NaN for a pack that cannot be read).

:class:`mapanything.datasets.procthor_unicol.ProcThorUnicol` takes this file as
``frame_stats_path`` and, when building its frame index, skips frames whose ``rgb_std`` is
below ``min_rgb_std`` (views facing a flat wall).

Usage:

    python scripts/precompute_mask_frac.py \
        --root /path/to/renders/scenes \
        --out  /path/to/renders/frame_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PACK_RE = re.compile(r"^(?P<base>.+)_pose(?P<pose>\d+)_pack\.npz$")


def scan_scene(scene_dir: str) -> tuple[str, dict[str, dict]]:
    """Return (scene_name, {tag: {valid_frac, depth_mean, depth_std, rgb_std}})."""
    name = os.path.basename(scene_dir.rstrip("/"))
    out: dict[str, dict] = {}
    for fn in sorted(os.listdir(scene_dir)):
        m = PACK_RE.match(fn)
        if not m:
            continue
        tag = f"{m['base']}_pose{m['pose']}"
        path = os.path.join(scene_dir, fn)
        try:
            with np.load(path) as pack:
                mask = np.asarray(pack["mask"]).astype(bool)  # (H,W)
                depth = np.asarray(pack["depth"]).astype(np.float32)  # (H,W)
                rgb = np.asarray(pack["rgb"]).astype(np.float32)  # (H,W,3) in [0,1]
            valid_frac = float(mask.mean())
            d_valid = depth[mask]
            if d_valid.size == 0:
                d_mean = d_std = 0.0
            else:
                d_mean = float(d_valid.mean())
                d_std = float(d_valid.std())
            # RGB std over valid pixels (mean across channels)
            rgb_v = rgb[mask]  # (N, 3)
            rgb_std = float(rgb_v.std(axis=0).mean()) if rgb_v.size else 0.0
            stats = dict(valid_frac=valid_frac, depth_mean=d_mean,
                         depth_std=d_std, rgb_std=rgb_std)
        except Exception as e:
            print(f"  [warn] {path}: {e}", file=sys.stderr)
            stats = dict(valid_frac=float("nan"), depth_mean=float("nan"),
                         depth_std=float("nan"), rgb_std=float("nan"))
        out[tag] = stats
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Directory containing scene_* / proc_scene_* folders.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output JSON cache path.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    scenes = sorted(
        str(root / d) for d in os.listdir(root)
        if (root / d).is_dir()
    )
    print(f"[scan] {len(scenes)} scenes under {root}")

    t0 = time.time()
    cache: dict[str, dict[str, float]] = {}
    n_packs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(scan_scene, s) for s in scenes]
        for i, f in enumerate(as_completed(futs)):
            name, d = f.result()
            cache[name] = d
            n_packs += len(d)
            if (i + 1) % 5 == 0 or (i + 1) == len(scenes):
                print(f"  [{i+1}/{len(scenes)}] {name} ({len(d)} packs)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, sort_keys=True, indent=2))
    dt = time.time() - t0
    print(f"\n[done] scanned {n_packs} packs in {dt:.1f}s -> {out_path}")

    # Print percentile summaries of the four statistics
    rows = [(sc, tag, s) for sc, d in cache.items() for tag, s in d.items()
            if not np.isnan(s["valid_frac"])]
    arr_v = np.array([r[2]["valid_frac"] for r in rows])
    arr_dm = np.array([r[2]["depth_mean"] for r in rows])
    arr_ds = np.array([r[2]["depth_std"] for r in rows])
    arr_rs = np.array([r[2]["rgb_std"] for r in rows])
    print(f"\n[stats] across {len(rows)} packs:")
    for name, arr in [("valid_frac", arr_v), ("depth_mean", arr_dm),
                       ("depth_std", arr_ds), ("rgb_std", arr_rs)]:
        print(f"  {name:<12} min={arr.min():.3f} p10={np.percentile(arr,10):.3f}"
              f" p25={np.percentile(arr,25):.3f} median={np.median(arr):.3f}"
              f" mean={arr.mean():.3f} max={arr.max():.3f}")
    # Wall-hit candidates: depth_mean < 1.5 m and depth_std < 0.5 m
    wall = [r for r in rows if r[2]["depth_mean"] < 1.5 and r[2]["depth_std"] < 0.5]
    print(f"\n[wall-hit suspects: depth_mean<1.5 AND depth_std<0.5] {len(wall)} packs")
    for sc, tag, s in sorted(wall, key=lambda r: r[2]["depth_mean"])[:10]:
        print(f"  d_mean={s['depth_mean']:.2f} d_std={s['depth_std']:.2f}"
              f" rgb_std={s['rgb_std']:.3f}  {sc}/{tag}")


if __name__ == "__main__":
    main()
