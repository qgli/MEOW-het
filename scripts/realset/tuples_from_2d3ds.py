#!/usr/bin/env python3
"""Export the wid3r-faithful 2D3DS panorama cases as a realset tuples.json so
the baseline predictors (VGGT/pi3/DUSt3R/MASt3R) run on exactly the cases of
eval_2d3ds_pose_v2 --wid3r-faithful.

Sampling replicates the wid3r-faithful branch of eval_2d3ds_pose_v2.main
(same rng seed, iteration order and shuffle/truncation), and GT poses come
from the same load_gt_c2w. Pose-only: the gt npz carries empty pointmap
fields, so eval_realset skips Acc/Comp/N.C. automatically.

Run in the mapanything env; the 2D3DS root is --stanford-root or, if omitted,
eval_2d3ds_pose.STANFORD_ROOT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
sys.path.insert(0, str(THIS.parent.parent))          # scripts/ for eval_2d3ds_pose
from common import save_gt_npz, save_tuples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-cases", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stanford-root", default=None)
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep repeated draws of the same tuple (the original 20-case set)")
    args = ap.parse_args()

    import eval_2d3ds_pose as v1
    if args.stanford_root:
        v1.STANFORD_ROOT = args.stanford_root
    root = Path(v1.STANFORD_ROOT)
    assert root.is_dir(), f"STANFORD_ROOT missing: {root}"

    def rgb_path(area, base):
        # same as eval_2d3ds_pose_v2.rgb_path. Defined locally because
        # eval_2d3ds_pose (v1) has no rgb_path, and importing
        # eval_2d3ds_pose_v2 for one path helper would pull torch/model
        # dependencies into this CPU exporter. Reads v1.STANFORD_ROOT at
        # call time, as eval_2d3ds_pose_v2 does.
        return str(Path(v1.STANFORD_ROOT) / area / "pano" / "rgb"
                   / f"{base}_rgb.png")

    # ---- same sampling as eval_2d3ds_pose_v2 --wid3r-faithful ---------------
    rng = np.random.default_rng(args.seed)
    cases = []
    lo, hi, reps = 10, 30, 10
    for area in args.areas:
        by_scene = v1.list_frames_by_scene(area)
        for _room, frames_with_loc in by_scene.items():
            n = len(frames_with_loc)
            if n < 2:
                continue
            bases = [b for b, _ in frames_with_loc]
            for _ in range(reps):
                k = min(n, int(rng.integers(lo, hi + 1)))
                if k < 2:
                    continue
                idx = rng.choice(n, size=k, replace=False)
                cases.append((area, [bases[i] for i in idx]))
    rng.shuffle(cases)
    if len(cases) > args.max_cases:
        cases = cases[: args.max_cases]
    cases = list(enumerate(cases))          # tuple ids keep the draw index
    if not args.keep_duplicates:            # as eval_2d3ds_pose_v2: a repeated tuple is kept once
        seen, unique = set(), []
        for ci, (area, sel) in cases:
            if (area, frozenset(sel)) not in seen:
                seen.add((area, frozenset(sel)))
                unique.append((ci, (area, sel)))
        cases = unique
    # ------------------------------------------------------------------------

    out = Path(args.out)
    tuples = []
    from PIL import Image
    for ci, (area, sel) in cases:
        tid = f"s2d3ds_{ci:03d}"
        views, c2ws = [], []
        for b in sel:
            p = Path(rgb_path(area, b))
            assert p.is_file(), f"missing pano {p}"
            with Image.open(p) as im:
                w0, h0 = im.size
            views.append({"img": str(p.relative_to(root)), "w": w0, "h": h0})
            c2ws.append(v1.load_gt_c2w(area, b))
        save_gt_npz(out / "gt" / f"{tid}.npz", np.stack(c2ws),
                    np.zeros((len(sel), 0, 0, 3), np.float16), 1,
                    np.zeros((0, 3), np.float32))
        tuples.append({"id": tid, "seq": f"{area}", "views": views,
                       "gt": f"gt/{tid}.npz"})
    nv = [len(t["views"]) for t in tuples]
    p = save_tuples(out, "2d3ds", "erp", str(root), tuples)
    print(f"[2d3ds-export] cases={len(tuples)} views min/med/max="
          f"{min(nv)}/{int(np.median(nv))}/{max(nv)} -> {p}")
    print("[2d3ds-export] PROTOCOL: wid3r-faithful, sampling bit-identical to "
          f"eval_2d3ds_pose_v2 (seed={args.seed}, areas={args.areas}, "
          f"max_cases={args.max_cases})")


if __name__ == "__main__":
    main()
