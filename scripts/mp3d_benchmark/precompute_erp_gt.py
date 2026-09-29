#!/usr/bin/env python3
"""One-off ERP GT precompute for the MP3D point-map evaluation (eval_mp3d_panoramas.py).

Fills the npz cache that eval_mp3d_panoramas.py --gt-cache consumes, so the eval's hot
path never touches the depth zips. Pano selection replays eval_mp3d_panoramas's exact
sampling (same seed, same K, same covisibility softmax) so the cached set matches
what the eval will request; --all-panos caches every pano instead (seed/K
agnostic, ~10x more work).

Usage:
  python scripts/mp3d_benchmark/precompute_erp_gt.py \
    --scans-root <matterport3d>/scans \
    --out <erp_gt_cache> \
    [--k 8 --seed 0 --erp-w 1024 --erp-h 512 --stride 4 --all-panos]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parse_conf import parse_conf_from_zip, scan_dir
from covis_sample import sample_covis, pano_centers
from gt_erp_pointmap import erp_point_maps_for_scan
from eval_mp3d_panoramas import TEST_18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-root", default=os.environ.get("MEOW_MP3D_ROOT", "."),
                    help="MP3D scans dir, one sub-directory per scan "
                         "(default: env MEOW_MP3D_ROOT)")
    ap.add_argument("--out", required=True, help="npz cache dir (feed eval_mp3d_panoramas --gt-cache)")
    ap.add_argument("--scans", nargs="*", default=None, help="default: TEST_18")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--erp-w", type=int, default=1024)
    ap.add_argument("--erp-h", type=int, default=512)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--all-panos", action="store_true")
    args = ap.parse_args()

    scans = args.scans or TEST_18
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for sid in scans:
        scan = os.path.join(args.scans_root, sid)
        if not os.path.isdir(scan):
            print(f"[{sid}] MISSING under {args.scans_root} — skipped")
            continue
        panos = parse_conf_from_zip(scan_dir(scan))
        if args.all_panos:
            sel_uuids = list(panos.keys())
        else:
            # replay eval_mp3d_panoramas.run_scan sampling exactly (fresh rng per scan)
            uuids, C = pano_centers(panos)
            sel = sample_covis(C, args.k, rng=np.random.default_rng(args.seed))
            sel_uuids = [uuids[i] for i in sel]
        t0 = time.time()
        n = 0
        for _u, _P, valid in erp_point_maps_for_scan(
                scan, sel_uuids, erp_w=args.erp_w, erp_h=args.erp_h,
                stride=args.stride, cache_dir=args.out):
            n += 1
        total += n
        print(f"[{sid}] cached {n} panos in {time.time()-t0:.1f}s "
              f"(coverage last={valid.mean()*100:.1f}%)")
    print(f"DONE: {total} pano GT maps -> {args.out}")


if __name__ == "__main__":
    main()
