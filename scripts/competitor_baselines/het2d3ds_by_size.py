#!/usr/bin/env python3
"""Heterogeneous 2D3DS benchmark by tuple size, comparing two result files.

Both inputs are raw result JSONs with per-tuple "records" (scripts/mp3d_benchmark/het_2d3ds_pose.py --out
or scripts/competitor_baselines/score_het2d3ds.py --out) over the same tuples. For each tuple-size range the
script prints the number of tuples, the mean over tuples of per-tuple mAA@30, RRA@15 and RTA@15 for
both models, and the number of tuples on which the first model's per-tuple mAA@30 is higher.

Usage: python scripts/competitor_baselines/het2d3ds_by_size.py FIRST.json SECOND.json [--out SUMMARY.json]
"""
import argparse
import json

import numpy as np

RANGES = ((3, 3), (4, 4), (5, 8), (9, 14), (15, 24))


def maa(r, t, mx=30):
    m = np.maximum(r, t)
    return float(np.mean([(m < th).mean() for th in range(1, mx + 1)]) * 100)


def per_tuple(rec):
    r = np.asarray(rec["rotation_errors"], float)
    t = np.asarray(rec["translation_errors"], float)
    return {"mAA@30": maa(r, t), "RRA@15": float((r < 15).mean() * 100), "RTA@15": float((t < 15).mean() * 100)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--out")
    a = ap.parse_args()
    A = json.load(open(a.first))["records"]
    B = json.load(open(a.second))["records"]
    key = lambda x: (x.get("area"), tuple(x["frames"]))
    bmap = {key(x): x for x in B}
    missing = [key(x) for x in A if key(x) not in bmap]
    if missing or len(A) != len(B):
        raise SystemExit(f"tuple lists differ ({len(A)} vs {len(B)} tuples, {len(missing)} unmatched)")
    rows = []
    print(f"{'views':>6} {'tuples':>6} | {'mAA@30':>7} {'RRA@15':>7} {'RTA@15':>7} | "
          f"{'mAA@30':>7} {'RRA@15':>7} {'RTA@15':>7} | first ahead")
    for lo, hi in RANGES:
        sel = [x for x in A if lo <= len(x["frames"]) <= hi]
        if not sel:
            continue
        pa = [per_tuple(x) for x in sel]
        pb = [per_tuple(bmap[key(x)]) for x in sel]
        mean = lambda p, k: float(np.mean([q[k] for q in p]))
        ahead = sum(qa["mAA@30"] > qb["mAA@30"] for qa, qb in zip(pa, pb))
        row = dict(views=f"{lo}" if lo == hi else f"{lo}-{hi}", tuples=len(sel),
                   first={k: mean(pa, k) for k in pa[0]}, second={k: mean(pb, k) for k in pb[0]},
                   first_ahead=ahead)
        rows.append(row)
        f, s = row["first"], row["second"]
        print(f"{row['views']:>6} {len(sel):>6} | {f['mAA@30']:7.1f} {f['RRA@15']:7.1f} {f['RTA@15']:7.1f} | "
              f"{s['mAA@30']:7.1f} {s['RRA@15']:7.1f} {s['RTA@15']:7.1f} | {ahead}/{len(sel)}")
    if a.out:
        json.dump(dict(first=a.first, second=a.second, by_size=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
