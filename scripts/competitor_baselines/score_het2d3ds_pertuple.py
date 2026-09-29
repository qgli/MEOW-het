#!/usr/bin/env python3
"""Per-tuple aggregation of the heterogeneous 2D3DS benchmark (88 tuples).

Reads the raw JSON written by scripts/mp3d_benchmark/het_2d3ds_pose.py --out (key "records":
one entry per tuple with "frames", "rotation_errors", "translation_errors" over the tuple's ordered
pairs and "ate") or the Wid3R score file of scripts/competitor_baselines/score_het2d3ds.py (same "records").
Prints, for all 88 tuples and for the 3-8-view and 9-24-view subsets: the pooled-pair metrics and the
mean over tuples of the per-tuple metrics (mAA@30, RRA@30, RTA@30, ATE), plus a per-size breakdown.
Optionally writes a JSON summary.

Usage: python scripts/competitor_baselines/score_het2d3ds_pertuple.py RESULT.json [--out SUMMARY.json]
"""
import argparse, json
import numpy as np


def maa(r, t, mx=30):
    m = np.maximum(r, t)
    return float(np.mean([(m < th).mean() for th in range(1, mx + 1)]) * 100)


def block(recs):
    R = [np.asarray(x["rotation_errors"], float) for x in recs]
    T = [np.asarray(x["translation_errors"], float) for x in recs]
    Rp, Tp = np.concatenate(R), np.concatenate(T)
    per = np.array([[maa(r, t), (r < 30).mean() * 100, (t < 30).mean() * 100] for r, t in zip(R, T)])
    ate = np.array([x.get("ate", np.nan) for x in recs], float)
    return {
        "n_tuples": len(recs), "n_pairs": int(len(Rp)),
        "pooled": {"mAA@30": maa(Rp, Tp), "RRA@30": float((Rp < 30).mean() * 100),
                   "RTA@30": float((Tp < 30).mean() * 100), "ATE": float(np.nanmean(ate))},
        "per_tuple_mean": {"mAA@30": float(per[:, 0].mean()), "RRA@30": float(per[:, 1].mean()),
                           "RTA@30": float(per[:, 2].mean()), "ATE": float(np.nanmean(ate))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = json.load(open(a.result))
    recs = d["records"]
    nviews = lambda x: len(x["frames"])
    out = {"source": a.result, "model": d.get("model"),
           "all": block(recs),
           "views_3_8": block([x for x in recs if nviews(x) <= 8]),
           "views_9_24": block([x for x in recs if nviews(x) >= 9]),
           "by_size": {}}
    for lo, hi in ((3, 3), (4, 4), (5, 8), (9, 14), (15, 24)):
        sel = [x for x in recs if lo <= nviews(x) <= hi]
        if sel:
            out["by_size"][f"{lo}-{hi}"] = block(sel)
    for k in ("all", "views_3_8", "views_9_24"):
        b = out[k]
        print(f"{k:11s} n={b['n_tuples']:3d} pairs={b['n_pairs']:5d} | pooled mAA {b['pooled']['mAA@30']:.1f} "
              f"RRA {b['pooled']['RRA@30']:.1f} RTA {b['pooled']['RTA@30']:.1f} | per-tuple mAA "
              f"{b['per_tuple_mean']['mAA@30']:.1f} RRA {b['per_tuple_mean']['RRA@30']:.1f} "
              f"RTA {b['per_tuple_mean']['RTA@30']:.1f} | ATE {b['pooled']['ATE']:.3f}")
    for k, b in out["by_size"].items():
        print(f"  {k:6s} n={b['n_tuples']:3d} per-tuple mAA {b['per_tuple_mean']['mAA@30']:.1f} "
              f"RRA {b['per_tuple_mean']['RRA@30']:.1f} RTA {b['per_tuple_mean']['RTA@30']:.1f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
