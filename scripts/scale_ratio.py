#!/usr/bin/env python3
"""Predicted metric scale without alignment: ratio of ground-truth to predicted camera-centre distances.

Per tuple, the ratio |C_i^gt - C_j^gt| / |C_i^pred - C_j^pred| is computed for every view pair and its
median taken; the statistic is the median over tuples, with the share of tuples whose ratio lies within
20 % and 30 % of 1. A ratio above 1 means the predicted scene is too small.

Predictions: one npz per tuple with "c2w" (V, 4, 4), named after the tuple (laser benchmark:
"<tuple id>.npz"; exported 2D3DS cases: "case_NNN.npz"), in the order of the tuple's views.
Ground truth (one of):
  --tuples T.json            laser benchmark tuples; poses from gt/<tuple id>.npz next to the file
  --cases-root DIR           exported heterogeneous 2D3DS cases (manifest.json with per-case gt_c2w)
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def tuple_ratio(c2w_gt, c2w_pred):
    cg, cp = np.asarray(c2w_gt)[:, :3, 3], np.asarray(c2w_pred)[:, :3, 3]
    r = []
    for i, j in combinations(range(len(cg)), 2):
        dp = np.linalg.norm(cp[i] - cp[j])
        if dp > 1e-9:
            r.append(np.linalg.norm(cg[i] - cg[j]) / dp)
    return float(np.median(r)) if r else float("nan")


def pairs_laser(tuples_json, pred_dir):
    tj = Path(tuples_json)
    for tup in json.loads(tj.read_text())["tuples"]:
        gt = np.load(tj.parent / "gt" / f"{tup['id']}.npz")["c2w"]
        yield tup["id"], gt, np.load(Path(pred_dir) / f"{tup['id']}.npz")["c2w"]


def pairs_cases(cases_root, pred_dir):
    root = Path(cases_root)
    for case in json.loads((root / "manifest.json").read_text())["cases"]:
        name = f"case_{int(case['case_id']):03d}"
        gt = np.load(root / case["gt_c2w"])
        yield name, gt, np.load(Path(pred_dir) / f"{name}.npz")["c2w"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred-dir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tuples")
    src.add_argument("--cases-root")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="optional JSON with the per-tuple ratios")
    a = ap.parse_args()

    it = pairs_laser(a.tuples, a.pred_dir) if a.tuples else pairs_cases(a.cases_root, a.pred_dir)
    per = {}
    for name, gt, pred in it:
        if len(gt) != len(pred):
            raise ValueError(f"{name}: {len(gt)} ground-truth views, {len(pred)} predicted")
        per[name] = tuple_ratio(gt, pred)
    r = np.array([v for v in per.values() if np.isfinite(v)])
    summary = dict(set=a.name, tuples=len(per), median_ratio=round(float(np.median(r)), 3),
                   within_20pct=round(float(np.mean(np.abs(r - 1) <= 0.2)), 3),
                   within_30pct=round(float(np.mean(np.abs(r - 1) <= 0.3)), 3))
    print(json.dumps(summary))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=summary, per_tuple=per), indent=1))


if __name__ == "__main__":
    main()
