#!/usr/bin/env python3
"""Training-convention covisibility of the laser-benchmark tuples.

The benchmark generator gates consecutive views with covis_pair() (min of the two directions,
depth agreement max(10 cm, 3 %), 1500 random pixels).  The training engine certifies tuples with the
pixel-grid test of mapanything/datasets/covis_gpu.py (nearest ray within 3.6 deg, depth agreement
10 cm + 5 %, fraction of the source view's valid pixels) and the mean of the two directions, gate 0.25.
This script evaluates the training-convention statistic on the benchmark tuples as released, from
gt/<tid>.npz alone (c2w per view + NaN-padded world points), so the two definitions can be compared
on the same tuples.  No prediction is involved.

Per tuple it reports the V x V directed matrix, the mean- and min-symmetrised matrices, the minimum
over all pairs and over consecutive pairs of each, and whether the graph with mean-symmetrised
covisibility > 0.25 is connected.  Per track it reports medians and pass fractions.

Usage:
  python scripts/blk_bench/covis_training_convention.py --track-dir <bench>/blk_mixed --out <json> [--device cuda:0]
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

COS_THRES, DEPTH_ABS, DEPTH_REL, GATE = 0.998, 0.10, 0.05, 0.25   # mapanything/datasets/covis_gpu.py + connectivity.py


def views_from_gt(npz, src_stride):
    c2w = np.asarray(npz["c2w"], np.float64)                  # (V,4,4)
    xyz = np.asarray(npz["xyz_ds"], np.float64)               # (V,Hd,Wd,3) NaN-padded world points
    out = []
    for v in range(c2w.shape[0]):
        X = xyz[v]; R, t = c2w[v, :3, :3], c2w[v, :3, 3]
        valid = np.isfinite(X).all(-1)
        P = (X - t) @ R                                         # camera frame: R^T (X - t)
        depth = np.linalg.norm(P, axis=-1)
        rays = np.where(valid[..., None], P / np.clip(depth[..., None], 1e-9, None), 0.0)
        src = np.zeros_like(valid); src[::src_stride, ::src_stride] = True; src &= valid
        out.append(dict(rays=rays.reshape(-1, 3), depth=depth.reshape(-1), valid=valid.reshape(-1),
                        src=src.reshape(-1), R=R, t=t))
    return out


def directed_covis(views, device, chunk=512, skip_fwd_for=()):
    """skip_fwd_for: indices of target views for which the forward-hemisphere condition is dropped
    (full panoramas; the engine keeps it for every target, so the default reproduces the engine)."""
    V = len(views)
    T = [{k: torch.as_tensor(w[k], dtype=torch.float64 if k != "valid" and k != "src" else torch.bool, device=device)
          for k in ("rays", "depth", "valid", "src", "R", "t")} for w in views]
    cov = np.eye(V)
    for i in range(V):
        fi = T[i]
        P_cam = fi["rays"][fi["src"]] * fi["depth"][fi["src"]].unsqueeze(-1)
        P_w = P_cam @ fi["R"].T + fi["t"]                     # world points of the subsampled source pixels
        n_src = P_w.shape[0]
        for j in range(V):
            if i == j or n_src == 0:
                continue
            fj = T[j]
            P_j = (P_w - fj["t"]) @ fj["R"]
            dist = P_j.norm(dim=-1).clamp_min(1e-8)
            d = P_j / dist.unsqueeze(-1)
            fwd = P_j[:, 2] > 0
            hits = 0.0
            for s in range(0, n_src, chunk):
                e = min(s + chunk, n_src)
                cos = d[s:e] @ fj["rays"].T
                mx, idx = cos.max(dim=-1)
                err = (dist[s:e] - fj["depth"][idx]).abs()
                ok = (mx > COS_THRES) & fj["valid"][idx] & (err <= DEPTH_ABS + DEPTH_REL * fj["depth"][idx])
                if j not in skip_fwd_for:
                    ok = ok & fwd[s:e]
                hits += float(ok.sum())
            cov[i, j] = hits / n_src
    return cov


def connected(adj):
    n = adj.shape[0]; seen = {0}; stack = [0]
    while stack:
        c = stack.pop()
        for j in np.flatnonzero(adj[c]):
            if j not in seen:
                seen.add(int(j)); stack.append(int(j))
    return len(seen) == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--src-stride", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.track_dir)
    meta = json.load(open(root / "tuples.json"))
    tuples = meta["tuples"][: a.limit or None]
    rows = []
    for tup in tuples:
        z = np.load(root / tup["gt"])
        views = views_from_gt(z, a.src_stride)
        cov = directed_covis(views, a.device)                                   # exact engine test (forward hemisphere for every target)
        erp_idx = tuple(k for k, v in enumerate(tup["views"]) if v["kind"] == "erp")
        cov_s = directed_covis(views, a.device, skip_fwd_for=erp_idx) if erp_idx else cov   # sphere-aware variant for panorama targets
        mean_sym = 0.5 * (cov + cov.T); min_sym = np.minimum(cov, cov.T)
        mean_sym_s = 0.5 * (cov_s + cov_s.T)
        V = cov.shape[0]; iu = np.triu_indices(V, 1)
        consec = [(k, k + 1) for k in range(V - 1)]
        adj = (mean_sym > GATE) & ~np.eye(V, dtype=bool)
        rows.append({
            "id": tup["id"], "kinds": [v["kind"] for v in tup["views"]],
            "cov_directed": np.round(cov, 4).tolist(),
            "mean_min_allpairs": float(mean_sym[iu].min()), "mean_min_consecutive": float(min(mean_sym[k, l] for k, l in consec)),
            "min_min_allpairs": float(min_sym[iu].min()), "min_min_consecutive": float(min(min_sym[k, l] for k, l in consec)),
            "connected_at_training_gate": bool(connected(adj)),
            "sphere_aware": {"cov_directed": np.round(cov_s, 4).tolist(), "mean_min_allpairs": float(mean_sym_s[iu].min()),
                             "mean_min_consecutive": float(min(mean_sym_s[k, l] for k, l in consec)),
                             "connected_at_training_gate": bool(connected((mean_sym_s > GATE) & ~np.eye(V, dtype=bool)))},
        })
        print(f"{tup['id']}: mean-sym min(all) {rows[-1]['mean_min_allpairs']:.3f} min(consec) {rows[-1]['mean_min_consecutive']:.3f} | "
              f"min-sym min(all) {rows[-1]['min_min_allpairs']:.3f} min(consec) {rows[-1]['min_min_consecutive']:.3f} | "
              f"connected@0.25 {rows[-1]['connected_at_training_gate']}", flush=True)
    med = lambda k: float(np.median([r[k] for r in rows]))
    summary = {
        "track": meta.get("track"), "n_tuples": len(rows), "src_stride": a.src_stride,
        "definition": "covis_gpu.py test (3.6 deg nearest ray, 10 cm + 5 % depth) on the released gt point maps; mean/min symmetrisation",
        "median_mean_min_allpairs": med("mean_min_allpairs"), "median_mean_min_consecutive": med("mean_min_consecutive"),
        "median_min_min_allpairs": med("min_min_allpairs"), "median_min_min_consecutive": med("min_min_consecutive"),
        "frac_connected_at_training_gate": float(np.mean([r["connected_at_training_gate"] for r in rows])),
        "frac_consecutive_mean_ge_0.25": float(np.mean([r["mean_min_consecutive"] >= 0.25 for r in rows])),
        "sphere_aware_median_mean_min_allpairs": float(np.median([r["sphere_aware"]["mean_min_allpairs"] for r in rows])),
        "sphere_aware_median_mean_min_consecutive": float(np.median([r["sphere_aware"]["mean_min_consecutive"] for r in rows])),
        "sphere_aware_frac_connected_at_training_gate": float(np.mean([r["sphere_aware"]["connected_at_training_gate"] for r in rows])),
    }
    json.dump({"summary": summary, "tuples": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
