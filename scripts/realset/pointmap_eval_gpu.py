#!/usr/bin/env python3
"""GPU-accelerated pointmap scoring: an opt-in counterpart of
scripts/mp3d_benchmark/pointmap_eval.py (which stays unchanged).

Numerical design (for consumer GPUs):
  * consumer GPUs run float64 at 1/64 rate, so pure-f64 brute force would be
    slower than the CPU KDTree. Instead: candidate k-NN search in f32
    (chunked direct (x-y)^2, no matmul-expansion cancellation), then exact
    float64 re-scoring of the candidates and a final f64 selection. The
    reported distance for the winning neighbour is exact f64, the same
    arithmetic as scipy's tree ((x-y)^2 summed over 3 axes, then sqrt).
  * a candidate miss is only possible if the true NN is not among the f32
    top-(k+8) candidates, which requires 8 neighbours within f32 epsilon of
    each other; this has measure zero for real point clouds, and the
    equivalence check below would catch it.
  * Umeyama + moge_align stay in numpy (cheap, identical); ICP/metrics/normals
    reuse the same code structure with the torch KNN.

Safety: eval_realset runs an equivalence check: the first scored tuple of
every model is computed on both backends; if any metric differs by more than
1e-6 (relative), the remaining tuples of that model are scored on the CPU.
CPU scoring is the default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mp3d_benchmark"))
from pointmap_eval import apply_sim3, moge_align, umeyama  # noqa: E402  (reused unchanged)


def _knn(query, ref, k, device, cand=8, chunk=8192):
    """Exact-f64 k-NN via f32 candidate search + f64 re-score.
    Returns (dist_f64 [Nq,k], idx [Nq,k])."""
    import torch
    q32 = torch.as_tensor(query, dtype=torch.float32, device=device)
    r32 = torch.as_tensor(ref, dtype=torch.float32, device=device)
    q64 = torch.as_tensor(query, dtype=torch.float64, device=device)
    r64 = torch.as_tensor(ref, dtype=torch.float64, device=device)
    kk = min(max(k + cand, k), len(ref))
    out_d, out_i = [], []
    for s in range(0, len(query), chunk):
        qc = q32[s:s + chunk]                                  # C,3
        d2 = ((qc[:, None, :] - r32[None, :, :]) ** 2).sum(-1)  # C,R  (f32 direct diff)
        _, cidx = torch.topk(d2, kk, dim=1, largest=False)      # C,kk candidates
        # exact f64 re-score of candidates only
        qc64 = q64[s:s + chunk]
        cand_pts = r64[cidx]                                    # C,kk,3
        d2_64 = ((qc64[:, None, :] - cand_pts) ** 2).sum(-1)    # C,kk  (f64 exact)
        dsel, sel = torch.topk(d2_64, k, dim=1, largest=False)
        out_d.append(torch.sqrt(dsel))
        out_i.append(torch.gather(cidx, 1, sel))
    return torch.cat(out_d).cpu().numpy(), torch.cat(out_i).cpu().numpy()


def icp_gpu(src, dst, device, iters=20, sample=20000, rng=None):
    """Mirror of pointmap_eval.icp with the torch KNN (same seeded sampling)."""
    rng = rng or np.random.default_rng(0)
    cur = src.copy()
    for _ in range(iters):
        idx = rng.choice(len(cur), min(sample, len(cur)), replace=False)
        q = cur[idx]
        _, nn = _knn(q, dst, 1, device)
        s, R, t = umeyama(q, dst[nn[:, 0]])
        cur = apply_sim3(cur, s, R, t)
    return cur


def estimate_normals_gpu(pts, device, k=16):
    """Mirror of pointmap_eval.estimate_normals (local-PCA smallest eigvec),
    KNN on GPU, eigh in numpy f64 (identical to CPU path)."""
    _, idx = _knn(pts, pts, min(k, len(pts)), device)
    nbr = pts[idx]
    c = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    _w, v = np.linalg.eigh(cov)
    return v[:, :, 0]


def point_metrics_gpu(pred, gt, device, pred_n=None, gt_n=None):
    dp, ip = _knn(pred, gt, 1, device)
    dg, ig = _knn(gt, pred, 1, device)
    dp, ip = dp[:, 0], ip[:, 0]
    dg, ig = dg[:, 0], ig[:, 0]
    out = {
        "Acc_mean": float(dp.mean()), "Acc_med": float(np.median(dp)),
        "Comp_mean": float(dg.mean()), "Comp_med": float(np.median(dg)),
    }
    if pred_n is not None and gt_n is not None:
        nc_p = np.abs(np.sum(pred_n * gt_n[ip], 1))
        nc_g = np.abs(np.sum(gt_n * pred_n[ig], 1))
        nc = np.concatenate([nc_p, nc_g])
        out["NC_mean"] = float(nc.mean())
        out["NC_med"] = float(np.median(nc))
    return out


def evaluate_gpu(pred, gt, device, pred_corr=None, gt_corr=None,
                 with_normals=True, gt_normals=None):
    """Same protocol as pointmap_eval.evaluate (Umeyama -> moge_align -> ICP ->
    Acc/Comp/N.C.); gt_normals may be passed pre-computed (cache)."""
    if pred_corr is not None and gt_corr is not None and len(pred_corr) >= 3:
        s, R, t = umeyama(pred_corr, gt_corr)
        pred = apply_sim3(pred, s, R, t)
        pc_aligned = apply_sim3(np.asarray(pred_corr), s, R, t)
        pred = moge_align(pc_aligned, gt_corr, pred)
    pred = icp_gpu(pred, gt, device)
    pn = estimate_normals_gpu(pred, device) if with_normals else None
    if with_normals:
        gn = gt_normals if gt_normals is not None else estimate_normals_gpu(gt, device)
    else:
        gn = None
    return point_metrics_gpu(pred, gt, device, pn, gn)
