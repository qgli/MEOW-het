#!/usr/bin/env python3
"""Point-map metrics after Wid3R §4.3: Umeyama -> scale/shift -> ICP, then Acc/Comp/N.C.

Wid3R: "For point cloud alignment, we sequentially apply Umeyama alignment, MoGe
optimal point alignment, and Iterative Closest Point. We report Accuracy (Acc.),
Completion (Comp.), and Normal Consistency (N.C.)."

This module implements:
- Umeyama: closed-form Sim(3) (scale s, rotation R, translation t) from the given
  predicted<->GT correspondences (eval_mp3d_panoramas.py passes the camera centres).
- Scale/shift (moge_align): closed-form least-squares global scale and 3D
  translation on the same correspondences. MoGe's optimal point alignment itself
  is not implemented; this unweighted least-squares fit takes its place.
- ICP: point-to-point refinement using nearest neighbours (Sim(3) per iteration).
- Acc:  mean/median of, for each predicted point, distance to nearest GT point.
- Comp: mean/median of, for each GT point, distance to nearest predicted point.
- N.C.: mean/median of |cos angle| between a point's normal and its nearest
  neighbour's normal, pooled over both directions (pred->GT and GT->pred).

Self-check (__main__): pred = Sim3 @ GT + noise must give Acc/Comp ~ 0 and N.C. ~ 1.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def umeyama(src, dst):
    """Closed-form Sim(3) mapping src -> dst (correspondences, equal length).
    Returns (s, R, t) with dst ~ s*R@src + t."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, Dm = src - mu_s, dst - mu_d
    cov = (Dm.T @ S) / len(src)
    U, Dsv, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = np.trace(np.diag(Dsv) @ W) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(pts, s, R, t):
    return (s * (R @ pts.T).T) + t


def moge_align(pred_corr, gt_corr, pred_all):
    """Least-squares scale/shift: closed-form global scale s and translation
    shift t minimizing ||s*pred + t - gt||^2 over the given correspondences
    (named after the scale/shift point alignment of MoGe, Wang et al. 2025, whose
    optimal solver is not reproduced). Applied after Umeyama as a residual
    scale/shift refinement. Returns transformed pred_all."""
    p = np.asarray(pred_corr, np.float64)
    g = np.asarray(gt_corr, np.float64)
    mp, mg = p.mean(0), g.mean(0)
    pc, gc = p - mp, g - mg
    s = (pc * gc).sum() / max((pc * pc).sum(), 1e-12)
    t = mg - s * mp
    return s * pred_all + t


def icp(src, dst, iters=20, sample=20000, rng=None):
    """Point-to-point ICP refining Sim(3) src->dst. Returns transformed src."""
    rng = rng or np.random.default_rng(0)
    tree = cKDTree(dst)
    cur = src.copy()
    for _ in range(iters):
        idx = rng.choice(len(cur), min(sample, len(cur)), replace=False)
        q = cur[idx]
        _, nn = tree.query(q, k=1, workers=-1)
        s, R, t = umeyama(q, dst[nn])
        cur = apply_sim3(cur, s, R, t)
    return cur


def estimate_normals(pts, k=16):
    """Per-point normal via local PCA (smallest eigenvector). Sign unoriented."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)), workers=-1)
    nbr = pts[idx]                      # N x k x 3
    c = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    w, v = np.linalg.eigh(cov)          # ascending
    return v[:, :, 0]                   # smallest-eigval direction


def point_metrics(pred, gt, pred_n=None, gt_n=None):
    """Acc/Comp/N.C. (mean+median) between two aligned point clouds."""
    tg = cKDTree(gt)
    tp = cKDTree(pred)
    dp, ip = tg.query(pred, k=1, workers=-1)        # pred -> nearest GT  (accuracy)
    dg, ig = tp.query(gt, k=1, workers=-1)          # gt   -> nearest pred (completion)
    out = {
        "Acc_mean": float(dp.mean()), "Acc_med": float(np.median(dp)),
        "Comp_mean": float(dg.mean()), "Comp_med": float(np.median(dg)),
    }
    if pred_n is not None and gt_n is not None:
        nc_p = np.abs(np.sum(pred_n * gt_n[ip], 1))   # pred normal vs matched GT normal
        nc_g = np.abs(np.sum(gt_n * pred_n[ig], 1))   # gt normal vs matched pred normal
        nc = np.concatenate([nc_p, nc_g])
        out["NC_mean"] = float(nc.mean())
        out["NC_med"] = float(np.median(nc))
    return out


def evaluate(pred, gt, pred_corr=None, gt_corr=None, with_normals=True):
    """Align pred to gt (Umeyama -> least-squares scale/shift -> ICP) and score Acc/Comp/N.C.
    pred_corr/gt_corr: equal-length correspondences (e.g. camera centres) used for
    Umeyama and the scale/shift fit. If None, ICP bootstraps from identity."""
    if pred_corr is not None and gt_corr is not None and len(pred_corr) >= 3:
        s, R, t = umeyama(pred_corr, gt_corr)
        pred = apply_sim3(pred, s, R, t)
        pc_aligned = apply_sim3(np.asarray(pred_corr), s, R, t)
        pred = moge_align(pc_aligned, gt_corr, pred)  # residual scale/shift
    pred = icp(pred, gt)
    pn = estimate_normals(pred) if with_normals else None
    gn = estimate_normals(gt) if with_normals else None
    return point_metrics(pred, gt, pn, gn)


if __name__ == "__main__":
    # Verification: pred = Sim3 @ GT (+ small noise). After alignment Acc/Comp ~ 0, NC ~ 1.
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(5000, 3)) * np.array([3.0, 3.0, 1.0])  # room-like slab
    # random Sim3
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=0).as_matrix()
    s, t = 2.3, np.array([5.0, -2.0, 1.0])
    pred = apply_sim3(gt, s, R, t) + rng.normal(scale=0.005, size=gt.shape)
    m = evaluate(pred.copy(), gt.copy(), pred_corr=pred, gt_corr=gt)
    print("pred = Sim3@GT + 5mm noise, after Umeyama+ICP:")
    for k, v in m.items():
        print(f"  {k:10s} = {v:.4f}")
    print("EXPECT: Acc/Comp ~ 0 (<~0.01m noise floor), NC ~ 1.0 => metric+alignment correct")
