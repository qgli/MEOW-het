"""Point-cloud helpers shared by the panorama evaluators: similarity alignment and Chamfer distance."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Solve dst ≈ s * R @ src + t. Returns (s, R, t) in float64."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = (src_c**2).sum() / src.shape[0]
    H = src_c.T @ dst_c / src.shape[0]
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = (D * np.diag(S)).sum() / (var_s + 1e-12)
    t = mu_d - s * R @ mu_s
    return s, R, t


def chamfer(a: np.ndarray, b: np.ndarray, max_pts: int = 100_000):
    """Symmetric nearest-neighbour distances between point sets a (prediction) and b (ground truth);
    each set is randomly subsampled (fixed seed) to at most max_pts points."""
    rng = np.random.default_rng(0)
    if a.shape[0] > max_pts:
        a = a[rng.choice(a.shape[0], max_pts, replace=False)]
    if b.shape[0] > max_pts:
        b = b[rng.choice(b.shape[0], max_pts, replace=False)]
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_ab, _ = tb.query(a, k=1, workers=-1)
    d_ba, _ = ta.query(b, k=1, workers=-1)
    return {
        "L1": float(0.5 * (d_ab.mean() + d_ba.mean())),
        "L2": float(0.5 * (np.sqrt((d_ab**2).mean()) + np.sqrt((d_ba**2).mean()))),
        "pred_to_gt_p50": float(np.median(d_ab)),
        "pred_to_gt_p90": float(np.percentile(d_ab, 90)),
        "gt_to_pred_p50": float(np.median(d_ba)),
        "gt_to_pred_p90": float(np.percentile(d_ba, 90)),
    }
