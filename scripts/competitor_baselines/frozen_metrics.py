"""Verbatim copies of the 2D3DS pose metric helpers of scripts/eval_2d3ds_pose.py (plus the Sim3/chamfer helpers of scripts/pointcloud_metrics.py) so baseline envs can score without mapanything imports."""
import numpy as np

def rel_pose(c2w_a, c2w_b):
    """Relative pose a->b: T = inv(c2w_b) @ c2w_a. Returns (R, t)."""
    T = np.linalg.inv(c2w_b) @ c2w_a
    return T[:3, :3], T[:3, 3]


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def trans_angle_deg(ta, tb):
    # Field-standard translation-angle error (PoseDiffusion util/metric.py
    # compare_translation_by_angle, used by VGGT / pi3 / Wid3R):
    #   normalize both, err = acos(sqrt(1 - (t . t_gt)^2)) in [0, 90 deg].
    # It is sign-agnostic: a reconstructed translation direction is only defined
    # up to sign, so +t and -t score identically (the (.)^2 removes the sign).
    # A degenerate predicted/GT translation (norm ~0) is the worst case (return
    # the max error), not a perfect match; returning 0 would let a near-identity
    # model trivially score RTA=100%.
    eps = 1e-15
    na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
    if na < 1e-9 or nb < 1e-9:
        return 90.0
    cos2 = (np.dot(ta, tb) / (na * nb)) ** 2
    loss_t = max(1.0 - cos2, 0.0)
    return float(np.degrees(np.arccos(np.sqrt(1.0 - loss_t + eps).clip(-1, 1))))


def eval_case(pred_c2w, gt_c2w):
    """All pairs relative pose; return per-pair rot/trans angular errors."""
    n = len(pred_c2w)
    rerr, terr = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            Rp, tp = rel_pose(pred_c2w[i], pred_c2w[j])
            Rg, tg = rel_pose(gt_c2w[i], gt_c2w[j])
            rerr.append(rot_angle_deg(Rp, Rg))
            terr.append(trans_angle_deg(tp, tg))
    return np.array(rerr), np.array(terr)


def auc_min(rerr, terr, max_thr=30):
    """AUC of min(RRA,RTA) curve over thresholds [1..max_thr]."""
    thrs = np.arange(1, max_thr + 1)
    accs = []
    m = np.maximum(rerr, terr)  # min(RRA,RTA) accuracy = fraction with both < thr
    for t in thrs:
        accs.append((m < t).mean())
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(accs, thrs) / (max_thr - 1) * 100), accs


def maa(rerr, terr, max_thr=30):
    """mean Average Accuracy (CAM3R/VGGT multi-view): mean over integer thresholds
    1..max_thr of the fraction of pairs with both rot & trans error < thr. This is
    the discrete mean of the min(RRA,RTA) accuracy curve (≈ auc_min but reported as
    a percentage mean, the convention CAM3R Table 2 uses)."""
    thrs = np.arange(1, max_thr + 1)
    m = np.maximum(rerr, terr)
    return float(np.mean([(m < t).mean() for t in thrs]) * 100)


def pose_metrics(rerr, terr):
    """Full metric bundle at multiple thresholds (field-standard). Returns dict."""
    out = {}
    for thr in (5, 10, 15, 30):
        out[f"RRA@{thr}"] = float((rerr < thr).mean() * 100)
        out[f"RTA@{thr}"] = float((terr < thr).mean() * 100)
    out["AUC@30"], _ = auc_min(rerr, terr, 30)
    out["AUC@10"], _ = auc_min(rerr, terr, 10)
    out["mAA@30"] = maa(rerr, terr, 30)
    return out


def ate_rmse(pred_c2w, gt_c2w):
    """Absolute Trajectory Error (RMSE) after Sim3 (Umeyama) alignment of camera
    centres (CAM3R Table3 / pi3 Table2 / Wid3R Table3 protocol)."""
    P = np.array([c[:3, 3] for c in pred_c2w], dtype=np.float64)
    G = np.array([c[:3, 3] for c in gt_c2w], dtype=np.float64)
    if len(P) < 2:
        return float("nan")
    muP, muG = P.mean(0), G.mean(0)
    P0, G0 = P - muP, G - muG
    H = P0.T @ G0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    varP = (P0 ** 2).sum()
    s = (S * np.array([1, 1, d])).sum() / (varP + 1e-12)
    t = muG - s * R @ muP
    P_al = (s * (R @ P.T).T) + t
    return float(np.sqrt(((P_al - G) ** 2).sum(1).mean()))


# --- ERP rays (scripts/eval_2d3ds_pose.py, same as eval_2d3ds_panorama.generate_rays_erp core) ---
def generate_rays_erp(H, W):
    j = np.arange(W, dtype=np.float64)
    i = np.arange(H, dtype=np.float64)
    jj, ii = np.meshgrid(j, i)
    lon = (jj + 0.5) / W * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (ii + 0.5) / H * np.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], axis=-1).astype(np.float32)

# --- Sim3 + chamfer (scripts/pointcloud_metrics.py) ---
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

