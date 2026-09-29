"""Multi-view relative pose on Stanford 2D-3D-S panoramas (Wid3R protocol).

Follows the Stanford 2D-3D-S pose protocol of Wid3R (arXiv 2602.05321, Sec. 4.2), a
public panorama benchmark also used by CAM3R. The dataset root is read from the
environment variable MEOW_2D3DS_ROOT (the directory containing area_*/pano).

Wid3R protocol (Sec. 4.2):
  - From area_5a and area_5b, per scene randomly pick 10-30 equirectangular (ERP)
    images; repeat the sampling 10 times per scene (about 20 cases).
  - Metrics: RRA@30 / RTA@30 / AUC@30 (area under the min(RRA, RTA) threshold curve)
    and ATE.
  - The model predicts all views in one forward pass (per-view cam2world cam_quats and
    cam_trans, OpenCV axes), so no pairwise inference or global alignment is needed.

Pose convention:
  - Model output: cam_quats (B,4) and cam_trans (B,3), the OpenCV cam2world pose of
    each view (right, down, forward).
  - 2D3DS GT: pose.json camera_rt_matrix (3x4 [R|t], world->cam); c2w is its inverse.
    2D3DS uses a different world axis convention than OpenCV; relative poses do not
    depend on the world frame, so predicted relative poses are compared with GT
    relative poses.

Usage:
  CUDA_VISIBLE_DEVICES=0 MEOW_2D3DS_ROOT=/path/to/2d3ds python scripts/eval_2d3ds_pose.py \
      --ckpts NAME=/path/to/checkpoint.pth MapAnything=checkpoints/facebook_map-anything-apache.pth \
      --areas area_5a area_5b --min-views 10 --max-views 30 --repeats 10 \
      --res 518 --out experiments/pub_bench/2d3ds_pose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))

from mapanything.models import _init_hydra_config, init_model  # noqa: E402

STANFORD_ROOT = os.environ.get("MEOW_2D3DS_ROOT", ".")


# --------------------------------------------------------------------------
# ERP rays (inlined): Z-forward, Y-up frame of the first-generation renderer,
# image centre = +Z
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Quaternion (OpenCV RDF cam2world) -> rotation matrix
# --------------------------------------------------------------------------
def quat_to_R(q):
    """Rotation matrix of a quaternion in MapAnything order (x, y, z, w), normalized first."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q  # MapAnything order: x, y, z, w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R


def build_model(device):
    cfg = _init_hydra_config(
        "configs/train.yaml",
        overrides=["model=mapanything", "machine=default",
                   "model/task=aug_training", "++model.task.ar_prob=1.0"],
    )
    model = init_model(model_str=cfg.model.model_str,
                       model_config=cfg.model.model_config,
                       torch_hub_force_reload=False)
    return model.to(device).eval()


def load_ckpt(model, sd):
    incompat = model.load_state_dict(sd, strict=False)
    bad = [k for k in incompat.missing_keys if not k.startswith("ar_encoder.")]
    if bad:
        raise RuntimeError(f"Unexpected missing keys: {bad[:8]}")
    if incompat.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {incompat.unexpected_keys[:8]}")


# --------------------------------------------------------------------------
# 2D3DS frame listing grouped by scene (room)
# --------------------------------------------------------------------------
def list_frames_by_scene(area):
    import json as _json
    pano = Path(STANFORD_ROOT) / area / "pano"
    by_scene = {}
    for p in sorted((pano / "pose").glob("*_pose.json")):
        base = p.name.replace("_pose.json", "")
        meta = _json.load(open(p))
        room = meta.get("room", "unknown")
        loc = np.array(meta["camera_location"], dtype=np.float64)
        by_scene.setdefault(room, []).append((base, loc))
    return by_scene


def sample_covisible_group(frames_with_loc, k, rng, bmin=0.1, bmax=2.5):
    """Pick k frames forming a spatially compact group (pairwise baseline in
    [bmin, bmax]), similar to the CAM3R/Wid3R indoor pairing range (0.1-2.2 m).
    Greedy: start from a random anchor, then repeatedly add the first frame, in a
    random order, whose distance to every chosen frame is at most bmax and at
    least bmin (non-degenerate)."""
    bases = [b for b, _ in frames_with_loc]
    locs = np.array([l for _, l in frames_with_loc])
    n = len(bases)
    if n < 2:
        return None
    anchor = int(rng.integers(0, n))
    chosen = [anchor]
    # add candidates within [bmin, bmax] of every chosen frame
    for _ in range(k - 1):
        best = None
        rng_order = list(rng.permutation(n))
        for c in rng_order:
            if c in chosen:
                continue
            d = np.linalg.norm(locs[c] - locs[chosen], axis=1)
            if d.max() <= bmax and d.min() >= bmin:
                best = c
                break
        if best is None:
            break
        chosen.append(best)
    if len(chosen) < 2:
        return None
    return [bases[i] for i in chosen]


def load_gt_c2w(area, base):
    import json as _json
    pano = Path(STANFORD_ROOT) / area / "pano"
    meta = _json.load(open(pano / "pose" / f"{base}_pose.json"))
    rt = np.array(meta["camera_rt_matrix"], dtype=np.float64)  # (3,4) w2c
    w2c = np.eye(4)
    w2c[:3, :4] = rt
    c2w = np.linalg.inv(w2c)
    return c2w


def load_img_tensor(area, base, res, device):
    pano = Path(STANFORD_ROOT) / area / "pano"
    H, W = res, res * 2
    rgb = Image.open(pano / "rgb" / f"{base}_rgb.png").convert("RGB").resize((W, H), Image.BILINEAR)
    rgb = np.asarray(rgb).astype(np.float32) / 255.0
    # 2D3DS panoramas fill the top and bottom poles (~27%) with pure black (no content).
    # Detect these pixels (the boundary is irregular) so that their rays can be set to a
    # placeholder, as for fisheye pixels outside the image circle in training
    # (RGB=0, ray=[0,0,1]).
    black = (rgb.sum(2) < 0.04)  # (H,W) bool
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = ((t - mean) / std).unsqueeze(0)
    return img.to(device), H, W, black


def build_views(area, bases, res, device, with_rays=False):
    views = []
    for k, b in enumerate(bases):
        img, H, W, black = load_img_tensor(area, b, res, device)
        v = {"img": img, "data_norm_type": ["dinov2"],
             "true_shape": torch.tensor([[H, W]]).to(device),
             "instance": [b], "idx": [k]}
        if with_rays:
            rays = generate_rays_erp(H, W)
            # black (pole) region: placeholder ray [0,0,1] (the convention for fisheye
            # pixels outside the image circle in training), not a valid ERP direction.
            rays[black] = np.array([0.0, 0.0, 1.0], dtype=rays.dtype)
            v["ray_directions_cam"] = torch.from_numpy(rays).unsqueeze(0).to(device)
        views.append(v)
    return views


@torch.no_grad()
def predict_c2w(model, views):
    """Return list of pred c2w (4x4) per view from cam_quats+cam_trans."""
    preds = model(views)
    out = []
    for p in preds:
        q = p["cam_quats"][0].detach().cpu().numpy()
        t = p["cam_trans"][0].detach().cpu().numpy()
        R = quat_to_R(q)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        out.append(c2w)
    return out


# --------------------------------------------------------------------------
# Relative pose metrics (RRA/RTA/AUC/ATE)
# --------------------------------------------------------------------------
def rel_pose(c2w_a, c2w_b):
    """Relative pose a->b: T = inv(c2w_b) @ c2w_a. Returns (R, t)."""
    T = np.linalg.inv(c2w_b) @ c2w_a
    return T[:3, :3], T[:3, 3]


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def trans_angle_deg(ta, tb):
    # Standard translation-angle error (PoseDiffusion util/metric.py
    # compare_translation_by_angle, used by VGGT / pi3 / Wid3R):
    #   normalize both, err = acos(sqrt(1 - (t . t_gt)^2)) in [0, 90 deg].
    # It is sign-agnostic: a reconstructed translation direction is only defined
    # up to sign, so +t and -t score identically (the (.)^2 removes the sign).
    # A degenerate predicted or GT translation (norm ~0) gets the maximum error,
    # not zero; otherwise a near-identity prediction would score RTA=100%.
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
    m = np.maximum(rerr, terr)  # min(RRA,RTA) accuracy = fraction with both errors < thr
    for t in thrs:
        accs.append((m < t).mean())
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(accs, thrs) / (max_thr - 1) * 100), accs


def maa(rerr, terr, max_thr=30):
    """mean Average Accuracy (CAM3R/VGGT multi-view): mean over integer thresholds
    1..max_thr of the fraction of pairs with both rotation and translation error
    < thr. This is the discrete mean of the min(RRA,RTA) accuracy curve (close to
    auc_min, reported as a percentage mean, the convention of CAM3R Table 2)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--areas", nargs="+", default=["area_5a", "area_5b"])
    ap.add_argument("--min-views", type=int, default=10)
    ap.add_argument("--max-views", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--baseline-min", type=float, default=0.1,
                    help="min pairwise camera baseline (m), CAM3R uses 0.1")
    ap.add_argument("--baseline-max", type=float, default=2.5,
                    help="max pairwise camera baseline (m), CAM3R uses 2.2 for 2D3DS")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-rays", action="store_true",
                    help="provide ERP ray directions (calibrated mode)")
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpts = []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        assert os.path.isfile(path), f"ckpt not found: {path}"
        ckpts.append((name, path))

    rng = np.random.default_rng(args.seed)
    # build cases: (area, [bases]) with covisible/baseline-constrained groups
    cases = []
    for area in args.areas:
        by_scene = list_frames_by_scene(area)
        for room, frames_with_loc in by_scene.items():
            if len(frames_with_loc) < args.min_views:
                continue
            for _ in range(args.repeats):
                k = int(rng.integers(args.min_views, args.max_views + 1))
                sel = sample_covisible_group(frames_with_loc, k, rng,
                                             bmin=args.baseline_min,
                                             bmax=args.baseline_max)
                if sel is not None and len(sel) >= max(3, args.min_views // 2):
                    cases.append((area, sel))
    rng.shuffle(cases)
    if len(cases) > args.max_cases:
        cases = cases[:args.max_cases]
    print(f"[2d3ds-pose] areas={args.areas} cases={len(cases)} "
          f"with_rays={args.with_rays} ckpts={[n for n,_ in ckpts]}")

    print("[model] building once")
    model = build_model(args.device)

    records = {}
    for name, path in ckpts:
        c = torch.load(path, map_location="cpu", weights_only=False)
        load_ckpt(model, c.get("model", c))
        rra30, rta30, aucs, n_ok = [], [], [], 0
        for area, sel in cases:
            try:
                views = build_views(area, sel, args.res, args.device, args.with_rays)
                pred = predict_c2w(model, views)
                gt = [load_gt_c2w(area, b) for b in sel]
                rerr, terr = eval_case(pred, gt)
                rra30.append((rerr < 30).mean())
                rta30.append((terr < 30).mean())
                auc, _ = auc_min(rerr, terr, 30)
                aucs.append(auc)
                n_ok += 1
            except Exception as e:
                print(f"  [warn] case {area} n={len(sel)} failed: {e}")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        records[name] = {
            "n_cases": n_ok,
            "RRA@30": float(np.mean(rra30) * 100) if rra30 else 0,
            "RTA@30": float(np.mean(rta30) * 100) if rta30 else 0,
            "AUC@30": float(np.mean(aucs)) if aucs else 0,
        }
        print(f"  {name:8s} cases={n_ok:2d}  RRA@30={records[name]['RRA@30']:.2f}  "
              f"RTA@30={records[name]['RTA@30']:.2f}  AUC@30={records[name]['AUC@30']:.2f}")

    out_json = args.out / "2d3ds_pose_results.json"
    with open(out_json, "w") as f:
        json.dump({"areas": args.areas, "with_rays": args.with_rays,
                   "protocol": "Wid3R-2D3DS", "records": records,
                   "reference": {"Wid3R": {"RRA@30": 94.05, "RTA@30": 93.29, "AUC@30": 79.93},
                                 "VGGT": {"RRA@30": 19.30, "AUC@30": 2.60},
                                 "pi3": {"RRA@30": 19.94, "AUC@30": 2.06}}},
                  f, indent=2)
    print(f"\n[out] {out_json}")
    print("[ref] Wid3R 94.05/93.29/79.93 | VGGT 19.30/-/2.60 | pi3 19.94/-/2.06")


if __name__ == "__main__":
    main()
