"""Single-panorama reconstruction scoring on the 2D3DS panorama set (inputs/2d3ds_single_panoramas), as in
eval_2d3ds_panorama_v2 (GT = depth/512 * the generate_rays_erp ray at the model's output (H,W),
black-pole + invalid-depth mask, Sim3-Umeyama -> chamfer L1/L2, median rel depth, delta1; mean
per area and combined), for baseline local point maps.
--backend wid3r: npz local_points [1,336,518,3] radial (rays*radius) in Wid3R ERP frame.
--backend panovggt: npz depth [1,518,1036] radial distance; local = PanoVGGT dirs * depth.
The baseline ERP frame is y-down, the GT frame y-up -> p_O = diag(1,-1,1) p_T before alignment
(--conv yflip, the default; a reflection cannot be absorbed by the proper-rotation Sim3)."""
import sys, json, os, argparse
from pathlib import Path
import numpy as np
from PIL import Image
S = os.environ['COMP_BASE_ROOT']
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_metrics import generate_rays_erp, umeyama_sim3, chamfer
SINGLE_PANOS = f'{S}/inputs/2d3ds_single_panoramas'; DEPTH_SCALE = 512.0; DEPTH_INVALID = 65535

def build_gt_at(area, base, H, W):
    pano = f'{SINGLE_PANOS}/stanford/{area}/pano'
    dep = Image.open(f'{pano}/depth/{base}_depth.png').resize((W, H), Image.NEAREST)
    dep_raw = np.asarray(dep).astype(np.float32)
    valid = (dep_raw > 0) & (dep_raw < DEPTH_INVALID)
    depth_m = np.where(valid, dep_raw / DEPTH_SCALE, 0.0)
    rays = generate_rays_erp(H, W).astype(np.float32)
    gt_cam = depth_m[..., None] * rays
    rgb = np.asarray(Image.open(f'{pano}/rgb/{base}_rgb.png').convert('RGB').resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    mask = valid & (rgb.sum(axis=2) >= 0.04)
    return gt_cam, mask

def eval_recon(pred_xyz, gt_cam, mask):
    gt = gt_cam[mask]; pr = pred_xyz[mask]
    if len(gt) < 500: return None
    s, R, t = umeyama_sim3(pr, gt); pr_al = (s * (R @ pr.T).T) + t
    ch = chamfer(pr_al, gt)
    gt_d = np.linalg.norm(gt, axis=1); pr_d = np.linalg.norm(pr_al, axis=1)
    rel = np.abs(pr_d - gt_d) / np.maximum(gt_d, 1e-6)
    d1 = (np.maximum(pr_d / np.maximum(gt_d, 1e-6), gt_d / np.maximum(pr_d, 1e-6)) < 1.25).mean()
    return {"chamfer_L1": float(ch["L1"]), "chamfer_L2": float(ch["L2"]), "depth_rel": float(np.median(rel)), "delta1": float(d1)}

def panovggt_dirs(H, W):
    u = np.arange(W) + 0.5; v = np.arange(H) + 0.5
    phi = (u / W - 0.5) * 2 * np.pi; theta = -(v / H - 0.5) * np.pi
    gt, gp = np.meshgrid(theta, phi, indexing='ij')
    return np.stack([np.cos(gt) * np.sin(gp), -np.sin(gt), np.cos(gt) * np.cos(gp)], -1).astype(np.float32)

ap = argparse.ArgumentParser(); ap.add_argument('--preds', required=True); ap.add_argument('--backend', required=True, choices=['wid3r', 'panovggt']); ap.add_argument('--name', required=True); ap.add_argument('--out', required=True); ap.add_argument('--conv', default='yflip', choices=['identity', 'yflip'])
a = ap.parse_args()
C = np.diag([1., -1., 1.]).astype(np.float32) if a.conv == 'yflip' else np.eye(3, dtype=np.float32)
items = json.load(open(f'{S}/runs/lists/single_panoramas.json'))
per_area, all_res, failed = {}, [], []
for it in items:
    p = f"{a.preds}/{it['id']}.npz"
    if not os.path.exists(p): failed.append(it['id']); continue
    d = np.load(p)
    if a.backend == 'wid3r':
        pts = d['local_points'][0].astype(np.float32)               # (336,518,3)
    else:
        dep = d['depth'][0].astype(np.float32); pts = panovggt_dirs(*dep.shape) * dep[..., None]
    H, W = pts.shape[:2]; pts = pts @ C.T
    gt_cam, mask = build_gt_at(it['area'], it['base'], H, W)
    m = eval_recon(pts, gt_cam, mask)
    if m is None: failed.append(it['id']); continue
    m['id'] = it['id']; m['sec'] = float(d['sec']); per_area.setdefault(it['area'], []).append(m); all_res.append(m)
keys = ['chamfer_L1', 'chamfer_L2', 'depth_rel', 'delta1']
rec = {'variant': f'{a.backend} conv={a.conv} HxW={H}x{W}', 'n_frames': len(all_res), **{k: float(np.mean([r[k] for r in all_res])) for k in keys},
       'per_area': {ar: {'n_frames': len(rs), **{k: float(np.mean([r[k] for r in rs])) for k in keys}} for ar, rs in per_area.items()}, 'failed': failed, 'frames': all_res}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump({'areas': list(per_area), 'pipeline': 'competitor official recipe + single-panorama scorer (score_single_panoramas.py)', 'records': {a.name: rec}}, open(a.out, 'w'), indent=1)
print(f"{a.name} single-pano n={rec['n_frames']} failed={len(failed)} chamferL1={rec['chamfer_L1']:.4f} L2={rec['chamfer_L2']:.4f} depth_rel={rec['depth_rel']:.4f} d1={rec['delta1']:.3f}  " + '  '.join(f"[{ar}] L1={v['chamfer_L1']:.4f} d1={v['delta1']:.3f}" for ar, v in rec['per_area'].items()))
