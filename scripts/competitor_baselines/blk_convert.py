"""Convert baseline (Wid3R / PanoVGGT) BLK predictions into the realset pred contract
(scripts/realset/common.py: c2w [V,4,4], pts [M,3] world, uv [M,2] original fed-image pixels,
vidx [M], proc, sec), mirroring scripts/realset/predict_vggt.py's sampling policy:
stride-4 pixel grid at the processed resolution, per-view confidence >= median (when the
model gives a confidence), finite points, in-bounds uv, cap 150k with rng seed 0.
Frame conventions: the defaults are the ones of the published laser-benchmark numbers, identity camera
convention (--pose-conv identity) and no reflection of the predicted world (the baseline's world and the
laser world have the same handedness). The alternatives, --pose-conv yflip (c2w_O = c2w_T blkdiag(C^-1,1))
and --world-reflect (c2w_O = W c2w_O, x_O = W x_T), with C = W = diag(1,-1,1), are kept for convention
probes."""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'realset'))
from common import save_pred_npz, grid_uv
C = np.diag([1., -1., 1.]); Wr = np.diag([1., -1., 1.])
BC = np.eye(4); BC[:3, :3] = C.T
BW = np.eye(4); BW[:3, :3] = Wr

def panovggt_dirs(H, W):
    u = np.arange(W) + 0.5; v = np.arange(H) + 0.5
    phi = (u / W - 0.5) * 2 * np.pi; theta = -(v / H - 0.5) * np.pi
    gt, gp = np.meshgrid(theta, phi, indexing='ij')
    return np.stack([np.cos(gt) * np.sin(gp), -np.sin(gt), np.cos(gt) * np.cos(gp)], -1)

ap = argparse.ArgumentParser(); ap.add_argument('--backend', required=True, choices=['wid3r', 'panovggt']); ap.add_argument('--preds', required=True); ap.add_argument('--list', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--world-reflect', action='store_true', help='reflect the predicted world by diag(1,-1,1)')
ap.add_argument('--no-world-reflect', dest='world_reflect', action='store_false', help='no world reflection (default)')
ap.add_argument('--pose-conv', default='identity', choices=['identity', 'yflip'])
ap.add_argument('--pt-stride', type=int, default=4); ap.add_argument('--pt-cap', type=int, default=150_000); ap.add_argument('--no-conf-filter', action='store_true')
a = ap.parse_args()
if not a.world_reflect: BW = np.eye(4); Wr = np.eye(3)
if a.pose_conv == 'identity': BC = np.eye(4)
items = json.load(open(a.list)); rng = np.random.default_rng(0); os.makedirs(a.out, exist_ok=True)
for it in items:
    p = f"{a.preds}/{it['id']}.npz"
    if not os.path.exists(p): print('missing', it['id']); continue
    d = np.load(p); c2w_T = d['c2w']; V = len(c2w_T)
    from PIL import Image
    wh = [Image.open(pth).size for pth in it['images']]
    if a.backend == 'wid3r':
        lp = d['local_points'].astype(np.float64); conf = -d['uncertain'].astype(np.float64)   # (V,336,518,3), (V,336,518)
    else:
        dep = d['depth'].astype(np.float64); lp = panovggt_dirs(*dep.shape[1:])[None] * dep[..., None]; conf = None
    Hp, Wp = lp.shape[1:3]
    c2w_O, pts_l, uv_l, vidx_l = [], [], [], []
    uvp = grid_uv(Hp, Wp, a.pt_stride); ii = uvp[:, 1].astype(int); jj = uvp[:, 0].astype(int)
    for i in range(V):
        T = c2w_T[i]
        c2w_O.append(BW @ T @ BC)
        pv_local = lp[i, ii, jj]                                   # competitor camera frame
        pv_world = (T[:3, :3] @ pv_local.T).T + T[:3, 3]           # competitor world
        pv = (Wr @ pv_world.T).T                                    # GT (reflected) world
        good = np.isfinite(pv).all(1) & (np.linalg.norm(pv_local, axis=1) > 0)
        if conf is not None and not a.no_conf_filter:
            cv = conf[i, ii, jj]; good &= cv >= np.median(cv)
        w0, h0 = wh[i]
        uo = np.stack([(uvp[good, 0] + 0.5) * (w0 / Wp) - 0.5, (uvp[good, 1] + 0.5) * (h0 / Hp) - 0.5], 1)
        inb = (uo[:, 0] >= 0) & (uo[:, 0] < w0) & (uo[:, 1] >= 0) & (uo[:, 1] < h0)
        pts_l.append(pv[good][inb]); uv_l.append(uo[inb]); vidx_l.append(np.full(inb.sum(), i, np.int32))
    pts = np.concatenate(pts_l); uv = np.concatenate(uv_l); vidx = np.concatenate(vidx_l)
    if len(pts) > a.pt_cap:
        idx = rng.choice(len(pts), a.pt_cap, replace=False); pts, uv, vidx = pts[idx], uv[idx], vidx[idx]
    save_pred_npz(Path(f"{a.out}/{it['id']}.npz"), np.stack(c2w_O), pts, uv, vidx, proc=f"{a.backend} {Hp}x{Wp} stride{a.pt_stride} pose_conv={a.pose_conv} world_reflect={a.world_reflect} {str(d['proc'])}", sec=float(d['sec']))
print('converted', len(items), '->', a.out)
