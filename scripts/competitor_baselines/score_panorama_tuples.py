"""Multi-view panorama pose scoring on the 2D3DS panorama cases (inputs/2d3ds_panorama_tuples), with the
eval_2d3ds_pose_v2 aggregation (per-case RRA@30 / RTA@30 / AUC@30, then mean over cases), for
baseline ERP predictions (npz per case id with c2w). GT from 2D3DS pose json."""
import sys, json, os, argparse
from pathlib import Path
import numpy as np
S = os.environ['COMP_BASE_ROOT']
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_metrics import eval_case, auc_min, maa
PANO_TUPLES = f'{S}/inputs/2d3ds_panorama_tuples'
def load_gt_c2w(area, base):
    meta = json.load(open(f'{PANO_TUPLES}/stanford/{area}/pano/pose/{base}_pose.json'))
    rt = np.array(meta['camera_rt_matrix'], dtype=np.float64); w2c = np.eye(4); w2c[:3, :4] = rt
    return np.linalg.inv(w2c)
ap = argparse.ArgumentParser(); ap.add_argument('--preds', required=True); ap.add_argument('--out', required=True); ap.add_argument('--name', required=True); ap.add_argument('--conv', default='identity', choices=['identity', 'yflip'], help='camera convention of the baseline poses (identity for the published numbers)')
a = ap.parse_args()
C = np.diag([1., -1., 1.]) if a.conv == 'yflip' else np.eye(3); B = np.eye(4); B[:3, :3] = C.T
items = json.load(open(f'{S}/runs/lists/panorama_tuples.json'))
rra, rta, aucs, maas, recs, failed = [], [], [], [], [], []
for it in items:
    p = f"{a.preds}/{it['id']}.npz"
    if not os.path.exists(p): failed.append(it['id']); continue
    d = np.load(p); pred = [M @ B for M in d['c2w']]; gt = [load_gt_c2w(it['area'], b) for b in it['frames']]
    rerr, terr = eval_case(pred, gt)
    rra.append((rerr < 30).mean()); rta.append((terr < 30).mean()); auc, _ = auc_min(rerr, terr, 30); aucs.append(auc); maas.append(maa(rerr, terr, 30))
    recs.append(dict(id=it['id'], area=it['area'], n_views=len(gt), rra30=float(rra[-1]), rta30=float(rta[-1]), auc30=float(auc), maa30=float(maas[-1]), sec=float(d['sec'])))
res = {a.name: {'n_cases': len(recs), 'failed': failed, 'RRA@30': float(np.mean(rra) * 100), 'RTA@30': float(np.mean(rta) * 100), 'AUC@30': float(np.mean(aucs)), 'mAA@30_case_mean': float(np.mean(maas)), 'convention': a.conv, 'cases': recs}}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)
r = res[a.name]; print(f"{a.name} 2D3DS-pano cases={r['n_cases']} failed={len(failed)} RRA@30={r['RRA@30']:.2f} RTA@30={r['RTA@30']:.2f} AUC@30={r['AUC@30']:.2f} (conv={a.conv})")
