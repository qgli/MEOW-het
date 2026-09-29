"""Score baseline predictions on the 88 cases of the heterogeneous 2D3DS benchmark (mixed
equirectangular / perspective / fisheye views, scripts/mp3d_benchmark/het_2d3ds_pose.py) with
the metric code of frozen_metrics.py (pooled pairs -> pose_metrics; ATE mean over cases), after
mapping the baseline's camera frames into the GT frames with a per-kind orthogonal C (from the
calibration probe, or a fixed --conv). Writes the same top-level fields as the het_2d3ds_pose.py JSON."""
import sys, json, os, argparse, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_metrics import eval_case, pose_metrics, ate_rmse, rel_pose, rot_angle_deg, trans_angle_deg
KMAP = {'erp': 'erp', 'persp': 'persp', 'pinhole': 'persp', 'fish': 'fish', 'fisheye': 'fish'}
CONV = {'identity': np.eye(3), 'yflip': np.diag([1., -1., 1.])}

def convert(c2w, kinds, Cs):
    out = []
    for M, k in zip(c2w, kinds):
        C = np.asarray(Cs[KMAP[k]], float); B = np.eye(4); B[:3, :3] = C.T  # c2w_O = c2w_T @ blkdiag(C^-1,1), C orthogonal
        out.append(M @ B)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preds', required=True); ap.add_argument('--cases-root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--probe', help='probe JSON (uses best C per kind)'); ap.add_argument('--conv', default=None, choices=[None, 'identity', 'yflip']); ap.add_argument('--name', default='competitor')
    a = ap.parse_args()
    if a.probe:
        b = json.load(open(a.probe))['best']; Cs = {'erp': b['C_erp'], 'persp': b['C_persp'], 'fish': b['C_fish']}; conv_desc = f'probe:{a.probe}'
    else:
        Cs = {k: CONV[a.conv or 'identity'].tolist() for k in ['erp', 'persp', 'fish']}; conv_desc = a.conv or 'identity'
    man = json.load(open(f'{a.cases_root}/manifest.json'))
    all_r, all_t, ates, recs, secs, missing = [], [], [], [], [], 0
    same_r, same_t, cross_r, cross_t = [], [], [], []
    for c in man['cases']:
        p = f"{a.preds}/case_{c['case_id']:03d}.npz"
        if not os.path.exists(p): missing += 1; continue
        d = np.load(p); gt = list(np.load(f"{a.cases_root}/{c['gt_c2w']}")); kinds = c['kinds']
        pred = convert(d['c2w'], kinds, Cs)
        rerr, terr = eval_case(pred, gt); ate = ate_rmse(pred, gt)
        all_r.append(rerr); all_t.append(terr); ates.append(ate); secs.append(float(d['sec']))
        # pair-type breakdown (same ordered-pair enumeration as eval_case)
        n = len(pred); idx = 0
        for i in range(n):
            for j in range(n):
                if i == j: continue
                (same_r if KMAP[kinds[i]] == KMAP[kinds[j]] else cross_r).append(rerr[idx]); (same_t if KMAP[kinds[i]] == KMAP[kinds[j]] else cross_t).append(terr[idx]); idx += 1
        recs.append(dict(area=c['area'], frames=c['frames'], kinds=kinds, rotation_errors=rerr.tolist(), translation_errors=terr.tolist(), ate=float(ate), sec=float(d['sec'])))
    R = np.concatenate(all_r); T = np.concatenate(all_t); m = pose_metrics(R, T)
    out = dict(model=a.name, convention=conv_desc, C=Cs, metrics=m, ate=float(np.nanmean(ates)), cases=len(recs), missing=missing, n_pairs=int(len(R)),
               sec_mean_excl_first=float(np.mean(secs[1:])) if len(secs) > 1 else None,
               by_pair_type=dict(same=pose_metrics(np.array(same_r), np.array(same_t)) if same_r else None, cross=pose_metrics(np.array(cross_r), np.array(cross_t)) if cross_r else None, n_same=len(same_r), n_cross=len(cross_r)),
               records=recs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, 'w'), indent=1)
    print(f"{a.name} conv={conv_desc} cases={len(recs)} missing={missing} pairs={len(R)}  RRA@30={m['RRA@30']:.2f} RTA@30={m['RTA@30']:.2f} mAA@30={m['mAA@30']:.2f} AUC@30={m['AUC@30']:.2f} ATE={out['ate']:.3f}  same-type mAA={out['by_pair_type']['same']['mAA@30']:.1f} cross-type mAA={out['by_pair_type']['cross']['mAA@30']:.1f}")
if __name__ == '__main__':
    main()
