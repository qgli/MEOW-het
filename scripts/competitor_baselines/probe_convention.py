"""Data-driven camera-frame convention probe (its output feeds score_het2d3ds.py --probe): the
convention is chosen from data, on a calibration split disjoint from the evaluation cases.
Model: GT-frame camera coords p_O = C_k p_T for a baseline camera of kind k, C_k in the
48 signed permutation matrices. Then rel_O(i->j) rotation = C_kj R_T C_ki^T, translation =
C_kj t_T (rel_pose convention T = inv(c2w_j) @ c2w_i as in frozen_metrics). The probe picks
the (C_erp, C_persp, C_fish) maximising the mAA-style score over all ordered pairs of the
calibration cases, and reports margins + the identity / all-y-flip scores."""
import json, os, itertools, argparse
import numpy as np
KMAP = {'erp': 'erp', 'persp': 'persp', 'pinhole': 'persp', 'fish': 'fish', 'fisheye': 'fish'}
KINDS = ['erp', 'persp', 'fish']

def signed_perms():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for r, c in enumerate(perm): M[r, c] = signs[r]
            out.append(M)
    return np.stack(out)  # (48,3,3)

def rel(c2w_i, c2w_j):
    T = np.linalg.inv(c2w_j) @ c2w_i
    return T[:3, :3], T[:3, 3]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preds', required=True); ap.add_argument('--cases-root', required=True); ap.add_argument('--out', required=True); ap.add_argument('--thr', type=int, default=30)
    a = ap.parse_args()
    C = signed_perms(); nC = len(C)
    man = json.load(open(f'{a.cases_root}/manifest.json'))
    pairs = {(x, y): dict(RT=[], RG=[], tT=[], tG=[]) for x in KINDS for y in KINDS}
    ncase = 0
    for c in man['cases']:
        p = f"{a.preds}/case_{c['case_id']:03d}.npz"
        if not os.path.exists(p): continue
        d = np.load(p); c2w = d['c2w']; gt = np.load(f"{a.cases_root}/{c['gt_c2w']}")
        kinds = [KMAP[k] for k in c['kinds']]; n = len(kinds); ncase += 1
        for i in range(n):
            for j in range(n):
                if i == j: continue
                RT, tT = rel(c2w[i], c2w[j]); RG, tG = rel(gt[i], gt[j])
                b = pairs[(kinds[i], kinds[j])]; b['RT'].append(RT); b['RG'].append(RG); b['tT'].append(tT); b['tG'].append(tG)
    thrs = np.arange(1, a.thr + 1)
    tables, counts = {}, {}
    for key, b in pairs.items():
        if not b['RT']: continue
        RT = np.stack(b['RT']); RG = np.stack(b['RG']); tT = np.stack(b['tT']); tG = np.stack(b['tG'])
        P = len(RT); counts[key] = P
        # rotation: R' = C[cb] @ RT @ C[ca]^T  -> (P, cb, ca, 3, 3)
        Rp = np.einsum('bij,pjk->pbik', C, RT)                     # (P,48,3,3)
        Rp = np.einsum('pbik,alk->pbail', Rp, C)                   # (P,48,48,3,3)  (C[ca]^T)
        tr = np.einsum('pji,pbaji->pba', RG, Rp)                   # trace(RG^T R')
        rerr = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))  # (P,48,48)
        # translation: t' = C[cb] @ tT, sign-agnostic angle vs tG (frozen_metrics.trans_angle_deg)
        tp = np.einsum('bij,pj->pbi', C, tT)                        # (P,48,3)
        na = np.linalg.norm(tp, axis=-1); nb = np.linalg.norm(tG, axis=-1)[:, None]
        cos2 = (np.einsum('pbi,pi->pb', tp, tG) / np.maximum(na * nb, 1e-30)) ** 2
        terr = np.degrees(np.arccos(np.clip(np.sqrt(np.clip(1 - np.clip(1 - cos2, 0, None) + 1e-15, 0, 1)), -1, 1)))
        terr = np.where((na < 1e-9) | (nb < 1e-9), 90.0, terr)     # (P,48)
        m = np.maximum(rerr, terr[:, :, None])                      # (P,cb,ca)
        score = np.mean([(m < t).mean(0) for t in thrs], axis=0)    # mAA-like (48,48)
        tables[key] = score
    N = sum(counts.values())
    total = np.zeros((nC, nC, nC))  # index order (C_erp, C_persp, C_fish)
    ax = {'erp': 0, 'persp': 1, 'fish': 2}
    for (ka, kb), sc in tables.items():
        w = counts[(ka, kb)] / N
        # sc[cb, ca]; expand to 3D
        ia, ib = ax[ka], ax[kb]
        if ia == ib:
            diag = np.diagonal(sc)  # ca == cb
            shape = [1, 1, 1]; shape[ia] = nC
            total += w * diag.reshape(shape)
        else:
            # want total[..., c_a at ia, c_b at ib] += sc[cb, ca]
            t = sc.T  # (ca, cb)
            shape = [1, 1, 1]; shape[ia] = nC; shape[ib] = nC
            if ia < ib: total += w * t.reshape(shape)
            else: total += w * t.T.reshape(shape)
    best = np.unravel_index(np.argmax(total), total.shape); bs = total[best]
    flat = np.sort(total.ravel())[::-1]
    ident = [i for i in range(nC) if np.allclose(C[i], np.eye(3))][0]
    yflip = [i for i in range(nC) if np.allclose(C[i], np.diag([1, -1, 1]))][0]
    res = dict(n_cases=ncase, n_pairs=N, pair_counts={f'{k[0]}->{k[1]}': v for k, v in counts.items()},
               best=dict(score=float(bs), C_erp=C[best[0]].tolist(), C_persp=C[best[1]].tolist(), C_fish=C[best[2]].tolist(), dets=[float(np.linalg.det(C[b])) for b in best]),
               second_best_score=float(flat[1]), identity_score=float(total[ident, ident, ident]), yflip_all_score=float(total[yflip, yflip, yflip]),
               per_kind_best_marginal={k: float(total.max(axis=tuple(j for j in range(3) if j != i)).max()) for k, i in ax.items()})
    # how unique: list all combos within 0.5 pt of best
    close = np.argwhere(total > bs - 0.005)
    res['n_combos_within_0.5pt'] = int(len(close))
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'best'}, indent=1)); print('best', json.dumps(res['best']))
if __name__ == '__main__':
    main()
