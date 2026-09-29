"""Same signed-permutation probe as probe_convention.py but for realset/BLK tuples
(kinds from the list JSON, GT c2w from gt npz, competitor c2w from raw preds npz)."""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_convention import signed_perms, rel, KMAP, KINDS
ap=argparse.ArgumentParser(); ap.add_argument('--preds',required=True); ap.add_argument('--list',required=True); ap.add_argument('--gt-dir',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
C=signed_perms(); nC=len(C); items=json.load(open(a.list)); thrs=np.arange(1,31)
pairs={(x,y):dict(RT=[],RG=[],tT=[],tG=[]) for x in KINDS for y in KINDS}
for it in items:
    d=np.load(f"{a.preds}/{it['id']}.npz"); g=np.load(f"{a.gt_dir}/{it['id']}.npz")['c2w']; c2w=d['c2w']; kinds=[KMAP[k] for k in it['kinds']]
    for i in range(len(kinds)):
        for j in range(len(kinds)):
            if i==j: continue
            RT,tT=rel(c2w[i],c2w[j]); RG,tG=rel(g[i],g[j]); b=pairs[(kinds[i],kinds[j])]; b['RT'].append(RT); b['RG'].append(RG); b['tT'].append(tT); b['tG'].append(tG)
tables,counts={},{}
for key,b in pairs.items():
    if not b['RT']: continue
    RT=np.stack(b['RT']); RG=np.stack(b['RG']); tT=np.stack(b['tT']); tG=np.stack(b['tG']); P=len(RT); counts[key]=P
    Rp=np.einsum('bij,pjk->pbik',C,RT); Rp=np.einsum('pbik,alk->pbail',Rp,C); tr=np.einsum('pji,pbaji->pba',RG,Rp); rerr=np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
    tp=np.einsum('bij,pj->pbi',C,tT); na=np.linalg.norm(tp,axis=-1); nb=np.linalg.norm(tG,axis=-1)[:,None]; cos2=(np.einsum('pbi,pi->pb',tp,tG)/np.maximum(na*nb,1e-30))**2
    terr=np.degrees(np.arccos(np.clip(np.sqrt(np.clip(1-np.clip(1-cos2,0,None)+1e-15,0,1)),-1,1))); terr=np.where((na<1e-9)|(nb<1e-9),90.0,terr)
    m=np.maximum(rerr,terr[:,:,None]); tables[key]=np.mean([(m<t).mean(0) for t in thrs],axis=0)
N=sum(counts.values()); total=np.zeros((nC,nC,nC)); ax={'erp':0,'persp':1,'fish':2}
for (ka,kb),sc in tables.items():
    w=counts[(ka,kb)]/N; ia,ib=ax[ka],ax[kb]
    if ia==ib: shape=[1,1,1]; shape[ia]=nC; total+=w*np.diagonal(sc).reshape(shape)
    else:
        t=sc.T; shape=[1,1,1]; shape[ia]=nC; shape[ib]=nC; total+=w*(t if ia<ib else t.T).reshape(shape)
best=np.unravel_index(np.argmax(total),total.shape); ident=[i for i in range(nC) if np.allclose(C[i],np.eye(3))][0]
res=dict(n_pairs=N,pair_counts={f'{k[0]}->{k[1]}':v for k,v in counts.items()},best_score=float(total[best]),identity_score=float(total[ident,ident,ident]),
         best=dict(C_erp=C[best[0]].tolist(),C_persp=C[best[1]].tolist(),C_fish=C[best[2]].tolist()),n_within_1pt=int((total>total[best]-0.01).sum()))
json.dump(res,open(a.out,'w'),indent=1); print(json.dumps(res,indent=1))
