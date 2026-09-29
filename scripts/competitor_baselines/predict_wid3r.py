"""Wid3R (official wid3r.bin, demo recipe: 518x336 LANCZOS resize, bf16 autocast,
per-view camera-type token) predictor for the benchmark inputs.
Modes: told  = kind -> {erp:Spherical, persp/pinhole:Pinhole, fish/fisheye:Fisheye624}
       pinhole = every view gets the same Pinhole token (no per-view type labels)
Inputs: --cases-root (case dirs as for the heterogeneous 2D3DS benchmark, images and kinds
from manifest.json), --list (JSON list of {id, images[, kinds]}) or --tuples (realset json).
Outputs: one npz per case/tuple: c2w [N,4,4] (Wid3R frame), kinds, sec, and with
--save-points: local_points [N,336,518,3] f16 (camera frame), uncertain [N,336,518] f16."""
import sys, os, glob, json, time, argparse
import numpy as np, torch
from PIL import Image
from torchvision import transforms
S = os.environ['COMP_BASE_ROOT']
sys.path.insert(0, f'{S}/competitors/Wid3R')
from wid3r.models.wid3r_training import Wid3R
from cam_utils.camera import Spherical, Fisheye624, Pinhole
TW, TH = 518, 336
KIND2CLS = {'erp': 'Spherical', 'persp': 'Pinhole', 'pinhole': 'Pinhole', 'fish': 'Fisheye624', 'fisheye': 'Fisheye624'}

def make_cam(cls):
    if cls == 'Spherical':
        return Spherical(params=torch.from_numpy(np.array([1., 1., 1., 1., TW, TH, np.pi, np.pi / 2.])))
    if cls == 'Pinhole':
        return Pinhole(params=torch.from_numpy(np.array([1, 1, TW, TH])))
    return Fisheye624(params=torch.from_numpy(np.zeros(16)).float())

def load_imgs(paths):
    tt = transforms.ToTensor(); out = []
    for p in paths:
        im = Image.open(p).convert('RGB').resize((TW, TH), Image.Resampling.LANCZOS)
        out.append(tt(im))
    return torch.stack(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases-root'); ap.add_argument('--tuples'); ap.add_argument('--frames-root'); ap.add_argument('--list')
    ap.add_argument('--out', required=True); ap.add_argument('--mode', default='told', choices=['told', 'pinhole'])
    ap.add_argument('--save-points', action='store_true'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Wid3R(pos_type="rope100", decoder_size="large", load_vggt=False, freeze_encoder=True, use_global_points=False, train_conf=False, num_dec_blk_not_to_checkpoint=4, ckpt=None, use_camera_gt=False)
    ck = torch.load(f'{S}/weights/wid3r/wid3r.bin', weights_only=False, map_location='cpu')
    print('load_state_dict:', model.load_state_dict(ck, strict=True)); del ck
    model = model.cuda().eval()
    items = []
    if a.cases_root:
        man = json.load(open(f'{a.cases_root}/manifest.json'))
        for c in man['cases']:
            paths = [f"{a.cases_root}/{p}" for p in c['images']]
            items.append((f"case_{c['case_id']:03d}", paths, c['kinds'], [Image.open(p).size for p in paths]))
    elif a.list:
        for it in json.load(open(a.list)):
            paths = it['images']; kinds = it.get('kinds', ['erp'] * len(paths))
            items.append((it['id'], paths, kinds, [Image.open(p).size for p in paths]))
    else:
        spec = json.load(open(a.tuples)); root = a.frames_root or spec['frames_root']
        for t in spec['tuples']:
            paths = [f"{root}/{v['img']}" for v in t['views']]
            items.append((t['id'], paths, [v['kind'] for v in t['views']], [(v['w'], v['h']) for v in t['views']]))
    if a.limit: items = items[:a.limit]
    secs = []
    for cid, paths, kinds, sizes in items:
        outp = f'{a.out}/{cid}.npz'
        if os.path.exists(outp): continue
        imgs = load_imgs(paths).cuda()
        clss = [KIND2CLS[k] if a.mode == 'told' else 'Pinhole' for k in kinds]
        cams = torch.cat([make_cam(c) for c in clss]).to('cuda')
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(imgs[None], cameras=cams)
        torch.cuda.synchronize(); sec = time.perf_counter() - t0; secs.append(sec)
        c2w = pred['camera_poses'][0].float().cpu().numpy().astype(np.float64)
        extra = {}
        if a.save_points:
            extra['local_points'] = pred['local_points'][0].float().cpu().numpy().astype(np.float16)
            extra['uncertain'] = pred['uncertain'][0, ..., 0].float().cpu().numpy().astype(np.float16)
        np.savez_compressed(outp, c2w=c2w, kinds=np.array(kinds), classes=np.array(clss), sizes=np.array(sizes), sec=sec, proc=f'wid3r {TW}x{TH} lanczos bf16 mode={a.mode}', **extra)
        print(f'{cid} N={len(paths)} {sec:.2f}s', flush=True)
    print(f'DONE {len(items)} items, mean sec (excl first) {np.mean(secs[1:]) if len(secs) > 1 else float("nan"):.3f}')
if __name__ == '__main__':
    main()
