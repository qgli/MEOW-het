"""PanoVGGT (official model.pt, official inference.py recipe: cv2 INTER_AREA resize to
518x1036, bf16 autocast) predictor. Input: a JSON list [{id, images:[paths]}].
Output: one npz per id: c2w [S,4,4] (PanoVGGT frame, camera-to-world as used by its own
world_points = c2w @ local_points), depth [S,518,1036] f16, sec. OOM is recorded, not fatal."""
import sys, os, json, time, argparse
import numpy as np, torch
S = os.environ['COMP_BASE_ROOT']
P = f'{S}/competitors/PanoVGGT'
sys.path.insert(0, P); os.chdir(P)
from inference import load_model, run_inference
ap = argparse.ArgumentParser(); ap.add_argument('--list', required=True); ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
model = load_model('training/config/default.yaml', f'{S}/weights/panovggt/model.pt', 'cuda'); model.eval().to('cuda')
items = json.load(open(a.list))
if a.limit: items = items[:a.limit]
secs, fails = [], []
for it in items:
    outp = f"{a.out}/{it['id']}.npz"
    if os.path.exists(outp): continue
    try:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pr = run_inference(model, it['images'], 'cuda')
        torch.cuda.synchronize(); sec = time.perf_counter() - t0; secs.append(sec)
        depth = pr['depth']; depth = depth[..., 0] if depth.ndim == 4 else depth
        np.savez_compressed(outp, c2w=pr['camera_poses'].astype(np.float64), depth=depth.astype(np.float16), sec=sec, proc='panovggt 518x1036 inter_area bf16')
        print(f"{it['id']} S={len(it['images'])} {sec:.2f}s", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        fails.append(it['id']); print(f"{it['id']} OOM S={len(it['images'])}", flush=True); torch.cuda.empty_cache()
json.dump(dict(failed=fails, mean_sec_excl_first=(float(np.mean(secs[1:])) if len(secs) > 1 else None)), open(f'{a.out}/_run.json', 'w'))
print('DONE', len(items), 'failed', fails)
