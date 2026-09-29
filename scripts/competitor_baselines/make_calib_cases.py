"""Build a convention-calibration set disjoint from the heterogeneous 2D3DS benchmark
(2D3DS areas 1-4 only; the benchmark uses 5a/5b/6), with the benchmark's recipe
(scripts/mp3d_benchmark/het_2d3ds_pose.py build_het_group:
ERP 2048x1024 source -> erp view 1024x512, persp fov90 512^2, fish fov180 512^2,
centroid-facing synth views, GT = pano_c2w @ FLIP_Y R_pano_from_cam FLIP_Y, yflip=False).
Each room is emitted 3x with the kind cycle rotated so every camera-type pair occurs."""
import sys, json, math, os
from pathlib import Path
import numpy as np, cv2
S = os.environ['COMP_BASE_ROOT']
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mp3d_benchmark'))
from het_synth import perspective_view, fisheye_view
PANO_TUPLES = f'{S}/inputs/2d3ds_panorama_tuples'
OUT = f'{S}/inputs/calib_cases'
man = json.load(open(f'{PANO_TUPLES}/manifest.json'))

def load_gt_c2w(area, base):
    meta = json.load(open(f'{PANO_TUPLES}/stanford/{area}/pano/pose/{base}_pose.json'))
    rt = np.array(meta['camera_rt_matrix'], dtype=np.float64)
    w2c = np.eye(4); w2c[:3, :4] = rt
    return np.linalg.inv(w2c)

def _dir_to_yawpitch(d_world, pano_c2w):
    R = pano_c2w[:3, :3]
    dp = R.T @ d_world
    dp = dp / (np.linalg.norm(dp) + 1e-12)
    return math.atan2(dp[0], dp[2]), math.asin(np.clip(dp[1], -1, 1))

def _synth_view(erp, kind, yaw, pitch, size=512, persp_fov=90.0, fish_fov=180.0):
    if kind == 'erp':
        return cv2.resize(erp, (size * 2, size)), np.eye(3)
    if kind == 'persp':
        img, R, _ = perspective_view(erp, yaw, pitch, persp_fov, size); return img, R
    img, R, _ = fisheye_view(erp, yaw, pitch, fish_fov, size); return img, R

rooms, seen = [], set()
for c in man['cases']:
    if c['area'] in ('area_1', 'area_2', 'area_3', 'area_4') and len(c['frames']) >= 3 and (c['area'], c['room']) not in seen:
        seen.add((c['area'], c['room'])); rooms.append(c)
MODELS = ['erp', 'persp', 'fish']
cases, cid = [], 0
rng = np.random.default_rng(0)
os.makedirs(OUT, exist_ok=True)
for r in rooms:
    frames = r['frames'][:8]
    locs = {b: load_gt_c2w(r['area'], b) for b in frames}
    centroid = np.mean([locs[b][:3, 3] for b in frames], axis=0)
    erps = {}
    for b in frames:
        img = cv2.imread(f"{PANO_TUPLES}/stanford/{r['area']}/pano/rgb/{b}_rgb.png")
        erps[b] = cv2.resize(img, (2048, 1024))
    for off in range(3):
        models = MODELS[off:] + MODELS[:off]
        d = f'{OUT}/case_{cid:03d}'; os.makedirs(d, exist_ok=True)
        gts, kinds, imgs = [], [], []
        for k, b in enumerate(frames):
            pano_c2w = locs[b]; kind = models[k % 3]
            if kind in ('persp', 'fish'):
                dv = centroid - pano_c2w[:3, 3]
                if np.linalg.norm(dv) < 1e-6:
                    yaw, pitch = float(rng.uniform(-math.pi, math.pi)), 0.0
                else:
                    yaw, pitch = _dir_to_yawpitch(dv, pano_c2w)
            else:
                yaw = pitch = 0.0
            img, R = _synth_view(erps[b], kind, yaw, pitch)
            FLIP_Y = np.diag([1., -1., 1.])  # het_synth frame (+Y up) -> panorama pose frame (+Y down)
            gt = pano_c2w.copy(); gt[:3, :3] = pano_c2w[:3, :3] @ (FLIP_Y @ R @ FLIP_Y)
            p = f'{d}/{k:02d}_{kind}.png'; cv2.imwrite(p, img)
            gts.append(gt); kinds.append(kind); imgs.append(os.path.relpath(p, OUT))
        np.save(f'{d}/gt_c2w.npy', np.stack(gts))
        cases.append(dict(case_id=cid, area=r['area'], room=r['room'], frames=frames, kinds=kinds, images=imgs, gt_c2w=os.path.relpath(f'{d}/gt_c2w.npy', OUT), kind_offset=off))
        cid += 1
json.dump(dict(source='2d3ds_panorama_tuples stanford areas 1-4 (disjoint from the heterogeneous benchmark areas 5a/5b/6)', recipe='het_2d3ds_pose build_het_group: centroid, persp90, fish180, size512, erp 1024x512, yflip False', cases=cases), open(f'{OUT}/manifest.json', 'w'), indent=1)
print('rooms', len(rooms), 'cases', len(cases), 'views', sum(len(c['kinds']) for c in cases))
