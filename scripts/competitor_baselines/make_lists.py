import json, os
S = os.environ['COMP_BASE_ROOT']
os.makedirs(f'{S}/runs/lists', exist_ok=True)
PANO_TUPLES = f'{S}/inputs/2d3ds_panorama_tuples'; m = json.load(open(f'{PANO_TUPLES}/manifest.json'))
json.dump([dict(id=f"case_{c['case_id']:03d}", area=c['area'], frames=c['frames'], images=[f"{PANO_TUPLES}/stanford/{c['area']}/pano/rgb/{b}_rgb.png" for b in c['frames']]) for c in m['cases']], open(f'{S}/runs/lists/panorama_tuples.json', 'w'), indent=1)
SINGLE_PANOS = f'{S}/inputs/2d3ds_single_panoramas'; t = json.load(open(f'{SINGLE_PANOS}/manifest.json'))
json.dump([dict(id=f'{area}__{b}', area=area, base=b, images=[f'{SINGLE_PANOS}/stanford/{area}/pano/rgb/{b}_rgb.png']) for area, bs in t['frames'].items() for b in bs], open(f'{S}/runs/lists/single_panoramas.json', 'w'), indent=1)
LASER_V1 = f'{S}/inputs/laser_benchmark/v1'
for track in ['blk_erp', 'blk_mixed', 'blk_pinhole', 'blk_fisheye']:
    spec = json.load(open(f'{LASER_V1}/{track}/tuples.json'))
    json.dump([dict(id=tp['id'], images=[f"{LASER_V1}/{track}/{v['img']}" for v in tp['views']], kinds=[v['kind'] for v in tp['views']]) for tp in spec['tuples']], open(f'{S}/runs/lists/{track}.json', 'w'), indent=1)
LASER_V11 = f'{S}/inputs/laser_benchmark/v1.1'  # v1.1 envelope tracks, read by run_chain2.sh
for track in ['blk_erp', 'blk_mixed']:
    spec = json.load(open(f'{LASER_V11}/{track}/tuples.json'))
    json.dump([dict(id=tp['id'], images=[f"{LASER_V11}/{track}/{v['img']}" for v in tp['views']], kinds=[v['kind'] for v in tp['views']]) for tp in spec['tuples']], open(f'{S}/runs/lists/v11_{track}.json', 'w'), indent=1)
for f in sorted(os.listdir(f'{S}/runs/lists')): print(f, len(json.load(open(f'{S}/runs/lists/{f}'))))
