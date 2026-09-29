# Data engine

The data engine (Lenscope in the paper, package `lenscope/`) renders indoor scenes with exact per-pixel rays and depth. Training tuples are not stored:
the online camera sampler (`mapanything/datasets/camera_sampler.py`) draws them from the stored renders
during training (see [TRAINING.md](TRAINING.md)). Two scene generations were used.

| | first generation | second generation |
|---|---|---|
| code | `lenscope/gen1/` | `lenscope/genesis/`, `lenscope/core/`, `lenscope/bpy/`, `lenscope/fixtures/` |
| scenes | procedural single rooms | procedural multi-room floor plans, CC0 materials, public-domain wall art |
| renders per scene | 8 poses x 8 cameras: five 1920x1080 pinholes, two 1080x1080 fisheyes, one 2160x1080 panorama, 64 samples per pixel | one 3072x1536 panorama per pose of the sampled pose graph (tens to over a hundred poses), 128 samples per pixel, denoised |
| ground truth | rays from the camera model, depth and mask from the renderer's depth pass | rays, depth, semantics, instances, material flags and normals by ray casting the scene mesh |
| covisibility | offline, per frame pair (`scripts/compute_covisibility.py`) | from the pose-graph sampler (ray-cast covisibility edges) |
| used in | Stages 1 and 2 | Stage 3 |
| split used in the paper | 1,603 train / 401 validation scenes (`resources/gen1_splits/`) | 469 / 15 / 15 (`resources/gen2_splits/`, `lenscope/genesis/make_splits.py`, seed 20260729) |

All randomness is seeded: a scene is determined by its seed, the code, the Blender version and, for the
second generation, the asset library (rendered RGB also carries the renderer's sampling noise). Blender
versions used for the checks in [REPRODUCIBILITY.md](REPRODUCIBILITY.md): 4.5.0 for the first generation,
4.1.1 for the second generation (the version the training scenes were built and rendered with).

## Pack format

Both generations write one directory per scene:

```
<root>/<scene>/
  <tag>_pack.npz          rgb (H,W,3) float16 in [0,1]; rays (H,W,3) float16, unit directions in the
                          camera frame (x right, y up, z forward); depth (H,W) float16, metres along the
                          ray; mask (H,W) uint8. Second-generation packs add sem, inst, flags, normal.
  metadata.json           frames: tag, base_name, base_type, pose_index, camera_to_world_unicol_4x4 (the
                          camera frame above, so the rotation has determinant -1), pack_file
  covisibility/v0/        covisibility.npy (N,N) float32 and frame_meta.json
```

The data loader (`mapanything/datasets/procthor_unicol.py`) converts the camera frame to OpenCV
(x right, y down, z forward) when it loads a view.

## First generation

Requirements: Blender 4.5 (the generator and renderer use only `bpy`, `bmesh`, `mathutils` and the
Python standard library) and, for the offline covisibility, the Python environment of
[INSTALL.md](INSTALL.md).

One scene (seed 0, written as `proc_scene_000000`):

```bash
export WORK=/path/to/work BLENDER=/path/to/blender-4.5/blender
"$BLENDER" -b --python lenscope/gen1/generate_scene.py -- \
    --seed 0 --output_dir "$WORK/gen1/scenes" --room_type random --n_scenes 1
python lenscope/gen1/make_pick.py --blend-dir "$WORK/gen1/scenes" \
    --scenes proc_scene_000000 --out "$WORK/gen1/pick.json"
"$BLENDER" -b --python lenscope/gen1/render_fair_focal.py -- \
    --pick "$WORK/gen1/pick.json" --out "$WORK/gen1/packs" --num_poses 8 --samples 64
python scripts/compute_covisibility.py --root "$WORK/gen1/packs" \
    --scenes proc_scene_000000 --device cuda:0 --overwrite
```

The first-generation data consists of the 2,004 scenes listed in `resources/gen1_splits/{train,val}.json`;
scene `proc_scene_NNNNNN` is generated from seed `NNNNNN`. Rendering one scene takes about nine minutes on
an RTX 3090.

The split files were written by `scripts/make_procthor_unicol_splits.py --root <renders>`
(seed 20260513, 20 % validation), where `<renders>/scenes` holds the scene folders.

Two further inputs of the training loader are derived from the renders:

- Per-frame statistics: rendered frames facing a flat wall (colour standard deviation below
  `min_rgb_std` = 0.05) are left out of the native-render tuples when the loader is given this file
  (`MEOW_GEN1_FRAME_STATS`, applied to every first-generation dataset; see [TRAINING.md](TRAINING.md)). The file
  used in training is `resources/gen1_frame_stats.json` (110 scenes); a new one is computed with
  ```bash
  python scripts/precompute_mask_frac.py --root "$WORK/gen1/packs" --out "$WORK/gen1/frame_stats.json" --workers 16
  ```
- Feasibility tables (first-generation scenes): the stratified native-render sampler (`n5_sampling_v2`)
  draws camera-type compositions from `resources/gen1_feasibility/` for first-generation scenes (Stage-2
  training and the validation sets of Stages 2 and 3). They were produced with
  ```bash
  python scripts/precompute_feasibility_map.py --data-root <renders>/scenes --splits-dir resources/gen1_splits \
      --splits train,val --k-list 2,3,4,5,6,7,8 --trials 30 --ma-thres 0.25 --seed 42 \
      --type-axis 3type --out feasibility_3type_full.parquet      # and --type-axis 6name
  ```
  The per-scene random stream depends on Python's string hashing, so a rerun reproduces the tables
  statistically (set `PYTHONHASHSEED` to make reruns repeatable); the tables in `resources/gen1_feasibility` are
  the ones the model was trained with.

## Second generation

Requirements: Blender 4.1.1 for scenes and renders identical to the training scenes (Blender 4.5 gives the
same geometry and ground truth, but newer defaults of the Bump shader node change the shading), the
Python environment of [INSTALL.md](INSTALL.md) with `embreex`, and optionally the asset library.

### Asset library (optional)

```bash
python lenscope/genesis/assets/fetch_assets.py --out /path/to/genesis-assets
python lenscope/genesis/assets/compute_set_stats.py --root /path/to/genesis-assets
export GENESIS_ASSETS=/path/to/genesis-assets
```

`fetch_assets.py` downloads the 935 CC0 and public-domain files listed with their provenance in
`lenscope/genesis/assets/manifest.json` (ambientCG materials, Poly Haven skies, The Metropolitan
Museum of Art Open Access images; about 0.5 GB) and checks each size. The museum has re-encoded a few
images since the library was assembled; the fetcher reports them, and scenes that use them differ only
in those pixels. `compute_set_stats.py` must run once with a Python that has Pillow: Blender's bundled
Python does not ship Pillow, and without the precomputed statistics the scene builder falls back to
constants and chooses different materials. Without a library (`GENESIS_ASSETS` unset) the builder uses
procedural materials only.

### One scene end to end

```bash
export BLENDER=/path/to/blender-4.1.1/blender
"$BLENDER" -b --python lenscope/genesis/build_scene.py -- --seed 1000 --out "$WORK/genesis/scenes"
"$BLENDER" -b "$WORK/genesis/scenes/genesis_001000.blend" \
    --python lenscope/fixtures/export_fixture.py -- --out "$WORK/genesis/fixtures"
python lenscope/sample_pose_graphs.py "$WORK/genesis/fixtures" "$WORK/genesis/samples" --only genesis_001000
GENESIS_RENDER_QUALITY=1 GENESIS_CLAMP_INDIRECT=100 GENESIS_HDR_SIDECAR=1 \
"$BLENDER" -b "$WORK/genesis/scenes/genesis_001000.blend" --python lenscope/bpy/render_v2.py -- \
    --poses "$WORK/genesis/samples/genesis_001000_sample.json" \
    --out "$WORK/genesis/render/genesis_001000" --erp-w 3072 --samples 128 --pins '' --exposure 0
python lenscope/bridge_v1_shard.py "$WORK/genesis/fixtures" "$WORK/genesis/samples" \
    "$WORK/genesis/render" "$WORK/genesis/shards" --W 3072 --only genesis_001000
```

`scripts/repro/subset_pose_graph.py <sample.json> <out.json> --count 4` keeps a connected subset of
the pose graph for a quick check (four panoramas instead of the full graph).

Render settings: `GENESIS_RENDER_QUALITY=1` selects the light-path profile of the training scenes,
`GENESIS_CLAMP_INDIRECT` the indirect clamp (100 for the training scenes), `GENESIS_HDR_SIDECAR=1`
writes a linear EXR next to every panorama, `GENESIS_RENDER_THREADS` limits CPU threads when several
renders share a machine. `GENESIS_PACK_COMPRESS=1` makes the bridge write compressed packs.

### Full dataset

```bash
for s in $(seq 1000 1499); do
  "$BLENDER" -b --python lenscope/genesis/build_scene.py -- --seed $s --out "$WORK/genesis/scenes"
done
GENESIS_RENDER_QUALITY=1 GENESIS_CLAMP_INDIRECT=100 GENESIS_HDR_SIDECAR=1 \
python lenscope/run_production.py --blend-dir "$WORK/genesis/scenes" --work-root "$WORK/genesis" \
    --erp-w 3072 --samples 128 --n 500 --blender "$BLENDER"
python lenscope/verify_batch_complete.py "$WORK/genesis/scenes" "$WORK/genesis"
python lenscope/genesis/make_splits.py --scenes-dir "$WORK/genesis/shards/scenes" \
    --out-dir "$WORK/genesis/shards/splits" --seed 20260729 --val-count 15 --test-count 15
```

`run_production.py` resumes after interruptions, skips completed scenes and keeps a manifest. The
second-generation set of the paper contains the 499 scenes that completed the chain (seeds 1000-1499 except
1474); `make_splits.py` draws its 469/15/15 split from that list (seed 20260729) and writes the files of
`resources/gen2_splits/` byte for byte. Its panoramas took 90-120 s per pose on an RTX 5090, about
37,000 poses in total. The scene export, the renderer and the pose-graph sampler log JSON event lines with
the prefix `AGEN ` (per stage and per pose, on stdout for the Blender scripts and on stderr
for the sampler), so `grep '^AGEN '` extracts the event log of a run.
`run_production.py` sets `AGEN_NO_THUMB=1`, which skips the top-down thumbnail of the scene export (its
OpenGL context can abort Blender on machines without a display).

### Tests

```bash
pytest -q lenscope/tests                 # ray casting, pose-graph sampler, floor plans, layout rules
```

## Laser-scan benchmark scenes

`lenscope/blk/` converts registered Leica BLK360 G2 stations (per-station E57 clouds with their
registered poses, and the station panoramas) into packs of the same format: `blk2pack.py` does the
conversion (requires `pye57`), `validate_blk_scene.py` runs five consistency checks, `analyze_holes.py`
measures the regions the scanner does not see, and `ingest_blk_scene.py` runs all three on a capture and
registers the scene in a library directory. The benchmark tuples are then drawn by
`scripts/blk_bench/make_blk_bench.py`; see [BENCHMARKS.md](BENCHMARKS.md).
