# Evaluation sets

Definition files of the evaluation sets of the paper: which frames form each tuple, the camera kind of
every view and the parameters that rebuild the set. They contain no pixels, no depth and no ground-truth
poses.

| folder | set | paper |
|---|---|---|
| `heterogeneous_2d3ds_88/` | 88 heterogeneous tuples built from Stanford 2D-3D-Semantics panoramas | Tables 2, 4, 6, 8, 10, 14; Appendix F |
| `2d3ds_panorama_tuples_19/` | 19 same-room tuples of 2D3DS panoramas | Tables 4 (16:9 column), 9, 10, 11 |
| `2d3ds_single_panoramas_40/` | 40 single 2D3DS panoramas | Table 9 |
| `laser_benchmark/v1/` | laser-scanned benchmark, four tracks of 24 tuples | Tables 3, 4, 6, 8, 10, 13; Appendix F |
| `laser_benchmark/v1.1/` | laser benchmark tuples inside the covisibility envelope | Table 8 (caption) |

## Stanford 2D-3D-Semantics sets

The 2D3DS images, depth and poses are not redistributed. Download Stanford 2D-3D-Semantics from its
authors (the panorama folders `area_*/pano/{rgb,depth,pose}` are used) and point the evaluators at it
(`MEOW_2D3DS_ROOT`; `eval_2d3ds_pose_v2.py` and `scripts/realset/tuples_from_2d3ds.py` also accept
`--stanford-root`). Frame names below are the 2D3DS panorama names
(`camera_<id>_<room>_frame_equirectangular_domain`).

### `heterogeneous_2d3ds_88/manifest.json`

88 tuples from areas 5a, 5b and 6: one tuple per room (the room label of the 2D3DS pose files) with at
least three panoramas, holding all panoramas of the room (a seeded subset of 24 for the rooms with more),
3 to 24 views, 503 views in total (189 panoramas, 163 perspective, 151 fisheye views). The views of a
tuple cycle through the kinds panorama, perspective, fisheye in the order of the frame list.

- `protocol`: the arguments of `scripts/mp3d_benchmark/het_2d3ds_pose.py` that define the set.
- `cases[i]`: `case_id`, `area`, `frames` (2D3DS panorama of each view), `kinds` (`erp`, `persp` or
  `fish`), and `images` / `gt_c2w`, the file names of the exported views and ground-truth poses.

Each view is synthesized from its panorama by `scripts/mp3d_benchmark/het_synth.py`: panoramas are
resized to 1024 x 512; perspective views are 512 x 512 pinhole crops with a 90-degree field of view;
fisheye views are 512 x 512 equidistant crops with a 180-degree field of view (black outside the image
circle). Perspective and fisheye views look from their panorama's centre toward the centroid of the
tuple's panorama centres, so their orientation follows from the 2D3DS poses; the ground-truth pose of a
synthesized view is the panorama pose composed with the view rotation, both in the +Y-down camera frame of
the 2D3DS poses. The subsets of the largest rooms and the order of the cases are drawn with seed 0.

```bash
python scripts/mp3d_benchmark/het_2d3ds_pose.py --areas area_5a area_5b area_6 --models erp,persp,fish \
    --min-views 3 --all-views --cap-views 24 --max-cases 300 --seed 0 \
    --export-dir <cases> --export-only        # writes <cases>/manifest.json, cases/case_NNN/*.png, gt_c2w.npy
python - <<'EOF'
import json
a = json.load(open("<cases>/manifest.json"))["cases"]
b = json.load(open("benchmarks/heterogeneous_2d3ds_88/manifest.json"))["cases"]
assert [(c["area"], c["frames"], c["kinds"]) for c in a] == [(c["area"], c["frames"], c["kinds"]) for c in b]
print("same 88 tuples")
EOF
```

The evaluator itself rebuilds the views on the fly; the export is needed only for models that run in
their own environments (Wid3R) and for `scripts/detector_audit.py` and `scripts/scale_ratio.py`.

### `2d3ds_panorama_tuples_19/`

19 tuples of 2 to 13 panoramas from one room each, from all seven areas (areas 1 to 4, 5a, 5b and 6; 81
views). The sampling follows the description of the 2D3DS protocol in the Wid3R paper (Wid3R's released code
instead evaluates a fixed list of 20 tuples from areas 5a and 5b): per room, a random subset of 10 to 30
panoramas (all of them for smaller rooms), ten draws per room, then shuffled and truncated to 20 draws with
seed 0. A room with fewer than ten panoramas yields the same tuple in every draw; one such tuple was drawn
twice (cases 0 and 4) and is kept once, so case ids keep the draw index (case 4 is absent).
`eval_2d3ds_pose_v2.py --wid3r-faithful` keeps the first of repeated draws by default; `--keep-duplicates`
evaluates all 20 draws.

- `manifest.json`: `protocol` and, per case, `case_id`, `area`, `room`, `frames`.
- `tuples.json`: the same cases in the format of the real-image harness (`scripts/realset`), tuple
  `s2d3ds_NNN` = case NNN, with `frames_root` set to the placeholder `<2D3DS_ROOT>` (replace it or pass
  `--frames-root`).

```bash
python scripts/eval_2d3ds_pose_v2.py --ckpts MEOW=<checkpoint>:wrap --wid3r-faithful \
    --areas area_1 area_2 area_3 area_4 area_5a area_5b area_6 --max-cases 20 --seed 0 --ar-mode auto --out <dir>
python scripts/realset/tuples_from_2d3ds.py --areas area_1 area_2 area_3 area_4 area_5a area_5b area_6 \
    --seed 0 --max-cases 20 --out <dir>      # tuples.json and ground-truth poses for the baselines
```

### `2d3ds_single_panoramas_40/manifest.json`

40 panoramas, 20 from area 5a and 20 from area 5b (`frames` per area), drawn with seed 0 by
`scripts/eval_2d3ds_panorama_v2.py --areas area_5a area_5b --num-frames 20 --seed 0`. This list is the
export used for the Wid3R and PanoVGGT runs of Table 9.

## Laser-scanned benchmark

Registered Leica BLK360 G2 scans of a multi-room office (a corridor chain and a meeting room), 12
stations; every view is resampled from a station panorama through the stated camera model, and the
ground truth (poses and pointmaps from the laser depth) is exact. **The frames and ground truth are
released as a separate data archive, not in this repository.** The files here are the tuple definitions
and construction reports of that archive; `frames_root` is the placeholder `<LASER_BENCHMARK>/<version>/<track>`:
point it (or `--frames-root`) at the track folder of the archive, which contains `frames/`, `gt/` and
`tuples.json`.

| version | track | tuples | views per tuple | view kinds and sizes |
|---|---|---|---|---|
| v1 | `blk_mixed` | 24 | 4 | one panorama 1024 x 512, one fisheye 720 x 720, two pinhole 640 x 480 |
| v1 | `blk_erp` | 24 | 4 | panoramas 1024 x 512 |
| v1 | `blk_pinhole` | 24 | 4 | pinhole 640 x 480 (75-degree horizontal field of view) |
| v1 | `blk_fisheye` | 24 | 4 | equidistant fisheye 720 x 720 (170 degrees) |
| v1.1 envelope | `blk_erp` | 16 | 4 | panoramas 1024 x 512 |
| v1.1 envelope | `blk_mixed` | 13 | 4 | as in v1 |

`tuples.json` lists per tuple the view images (`img`, width `w`, height `h`), the camera `kind`, the
scanner `station` of each view and the ground-truth file (`gt/<tuple id>.npz`: camera-to-world poses,
per-view world points and the fused ground-truth cloud). `BENCH_REPORT.json` records the seed, the
convention check of the builder, the valid elevation band of the scans and, per tuple, the station chain
and the covisibility of its consecutive views.

Every tuple is a chain of stations whose consecutive views pass a covisibility test computed from
the laser ground truth: the fraction of one view's valid points that project into the other view with
a depth difference below the larger of 10 cm and 3 % of the depth, estimated from 1,500 sampled points
per direction (`covis_pair` in `make_blk_bench.py`) and symmetrized by the minimum. v1 gates consecutive views at 0.25 (pinhole and fisheye tracks), 0.10
(panorama track) and 0.15 (mixed track). v1.1 adds a floor of 0.10 on every view pair of a tuple (`--allpairs-gate`), with
consecutive gates of 0.25 (panorama track) and 0.20 (mixed track) and seed 1; 16 panorama and 13 mixed
tuples pass.

```bash
python scripts/blk_bench/make_blk_bench.py --scene-dir <station packs> --out <out>/v1 --tuples-per-track 24 --seed 0
pytest -q scripts/blk_bench/test_conventions.py      # camera conventions of the construction
```

The command line of the v1.1 build was not recorded in its output: the seed is in its report, and the
gates above are those of the builder's envelope mode (`--allpairs-gate 0.10`, panorama and mixed tracks
only).

## Other sets

- Matterport3D: the 18 official test scans listed in `scripts/mp3d_benchmark/eval_mp3d_panoramas.py`, eight
  panoramas per tuple around a random anchor, drawn with probabilities that decrease with the distance to
  the anchor (sampling in `scripts/mp3d_benchmark/covis_sample.py`; `eval_mp3d_panoramas.py --k 8 --seed 0`);
  download Matterport3D from its authors.
- Replica and Aria Digital Twin: built by `scripts/realset/replica_prepare.py` and
  `scripts/realset/adt_prepare.py` (48 Replica tuples, 80 ADT tuples per camera, eight views each);
  download the datasets from their authors.
