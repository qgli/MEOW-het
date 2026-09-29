# Evaluation sets

Commands and settings of every evaluation are in [EVALUATION.md](EVALUATION.md); the definition files
(frame lists and tuples, no pixels) are in [`benchmarks/`](../benchmarks/README.md).

## Heterogeneous 2D3DS tuples (88 tuples, 503 views)

Built from the Stanford 2D-3D-Semantics panoramas of areas 5a, 5b and 6 by
`scripts/mp3d_benchmark/het_2d3ds_pose.py` (seed 0): one tuple per room with at least three panoramas,
all of the room's panoramas up to 24 (a seeded subset beyond that), view kinds cycling panorama,
perspective (90-degree field of view) and fisheye (180 degrees). Perspective and fisheye views
(512 px) are resampled from their panoramas, oriented toward the centroid of the tuple's camera
positions; ground truth comes from the 2D3DS camera poses. The 503 views are 189 panoramas, 163
perspective and 151 fisheye views; tuples have 3 to 24 views. The calibration cases of the baseline
camera conventions come from areas 1-4 (`scripts/competitor_baselines/make_calib_cases.py`) and do not
overlap. `--export-dir` writes the synthesized views and poses for models evaluated outside the script.

## Laser-scan benchmark

Twelve Leica BLK360 G2 stations of a multi-room office with registered poses and point clouds, converted
into packs by `lenscope/blk/` and cut into tuples of four views by `scripts/blk_bench/make_blk_bench.py`
(`--tuples-per-track 24 --seed 0`):

| track | views | resolution |
|---|---|---|
| panorama (`blk_erp`) | 4 full panoramas | 1024 x 512 |
| pinhole (`blk_pinhole`) | 4 pinhole views | 640 x 480 |
| fisheye (`blk_fisheye`) | 4 fisheye views | 720 x 720 |
| mixed (`blk_mixed`) | 1 panorama, 1 fisheye, 2 pinhole views | as above |

Views are resampled from the station scans, so each carries exact rays and metric ground truth; regions
the scanner does not see are masked. Tuples are chained by covisibility computed from the laser ground
truth (on 1,500 sampled valid points per view and direction): consecutive views must reach 0.25 (pinhole
and fisheye tracks), 0.10 (panorama track) or 0.15 (mixed track). The envelope set v1.1 (16 panorama and
13 mixed tuples, seed 1) was built in the builder's envelope mode (`--allpairs-gate 0.10`): every view pair, not only consecutive
ones, reaches 0.10, and the consecutive gates rise to 0.25 (panorama track) and 0.20 (mixed track).
`scripts/blk_bench/test_conventions.py` checks the camera conventions of the construction on a synthetic
pack. The benchmark data (frames and
ground truth of all tracks) is distributed as a separate archive under CC BY 4.0: [meow_laser_benchmark.tar.gz](https://github.com/qgli/MEOW-het/releases/download/v1.0/meow_laser_benchmark.tar.gz).

## Real panoramas and perspective sets

- **2D3DS panorama poses:** 19 tuples of 2 to 13 panoramas from rooms of all areas (20 draws, one repeat kept once)
  (`scripts/eval_2d3ds_pose_v2.py --wid3r-faithful --areas area_1 area_2 area_3 area_4 area_5a area_5b
  area_6 --max-cases 20 --seed 0`).
- **2D3DS single panoramas:** 20 panoramas each from areas 5a and 5b
  (`scripts/eval_2d3ds_panorama_v2.py --areas area_5a area_5b --num-frames 20`).
- **Matterport3D:** the 18 test scans; per scan eight panoramas (skyboxes converted to equirectangular
  images) chosen near a seeded anchor panorama (`scripts/mp3d_benchmark/eval_mp3d_panoramas.py --k 8 --seed 0`;
  ground-truth pointmaps from `precompute_erp_gt.py`).
- **Replica:** tuples of 8 perspective views, 6 per scene, frame stride 40, seed 0
  (`scripts/realset/replica_prepare.py`).
- **Aria Digital Twin:** 20 sequences, 4 tuples of 8 views each, 0.8 s apart, as 640 px pinhole views
  (focal length 280 px) or the original fisheye images (`scripts/realset/adt_prepare.py`, seed 20260804).

The datasets are not redistributed; download them from their providers.
