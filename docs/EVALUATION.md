# Evaluation

All evaluators run from the repository root with the environment of [INSTALL.md](INSTALL.md). The evaluation
sets are described in [BENCHMARKS.md](BENCHMARKS.md). The checkpoints are not part of this release yet
(coming soon).

## Checkpoints and model specification

Evaluators take checkpoints as `NAME=path[:variant]`:

| variant | model |
|---|---|
| (none) | MapAnything as released (images-only task), e.g. `MapAnything=checkpoints/facebook_map-anything-apache.pth` |
| `:wrap` | MEOW: aspect-ratio embedding and panorama wrap of the dense head (the final model) |
| `:ar` | aspect-ratio embedding without the panorama wrap (Stage-2 checkpoints) |

The MEOW checkpoints are not part of this release (see the [README](../README.md)). A checkpoint loaded
with a variant must contain the aspect-ratio encoder and one loaded without a variant must not;
mismatches stop the evaluator (`eval_mp3d_panoramas.py` only prints the number of aspect-ratio encoder
weight tensors it had to leave untrained).

## Preprocessing and panorama routing

- Input resizing: both modes send all views of a tuple to one aspect bucket, the one closest to the mean
  aspect ratio of the tuple. `squeeze` resizes every view to that bucket without cropping (full field of
  view); `crop` is MapAnything's loader: each view is scaled to cover the bucket and centre-cropped (a 2:1 panorama
  in an all-panorama tuple loses a few rows; in a mixed tuple panoramas lose about a third of their width).
- Panorama routing decides which views get the panorama flag (panorama wrap of the dense head). `auto`
  runs the image-only detector (`scripts/erp_detect.py`) on each view; `label` uses the set's camera
  labels; `all` (heterogeneous 2D3DS) and `force` (real-image sets) flag every view, `none`/`off` no view;
  `on` (real-image sets) turns on the panorama wrap for every view. Views flagged by `auto`, `all` or
  `force` get the aspect-ratio input 2.0, the aspect of a full panorama; with the other routes every
  view keeps its image width/height as aspect-ratio input.

Settings behind the reported MEOW numbers:

| set | evaluator | resizing | panorama routing |
|---|---|---|---|
| heterogeneous 2D3DS tuples | `scripts/mp3d_benchmark/het_2d3ds_pose.py` | `squeeze` | `auto` |
| laser benchmark, four tracks | `scripts/realset/predict_ours.py` + `eval_realset.py` | `squeeze` | `auto` |
| 2D3DS panorama poses (19 tuples) | `scripts/eval_2d3ds_pose_v2.py` | `crop` | `auto` (`--ar-mode auto`) |
| 2D3DS single panoramas (40) | `scripts/eval_2d3ds_panorama_v2.py` | `crop` | every view is a panorama |
| Matterport3D panoramas | `scripts/mp3d_benchmark/eval_mp3d_panoramas.py` | `crop` | every view is a panorama |
| Replica, Aria Digital Twin | `scripts/realset/predict_ours.py` + `eval_realset.py` | `crop` | `off` |

## Pose metrics

The evaluators score every ordered pair of views in a tuple (`eval_case` in `scripts/eval_2d3ds_pose.py`, also
used by `scripts/realset/eval_realset.py`). The rotation error is the geodesic angle between the predicted and
the ground-truth relative rotation; the translation error is the angle between the relative translation
directions, taken without sign (0 to 90 degrees). RRA@t and RTA@t are the fractions of pairs whose rotation or
translation error is below t degrees. mAA@30 is the mean, over the integer thresholds 1 to 30 degrees, of the
fraction of pairs with both errors below the threshold; AUC@30 is the trapezoidal integral of the same curve
over 1 to 30 degrees, normalised to that interval. ATE is the root-mean-square error of the camera centres
after one Umeyama similarity alignment per tuple. Each evaluator averages the per-tuple values over the tuples
of a set.

## Heterogeneous 2D3DS tuples

Requires Stanford 2D-3D-Semantics (`MEOW_2D3DS_ROOT`, areas 5a, 5b and 6). The 88 tuples are built
deterministically from the panoramas (one tuple per room, up to 24 views, kinds cycling panorama,
perspective, fisheye):

```bash
HET="--areas area_5a area_5b area_6 --models erp,persp,fish --all-views --min-views 3 --cap-views 24 \
    --max-cases 300 --seed 0"
python scripts/mp3d_benchmark/het_2d3ds_pose.py $HET --backend ma --ckpts MEOW=$CKPT:wrap \
    --input-resize squeeze --pano-route auto --out het2d3ds_meow.json
python scripts/mp3d_benchmark/het_2d3ds_pose.py $HET --backend ma --ckpts MapAnything=$MA_CKPT \
    --input-resize crop --out het2d3ds_mapanything.json
python scripts/mp3d_benchmark/het_2d3ds_pose.py $HET --backend vggt --out het2d3ds_vggt.json   # needs vggt
python scripts/mp3d_benchmark/het_2d3ds_pose.py $HET --backend pi3 --out het2d3ds_pi3.json     # needs pi3
python scripts/competitor_baselines/score_het2d3ds_pertuple.py het2d3ds_meow.json      # per-tuple and pooled metrics
python scripts/competitor_baselines/het2d3ds_by_size.py het2d3ds_meow.json het2d3ds_wid3r.json   # by tuple size
```

The paper reports the mean over tuples of per-tuple mAA@30, RRA@30, RTA@30 and ATE (`score_het2d3ds_pertuple.py`
also prints the metrics pooled over all view pairs). Models that run in their own environments read the
same tuples from an export: `--export-dir DIR --export-only` writes every case's synthesized views and
ground-truth poses (`cases/case_NNN/*.png`, `gt_c2w.npy`, `manifest.json`).

## Laser benchmark

The benchmark data (tuples, frames and ground truth of four tracks) is distributed separately; see
[BENCHMARKS.md](BENCHMARKS.md). For each track of `v1/` (`blk_erp`, `blk_pinhole`, `blk_fisheye`, `blk_mixed`):

```bash
B=/path/to/laser_benchmark/v1/blk_mixed
python scripts/realset/predict_ours.py --tuples $B/tuples.json --frames-root $B \
    --ckpt MEOW=$CKPT:wrap --input-resize squeeze --pano auto --out preds
python scripts/realset/predict_ours.py --tuples $B/tuples.json --frames-root $B \
    --ckpt MapAnything=$MA_CKPT --out preds            # MapAnything: crop loader, no panorama flag
python scripts/realset/predict_vggt.py --tuples $B/tuples.json --frames-root $B --out preds   # likewise predict_pi3.py, predict_dust3r.py
python scripts/realset/predict_mast3r.py --tuples $B/tuples.json --frames-root $B --out preds \
    --weights /path/to/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --ga sparse
python scripts/realset/eval_realset.py --tuples $B/tuples.json --preds-root preds \
    --models MEOW MapAnything VGGT --out eval --scoring-device cpu
python scripts/realset/make_report.py --combined eval/combined_*.json --out report.md
```

Pose metrics are all-pairs relative rotation and translation accuracy and AUC@30; pointmap metrics
(accuracy, completeness, normal consistency) follow a similarity alignment (Umeyama, a least-squares
scale and shift, then ICP) computed by `scripts/mp3d_benchmark/pointmap_eval.py`. CPU scoring is the
reference; `--scoring-device cuda:0` checks the first tuple against the CPU result and falls back to
the CPU for a model whose check fails. `scripts/blk_bench/test_conventions.py` tests the camera
conventions of the benchmark construction (`pytest -q scripts/blk_bench/test_conventions.py`).

## Resizing and routing controls

Table 8 and the crop-loader row of Table 4 change one input setting of the final model at a time. The
Stage-1 checkpoint and MapAnything have no panorama wrap and no aspect-ratio input, so the panorama
routing does not apply to them.

| row | heterogeneous 2D3DS (`het_2d3ds_pose.py`) | laser benchmark (`predict_ours.py`) |
|---|---|---|
| full-FoV resizing, detector flag | `--input-resize squeeze --pano-route auto` | `--input-resize squeeze --pano auto` |
| full-FoV resizing, annotation flag | `--input-resize squeeze --pano-route label` | `--input-resize squeeze --pano label` |
| crop loader, annotation flag | `--input-resize crop --pano-route label` | `--input-resize crop --pano label` |
| Stage-1 checkpoint and MapAnything, crop loader | `--input-resize crop` | defaults (crop loader, no flag) |
| MapAnything, full-FoV resizing (diagnostic) | `--input-resize squeeze` | `--input-resize squeeze` |

The single-camera laser tracks with the annotation flag use `--input-resize squeeze --pano label`. The 16
panorama tuples inside the covisibility envelope (`v1.1/blk_erp`) use the crop loader with every view
flagged as a panorama, as the track holds panoramas only:

```bash
B=/path/to/laser_benchmark/v1.1/blk_erp
python scripts/realset/predict_ours.py --tuples $B/tuples.json --frames-root $B \
    --ckpt MEOW=$CKPT:wrap --pano on --out preds
```

## 2D3DS panoramas and Matterport3D

```bash
python scripts/eval_2d3ds_pose_v2.py --ckpts MEOW=$CKPT:wrap --wid3r-faithful \
    --areas area_1 area_2 area_3 area_4 area_5a area_5b area_6 --max-cases 20 --seed 0 \
    --ar-mode auto --out panorama_tuples/                                   # add --pre-squeeze 16:9 | 4:3 | 1:1
python scripts/eval_2d3ds_panorama_v2.py --ckpts MEOW=$CKPT:wrap --areas area_5a area_5b \
    --num-frames 20 --out single_panoramas/
python scripts/mp3d_benchmark/precompute_erp_gt.py --scans-root $MEOW_MP3D_ROOT --out mp3d_gt/
python scripts/mp3d_benchmark/eval_mp3d_panoramas.py --ckpts MEOW=$CKPT:wrap --k 8 --seed 0 \
    --gt-cache mp3d_gt/ --out mp3d.json
```

The squeezed-panorama rows add `--pre-squeeze 16:9 | 4:3 | 1:1` to the panorama-tuple command: rows with the
content aspect use `--ar-mode auto`, the row with the tensor aspect uses `--ar-mode asfed` (the aspect input is
the width/height of the squeezed image), and the row with the embedding withheld wraps the command in
`scripts/run_without_ar.py`.

Single panoramas: the crop loader maps each 4096 x 2048 panorama to 518 x 252 (scaled to cover 518 x 259, rows
3 to 254 kept). The ground truth (depth/512 along the panorama rays, with invalid depth and the black pole
pixels masked) is cropped the same way; `--uncropped-gt` resizes the whole panorama instead. Each prediction is
aligned to the ground truth with one Umeyama similarity, and the evaluator reports the Chamfer L1 distance and
the median relative depth error per panorama, averaged over the panoramas.

## Replica and Aria Digital Twin

```bash
python scripts/realset/replica_prepare.py --root $REPLICA --out replica/ \
    --views 8 --tuples-per-scene 6 --frame-stride 40 --seed 0
python scripts/realset/adt_prepare.py --adt-root $ADT --out adt/ --n-seqs 20 --tuples-per-seq 4 \
    --views 8 --stride-s 0.8 --pin-size 640 --pin-f 280      # seed default 20260804
python scripts/realset/predict_ours.py --tuples replica/tuples.json --ckpt MEOW=$CKPT:wrap --out preds
python scripts/realset/eval_realset.py --tuples replica/tuples.json --preds-root preds --models MEOW \
    --out eval --scoring-device cpu
```

## Baselines in their own environments

Wid3R and PanoVGGT run from their released code and weights; `scripts/competitor_baselines/` holds the
drivers and converters. They expect `COMP_BASE_ROOT` to contain `competitors/{Wid3R,PanoVGGT}`,
`weights/{wid3r/wid3r.bin,panovggt/model.pt}`, `env/{wid3r,panovggt}` (Python environments of the two
projects) and

```
inputs/heterogeneous_2d3ds/      het_2d3ds_pose.py --export-dir output for the 88 tuples
inputs/2d3ds_panorama_tuples/    manifest.json of benchmarks/2d3ds_panorama_tuples_19, stanford -> 2D3DS root
inputs/2d3ds_single_panoramas/   manifest.json of benchmarks/2d3ds_single_panoramas_40, stanford -> 2D3DS root
inputs/laser_benchmark/          the extracted laser benchmark (v1/, v1.1/)
inputs/calib_cases/              written by make_calib_cases.py (2D3DS areas 1-4)
```

The camera-frame convention of each baseline was fixed once on calibration cases disjoint from the
heterogeneous benchmark (`make_calib_cases.py`, `probe_convention.py`) and then frozen: identity camera
convention for poses, no reflection of the predicted world; single-panorama pointmaps are compared after
flipping the camera y axis (`score_single_panoramas.py --conv yflip`). `probe_convention_blk.py` runs
the same probe on laser-benchmark tuples.

```bash
python scripts/competitor_baselines/make_lists.py                   # image lists of the frozen inputs
cd $COMP_BASE_ROOT/competitors/Wid3R
$COMP_BASE_ROOT/env/wid3r/bin/python $REPO/scripts/competitor_baselines/predict_wid3r.py \
    --cases-root $COMP_BASE_ROOT/inputs/heterogeneous_2d3ds --out wid3r_het2d3ds --mode told
python $REPO/scripts/competitor_baselines/score_het2d3ds.py --preds wid3r_het2d3ds \
    --cases-root $COMP_BASE_ROOT/inputs/heterogeneous_2d3ds --conv identity --name Wid3R --out het2d3ds_wid3r.json
$COMP_BASE_ROOT/env/wid3r/bin/python $REPO/scripts/competitor_baselines/predict_wid3r.py \
    --list $COMP_BASE_ROOT/runs/lists/blk_mixed.json --out wid3r_blk_mixed --mode told --save-points
python $REPO/scripts/competitor_baselines/blk_convert.py --backend wid3r --preds wid3r_blk_mixed \
    --list $COMP_BASE_ROOT/runs/lists/blk_mixed.json --out preds/Wid3R
```

`run_chain1.sh` and `run_chain2.sh` run all baseline predictions; scoring and conversion follow as above.

## Other tables

- Panorama detector audit: `python scripts/detector_audit.py --cases-root <exported 2D3DS cases>` or
  `--tuples <tuples.json> --frames-root <dir>` prints the panoramas missed and other views flagged
  (`--default-kind` for tuple files without per-view kinds). Counts of the released detector:

  | set | views | panoramas missed | other views flagged |
  |---|---|---|---|
  | heterogeneous 2D3DS, 88 tuples | 503 | 0 of 189 | 2 of 314 |
  | laser benchmark v1, mixed | 96 | 0 of 24 | 0 of 72 |
  | laser benchmark v1, panorama | 96 | 2 of 96 | - |
  | laser benchmark v1, pinhole | 96 | - | 1 of 96 |
  | laser benchmark v1, fisheye | 96 | - | 0 of 96 |
  | laser benchmark v1.1, panorama | 64 | 0 of 64 | - |
  | laser benchmark v1.1, mixed | 52 | 0 of 13 | 0 of 39 |
  | 2D3DS panorama tuples, 19 | 81 | 0 of 81 | - |
  | 2D3DS single panoramas, 40 | 40 | 0 of 40 | - |
- Metric scale: `python scripts/scale_ratio.py --pred-dir <npz with c2w per tuple> --tuples <tuples.json>`
  (or `--cases-root`) gives the ratio of ground-truth to predicted camera-centre distances.
- Aspect-ratio embedding withheld: `python scripts/run_without_ar.py <evaluator> <evaluator arguments>`
  runs any evaluator with the embedding disabled.
- Latency: `python scripts/bench_latency.py --ckpt $CKPT --num-views N --res 518 --iters 30 --warmup 8`
  (`--meow` builds the aspect-ratio encoder and the panorama wrap and flags every view as a panorama);
  the per-tuple wall-clock times of the real-image sets come from the predictors' timing fields (image
  loading, inference and outputs on the CPU for each tuple), averaged over the tuples of a set with the first
  tuple excluded as warm-up (`sec_mean` of `eval_realset.py`).
