# Training

MEOW adapts the public MapAnything checkpoint in three stages. Every stage is a Hydra configuration in
`configs/meow_stage/`, combined with a hardware profile in `configs/meow_hardware/`, and launched with
`scripts/train/meow_train.sh`. Data preparation is described in [DATA_ENGINE.md](DATA_ENGINE.md).

## Lineage of the final model

| step | init | data | epochs | hardware | configuration |
|---|---|---|---|---|---|
| Stage 1 | MapAnything (Apache-2.0 checkpoint) | first generation, native renders, 4 views, centre-crop loader | 35 (7 warm-up) | 2 x RTX 5090 | `stage1` + `rtx5090x2` |
| Stage 2 | Stage 1, best checkpoint | first generation, camera-sampled tuples, K = 2..8 curriculum, aspect-ratio embedding, one resolution bucket per view | 100 (15 warm-up) | 4 x H200 | `stage2` + `h200x4` |
| Stage 3, run A | Stage 2, best checkpoint | 189-scene early batch of the second generation (2048x1024 panoramas; scene lists in `resources/gen2_early_batch_splits/`) | 15 | 8 x H200 | `stage3` + `h200x8` |
| Stage 3, run B | Stage 2, best checkpoint | second-generation training split (469 scenes) | 859, as the relay below | 8 x H200 and 4 x H200 | `stage3` + `h200x8` / `h200x4` |
| final model | weight interpolation 0.25 x run A + 0.75 x run B (after 859 epochs) | | | | released with the checkpoints |

Stage 3 uses the solid-angle weighted loss with the von Mises-Fisher ray term
(`configs/loss/overall_loss_solid_angle_vmf.yaml`) and the panorama wrap of the dense head; its validation
sets stay on first-generation scenes.

## Launching a stage

```bash
export MEOW_GEN1_ROOT=/path/to/gen1/scenes MEOW_GEN1_SPLITS=$PWD/resources/gen1_splits
export MEOW_GEN1_FRAME_STATS=$PWD/resources/gen1_frame_stats.json
export MEOW_EXPERIMENTS_DIR=/path/to/experiments

# Stage 1
MEOW_INIT_CKPT=checkpoints/facebook_map-anything-apache.pth \
  scripts/train/meow_train.sh stage1 rtx5090x2 stage1
# Stage 2
MEOW_INIT_CKPT=$MEOW_EXPERIMENTS_DIR/stage1/checkpoint-best.pth \
  scripts/train/meow_train.sh stage2 h200x4 stage2
# Stage 3, one leg
export MEOW_GEN2_ROOT=/path/to/gen2/shards/scenes MEOW_GEN2_SPLITS=$PWD/resources/gen2_splits
MEOW_INIT_CKPT=$MEOW_EXPERIMENTS_DIR/stage2/checkpoint-best.pth \
  scripts/train/meow_train.sh stage3 h200x8 stage3_runA
```

Extra arguments are passed to Hydra (for example `train_params.epochs=90`). `MEOW_DRY_RUN=1` prints the
command instead of running it; `NPROC` overrides the number of processes. A run directory that already
holds `checkpoint-last.pth` resumes from it. The trainer writes `checkpoint-last.pth` every epoch,
`checkpoint-best.pth` on the lowest validation loss (the mean over the validation sets of each set's
median loss) and `checkpoint-<epoch>.pth` every 10 epochs in Stage 1 and every 25 epochs in Stages 2 and 3.

`MEOW_GEN1_FRAME_STATS` points at the per-frame statistics of the first-generation renders
(`scripts/precompute_mask_frac.py`, see [DATA_ENGINE.md](DATA_ENGINE.md)); rendered frames facing a flat wall
are then left out of the native-render tuples of first-generation scenes (the online camera sampler does
not use the statistics). `resources/gen1_frame_stats.json` is the file used in Stages 1 and 2; it covers the
110 scenes rendered first, and frames of the other scenes are not filtered. Without it every rendered frame
can enter a native-render tuple.

## Stage-3 relay

Run B was trained as a sequence of warm restarts: each leg starts from the previous leg's
`checkpoint-last.pth` with a fresh optimiser, one warm-up epoch and its own cosine schedule.

| leg | hardware | epochs |
|---|---|---|
| 1 | 8 x H200 | 90 |
| 2 | 4 x H200 | 15 |
| 3 | 4 x H200 | 4 of a 15-epoch schedule |
| 4-9 | 8 x H200 | 125 each |

`scripts/train/meow_stage3_relay.sh RUN_PREFIX` runs the legs in order (environment as above,
`MEOW_INIT_CKPT` = the Stage-2 checkpoint). The 8-GPU and 4-GPU profiles put the same number of tuples into
an optimiser step for every tuple size except K = 5 (40 and 36 tuples, 200 and 180 images; otherwise at most
192 images), so a 4-GPU machine can run every leg with `h200x4`.

## Hardware profiles

| profile | GPUs | images per GPU and step | notes |
|---|---|---|---|
| `rtx5090x2` | 2 x 32 GB | 4 (one 4-view tuple) | Stage 1 |
| `h200x4` | 4 x 141 GB | at most 48 | Stages 2 and 3; tuples per GPU for K = 2..8: 24, 16, 12, 9, 8, 6, 6; a Stage-2 epoch takes about 33 minutes |
| `h200x8` | 8 x 141 GB | 24 (K = 5: 25, K = 7: 21) | Stages 2 and 3; tuples per GPU: 12, 8, 6, 5, 4, 3, 3 |
| `a100x8` | 8 x 40 GB | one tuple at a time | the `h200x4` recipe split over 8 GPUs (see below) |
| `a100x8_w8` | 8 x 40 GB | one tuple at a time | the `h200x8` recipe |

Memory: with gradient checkpointing, Stage 2 and 3 steps peak at about 28 GB per tuple of up to eight views
at 518 px, so GPUs with 40 GB use the logical-split mode. The ten-step pipeline check
(`scripts/repro/run_train_10steps.sh`) peaks at about 27 GB; on 24 GB GPUs run it with `ENCODER_LR=0`
(frozen image encoder, 18 GB).

### Logical-split mode (A100 40 GB)

`train_params.logical_world_size` makes the data pipeline behave as in the recipe's GPU count while
training runs on more GPUs: physical rank p samples as logical rank p // s (s = physical / logical world
size), takes every s-th tuple of that logical batch and processes its share one tuple at a time, with the
gradient all-reduce only after the last tuple. Each tuple's loss is scaled so that the update equals the
mean over the logical batch, and the learning-rate schedule advances once per logical batch. Per-tuple
losses are exact, and the update equals that of processing the tuples one by one on one GPU up to the
run-to-run noise of the GPU kernels. Loss terms that average over all valid pixels of a batch average over
each tuple instead (every tuple gets the same weight), so gradients differ from those of a whole-batch
step (1-8 % relative L2 on the batches we measured). With this scheme the Stage-2 recipe ran on 8 x A100
40 GB (PCIe) at about 11 s per optimiser step (77 minutes per epoch, 27.8 GB peak memory). The trainer
stops on a non-finite loss and saves the offending tuple; resume from `checkpoint-last.pth` by relaunching
with the same run name. `tests/logical_split/` holds the unit tests of the mode and a check of the split
sampler on a dataset (`check_split_sampler.py`).

## Pipeline check

`scripts/repro/run_train_10steps.sh LABEL DATA_ROOT SPLITS_DIR START_CKPT WORK_ROOT` runs ten complete
optimiser steps (camera-sampled 4-view tuples, one tuple per step) on a small shard without writing
checkpoints, for example on the scenes produced by the commands of [DATA_ENGINE.md](DATA_ENGINE.md)
(`scripts/repro/make_smoke_splits.py --scene <name> --out-dir <splits>` writes the split files for one scene).
`scripts/repro/smoke_injector_tuple.py` prints the first training tuple of a shard (`camera_sampling_applied` marks the
views drawn by the online camera sampler; a scene can fall back to its native renders).
