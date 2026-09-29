# Reproducibility checks

These checks were run with the code of this repository on a single RTX 3090 (24 GB), unless noted.
They compare against the artefacts of the original runs: training packs and scene files, logged Hydra
configurations and sampling logs. These artefacts are not distributed, so the
comparison scripts are not part of the release; the results are summarized here.

## Data engine

- **First generation.** Regenerating scenes from their seeds reproduces the rays, depth and validity
  masks of the original training packs exactly (seeds 0-2, 192 packs); RGB differs only by renderer
  sampling noise (PSNR 75-82 dB). The offline covisibility matrices of the regenerated scenes are
  bitwise identical between runs.
- **Second generation.** Rebuilding 16 archived scenes of the second-generation training data from their seeds with
  Blender 4.1.1, the pinned asset library and precomputed material statistics reproduces each scene
  file in geometry (vertex positions), objects, materials, texture references, every shader-node input,
  lights and sky. The pose-graph sampler run on the rebuilt scenes reproduces all 1,863 camera centres
  of the archived evaluation tuples of these scenes (largest deviation 6.4e-7 m), and ray casting the
  rebuilt geometry along the stored per-pixel rays reproduces the stored depth (median relative
  difference 1.5e-3, the precision of the float16 packs and panorama resampling).
- **Splits.** `scripts/make_procthor_unicol_splits.py` reproduces the first-generation split in
  `resources/gen1_splits` (1,603 / 401 scenes), and `lenscope/genesis/make_splits.py`, given the 499
  completed second-generation scenes, reproduces `resources/gen2_splits` (469 / 15 / 15) byte for byte.

## Training

- **Configurations.** `scripts/train/meow_train.sh` composes Hydra configurations equivalent to the
  logged configurations or launch scripts of every run of the final model's lineage: Stage 1,
  Stage 2, both Stage-3 runs and all legs of the relay (dataset definitions, model, loss and all
  training parameters).
- **Computation.** On fixed batches of each stage, the training data (images, rays, depth, masks,
  camera sampling) and the losses are bitwise identical to those of the development code, and the
  gradients agree within the run-to-run nondeterminism of the GPU kernels. Over ten optimiser steps the
  losses of the two code versions differ by as much as two runs of the same code do.
- **Sampler statistics.** Over 3,000 sampled tuples, the camera-model shares of the camera sampler agree
  with the sampling log of the Stage-2 run within 0.5 percentage points; the sampled tuples of the
  development and released code are identical.
- **Logical-split mode.** See [TRAINING.md](TRAINING.md#logical-split-mode-a100-40-gb).
