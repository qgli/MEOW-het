# MEOW: Many Eyes, One World

**Feed-Forward 3D Reconstruction from Mixed Cameras**

Qiaoge Li<sup>1</sup>, Yifan Zhan<sup>2</sup>, Haijun Yang<sup>1</sup>, Haiyang Liu<sup>2</sup>, Yiyi Cai<sup>2</sup>, Chenchi Luo<sup>1</sup>

<sup>1</sup>China Mobile Communications Company Limited Research Institute &nbsp;&nbsp; <sup>2</sup>The University of Tokyo

Paper: [arXiv:2609.35658](https://arxiv.org/abs/2609.35658)

MEOW reconstructs metric pointmaps and camera poses from one tuple of views that mixes perspective,
fisheye and full 360-degree panoramic images, in a single forward pass and from the images alone: no
calibration, distortion parameters, camera-type labels or poses are needed for any view (full panoramas are recognised from their
pixels; some evaluation sets set the panorama flag for the whole set, see
[docs/EVALUATION.md](docs/EVALUATION.md)). It keeps a
perspective-pretrained backbone (MapAnything) and learns heterogeneous cameras from a procedural data
engine that renders each scene across a continuous range of camera models with exact rays and depth,
and certifies the covisibility of every camera-sampled training tuple.

## Contents

| | |
|---|---|
| Data engine | Lenscope in the paper: two scene generators with Blender, exact per-pixel rays and depth, offline covisibility, the online camera sampler used during training, and the conversion of laser scans into the same format ([docs/DATA_ENGINE.md](docs/DATA_ENGINE.md)) |
| Training | the three training stages as Hydra configurations, hardware profiles for 2 x RTX 5090, 4 and 8 x H200 and 8 x A100 40 GB, launch scripts and the Stage-3 relay ([docs/TRAINING.md](docs/TRAINING.md)) |
| Evaluation | benchmark construction scripts with their convention tests, evaluators and settings for every reported set, baseline drivers ([docs/EVALUATION.md](docs/EVALUATION.md), [docs/BENCHMARKS.md](docs/BENCHMARKS.md)) |
| Results | model checkpoints are not part of this release yet (coming soon); every reported number can be recomputed from scratch with the code, configurations and benchmark definitions here |
| Reproducibility | the checks run with this code ([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)) |

Model checkpoints, including both parents of the final model and the interpolation script, are not part
of this release yet (coming soon). The training data is not distributed: the scenes of both generations regenerate from their seeds with the
engine (see [docs/DATA_ENGINE.md](docs/DATA_ENGINE.md) for the Blender version and asset library this
requires). The laser-scan benchmark data is distributed as a separate archive under the Creative Commons
Attribution 4.0 license (CC BY 4.0): [meow_laser_benchmark.tar.gz](https://github.com/qgli/MEOW-het/releases/download/v1.0/meow_laser_benchmark.tar.gz) (636 MB,
SHA-256 `5279045d2c92d4290d83564422f91054cbdad6ca1930257575187feb17d36aa3`).

## Quick start

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/third_party/uniception
python scripts/convert_hf_to_benchmark_checkpoint.py --apache \
    --output_path checkpoints/facebook_map-anything-apache.pth
```

Generate and render one first-generation scene, build its covisibility, draw one training tuple with the
online camera sampler and run ten optimiser steps as a pipeline check:

```bash
export WORK=$PWD/work BLENDER=/path/to/blender-4.5/blender
"$BLENDER" -b --python lenscope/gen1/generate_scene.py -- --seed 3 \
    --output_dir "$WORK/gen1/scenes" --room_type random --n_scenes 1
python lenscope/gen1/make_pick.py --blend-dir "$WORK/gen1/scenes" \
    --scenes proc_scene_000003 --out "$WORK/gen1/pick.json"
"$BLENDER" -b --python lenscope/gen1/render_fair_focal.py -- \
    --pick "$WORK/gen1/pick.json" --out "$WORK/gen1/packs" --num_poses 8 --samples 64
python scripts/compute_covisibility.py --root "$WORK/gen1/packs" --scenes proc_scene_000003 --overwrite
python scripts/repro/make_smoke_splits.py --scene proc_scene_000003 --out-dir "$WORK/gen1/splits"
python scripts/repro/smoke_injector_tuple.py --repo . --root "$WORK/gen1/packs" \
    --splits "$WORK/gen1/splits" --out "$WORK/gen1/tuple.json"
ENCODER_LR=0 scripts/repro/run_train_10steps.sh gen1 "$WORK/gen1/packs" "$WORK/gen1/splits" \
    checkpoints/facebook_map-anything-apache.pth "$WORK"      # ENCODER_LR=0 on 24 GB GPUs
```

## Repository layout

```
lenscope/          data engine: gen1/ (first generation), genesis/, core/, bpy/, fixtures/ (second
                    generation), blk/ (laser scans), tests/
mapanything/        MapAnything with the MEOW changes: model, losses, trainer, datasets, online camera
                    sampler (datasets/camera_sampler.py, connectivity.py)
third_party/        UniCeption 0.1.7 with the variable-resolution change
configs/            Hydra configurations; meow_stage/ and meow_hardware/ hold the MEOW recipes
scripts/            training launchers (train/), data preparation, evaluators, benchmark construction
                    (blk_bench/), baselines (competitor_baselines/), pipeline checks (repro/)
resources/          split files of both generations, first-generation frame statistics and feasibility
                    tables, all used by the training loader
benchmarks/         definitions of the evaluation sets (frame lists and tuples, no pixels)
docs/               installation, data engine, training, evaluation, benchmarks, reproducibility
```

## Citation

```bibtex
@article{li2026meow,
  title   = {Many Eyes, One World: Feed-Forward 3D Reconstruction from Mixed Cameras},
  author  = {Li, Qiaoge and Zhan, Yifan and Yang, Haijun and Liu, Haiyang and Cai, Yiyi and Luo, Chenchi},
  journal = {arXiv preprint arXiv:2609.35658},
  year    = {2026}
}
```

## License

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)). It builds on MapAnything
(Apache License 2.0) and UniCeption (BSD 3-Clause); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for these and for the data and assets used at run time.
