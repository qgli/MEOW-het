# Third-party code, models and data

## Included in this repository

| component | location | license |
|---|---|---|
| MapAnything (Meta Platforms, Inc. and affiliates), commit f7ebafb4 | `mapanything/` and `configs/` except the files added by this project (below), `scripts/train.py`, `scripts/convert_hf_to_benchmark_checkpoint.py` | Apache License 2.0 (`LICENSE`). Files changed by this project carry a "Modified by the MEOW authors" notice. |
| External model code shipped with MapAnything | `mapanything/models/external/` | the licenses of the projects the code comes from; see `mapanything/models/external/README.md` |
| UniCeption 0.1.7 (AirLab Stacks) | `third_party/uniception/` | BSD 3-Clause (`third_party/uniception/LICENSE`); changes in `third_party/uniception/MODIFICATIONS.md` |

All other files are the work of the MEOW authors and are released under the Apache License 2.0. Inside
`mapanything/` and `configs/` these are:

- `mapanything/datasets/`: `camera_sampler.py`, `camera_models.py`, `connectivity.py`, `covis_gpu.py`,
  `geom_aug.py`, `gpu_aug.py`, `procthor_unicol.py`
- `mapanything/train/logical_split.py`, `mapanything/utils/spherical.py`
- `configs/dataset/procthor_unicol_518_many_ar.yaml`, `configs/dataset/procthor_unicol/`,
  `configs/loss/overall_loss_solid_angle_vmf.yaml`, `configs/machine/meow.yaml`, `configs/meow_hardware/`,
  `configs/meow_stage/`

## Downloaded at run time (not included)

- DINOv2 code and weights through `torch.hub` (facebookresearch/dinov2, Apache License 2.0).
- The public MapAnything checkpoint (`facebook/map-anything-apache` on the Hugging Face Hub, Apache
  License 2.0), converted with `scripts/convert_hf_to_benchmark_checkpoint.py --apache`.
- The optional asset library of the second-generation scenes, fetched by
  `lenscope/genesis/assets/fetch_assets.py`: materials from ambientCG (CC0 1.0), sky images from
  Poly Haven (CC0 1.0) and artworks from The Metropolitan Museum of Art Open Access (public domain).
  `lenscope/genesis/assets/manifest.json` records the provider, license and source of every file.
- Blender (GNU GPL), used as an external program to build and render scenes.

## Datasets and baselines (not included)

Evaluation uses Stanford 2D-3D-Semantics, Matterport3D, Replica and Aria Digital Twin. None of their
data is redistributed here; obtain them from their providers under their terms. `benchmarks/` lists
which frames the evaluation sets use.

Baselines (Wid3R, PanoVGGT, VGGT, pi3, DUSt3R, MASt3R) run from their own released code and weights under
their licenses; `scripts/competitor_baselines/` and `scripts/realset/` only contain drivers and
converters.
