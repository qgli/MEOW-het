# Installation

The code runs from the repository root without installing a package. Use Linux x86-64, Python 3.10 and
the CUDA 12.8 wheels of PyTorch 2.7.1 (they cover the RTX 30/40/50 series, A100 and H100/H200). The
final model was trained on RTX 5090 and H200 GPUs; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists the
checks that were run with this code.

## Python environment

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/third_party/uniception
```

UniCeption 0.1.7 is vendored in `third_party/uniception` with one addition for variable-resolution
batches (see `third_party/uniception/MODIFICATIONS.md`); do not install the PyPI package alongside it.

`requirements.txt` pins the package versions of the environment used for Stages 2 and 3. `xformers` is
optional at inference time (the DINOv2 layers fall back to PyTorch attention without it) but part of
the training environment. `embreex` accelerates ray casting in the data engine; `pye57` is needed only
to convert laser scans; `pandas` and `pyarrow` read the feasibility tables of the Stage-2/3 sampler.
Two packages are needed only by single scripts and are not in `requirements.txt`: `pypdf` for the
comparison with the Leica registration report (`lenscope/blk/validate_blk_scene.py --report`, passed by
`ingest_blk_scene.py` when the scan archive holds a `FinalizeReport.pdf`) and `projectaria-tools` for
`scripts/realset/adt_prepare.py`. The baselines run with their own code and environments (see
[EVALUATION.md](EVALUATION.md)).

The image encoder is created through `torch.hub` (DINOv2). The first run downloads the DINOv2 hub
repository (and, where a configuration asks for them, its weights) into `$TORCH_HOME` (default `~/.cache/torch`); set `TORCH_HOME` to a shared directory on
clusters without internet access on the compute nodes and populate it once.

## Starting checkpoint

All stages start from the public MapAnything checkpoint (Apache-2.0 release). Convert it once:

```bash
python scripts/convert_hf_to_benchmark_checkpoint.py --apache \
    --output_path checkpoints/facebook_map-anything-apache.pth
```

## Blender

The data engine runs inside Blender (see [DATA_ENGINE.md](DATA_ENGINE.md)): Blender 4.1.1 for the second
generation, Blender 4.5 for the first. Download the portable Linux builds from blender.org; nothing has
to be installed into Blender's Python.

## Paths

Scripts take data locations as arguments. The training configurations read these environment variables
(see [TRAINING.md](TRAINING.md)):

| variable | meaning |
|---|---|
| `MEOW_GEN1_ROOT`, `MEOW_GEN1_SPLITS` | first-generation scene directory and split directory |
| `MEOW_GEN1_FRAME_STATS` | per-frame statistics of the first-generation renders (optional flat-wall filter of every first-generation dataset) |
| `MEOW_GEN2_ROOT`, `MEOW_GEN2_SPLITS` | second-generation shard scenes and split directory (Stage 3) |
| `MEOW_INIT_CKPT` | checkpoint a stage starts from |
| `MEOW_EXPERIMENTS_DIR` | output root of training runs |
| `MEOW_DATA_ROOT`, `MEOW_CHECKPOINT_DIR` | data and checkpoint roots of the upstream MapAnything datasets, benchmarks and baseline models (`configs/machine/meow.yaml`); the MEOW stages do not read them, and unset variables fall back to `.` |

Evaluation scripts use `MEOW_2D3DS_ROOT` (Stanford 2D-3D-Semantics) and `MEOW_MP3D_ROOT` (Matterport3D
scans); see [EVALUATION.md](EVALUATION.md).
