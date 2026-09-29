#!/usr/bin/env bash
# Pipeline check: ten real forward/loss/backward/optimizer steps on one shard (no checkpoint written).
# START_CKPT is any MapAnything-architecture checkpoint, e.g. the public MapAnything weights converted
# with scripts/convert_hf_to_benchmark_checkpoint.py. The data settings follow the camera-sampled
# training stream with four views and one tuple per step.
# Peak memory is about 27 GB. On 24 GB GPUs set ENCODER_LR=0, which freezes the image encoder
# (no encoder gradients or optimizer state; the encoder learning rate here is 8e-7 otherwise).
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 LABEL DATA_ROOT SPLITS_DIR START_CKPT WORK_ROOT" >&2
  exit 2
fi

LABEL=$1
DATA_ROOT=$2
SPLITS=$3
CKPT=$4
WORK_ROOT=$5
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}
ENCODER_LR=${ENCODER_LR:-8e-7}
RUN_DIR="$WORK_ROOT/train_${LABEL}"
LOG="$WORK_ROOT/train_${LABEL}.log"

mkdir -p "$WORK_ROOT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$REPO:$REPO/third_party/uniception${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node 1 \
  scripts/repro/train_no_save.py \
  machine=meow \
  dataset=procthor_unicol_518_many_ar \
  dataset.num_workers=0 \
  dataset.num_views=4 \
  dataset.train.variable_num_views=False \
  dataset.procthor_unicol.train.ROOT="$DATA_ROOT" \
  dataset.procthor_unicol.train.splits_dir="$SPLITS" \
  dataset.procthor_unicol.train.n5_sampling=true \
  dataset.procthor_unicol.train.n5_sampling_v2=true \
  'dataset.procthor_unicol.train.n5_v2_tier_weights=[0.0,0.5,0.5]' \
  dataset.procthor_unicol.train.preserve_info_resize=True \
  dataset.procthor_unicol.train.variable_resolution=False \
  dataset.procthor_unicol.train.camera_sampling=True \
  dataset.procthor_unicol.train.camera_sampling_p=1.0 \
  dataset.procthor_unicol.train.camera_sampling_retries=1 \
  dataset.procthor_unicol.train.photo_mtf_p=0.5 \
  dataset.procthor_unicol.train.photo_mtf_vmin=0.35 \
  dataset.procthor_unicol.val.ROOT="$DATA_ROOT" \
  dataset.procthor_unicol.val.splits_dir="$SPLITS" \
  dataset.procthor_unicol.val.preserve_info_resize=True \
  dataset.procthor_unicol.val.n5_sampling=true \
  dataset.procthor_unicol.val.n5_sampling_v2=true \
  'dataset.procthor_unicol.val.n5_v2_tier_weights=[0.0,0.5,0.5]' \
  'dataset.train_dataset=+ 10 @ ${dataset.procthor_unicol.train.dataset_str}' \
  'dataset.test_dataset=+ 1 @ ${dataset.procthor_unicol.val.dataset_str}' \
  loss=overall_loss_highpm_plus_rel_pose \
  model=mapanything \
  model/task=aug_training \
  '++model.task.ar_prob=1.0' \
  '++model.task.ar_fuse_mode=additive' \
  model.model_config.variable_resolution=False \
  model.encoder.uses_torch_hub=true \
  ++model.encoder.torch_hub_pretrained=False \
  model.encoder.gradient_checkpointing=true \
  model.info_sharing.module_args.gradient_checkpointing=true \
  model.pred_head.gradient_checkpointing=true \
  model.model_config.pretrained_checkpoint_path="$CKPT" \
  train_params=lower_encoder_lr \
  train_params.resume=False \
  train_params.epochs=1 \
  train_params.warmup_epochs=0 \
  train_params.lr=1.6e-5 \
  train_params.min_lr=1.6e-7 \
  train_params.submodule_configs.encoder.lr="$ENCODER_LR" \
  train_params.submodule_configs.encoder.min_lr=8e-9 \
  train_params.print_freq=1 \
  train_params.eval_freq=0 \
  train_params.save_freq=0 \
  train_params.keep_freq=0 \
  train_params.max_num_of_imgs_per_gpu=4 \
  '++train_params.feature_to_batch_size_map_override={0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1}' \
  "hydra.run.dir=$RUN_DIR" \
  >"$LOG" 2>&1

grep -E "Epoch: \[0\].*\[[[:space:]]*[0-9]+/10\]|Training time|checkpoint write skipped" \
  "$LOG" | tail -20
