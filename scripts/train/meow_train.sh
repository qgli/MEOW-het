#!/usr/bin/env bash
# Launch one MEOW training stage, or one leg of the Stage-3 relay, on a single node.
#
# usage: scripts/train/meow_train.sh STAGE HARDWARE RUN_NAME [extra Hydra overrides ...]
#   STAGE     stage1 | stage2 | stage3                       (configs/meow_stage)
#   HARDWARE  rtx5090x2 | h200x4 | h200x8 | a100x8 | a100x8_w8   (configs/meow_hardware)
#   RUN_NAME  output directory name under $MEOW_EXPERIMENTS_DIR
#
# Environment:
#   MEOW_INIT_CKPT                      checkpoint the run starts from (required)
#   MEOW_GEN1_ROOT, MEOW_GEN1_SPLITS    first-generation scenes and split directory (required: training
#                                       data of Stages 1 and 2, validation data of all stages)
#   MEOW_GEN1_FRAME_STATS               per-frame statistics of the first generation (flat-wall filter)
#   MEOW_GEN2_ROOT, MEOW_GEN2_SPLITS    second-generation shard scenes and split directory (Stage 3)
#   MEOW_EXPERIMENTS_DIR                output root (default: experiments/ in the repository)
#   NPROC                               processes on this node (default: GPUs of the hardware profile)
#   MEOW_DRY_RUN=1                      print the command instead of running it
# A run directory that already holds checkpoint-last.pth is resumed.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,18p' "$0" >&2
  exit 2
fi
STAGE=$1; HW=$2; RUN_NAME=$3; shift 3

REPO=$(cd "$(dirname "$0")/../.." && pwd)
: "${MEOW_INIT_CKPT:?set MEOW_INIT_CKPT}"
: "${MEOW_GEN1_ROOT:?set MEOW_GEN1_ROOT}"
: "${MEOW_GEN1_SPLITS:?set MEOW_GEN1_SPLITS}"
EXP=${MEOW_EXPERIMENTS_DIR:-$REPO/experiments}

case "$HW" in
  rtx5090x2) GPUS=2 ;;
  h200x4) GPUS=4 ;;
  h200x8|a100x8|a100x8_w8) GPUS=8 ;;
  *) echo "unknown hardware profile: $HW" >&2; exit 2 ;;
esac
NPROC=${NPROC:-$GPUS}

DATA=(
  "dataset.procthor_unicol.val.ROOT=$MEOW_GEN1_ROOT"
  "dataset.procthor_unicol.val.splits_dir=$MEOW_GEN1_SPLITS"
  "dataset.procthor_unicol.val_camera_sampled.ROOT=$MEOW_GEN1_ROOT"
  "dataset.procthor_unicol.val_camera_sampled.splits_dir=$MEOW_GEN1_SPLITS"
)
case "$STAGE" in
  stage1|stage2)
    DATA+=("dataset.procthor_unicol.train.ROOT=$MEOW_GEN1_ROOT"
           "dataset.procthor_unicol.train.splits_dir=$MEOW_GEN1_SPLITS") ;;
  stage3)
    : "${MEOW_GEN2_ROOT:?set MEOW_GEN2_ROOT for stage3}"
    : "${MEOW_GEN2_SPLITS:?set MEOW_GEN2_SPLITS for stage3}"
    DATA+=("dataset.procthor_unicol.train.ROOT=$MEOW_GEN2_ROOT"
           "dataset.procthor_unicol.train.splits_dir=$MEOW_GEN2_SPLITS") ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
# First-generation frame statistics apply to every first-generation dataset of the run (Stage 3 keeps
# its second-generation training set unfiltered in configs/meow_stage/stage3.yaml).
if [[ -n "${MEOW_GEN1_FRAME_STATS:-}" ]]; then
  DATA+=("dataset.procthor_unicol.frame_stats_path=$MEOW_GEN1_FRAME_STATS")
fi

CMD=(torchrun --standalone --nproc_per_node "$NPROC" scripts/train.py
     "+meow_stage=$STAGE" "+meow_hardware=$HW"
     "model.model_config.pretrained_checkpoint_path=$MEOW_INIT_CKPT"
     "${DATA[@]}"
     "hydra.run.dir=$EXP/$RUN_NAME"
     "$@")
if [[ "${MEOW_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/third_party/uniception${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
mkdir -p "$EXP/$RUN_NAME"
"${CMD[@]}" 2>&1 | tee -a "$EXP/$RUN_NAME/train.log"
