#!/usr/bin/env bash
# Stage-3 relay of the final model (run B): warm restarts on the second-generation training split.
# Each leg starts from the previous leg's checkpoint-last.pth with a fresh optimiser, one warm-up epoch
# and its own cosine schedule.
#
# usage: scripts/train/meow_stage3_relay.sh RUN_PREFIX
# Environment: as scripts/train/meow_train.sh, with MEOW_INIT_CKPT = the Stage-2 checkpoint, plus
#   HW8        profile of the 8-GPU legs (default h200x8; a100x8_w8 on 8 x A100 40 GB; h200x4 on 4 GPUs)
#   HW4        profile of the 4-GPU legs (default h200x4; a100x8 on 8 x A100 40 GB)
#   FIRST_LEG  first leg to run (default 1); earlier legs must have finished
set -euo pipefail

if [[ $# -ne 1 ]]; then
  sed -n '2,10p' "$0" >&2
  exit 2
fi
PREFIX=$1
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=${MEOW_EXPERIMENTS_DIR:-$(cd "$HERE/../.." && pwd)/experiments}
HW8=${HW8:-h200x8}
HW4=${HW4:-h200x4}
FIRST_LEG=${FIRST_LEG:-1}
: "${MEOW_INIT_CKPT:?set MEOW_INIT_CKPT to the Stage-2 checkpoint}"
STAGE2_CKPT=$MEOW_INIT_CKPT

# leg: profile, scheduled epochs, epochs trained (leg 3 trains 4 of its 15 scheduled epochs)
LEGS=(
  "$HW8 90 90"
  "$HW4 15 15"
  "$HW4 15 4"
  "$HW8 125 125"
  "$HW8 125 125"
  "$HW8 125 125"
  "$HW8 125 125"
  "$HW8 125 125"
  "$HW8 125 125"
)

for ((i = FIRST_LEG; i <= ${#LEGS[@]}; i++)); do
  read -r hw epochs trained <<< "${LEGS[i-1]}"
  if (( i == 1 )); then
    init=$STAGE2_CKPT
  else
    init=$EXP/${PREFIX}_leg$((i - 1))/checkpoint-last.pth
  fi
  [[ -f "$init" || "${MEOW_DRY_RUN:-0}" == 1 ]] || { echo "missing $init" >&2; exit 1; }
  extra=("train_params.epochs=$epochs")
  if (( trained < epochs )); then
    extra+=("++train_params.stop_epoch=$trained")
  fi
  echo "[relay] leg $i/${#LEGS[@]}: $hw, $trained of $epochs epochs, init $init"
  MEOW_INIT_CKPT=$init "$HERE/meow_train.sh" stage3 "$hw" "${PREFIX}_leg$i" "${extra[@]}"
done
