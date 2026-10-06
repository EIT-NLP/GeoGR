#!/usr/bin/env bash
# LLaVA-OneVision Voxel-VTC post-training on ScanQA and SQA3D.
# Compressor settings match ov-eval/eval_voxel_vtc.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"

# A larger voxel size merges more tokens. Inspect compression profiling for
# the measured token count. Example overrides:
#   VOXEL_SIZE=0.20 bash bash-paper/ov-train/train_voxel_vtc.sh
#   VOXEL_SIZE=0.25 bash bash-paper/ov-train/train_voxel_vtc.sh
VOXEL_SIZE="${VOXEL_SIZE:-0.35}"
VTC_ORDER_STRATEGY_DEFAULT="representative"
VTC_ORDER_STRATEGY="${VTC_ORDER_STRATEGY:-${VTC_ORDER_STRATEGY_DEFAULT}}"
VTC_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_TYPE="voxel_vtc"
COMPRESSOR_CONFIG="{\"voxel_size\":${VOXEL_SIZE},\"order_strategy\":\"${VTC_ORDER_STRATEGY}\",\"newline_strategy\":\"${VTC_NEWLINE_STRATEGY}\"}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_voxel_vtc_v${VOXEL_SIZE}_${VTC_ORDER_STRATEGY}_${VTC_NEWLINE_STRATEGY}_proj_llm_ft}"

run_ov_train "${DATA_YAML}" "${RUN_NAME_PREFIX}" "${COMPRESSOR_TYPE}" "${COMPRESSOR_CONFIG}"
