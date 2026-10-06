#!/usr/bin/env bash
# Video3D post-training under a fixed Voxel-VTC policy. VTC has no learned
# parameters; model weights adapt to its compressed sequence distribution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
# FREEZE_WORLD_POSITION_EMBEDDING=1
# FREEZE_GROUND_HEAD=1
source "${SCRIPT_DIR}/_common_video3d_train.sh"

VOXEL_SIZE="${VOXEL_SIZE:-0.35}"
VTC_ORDER_STRATEGY="${VTC_ORDER_STRATEGY:-representative}"
VTC_NEWLINE_STRATEGY="grid_drop"
DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
COMPRESSOR_CONFIG="{\"voxel_size\":${VOXEL_SIZE},\"order_strategy\":\"${VTC_ORDER_STRATEGY}\",\"newline_strategy\":\"${VTC_NEWLINE_STRATEGY}\"}"

run_video3d_training \
    projector \
    voxel_vtc \
    "${COMPRESSOR_CONFIG}" \
    "${DATA_YAML}" \
    "video3d_voxel_vtc_train"
