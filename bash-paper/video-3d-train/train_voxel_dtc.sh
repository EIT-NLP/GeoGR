#!/usr/bin/env bash
# Video3D post-training under a fixed Voxel-DTC policy. DTC has no learned
# parameters; model weights adapt to its compressed sequence distribution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
# FREEZE_WORLD_POSITION_EMBEDDING=1
# FREEZE_GROUND_HEAD=1
source "${SCRIPT_DIR}/_common_video3d_train.sh"

INITIAL_VOXEL_SIZE="${INITIAL_VOXEL_SIZE:-0.1}"
VOXEL_SIZE_STEP="${VOXEL_SIZE_STEP:-0.02}"
EDGE_KEEP_RATIO="${EDGE_KEEP_RATIO:-0.4}"
TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO-0.12}"
TARGET_TOKENS="${TARGET_TOKENS:-1000}"
DTC_NEWLINE_STRATEGY="grid_drop"
DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"

if [[ -n "${TARGET_KEEP_RATIO}" ]]; then
    TARGET_CONFIG="\"target_keep_ratio\":${TARGET_KEEP_RATIO}"
    RUN_TARGET="k${TARGET_KEEP_RATIO}"
else
    TARGET_CONFIG="\"target_tokens\":${TARGET_TOKENS}"
    RUN_TARGET="t${TARGET_TOKENS}"
fi
COMPRESSOR_CONFIG="{\"initial_voxel_size\":${INITIAL_VOXEL_SIZE},\"voxel_size_step\":${VOXEL_SIZE_STEP},\"edge_keep_ratio\":${EDGE_KEEP_RATIO},\"newline_strategy\":\"${DTC_NEWLINE_STRATEGY}\",${TARGET_CONFIG}}"

run_video3d_training \
    projector \
    voxel_dtc \
    "${COMPRESSOR_CONFIG}" \
    "${DATA_YAML}" \
    "video3d_voxel_dtc_${RUN_TARGET}_${DTC_NEWLINE_STRATEGY}_train"
