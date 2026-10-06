#!/usr/bin/env bash
# LLaVA-OneVision Voxel-DTC post-training on ScanQA and SQA3D.
# Compressor settings match ov-eval/eval_voxel_dtc.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"

# Set TARGET_KEEP_RATIO= to use a fixed TARGET_TOKENS budget.
INITIAL_VOXEL_SIZE="${INITIAL_VOXEL_SIZE:-0.1}"
VOXEL_SIZE_STEP="${VOXEL_SIZE_STEP:-0.02}"
EDGE_KEEP_RATIO="${EDGE_KEEP_RATIO:-0.4}"

TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO-0.22}"
TARGET_TOKENS="${TARGET_TOKENS:-1000}"
DTC_NEWLINE_STRATEGY="grid_drop"

if [[ -n "${TARGET_KEEP_RATIO}" ]]; then
    TARGET_CONFIG="\"target_keep_ratio\":${TARGET_KEEP_RATIO}"
    RUN_TARGET="k${TARGET_KEEP_RATIO}"
else
    TARGET_CONFIG="\"target_tokens\":${TARGET_TOKENS}"
    RUN_TARGET="t${TARGET_TOKENS}"
fi

COMPRESSOR_TYPE="voxel_dtc"
COMPRESSOR_CONFIG="{\"initial_voxel_size\":${INITIAL_VOXEL_SIZE},\"voxel_size_step\":${VOXEL_SIZE_STEP},\"edge_keep_ratio\":${EDGE_KEEP_RATIO},\"newline_strategy\":\"${DTC_NEWLINE_STRATEGY}\",${TARGET_CONFIG}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_voxel_dtc_${RUN_TARGET}_${DTC_NEWLINE_STRATEGY}_proj_llm_ft}"

run_ov_train "${DATA_YAML}" "${RUN_NAME_PREFIX}" "${COMPRESSOR_TYPE}" "${COMPRESSOR_CONFIG}"
