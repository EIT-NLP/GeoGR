#!/usr/bin/env bash
set -euo pipefail

# Voxel-DTC progressively increases the voxel size and keeps the most similar
# candidate edges until it reaches the requested token budget.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_eval.sh"

INITIAL_VOXEL_SIZE="${INITIAL_VOXEL_SIZE:-0.1}"
VOXEL_SIZE_STEP="${VOXEL_SIZE_STEP:-0.02}"
EDGE_KEEP_RATIO="${EDGE_KEEP_RATIO:-0.4}"
DTC_NEWLINE_STRATEGY="grid_drop"

# Set TARGET_KEEP_RATIO= to use TARGET_TOKENS instead.
TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO-0.12}"
TARGET_TOKENS="${TARGET_TOKENS:-1000}"

if [[ -n "${TARGET_KEEP_RATIO}" ]]; then
    TARGET_CONFIG="\"target_keep_ratio\":${TARGET_KEEP_RATIO}"
else
    TARGET_CONFIG="\"target_tokens\":${TARGET_TOKENS}"
fi

COMPRESSOR_ARGS="mm_projector_compressor_type=voxel_dtc,mm_projector_compressor_config={\"initial_voxel_size\":${INITIAL_VOXEL_SIZE},\"voxel_size_step\":${VOXEL_SIZE_STEP},\"edge_keep_ratio\":${EDGE_KEEP_RATIO},\"newline_strategy\":\"${DTC_NEWLINE_STRATEGY}\",${TARGET_CONFIG}}"

run_video3d_eval "voxel_dtc_${DTC_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
