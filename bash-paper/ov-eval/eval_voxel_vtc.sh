#!/usr/bin/env bash
set -euo pipefail

# Voxel-VTC groups pooled visual tokens in world-coordinate voxels. A larger
# voxel size merges more tokens; enable OV_LOG_COMPRESSION_PROFILE to inspect
# the measured keep ratio.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export TASKS="${TASKS:-scanqa_val}"
export LIMIT="${LIMIT:-}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_eval.sh"

VOXEL_SIZE="${VOXEL_SIZE:-0.35}"

# Preserve the source sequence order through each voxel representative.
VTC_ORDER_STRATEGY_DEFAULT="representative"
VTC_ORDER_STRATEGY="${VTC_ORDER_STRATEGY:-${VTC_ORDER_STRATEGY_DEFAULT}}"

VTC_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_ARGS="mm_projector_compressor_type=voxel_vtc,mm_projector_compressor_config={\"voxel_size\":${VOXEL_SIZE},\"order_strategy\":\"${VTC_ORDER_STRATEGY}\",\"newline_strategy\":\"${VTC_NEWLINE_STRATEGY}\"}"

run_ov_eval "ov_voxel_vtc_v${VOXEL_SIZE}_${VTC_ORDER_STRATEGY}_${VTC_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
