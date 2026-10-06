#!/usr/bin/env bash
set -euo pipefail

# Voxel-VTC groups pooled tokens by world-coordinate voxel. `voxel_size` is
# the native compression control; the measured token ratio is data-dependent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_eval.sh"

VOXEL_SIZE="${VOXEL_SIZE:-0.35}"

VTC_ORDER_STRATEGY_DEFAULT="representative"
VTC_ORDER_STRATEGY="${VTC_ORDER_STRATEGY:-${VTC_ORDER_STRATEGY_DEFAULT}}"

VTC_NEWLINE_STRATEGY_DEFAULT="grid_drop"
VTC_NEWLINE_STRATEGY="${VTC_NEWLINE_STRATEGY_DEFAULT}"

COMPRESSOR_ARGS="mm_projector_compressor_type=voxel_vtc,mm_projector_compressor_config={\"voxel_size\":${VOXEL_SIZE},\"order_strategy\":\"${VTC_ORDER_STRATEGY}\",\"newline_strategy\":\"${VTC_NEWLINE_STRATEGY}\"}"

run_video3d_eval "voxel_vtc_${VTC_ORDER_STRATEGY}_${VTC_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
