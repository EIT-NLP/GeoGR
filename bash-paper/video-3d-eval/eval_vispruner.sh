#!/usr/bin/env bash
set -euo pipefail

# VisPruner operates on each native pooled 14x14 Video3D frame. The visual
# token budget is specified per frame.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_eval.sh"

VISUAL_TOKEN_NUM="${VISUAL_TOKEN_NUM:-22}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.5}"
PRUNE_STEP="${PRUNE_STEP:-8}"

VISPRUNER_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_ARGS="mm_projector_compressor_type=vispruner,mm_projector_compressor_config={\"visual_token_num\":${VISUAL_TOKEN_NUM},\"important_ratio\":${IMPORTANT_RATIO},\"prune_step\":${PRUNE_STEP},\"newline_strategy\":\"${VISPRUNER_NEWLINE_STRATEGY}\"}"

run_video3d_eval "vispruner_t${VISUAL_TOKEN_NUM}_${VISPRUNER_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
