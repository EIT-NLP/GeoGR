#!/usr/bin/env bash
set -euo pipefail

# VisPruner keeps attention-important tokens and removes semantic redundancy
# from the remaining pooled 14x14 visual grid.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENABLE_3D_AUX="${ENABLE_3D_AUX:-false}"
export LIMIT="${LIMIT:-}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_eval.sh"

VISUAL_TOKEN_NUM="${VISUAL_TOKEN_NUM:-39}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.5}"
PRUNE_STEP="${PRUNE_STEP:-8}"

VISPRUNER_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_ARGS="mm_projector_compressor_type=vispruner,mm_projector_compressor_config={\"visual_token_num\":${VISUAL_TOKEN_NUM},\"important_ratio\":${IMPORTANT_RATIO},\"prune_step\":${PRUNE_STEP},\"newline_strategy\":\"${VISPRUNER_NEWLINE_STRATEGY}\"}"

run_ov_eval "ov_vispruner_t${VISUAL_TOKEN_NUM}_${VISPRUNER_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
