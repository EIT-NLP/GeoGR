#!/usr/bin/env bash
set -euo pipefail

# VisionZip selects dominant tokens and contextual anchors from each pooled
# 14x14 frame. Token budgets exclude inserted image-newline tokens.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENABLE_3D_AUX="${ENABLE_3D_AUX:-false}"
export LIMIT="${LIMIT:-}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_eval.sh"

DOMINANT_TOKENS="${DOMINANT_TOKENS:-17}"
CONTEXTUAL_TOKENS="${CONTEXTUAL_TOKENS:-3}"

# grid_drop is the maintained default; override only for legacy checkpoints.
VISIONZIP_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_ARGS="mm_projector_compressor_type=visionzip,mm_projector_compressor_config={\"dominant_tokens\":${DOMINANT_TOKENS},\"contextual_tokens\":${CONTEXTUAL_TOKENS},\"newline_strategy\":\"${VISIONZIP_NEWLINE_STRATEGY}\"}"

run_ov_eval "ov_visionzip_d${DOMINANT_TOKENS}_c${CONTEXTUAL_TOKENS}_${VISIONZIP_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
