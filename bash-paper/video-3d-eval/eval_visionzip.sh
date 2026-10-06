#!/usr/bin/env bash
set -euo pipefail

# VisionZip over the native pooled 14x14 Video3D visual grid. The default
# 19+3 budget preserves roughly the reference dominant/contextual allocation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_eval.sh"

DOMINANT_TOKENS="${DOMINANT_TOKENS:-19}"
CONTEXTUAL_TOKENS="${CONTEXTUAL_TOKENS:-3}"

VISIONZIP_NEWLINE_STRATEGY_DEFAULT="grid_drop"
VISIONZIP_NEWLINE_STRATEGY="${VISIONZIP_NEWLINE_STRATEGY_DEFAULT}"

COMPRESSOR_ARGS="mm_projector_compressor_type=visionzip,mm_projector_compressor_config={\"dominant_tokens\":${DOMINANT_TOKENS},\"contextual_tokens\":${CONTEXTUAL_TOKENS},\"newline_strategy\":\"${VISIONZIP_NEWLINE_STRATEGY}\"}"

run_video3d_eval "visionzip_${VISIONZIP_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
