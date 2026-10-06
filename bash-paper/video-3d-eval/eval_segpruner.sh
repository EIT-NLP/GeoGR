#!/usr/bin/env bash
set -euo pipefail

# Video3D pooled-token SegPruner-style evaluation. The compressor is shared
# with OV and uses attention, 3D coordinates, and pre-projector features.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_eval.sh"

# visual_token_num and token_keep_ratio are mutually exclusive. The ratio
# is applied independently to every pooled frame, matching SegPruner.
TOKEN_KEEP_RATIO="${TOKEN_KEEP_RATIO:-0.12}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.35}"
LAM="${LAM:-0.5}"
SEGPRUNER_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_ARGS="mm_projector_compressor_type=segpruner,mm_projector_compressor_config={\"token_keep_ratio\":${TOKEN_KEEP_RATIO},\"important_ratio\":${IMPORTANT_RATIO},\"lam\":${LAM},\"newline_strategy\":\"${SEGPRUNER_NEWLINE_STRATEGY}\"}"
run_video3d_eval "segpruner_k${TOKEN_KEEP_RATIO}_i${IMPORTANT_RATIO}_l${LAM}_${SEGPRUNER_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
