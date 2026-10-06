#!/usr/bin/env bash
set -euo pipefail

# SegPruner-style compression over the pooled 14x14 OV projector grid.
# IMPORTANT_RATIO controls attention-selected tokens; LAM balances geometric
# and semantic distances.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIMIT="${LIMIT:-}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_eval.sh"

TOKEN_KEEP_RATIO="${TOKEN_KEEP_RATIO:-0.30}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.35}"
LAM="${LAM:-0.5}"

SEGPRUNER_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_ARGS="mm_projector_compressor_type=segpruner,mm_projector_compressor_config={\"token_keep_ratio\":${TOKEN_KEEP_RATIO},\"important_ratio\":${IMPORTANT_RATIO},\"lam\":${LAM},\"newline_strategy\":\"${SEGPRUNER_NEWLINE_STRATEGY}\"}"

run_ov_eval "ov_segpruner_k${TOKEN_KEEP_RATIO}_i${IMPORTANT_RATIO}_l${LAM}_${SEGPRUNER_NEWLINE_STRATEGY}" "${COMPRESSOR_ARGS}"
