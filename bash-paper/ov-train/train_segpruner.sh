#!/usr/bin/env bash
# LLaVA-OneVision SegPruner post-training on ScanQA and SQA3D.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override GPU_IDS at launch time to select another GPU group.
GPU_IDS="${GPU_IDS:-0,1,2,3}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"

# The ratio applies independently to each pooled 14x14 frame.
TOKEN_KEEP_RATIO="${TOKEN_KEEP_RATIO:-0.40}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.35}"
LAM="${LAM:-0.5}"
SEGPRUNER_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_TYPE="segpruner"
COMPRESSOR_CONFIG="{\"token_keep_ratio\":${TOKEN_KEEP_RATIO},\"important_ratio\":${IMPORTANT_RATIO},\"lam\":${LAM},\"newline_strategy\":\"${SEGPRUNER_NEWLINE_STRATEGY}\"}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_segpruner_k${TOKEN_KEEP_RATIO}_i${IMPORTANT_RATIO}_l${LAM}_${SEGPRUNER_NEWLINE_STRATEGY}_proj_llm_ft}"

run_ov_train "${DATA_YAML}" "${RUN_NAME_PREFIX}" "${COMPRESSOR_TYPE}" "${COMPRESSOR_CONFIG}"
