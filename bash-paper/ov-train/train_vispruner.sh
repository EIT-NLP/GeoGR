#!/usr/bin/env bash
# LLaVA-OneVision VisPruner post-training on ScanQA and SQA3D.
# VisPruner does not use the 3D coordinate cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"

# OV pooling produces a 14x14 visual grid per frame. Example overrides:
#   VISUAL_TOKEN_NUM=39 bash bash-paper/ov-train/train_vispruner.sh
#   VISUAL_TOKEN_NUM=78 bash bash-paper/ov-train/train_vispruner.sh
VISUAL_TOKEN_NUM="${VISUAL_TOKEN_NUM:-39}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.5}"
PRUNE_STEP="${PRUNE_STEP:-8}"
VISPRUNER_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_TYPE="vispruner"
COMPRESSOR_CONFIG="{\"visual_token_num\":${VISUAL_TOKEN_NUM},\"important_ratio\":${IMPORTANT_RATIO},\"prune_step\":${PRUNE_STEP},\"newline_strategy\":\"${VISPRUNER_NEWLINE_STRATEGY}\"}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_vispruner_t${VISUAL_TOKEN_NUM}_${VISPRUNER_NEWLINE_STRATEGY}_proj_llm_ft}"

run_ov_train "${DATA_YAML}" "${RUN_NAME_PREFIX}" "${COMPRESSOR_TYPE}" "${COMPRESSOR_CONFIG}"
