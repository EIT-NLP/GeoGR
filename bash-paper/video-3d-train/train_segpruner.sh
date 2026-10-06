#!/usr/bin/env bash
set -euo pipefail

# Defaults match video-3d-eval/eval_segpruner.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints/video3d}"
# FREEZE_WORLD_POSITION_EMBEDDING=1
# FREEZE_GROUND_HEAD=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_train.sh"

TOKEN_KEEP_RATIO="${TOKEN_KEEP_RATIO:-0.11}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.35}"
LAM="${LAM:-0.5}"
SEGPRUNER_NEWLINE_STRATEGY="grid_drop"
DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
COMPRESSOR_CONFIG="{\"token_keep_ratio\":${TOKEN_KEEP_RATIO},\"important_ratio\":${IMPORTANT_RATIO},\"lam\":${LAM},\"newline_strategy\":\"${SEGPRUNER_NEWLINE_STRATEGY}\"}"

run_video3d_training \
    projector \
    segpruner \
    "${COMPRESSOR_CONFIG}" \
    "${DATA_YAML}" \
    "video3d_segpruner_k${TOKEN_KEEP_RATIO}_i${IMPORTANT_RATIO}_l${LAM}_${SEGPRUNER_NEWLINE_STRATEGY}_proj_llm_ft"
