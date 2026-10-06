#!/usr/bin/env bash
set -euo pipefail

# Video3D VisPruner post-training. The default budget is 22 tokens per frame.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints/video3d}"
# FREEZE_WORLD_POSITION_EMBEDDING=1
# FREEZE_GROUND_HEAD=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_train.sh"

VISUAL_TOKEN_NUM="${VISUAL_TOKEN_NUM:-22}"
IMPORTANT_RATIO="${IMPORTANT_RATIO:-0.5}"
PRUNE_STEP="${PRUNE_STEP:-8}"
VISPRUNER_NEWLINE_STRATEGY="grid_drop"
DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
COMPRESSOR_CONFIG="{\"visual_token_num\":${VISUAL_TOKEN_NUM},\"important_ratio\":${IMPORTANT_RATIO},\"prune_step\":${PRUNE_STEP},\"newline_strategy\":\"${VISPRUNER_NEWLINE_STRATEGY}\"}"

run_video3d_training \
    projector \
    vispruner \
    "${COMPRESSOR_CONFIG}" \
    "${DATA_YAML}" \
    "video3d_vispruner_t${VISUAL_TOKEN_NUM}_${VISPRUNER_NEWLINE_STRATEGY}_proj_llm_ft"
