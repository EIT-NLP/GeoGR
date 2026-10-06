#!/usr/bin/env bash
set -euo pipefail

# Video3D VisionZip post-training. The default budget is 22 tokens per frame.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints/video3d}"
# FREEZE_WORLD_POSITION_EMBEDDING=1
# FREEZE_GROUND_HEAD=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_train.sh"

DOMINANT_TOKENS="${DOMINANT_TOKENS:-19}"; CONTEXTUAL_TOKENS="${CONTEXTUAL_TOKENS:-3}"
VISIONZIP_NEWLINE_STRATEGY="grid_drop"
DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
COMPRESSOR_CONFIG="{\"dominant_tokens\":${DOMINANT_TOKENS},\"contextual_tokens\":${CONTEXTUAL_TOKENS},\"newline_strategy\":\"${VISIONZIP_NEWLINE_STRATEGY}\"}"

run_video3d_training \
    projector \
    visionzip \
    "${COMPRESSOR_CONFIG}" \
    "${DATA_YAML}" \
    "video3d_visionzip_d${DOMINANT_TOKENS}_c${CONTEXTUAL_TOKENS}_${VISIONZIP_NEWLINE_STRATEGY}_proj_llm_ft"
