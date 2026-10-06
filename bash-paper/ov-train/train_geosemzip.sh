#!/usr/bin/env bash
set -euo pipefail

# GeoSemZip Stage-I post-training for LLaVA-OneVision.
# The shared launcher freezes the vision encoder and trains the projector and
# full language model for one epoch with global batch 16 and learning rate 1e-5.
#
# Example:
# Replace xxx with your actual checkpoint path.
#   MODEL_PATH=xxx \
#   OUTPUT_ROOT=/path/to/checkpoints \
#   GPU_IDS=0,1,2,3 bash "$0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
VOXEL_SIZE="${VOXEL_SIZE:-0.1}"
TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO:-0.30}"
DOMINANT_RATIO="${DOMINANT_RATIO:-0.85}"
ATTENTION_REDUCE="${ATTENTION_REDUCE:-max}"
NEWLINE_STRATEGY="grid_drop"
RESIDUAL_MERGE="${RESIDUAL_MERGE:-true}"
COVERAGE_RULE="${COVERAGE_RULE:-morton}"
RANDOM_SEED="${RANDOM_SEED:-0}"

COMPRESSOR_TYPE="voxel_vtc_visionzip"
COMPRESSOR_CONFIG="{\"voxel_size\":${VOXEL_SIZE},\"target_keep_ratio\":${TARGET_KEEP_RATIO},\"dominant_ratio\":${DOMINANT_RATIO},\"attention_reduce\":\"${ATTENTION_REDUCE}\",\"newline_strategy\":\"${NEWLINE_STRATEGY}\",\"residual_merge\":${RESIDUAL_MERGE},\"coverage_rule\":\"${COVERAGE_RULE}\",\"random_seed\":${RANDOM_SEED}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_geosemzip_v${VOXEL_SIZE}_k${TARGET_KEEP_RATIO}_d${DOMINANT_RATIO}_${NEWLINE_STRATEGY}_proj_llm_ft}"

run_ov_train \
    "${DATA_YAML}" \
    "${RUN_NAME_PREFIX}" \
    "${COMPRESSOR_TYPE}" \
    "${COMPRESSOR_CONFIG}"
