#!/usr/bin/env bash
set -euo pipefail

# GeoSemZip (Stage I) on Video3D-LLM.
#
# Model path examples:
# Replace xxx with your actual checkpoint path.
#   MODEL_PATH=xxx bash "$0"
# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
#   MODEL_PATH=xxx bash "$0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_eval.sh"

VOXEL_SIZE="${VOXEL_SIZE:-0.1}"
TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO:-0.30}"
DOMINANT_RATIO="${DOMINANT_RATIO:-0.85}"
ATTENTION_REDUCE="${ATTENTION_REDUCE:-max}"
NEWLINE_STRATEGY="grid_drop"
RESIDUAL_MERGE="${RESIDUAL_MERGE:-true}"
COVERAGE_RULE="${COVERAGE_RULE:-morton}"
RANDOM_SEED="${RANDOM_SEED:-0}"

COMPRESSOR_ARGS="mm_projector_compressor_type=voxel_vtc_visionzip,mm_projector_compressor_config={\"voxel_size\":${VOXEL_SIZE},\"target_keep_ratio\":${TARGET_KEEP_RATIO},\"dominant_ratio\":${DOMINANT_RATIO},\"attention_reduce\":\"${ATTENTION_REDUCE}\",\"newline_strategy\":\"${NEWLINE_STRATEGY}\",\"residual_merge\":${RESIDUAL_MERGE},\"coverage_rule\":\"${COVERAGE_RULE}\",\"random_seed\":${RANDOM_SEED}}"

run_video3d_eval \
    "video3d_geosemzip_v${VOXEL_SIZE}_k${TARGET_KEEP_RATIO}_d${DOMINANT_RATIO}_${NEWLINE_STRATEGY}" \
    "${COMPRESSOR_ARGS}"
