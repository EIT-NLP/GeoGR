#!/usr/bin/env bash
# LLaVA-OneVision VisionZip post-training on ScanQA and SQA3D.
# VisionZip does not use the 3D coordinate cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"

# OV pooling produces a 14x14 visual grid per frame.
DOMINANT_TOKENS="${DOMINANT_TOKENS:-33}"
CONTEXTUAL_TOKENS="${CONTEXTUAL_TOKENS:-6}"
# DOMINANT_TOKENS="${DOMINANT_TOKENS:-17}"; CONTEXTUAL_TOKENS="${CONTEXTUAL_TOKENS:-3}" 
# DOMINANT_TOKENS="${DOMINANT_TOKENS:-49}"
# CONTEXTUAL_TOKENS="${CONTEXTUAL_TOKENS:-9}"
VISIONZIP_NEWLINE_STRATEGY="grid_drop"

COMPRESSOR_TYPE="visionzip"
COMPRESSOR_CONFIG="{\"dominant_tokens\":${DOMINANT_TOKENS},\"contextual_tokens\":${CONTEXTUAL_TOKENS},\"newline_strategy\":\"${VISIONZIP_NEWLINE_STRATEGY}\"}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_visionzip_d${DOMINANT_TOKENS}_c${CONTEXTUAL_TOKENS}_${VISIONZIP_NEWLINE_STRATEGY}_proj_llm_ft}"

run_ov_train "${DATA_YAML}" "${RUN_NAME_PREFIX}" "${COMPRESSOR_TYPE}" "${COMPRESSOR_CONFIG}"
