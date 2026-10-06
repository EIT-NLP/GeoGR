#!/usr/bin/env bash
set -euo pipefail

# GeoGR evaluation on LLaVA-OneVision:
#   Stage I  - GeoSemZip at the projector output.
#   Stage II - GroupRoute with [8,23], BARS anchors [9,13], and RGS k=0.5.
#
# MODEL_PATH must be the Stage-II late-entry/early-exit adapted checkpoint.
# Example:
# Replace xxx with your actual Stage-II late-entry/early-exit checkpoint path.
#   MODEL_PATH=xxx \
#   GPU_IDS=0,1,2,3 bash "$0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${LOCAL_OV_MODEL_PATH:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] Set MODEL_PATH to the Stage-II adapted LLaVA-OV checkpoint." >&2
    exit 1
fi
export MODEL_PATH
export LOCAL_OV_MODEL_PATH="${LOCAL_OV_MODEL_PATH:-${MODEL_PATH}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/ov/geogr}"
export TASKS="${TASKS:-scanqa_val,sqa3d_test}"
export LIMIT="${LIMIT-}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export USE_DP="${USE_DP:-true}"
export BATCH_SIZE=1
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export ENABLE_3D_AUX=true
export SCANNET3D_COORDS_CACHE_ROOT="${SCANNET3D_COORDS_CACHE_ROOT:-${PROJECT_ROOT}/cache/scannet3d_coords}"
export OV_POOLED_COORDS_ROOT="${OV_POOLED_COORDS_ROOT:-${SCANNET3D_COORDS_CACHE_ROOT}}"

MODE="${MODE:-query_attention}"
LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-8}"
# early_exit_layer is exclusive, so 24 implements the paper window [8,23].
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-24}"
ANCHOR_LAYERS="${ANCHOR_LAYERS:-9,13}"
KEEP_RATIOS="${KEEP_RATIOS:-0.50,0.50}"
RECOVERY_LAYERS="${RECOVERY_LAYERS:-0}"
RANDOM_SEED="${RANDOM_SEED:-0}"
PROJECTOR_GROUP_DOMINANT_RATIO="${PROJECTOR_GROUP_DOMINANT_RATIO:-0.85}"
PROJECTOR_GROUP_VOXEL_SIZE="${PROJECTOR_GROUP_VOXEL_SIZE:-0.1}"
PROJECTOR_RESIDUAL_MERGE="${PROJECTOR_RESIDUAL_MERGE:-true}"
PROJECTOR_COVERAGE_RULE="${PROJECTOR_COVERAGE_RULE:-morton}"
PROJECTOR_RANDOM_SEED="${PROJECTOR_RANDOM_SEED:-0}"

if [[ -n "${LIMIT}" && ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] LIMIT must be empty or a positive integer." >&2
    exit 1
fi
case "${MODE}" in
    query_attention|projector_vtc_visionzip|random|baseline) ;;
    *)
        echo "[ERROR] MODE must be query_attention, projector_vtc_visionzip, random, or baseline." >&2
        exit 1
        ;;
esac
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Checkpoint is missing config.json: ${MODEL_PATH}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/bash-paper/ov-eval/_common_ov_eval.sh"

PROJECTOR_ARGS="mm_projector_compressor_type=voxel_vtc_visionzip,mm_projector_compressor_config={\"attention_reduce\":\"max\",\"dominant_ratio\":0.85,\"newline_strategy\":\"grid_drop\",\"target_keep_ratio\":0.3,\"voxel_size\":0.1,\"residual_merge\":${PROJECTOR_RESIDUAL_MERGE},\"coverage_rule\":\"${PROJECTOR_COVERAGE_RULE}\",\"random_seed\":${PROJECTOR_RANDOM_SEED}}"

if [[ "${MODE}" == "baseline" ]]; then
    LLM_ARGS="mm_llm_compressor_type=late_entry_early_exit,mm_llm_compressor_config={\"late_entry_layer\":${LATE_ENTRY_LAYER},\"early_exit_layer\":${EARLY_EXIT_LAYER}}"
    METHOD_NAME="ov_geosemzip_window_l${LATE_ENTRY_LAYER}_e${EARLY_EXIT_LAYER}"
else
    LLM_ARGS="mm_llm_compressor_type=group_wise_skip_recovery,mm_llm_compressor_config={\"late_entry_layer\":${LATE_ENTRY_LAYER},\"early_exit_layer\":${EARLY_EXIT_LAYER},\"anchor_layers\":[${ANCHOR_LAYERS}],\"keep_ratios\":[${KEEP_RATIOS}],\"recovery_layers\":${RECOVERY_LAYERS},\"score_mode\":\"${MODE}\",\"projector_dominant_ratio\":${PROJECTOR_GROUP_DOMINANT_RATIO},\"projector_voxel_size\":${PROJECTOR_GROUP_VOXEL_SIZE},\"random_seed\":${RANDOM_SEED}}"
    METHOD_NAME="ov_geogr_${MODE}_a${ANCHOR_LAYERS//,/-}_k${KEEP_RATIOS//,/-}_r${RECOVERY_LAYERS}_l${LATE_ENTRY_LAYER}_e${EARLY_EXIT_LAYER}"
fi

run_ov_eval "${METHOD_NAME}" "${PROJECTOR_ARGS},${LLM_ARGS}"
