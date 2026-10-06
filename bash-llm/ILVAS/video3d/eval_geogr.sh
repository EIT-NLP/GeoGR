#!/usr/bin/env bash
set -euo pipefail

# GeoGR evaluation on Video3D-LLM. Layer boundaries are one-based and the
# early-exit boundary is exclusive, so [8,25) is the paper window [8,24].
#
# MODEL_PATH must be the Stage-II late-entry/early-exit adapted checkpoint.
# Example:
# Replace xxx with your actual Stage-II late-entry/early-exit checkpoint path.
#   MODEL_PATH=xxx \
#   GPU_IDS=0,1,2,3 bash "$0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] Set MODEL_PATH to the Stage-II adapted Video3D checkpoint." >&2
    exit 1
fi
export MODEL_PATH
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
export USE_DP="${USE_DP:-true}"
export TASKS="${TASKS:-scan2cap_val,scanqa_val,sqa3d_test,multi3drefer_val,scanrefer_val}"
export LIMIT="${LIMIT-}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/video3d/geogr}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-8}"
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-25}"
ANCHOR_LAYERS="${ANCHOR_LAYERS:-9,13}"
KEEP_RATIOS="${KEEP_RATIOS:-0.50,0.50}"
RECOVERY_LAYERS="${RECOVERY_LAYERS:-0}"
SCORE_MODE="${SCORE_MODE:-query_attention}"
PROJECTOR_RESIDUAL_MERGE="${PROJECTOR_RESIDUAL_MERGE:-true}"
PROJECTOR_COVERAGE_RULE="${PROJECTOR_COVERAGE_RULE:-morton}"
PROJECTOR_RANDOM_SEED="${PROJECTOR_RANDOM_SEED:-0}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Checkpoint is missing config.json: ${MODEL_PATH}" >&2
    exit 1
fi
if (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > ${#GPU_ARRAY[@]} )); then
    echo "[ERROR] NPROC_PER_NODE must be between 1 and the number of GPU_IDS." >&2
    exit 1
fi
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ || -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]]; then
        echo "[ERROR] GPU_IDS must contain unique non-negative integers: ${GPU_IDS}" >&2
        exit 1
    fi
    SEEN_GPU_IDS[${gpu_id}]=1
done

PROJECTOR_CONFIG="{\"voxel_size\":0.1,\"target_keep_ratio\":0.3,\"dominant_ratio\":0.85,\"attention_reduce\":\"max\",\"newline_strategy\":\"grid_drop\",\"residual_merge\":${PROJECTOR_RESIDUAL_MERGE},\"coverage_rule\":\"${PROJECTOR_COVERAGE_RULE}\",\"random_seed\":${PROJECTOR_RANDOM_SEED}}"
LLM_CONFIG="{\"late_entry_layer\":${LATE_ENTRY_LAYER},\"early_exit_layer\":${EARLY_EXIT_LAYER},\"anchor_layers\":[${ANCHOR_LAYERS}],\"keep_ratios\":[${KEEP_RATIOS}],\"recovery_layers\":${RECOVERY_LAYERS},\"score_mode\":\"${SCORE_MODE}\"}"
COMPRESSOR_ARGS="mm_projector_compressor_type=voxel_vtc_visionzip,mm_projector_compressor_config=${PROJECTOR_CONFIG},mm_llm_compressor_type=group_wise_skip_recovery,mm_llm_compressor_config=${LLM_CONFIG}"
METHOD_NAME="video3d_geogr_${SCORE_MODE}_a${ANCHOR_LAYERS//,/-}_k${KEEP_RATIOS//,/-}_r${RECOVERY_LAYERS}_l${LATE_ENTRY_LAYER}_e${EARLY_EXIT_LAYER}"

echo "[CHECK] model=${MODEL_PATH}"
echo "[CHECK] tasks=${TASKS}"
echo "[CHECK] projector=${PROJECTOR_CONFIG}"
echo "[CHECK] llm=${LLM_CONFIG}"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    exit 0
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/bash-paper/video-3d-eval/_common_video3d_eval.sh"
run_video3d_eval "${METHOD_NAME}" "${COMPRESSOR_ARGS}"
