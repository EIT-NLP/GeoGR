#!/usr/bin/env bash
set -euo pipefail

# Eight-rank launcher. Each rank owns one GPU; lmms-eval merges samples/metrics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export USE_DP=true
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-30141}"
export SHOW_ALL_RANK_PROGRESS="${SHOW_ALL_RANK_PROGRESS:-false}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 || "${NPROC_PER_NODE}" -ne 8 ]]; then
    echo "[ERROR] The 8-GPU launcher requires 8 GPU_IDS entries and NPROC_PER_NODE=8." >&2
    exit 1
fi
declare -A SEEN=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ || -n "${SEEN[${gpu_id}]:-}" ]]; then
        echo "[ERROR] GPU_IDS must contain 8 unique non-negative integers: ${GPU_IDS}" >&2
        exit 1
    fi
    SEEN[${gpu_id}]=1
done

exec bash "${SCRIPT_DIR}/eval_llava_ov_group_wise_skip_recovery.sh"
