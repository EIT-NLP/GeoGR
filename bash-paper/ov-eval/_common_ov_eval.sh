#!/usr/bin/env bash
set -euo pipefail

# Replace xxx with your actual checkpoint path before running a launcher.
#   export MODEL_PATH=xxx
# For local vision weights, replace xxx with your actual SigLIP checkpoint path.
#   export SIGLIP_MODEL_PATH=xxx


# Shared LLaVA-OneVision evaluation launcher.
# Projector compression operates on the pooled per-frame visual sequence.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_BASE="${CONDA_BASE:-}"
ENV_NAME="${ENV_NAME:-compress3d}"

LMMS_ROOT="${LMMS_ROOT:-${PROJECT_ROOT}/lmms-eval}"
LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-${PROJECT_ROOT}/algorithm/LLaVA-NeXT}"
VIDEO3D_COMP_ROOT="${VIDEO3D_COMP_ROOT:-${PROJECT_ROOT}}"
THREE_D_CONFIG="${THREE_D_CONFIG:-${LMMS_ROOT}/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml}"

# Set MODEL_PATH (or LOCAL_OV_MODEL_PATH) to a local checkpoint.
LOCAL_OV_MODEL_PATH="${LOCAL_OV_MODEL_PATH:-${MODEL_PATH:-}}"
MODEL_PATH="${MODEL_PATH:-${LOCAL_OV_MODEL_PATH}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] Set MODEL_PATH or LOCAL_OV_MODEL_PATH to a local checkpoint." >&2
    return 1 2>/dev/null || exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-llava_qwen}"
# For an unmerged LoRA adapter, set MODEL_PATH to the adapter directory and
# MODEL_BASE to the complete LLaVA-OV checkpoint. The loader merges LoRA at load time.
MODEL_BASE="${MODEL_BASE:-}"
if [[ -n "${MODEL_BASE}" ]]; then
    if [[ ! -d "${MODEL_BASE}" ]]; then
        echo "[ERROR] MODEL_BASE does not exist: ${MODEL_BASE}" >&2
        exit 1
    fi
    if [[ "${MODEL_NAME,,}" != *lora* ]]; then
        MODEL_NAME="${MODEL_NAME}_lora"
    fi
fi
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"
# The cache may contain both OV and Video3D coordinate payloads.
SCANNET3D_COORDS_CACHE_ROOT="${SCANNET3D_COORDS_CACHE_ROOT:-${PROJECT_ROOT}/cache/scannet3d_coords}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
# GPU_IDS="${GPU_IDS:-0,1,2,3}"
# GPU_IDS="${GPU_IDS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAME_NUM="${MAX_FRAME_NUM:-32}"
MM_SPATIAL_POOL_STRIDE="${MM_SPATIAL_POOL_STRIDE:-2}"
MM_SPATIAL_POOL_MODE="${MM_SPATIAL_POOL_MODE:-bilinear}"
TOKEN_STRATEGY="${TOKEN_STRATEGY:-single}"
VIDEO_DECODE_BACKEND="${VIDEO_DECODE_BACKEND:-decord}"
OV_SCANNET_VISUAL_MODE="${OV_SCANNET_VISUAL_MODE:-video}"
ENABLE_3D_AUX="${ENABLE_3D_AUX:-true}"
OV_POOLED_COORDS_ROOT="${OV_POOLED_COORDS_ROOT:-${SCANNET3D_COORDS_CACHE_ROOT}}"

# OV defaults to the text-answer benchmarks.
TASKS="${TASKS:-scanqa_val,sqa3d_test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/ov}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi
LIMIT="${LIMIT:-}"
VERBOSITY="${VERBOSITY:-INFO}"
# ScanQA metric normalization affects scoring only, not model outputs.
SCANNET3D_NORMALIZE_SCANQA_METRICS="${SCANNET3D_NORMALIZE_SCANQA_METRICS:-1}"
TEXT_MAX_NEW_TOKENS="${TEXT_MAX_NEW_TOKENS:-512}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-512}"
# The default prompt matches the 3D benchmark protocol.
DEFAULT_EXTRA_PROMPT="The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions."
EXTRA_PROMPT="${EXTRA_PROMPT-${DEFAULT_EXTRA_PROMPT}}"

USE_DP="${USE_DP:-true}"
SHOW_ALL_RANK_PROGRESS="${SHOW_ALL_RANK_PROGRESS:-false}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29601}"

GPU_ARR=()
if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
else
    GPU_ARR=("0")
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARR[@]}}"

activate_conda_env() {
    # Temporarily disable nounset while sourcing conda activation hooks.
    local nounset_was_enabled=0
    if [[ $- == *u* ]]; then
        nounset_was_enabled=1
        set +u
    fi
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        : "Using the already active virtual environment."
    else
        echo "[ERROR] conda is unavailable; set CONDA_BASE or activate an environment first." >&2
        return 1
    fi
    if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
        conda activate "${ENV_NAME}"
    fi
    if [[ "${nounset_was_enabled}" -eq 1 ]]; then
        set -u
    fi
}

task_gen_kwargs() {
    local task_name="$1"
    case "${task_name}" in
        scan2cap_*)
            # Captioning tasks use an independently configurable output limit.
            echo "do_sample=false,num_beams=1,temperature=0.0,max_new_tokens=${CAPTION_MAX_NEW_TOKENS}"
            ;;
        *)
            echo "do_sample=false,num_beams=1,temperature=0.0,max_new_tokens=${TEXT_MAX_NEW_TOKENS}"
            ;;
    esac
}

trim_task_name() {
    local raw="$1"
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    echo "${raw}"
}

run_ov_eval() {
    local method_name="$1"
    local compressor_args="$2"

    activate_conda_env

    export TOKENIZERS_PARALLELISM=false
    export PYTHONWARNINGS=ignore
    export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
    export SIGLIP_MODEL_PATH
    export VIDEO3D_COMP_ROOT
    export SCANNET3D_COORDS_CACHE_ROOT
    export OV_POOLED_COORDS_ROOT
    export SCANNET3D_NORMALIZE_SCANQA_METRICS
    export LMMS_EVAL_SCANNET3D_CONFIG="${THREE_D_CONFIG}"
    export LMMS_EVAL_PLUGINS="ov_lmms_plugin${LMMS_EVAL_PLUGINS:+,${LMMS_EVAL_PLUGINS}}"
    export PYTHONPATH="${LLAVA_NEXT_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
    mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
    if [[ "${OFFLINE_MODE}" == "1" ]]; then
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    fi
    if [[ -n "${GPU_IDS}" ]]; then
        export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    fi
    if [[ "${USE_DP}" == "true" && "${NPROC_PER_NODE}" -gt 1 && "${NPROC_PER_NODE}" -gt "${#GPU_ARR[@]}" ]]; then
        echo "[ERROR] NPROC_PER_NODE (${NPROC_PER_NODE}) exceeds the number of GPU_IDS (${#GPU_ARR[@]})." >&2
        exit 1
    fi
    if [[ "${BATCH_SIZE}" != "1" ]]; then
        echo "[ERROR] LLaVA-OneVision generation requires BATCH_SIZE=1; got ${BATCH_SIZE}." >&2
        exit 1
    fi

    mkdir -p "${OUTPUT_ROOT}"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local run_root="${OUTPUT_ROOT}/${method_name}_${ts}"
    mkdir -p "${run_root}"
    OV_LAST_RUN_ROOT="${run_root}"

    local model_args
    model_args="pretrained=${MODEL_PATH},conv_template=${CONV_TEMPLATE},device_map=${DEVICE_MAP},model_name=${MODEL_NAME},attn_implementation=${ATTN_IMPLEMENTATION},max_frames_num=${MAX_FRAME_NUM},mm_spatial_pool_stride=${MM_SPATIAL_POOL_STRIDE},mm_spatial_pool_mode=${MM_SPATIAL_POOL_MODE},token_strategy=${TOKEN_STRATEGY},video_decode_backend=${VIDEO_DECODE_BACKEND},three_d_config=${THREE_D_CONFIG},enable_3d_aux=${ENABLE_3D_AUX},mm_patch_merge_type=spatial_unpad,mm_newline_position=grid,scannet_visual_mode=${OV_SCANNET_VISUAL_MODE}"
    if [[ -n "${MODEL_BASE}" ]]; then
        model_args="${model_args},model_base=${MODEL_BASE}"
    fi
    if [[ -n "${EXTRA_PROMPT}" ]]; then
        model_args="${model_args},extra_prompt=${EXTRA_PROMPT}"
    fi
    if [[ -n "${OV_POOLED_COORDS_ROOT}" ]]; then
        model_args="${model_args},pooled_coords_root=${OV_POOLED_COORDS_ROOT}"
    fi
    if [[ -n "${compressor_args}" ]]; then
        model_args="${model_args},${compressor_args}"
    fi

    echo "[INFO] method=${method_name}"
    echo "[INFO] model_path=${MODEL_PATH}"
    echo "[INFO] model_base=${MODEL_BASE:-<none>}"
    echo "[INFO] siglip_model_path=${SIGLIP_MODEL_PATH}"
    echo "[INFO] llava_next_root=${LLAVA_NEXT_ROOT}"
    echo "[INFO] lmms_root=${LMMS_ROOT}"
    echo "[INFO] scannet_visual_mode=${OV_SCANNET_VISUAL_MODE}"
    echo "[INFO] coords_cache_root=${OV_POOLED_COORDS_ROOT:-<none>}"
    echo "[INFO] normalize_scanqa_metrics=${SCANNET3D_NORMALIZE_SCANQA_METRICS}"
    echo "[INFO] tasks=${TASKS}"
    echo "[INFO] output_root=${run_root}"
    echo "[INFO] use_dp=${USE_DP}"
    echo "[INFO] nproc_per_node=${NPROC_PER_NODE}"

    local task
    local idx=0
    IFS=',' read -r -a _TASK_ARRAY <<< "${TASKS}"
    for task in "${_TASK_ARRAY[@]}"; do
        task="$(trim_task_name "${task}")"
        [[ -z "${task}" ]] && continue
        idx=$((idx + 1))

        local task_out="${run_root}/${task}"
        local task_log="${run_root}/${task}.log"
        local task_rank_log_dir="${run_root}/${task}_ranks"
        local task_worker_script="${run_root}/${task}_worker.sh"
        local task_master_port=$((MASTER_PORT_BASE + idx))
        local gen_kwargs
        gen_kwargs="$(task_gen_kwargs "${task}")"
        mkdir -p "${task_out}" "${task_rank_log_dir}"

        echo "============================================================"
        echo "[INFO] task=${task}"
        echo "[INFO] output=${task_out}"
        echo "[INFO] log=${task_log}"
        echo "[INFO] gen_kwargs=${gen_kwargs}"
        echo "============================================================"

        if [[ "${USE_DP}" == "true" && "${NPROC_PER_NODE}" -gt 1 ]]; then
            cat > "${task_worker_script}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IFS=',' read -r -a _GPU_ARR <<< "${DP_GPU_IDS}"
_LOCAL_RANK="${LOCAL_RANK:-0}"
if [[ "${_LOCAL_RANK}" -ge "${#_GPU_ARR[@]}" ]]; then
  echo "[ERROR][rank ${RANK:-?}] LOCAL_RANK ${_LOCAL_RANK} is outside GPU_IDS." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${_GPU_ARR[${_LOCAL_RANK}]}"
# Expose one GPU per worker and reset the rank seen by Accelerate.
export LOCAL_RANK=0

CMD=(
  python -m lmms_eval eval
  --model llava_onevision_3d
  --tasks "${DP_TASK_NAME}"
  --model_args "${DP_TASK_MODEL_ARGS}"
  --batch_size "${DP_TASK_BATCH_SIZE}"
  --output_path "${DP_TASK_OUT}"
  --verbosity "${DP_TASK_VERBOSITY}"
  --log_samples
)

if [[ -n "${DP_TASK_GEN_KWARGS}" ]]; then
  CMD+=(--gen_kwargs "${DP_TASK_GEN_KWARGS}")
fi
if [[ -n "${DP_TASK_LIMIT}" ]]; then
  CMD+=(--limit "${DP_TASK_LIMIT}")
fi

exec "${CMD[@]}"
EOF
            chmod +x "${task_worker_script}"

            set +e
            (
                cd "${LMMS_ROOT}"
                DP_GPU_IDS="${GPU_IDS}" \
                DP_TASK_NAME="${task}" \
                DP_TASK_MODEL_ARGS="${model_args}" \
                DP_TASK_GEN_KWARGS="${gen_kwargs}" \
                DP_TASK_BATCH_SIZE="${BATCH_SIZE}" \
                DP_TASK_OUT="${task_out}" \
                DP_TASK_VERBOSITY="${VERBOSITY}" \
                DP_TASK_LIMIT="${LIMIT}" \
                LMMS_SHOW_ALL_RANK_PROGRESS="${SHOW_ALL_RANK_PROGRESS}" \
                torchrun \
                    --standalone \
                    --nnodes=1 \
                    --nproc-per-node="${NPROC_PER_NODE}" \
                    --master-port="${task_master_port}" \
                    --no-python \
                    --tee 3 \
                    --log-dir "${task_rank_log_dir}" \
                    "${task_worker_script}"
            ) > "${task_log}" 2>&1
            local rc=$?
            set -e
            if [[ "${rc}" -ne 0 ]]; then
                echo "[ERROR] Task failed: ${task}, rc=${rc}, log=${task_log}" >&2
                return "${rc}"
            fi
        else
            local cmd=(
                python -m lmms_eval eval
                --model llava_onevision_3d
                --tasks "${task}"
                --model_args "${model_args}"
                --batch_size "${BATCH_SIZE}"
                --output_path "${task_out}"
                --verbosity "${VERBOSITY}"
                --log_samples
            )
            if [[ -n "${gen_kwargs}" ]]; then
                cmd+=(--gen_kwargs "${gen_kwargs}")
            fi
            if [[ -n "${LIMIT}" ]]; then
                cmd+=(--limit "${LIMIT}")
            fi
            (
                cd "${LMMS_ROOT}"
                "${cmd[@]}"
            ) 2>&1 | tee "${task_log}"
        fi

        if grep -q "Traceback (most recent call last)" "${task_log}"; then
            echo "[ERROR] Python traceback detected: ${task}, log=${task_log}" >&2
            return 1
        fi
    done

    echo "[INFO] Completed: ${run_root}"
}
