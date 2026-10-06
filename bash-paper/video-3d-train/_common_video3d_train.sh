#!/usr/bin/env bash
# Shared Video3D-LLM compression training launcher.
# It passes a fixed compressor configuration to the upstream training entry
# point and keeps the vision encoder frozen by default.

set -euo pipefail

# Replace xxx with your actual checkpoint path before running a launcher.
#   export MODEL_PATH=xxx
# For local vision weights, replace xxx with your actual SigLIP checkpoint path.
#   export VISION_TOWER=xxx


TRAIN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO3D_COMP_ROOT="$(cd "${TRAIN_SCRIPT_DIR}/../.." && pwd)"
TRAIN_REPO_ROOT="${VIDEO3D_COMP_ROOT}/algorithm/Video-3D-LLM"
REFERENCE_VIDEO3D_ROOT="${VIDEO3D_COMP_ROOT}"
LMMS_ROOT="${LMMS_ROOT:-${VIDEO3D_COMP_ROOT}/lmms-eval}"
DEFAULT_DATA_ROOT="${VIDEO3D_COMP_ROOT}/data"
DEFAULT_HF_HOME="${VIDEO3D_COMP_ROOT}/.cache/huggingface"
DEFAULT_DATA_YAML="${TRAIN_SCRIPT_DIR}/data/multi_full.yaml"
DEFAULT_VISION_TOWER="${DEFAULT_HF_HOME}/siglip-so400m-patch14-384"
# Training and evaluation share versioned coordinate-cache payloads.
SCANNET3D_COORDS_CACHE_ROOT="${SCANNET3D_COORDS_CACHE_ROOT:-${VIDEO3D_COMP_ROOT}/cache/scannet3d_coords}"
CONDA_BASE="${CONDA_BASE:-}"
ENV_NAME="${ENV_NAME:-compress3d}"

count_csv_items() {
    local csv="$1"
    local -a items=()
    IFS=',' read -r -a items <<< "${csv}"
    echo "${#items[@]}"
}

require_divisible_global_batch() {
    local global_batch_size="$1"
    local num_gpus="$2"
    local per_device_batch_size="$3"
    local data_parallel_batch_size=$((num_gpus * per_device_batch_size))
    if (( global_batch_size < data_parallel_batch_size )); then
        echo "[train] GLOBAL_BATCH_SIZE=${global_batch_size} is smaller than the data-parallel micro-batch ${data_parallel_batch_size}." >&2
        exit 1
    fi
    if (( global_batch_size % data_parallel_batch_size != 0 )); then
        echo "[train] GLOBAL_BATCH_SIZE=${global_batch_size} is not divisible by ${num_gpus} GPUs * PER_DEVICE_TRAIN_BATCH_SIZE=${per_device_batch_size}." >&2
        exit 1
    fi
}

is_port_free() {
    local port="$1"
    python - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", port))
    except OSError:
        raise SystemExit(1)
PY
}

resolve_master_port() {
    local requested_port="$1"
    if is_port_free "${requested_port}"; then
        echo "${requested_port}"
        return 0
    fi

    if [[ "${AUTO_MASTER_PORT:-1}" != "1" ]]; then
        echo "[train] MASTER_PORT=${requested_port} is in use; choose another port or set AUTO_MASTER_PORT=1." >&2
        return 1
    fi

    python - "${requested_port}" <<'PY'
import socket
import sys

requested = int(sys.argv[1])
candidates = list(range(requested + 1, min(65535, requested + 2000)))
candidates.extend(range(20000, 50000))
seen = set()
for port in candidates:
    if port in seen or not (1 <= port <= 65535):
        continue
    seen.add(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

validate_data_yaml() {
    local data_yaml="$1"
    [[ -f "${data_yaml}" ]] || { echo "[train] DATA_YAML does not exist: ${data_yaml}" >&2; exit 1; }
    python - "${data_yaml}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

yaml_path = Path(os.path.expandvars(sys.argv[1])).expanduser().resolve()
data = yaml.safe_load(yaml_path.read_text()) or {}
datasets = data.get("datasets")
if not isinstance(datasets, list) or not datasets:
    raise SystemExit(f"[train] DATA_YAML contains no datasets: {yaml_path}")
for item in datasets:
    raw_path = item.get("json_path", "")
    path = Path(os.path.expandvars(raw_path)).expanduser()
    if not path.is_absolute():
        path = yaml_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"[train] Dataset file does not exist: {path}")
print(f"[train] dataset_count={len(datasets)}")
PY
}

setup_train_environment() {
    # Temporarily disable nounset while sourcing conda activation hooks.
    local nounset_was_on=0
    if [[ $- == *u* ]]; then
        nounset_was_on=1
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
        echo "[train] conda is unavailable; set CONDA_BASE or activate an environment first." >&2
        return 1
    fi
    if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
        conda activate "${ENV_NAME}"
    fi
    if [[ ${nounset_was_on} -eq 1 ]]; then
        set -u
    fi

    export WANDB_MODE="${WANDB_MODE:-offline}"
    export PYTHONPATH="${TRAIN_REPO_ROOT}:${LMMS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export LLAVA_SAFE_TOKENIZER_LOCAL_ONLY="${LLAVA_SAFE_TOKENIZER_LOCAL_ONLY:-1}"
    export SCANNET3D_COORDS_CACHE_ROOT

    # Keep all Hugging Face caches under one configurable root.
    export HF_HOME="${HF_HOME:-${DEFAULT_HF_HOME}}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
    mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

    # Offline mode prevents implicit network requests from from_pretrained().
    export OFFLINE_MODE="${OFFLINE_MODE:-1}"
    if [[ "${OFFLINE_MODE}" == "1" ]]; then
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export HF_DATASETS_OFFLINE=1
    else
        unset HF_HUB_OFFLINE
        unset TRANSFORMERS_OFFLINE
        unset HF_DATASETS_OFFLINE
    fi
}

run_video3d_training_impl() {
    local compressor_scope="$1"
    local compressor_type="$2"
    local compressor_config="$3"
    local data_yaml="$4"
    local run_name_prefix="$5"

    setup_train_environment
    validate_data_yaml "${data_yaml}"

    local gpu_ids="${GPU_IDS:-0,1,2,3}"
    local num_gpus
    num_gpus="$(count_csv_items "${gpu_ids}")"
    local per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    local global_batch_size="${GLOBAL_BATCH_SIZE:-16}"
    require_divisible_global_batch "${global_batch_size}" "${num_gpus}" "${per_device_train_batch_size}"
    local gradient_accumulation_steps=$((global_batch_size / (num_gpus * per_device_train_batch_size)))

    local requested_master_port="${MASTER_PORT:-43000}"
    local master_port
    master_port="$(resolve_master_port "${requested_master_port}")"
    if [[ "${master_port}" != "${requested_master_port}" ]]; then
        echo "[train] MASTER_PORT=${requested_master_port} is in use; selected ${master_port}."
    fi
    local prompt_version="${PROMPT_VERSION:-qwen_1_5}"
    local model_path="${MODEL_PATH:-${VIDEO3D_MODEL_PATH:-}}"
    if [[ -z "${model_path}" || ! -f "${model_path}/config.json" ]]; then
        echo "[train] Set MODEL_PATH (or VIDEO3D_MODEL_PATH) to a Video3D checkpoint containing config.json." >&2
        exit 1
    fi
    local default_vision_tower="google/siglip-so400m-patch14-384"
    if [[ -d "${DEFAULT_VISION_TOWER}" ]]; then
        default_vision_tower="${DEFAULT_VISION_TOWER}"
    fi
    local vision_tower="${VISION_TOWER:-${default_vision_tower}}"
    local image_folder="${IMAGE_FOLDER:-${DEFAULT_DATA_ROOT}}"
    local video_folder="${VIDEO_FOLDER:-${DEFAULT_DATA_ROOT}}"
    local embodiedscan_folder="${EMBODIEDSCAN_FOLDER:-${DEFAULT_DATA_ROOT}/embodiedscan}"
    local cache_dir="${CACHE_DIR:-${TRANSFORMERS_CACHE}}"
    local output_root="${OUTPUT_ROOT:-${VIDEO3D_COMP_ROOT}/video3d-checkpoint}"
    local timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    local run_name="${RUN_NAME:-${run_name_prefix}_${timestamp}}"
    local output_dir="${output_root}/${run_name}"
    local shell_log_dir="${SHELL_LOG_DIR:-${output_dir}/shell_logs}"
    local shell_log_file="${SHELL_LOG_FILE:-${shell_log_dir}/train_${timestamp}.log}"

    mkdir -p "${output_dir}" "${shell_log_dir}"
    mkdir -p "$(dirname "${shell_log_file}")"

    cd "${TRAIN_REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"

    # Follow the Video3D optimization recipe while freezing the vision encoder.
    # Dynamic compressed sequences are run without torch.compile.
    local -a cmd=(
        torchrun
        --nnodes=1
        --nproc_per_node="${num_gpus}"
        --master_port "${master_port}"
        llava/train/train_3d.py
        --deepspeed scripts/zero3.json
        --model_name_or_path "${model_path}"
        --cache_dir "${cache_dir}"
        --version "${prompt_version}"
        --data_path "${data_yaml}"
        --image_folder "${image_folder}"
        --video_folder "${video_folder}"
        --embodiedscan_folder "${embodiedscan_folder}"
        --mm_tunable_parts "${MM_TUNABLE_PARTS:-mm_mlp_adapter,mm_language_model}"
        --mm_vision_tower_lr 2e-6
        --vision_tower "${vision_tower}"
        --mm_projector_type mlp2x_gelu
        --mm_vision_select_layer -2
        --mm_use_im_start_end False
        --mm_use_im_patch_token False
        --image_aspect_ratio anyres_max_9
        --image_grid_pinpoints "(1x1),...,(6x6)"
        --mm_patch_merge_type spatial_unpad
        --bf16 True
        --run_name "${run_name}"
        --output_dir "${output_dir}"
        --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
        --per_device_train_batch_size "${per_device_train_batch_size}"
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
        --gradient_accumulation_steps "${gradient_accumulation_steps}"
        --evaluation_strategy no
        --save_strategy steps
        --save_steps "${SAVE_STEPS:-500}"
        --save_total_limit "${SAVE_TOTAL_LIMIT:-1}"
        --learning_rate "${LEARNING_RATE:-1e-5}"
        --weight_decay "${WEIGHT_DECAY:-0.0}"
        --warmup_ratio "${WARMUP_RATIO:-0.03}"
        --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
        --logging_steps "${LOGGING_STEPS:-1}"
        --tf32 True
        --model_max_length 32768
        --gradient_checkpointing True
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"
        --lazy_preprocess True
        --dataloader_drop_last True
        --mm_newline_position grid
        --add_spatial_instruction True
        --force_sample True
        --mm_spatial_pool_stride 2
        --world_position_embedding_type avg-discrete-sin3d
        --object_feature_type patch14-pe
        --ground_head_type infonce
        --group_by_task_length True
        --frame_sampling_strategy uniform
        --frames_upbound "${FRAMES_UPBOUND:-32}"
        --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
    )

    # Auxiliary 3D modules remain trainable unless explicitly frozen.
    if [[ "${FREEZE_WORLD_POSITION_EMBEDDING:-0}" == "1" ]]; then
        cmd+=(--freeze_world_position_embedding True)
    fi
    if [[ "${FREEZE_GROUND_HEAD:-0}" == "1" ]]; then
        cmd+=(--freeze_ground_head True)
    fi

    case "${compressor_scope}" in
        projector)
            cmd+=(
                --mm_projector_compressor_type "${compressor_type}"
                --mm_projector_compressor_config "${compressor_config}"
            )
            ;;
        llm)
            cmd+=(
                --mm_llm_compressor_type "${compressor_type}"
                --mm_llm_compressor_config "${compressor_config}"
            )
            ;;
        none)
            if [[ -n "${compressor_type}" || -n "${compressor_config}" ]]; then
                echo "[train] compressor type/config must be empty when compressor_scope=none." >&2
                exit 1
            fi
            ;;
        *)
            echo "[train] Unknown compressor scope: ${compressor_scope}" >&2
            exit 1
            ;;
    esac

    echo "[train] RUN_NAME=${run_name}"
    echo "[train] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[train] MASTER_PORT=${master_port}"
    echo "[train] shell_log_file=${shell_log_file}"
    echo "[train] DATA_YAML=${data_yaml}"
    echo "[train] MODEL_PATH=${model_path}"
    echo "[train] COMPRESSOR_SCOPE=${compressor_scope}"
    echo "[train] COMPRESSOR_TYPE=${compressor_type}"
    echo "[train] COMPRESSOR_CONFIG=${compressor_config}"
    echo "[train] coords_cache_root=${SCANNET3D_COORDS_CACHE_ROOT:-<none>}"
    echo "[train] GLOBAL_BATCH_SIZE=${global_batch_size}"
    echo "[train] PER_DEVICE_TRAIN_BATCH_SIZE=${per_device_train_batch_size}"
    echo "[train] GRADIENT_ACCUMULATION_STEPS=${gradient_accumulation_steps}"

    # DRY_RUN prints the final command without starting training.
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[train] dry_run=1"
        printf '[train] CMD='
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    printf '[train] CMD='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
}

run_video3d_training() {
    local compressor_scope="$1"
    local compressor_type="$2"
    local compressor_config="$3"
    local data_yaml="$4"
    local run_name_prefix="$5"
    local output_root="${OUTPUT_ROOT:-${VIDEO3D_COMP_ROOT}/video3d-checkpoint}"
    local timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    local run_name="${RUN_NAME:-${run_name_prefix}_${timestamp}}"
    local output_dir="${output_root}/${run_name}"
    local shell_log_dir="${SHELL_LOG_DIR:-${output_dir}/shell_logs}"
    local shell_log_file="${SHELL_LOG_FILE:-${shell_log_dir}/train_${timestamp}.log}"
    local exit_code

    mkdir -p "${output_dir}" "${shell_log_dir}" "$(dirname "${shell_log_file}")"

    # Store validation, command, and torchrun output in one log. The subshell
    # keeps errexit enabled so setup and training failures propagate.
    set +e
    (
        set -euo pipefail
        echo "[train] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
        RUN_TIMESTAMP="${timestamp}" \
        RUN_NAME="${run_name}" \
        SHELL_LOG_DIR="${shell_log_dir}" \
        SHELL_LOG_FILE="${shell_log_file}" \
        run_video3d_training_impl \
            "${compressor_scope}" \
            "${compressor_type}" \
            "${compressor_config}" \
            "${data_yaml}" \
            "${run_name_prefix}"
    ) 2>&1 | tee -a "${shell_log_file}"
    exit_code=${PIPESTATUS[0]}
    set -e

    echo "[train] finished_at=$(date '+%Y-%m-%d %H:%M:%S') exit_code=${exit_code}" | tee -a "${shell_log_file}"
    return "${exit_code}"
}
