#!/usr/bin/env bash
# Shared LLaVA-OneVision post-training launcher.
# The default freezes the vision encoder and tunes the projector and LLM.

set -euo pipefail

# Replace xxx with your actual checkpoint path before running a launcher.
#   export MODEL_PATH=xxx
# For local vision weights, replace xxx with your actual SigLIP checkpoint path.
#   export SIGLIP_MODEL_PATH=xxx


TRAIN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAIN_SCRIPT_DIR}/../.." && pwd)"

CONDA_BASE="${CONDA_BASE:-}"
ENV_NAME="${ENV_NAME:-compress3d}"

LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-${PROJECT_ROOT}/algorithm/LLaVA-NeXT}"
MODEL_PATH="${MODEL_PATH:-}"
VISION_TOWER="${VISION_TOWER:-${SIGLIP_MODEL_PATH:-}}"
SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-${VISION_TOWER}}"

DEFAULT_DATA_YAML="${TRAIN_SCRIPT_DIR}/data/multi_full.yaml"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-${PROJECT_ROOT}/bash-paper/ov-train/data/prepared_full_ov}"
PREPARED_JSON_DIR="${PREPARED_JSON_DIR:-${PREPARED_DATA_ROOT}/json}"
PREPARED_FRAME_ROOT="${PREPARED_FRAME_ROOT:-${PREPARED_DATA_ROOT}/frames}"
PREPARED_DATA_YAML="${PREPARED_DATA_YAML:-${PREPARED_DATA_ROOT}/multi_full_ov_video.yaml}"
THREE_D_CONFIG="${THREE_D_CONFIG:-${PROJECT_ROOT}/lmms-eval/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints/ov}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-43100}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
FRAMES_UPBOUND="${FRAMES_UPBOUND:-32}"
MM_SPATIAL_POOL_STRIDE="${MM_SPATIAL_POOL_STRIDE:-2}"
MM_SPATIAL_POOL_MODE="${MM_SPATIAL_POOL_MODE:-bilinear}"
MM_NEWLINE_POSITION="${MM_NEWLINE_POSITION:-grid}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MODEL_CLASS_NAME="${MODEL_CLASS_NAME:-LlavaQwen}"
MM_TUNABLE_PARTS="${MM_TUNABLE_PARTS:-mm_mlp_adapter,mm_language_model}"
SCANNET3D_COORDS_CACHE_ROOT="${SCANNET3D_COORDS_CACHE_ROOT:-${PROJECT_ROOT}/cache/scannet3d_coords}"

count_csv_items() {
    local csv="$1"
    local -a items=()
    IFS=',' read -r -a items <<< "${csv}"
    echo "${#items[@]}"
}

compressor_requires_scannet_coords() {
    local compressor_type="$1"
    case "${compressor_type}" in
        # SegPruner also consumes patch-level 3D coordinates for its
        # geometry-aware FPS stage; make the pooled-coordinate preparation
        # automatic just like the voxel and k-center compressors.
        segpruner|voxel_*|spatial_kcenter_merge)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

require_divisible_global_batch() {
    local global_batch_size="$1"
    local num_gpus="$2"
    local per_device_batch_size="$3"
    local data_parallel_batch_size=$((num_gpus * per_device_batch_size))
    if (( global_batch_size < num_gpus )); then
        echo "[train] GLOBAL_BATCH_SIZE=${global_batch_size} is smaller than the GPU count ${num_gpus}." >&2
        exit 1
    fi
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

setup_ov_train_environment() {
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

    export PYTHONPATH="${LLAVA_NEXT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export SIGLIP_MODEL_PATH
    export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export LLAVA_SAFE_TOKENIZER_LOCAL_ONLY="${LLAVA_SAFE_TOKENIZER_LOCAL_ONLY:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${OUTPUT_ROOT}"

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

require_path() {
    local path="$1"
    local desc="$2"
    if [[ ! -e "${path}" ]]; then
        echo "[train] Missing ${desc}: ${path}" >&2
        exit 1
    fi
}

prepared_ov_scannet_data_exists() {
    [[ -s "${PREPARED_DATA_YAML}" ]] || return 1
    [[ -d "${PREPARED_JSON_DIR}" ]] || return 1
    [[ -d "${PREPARED_FRAME_ROOT}/shareVideoGPTV" ]] || return 1

    python - "${PREPARED_DATA_YAML}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

yaml_path = Path(os.path.expandvars(sys.argv[1])).expanduser().resolve()
data = yaml.safe_load(yaml_path.read_text()) or {}
datasets = data.get("datasets")
if not isinstance(datasets, list) or not datasets:
    raise SystemExit(1)
for dataset in datasets:
    json_path = Path(os.path.expandvars(dataset.get("json_path", ""))).expanduser()
    if not json_path.is_absolute():
        json_path = yaml_path.parent / json_path
    json_path = json_path.resolve()
    if not json_path.is_file() or json_path.stat().st_size <= 0:
        raise SystemExit(1)
PY
}

prepare_ov_scannet_data() {
    local input_yaml="$1"
    setup_ov_train_environment

    require_path "${input_yaml}" "training data YAML"
    require_path "${DATA_ROOT}" "data root"
    require_path "${TRAIN_SCRIPT_DIR}/prepare_ov_scannet_train_data.py" "OV data-preparation script"

    if [[ "${FORCE_PREPARE_OV_DATA:-0}" != "1" ]] && prepared_ov_scannet_data_exists; then
        echo "[train] prepared OV data exists, skip prepare: ${PREPARED_DATA_YAML}"
        return 0
    fi

    local prepare_overwrite_flag=()
    if [[ "${PREPARE_OVERWRITE_LINKS:-1}" == "1" ]]; then
        prepare_overwrite_flag=(--overwrite-links)
    fi

    python "${TRAIN_SCRIPT_DIR}/prepare_ov_scannet_train_data.py" \
        --input-yaml "${input_yaml}" \
        --output-yaml "${PREPARED_DATA_YAML}" \
        --output-json-dir "${PREPARED_JSON_DIR}" \
        --frame-root "${PREPARED_FRAME_ROOT}" \
        --data-root "${DATA_ROOT}" \
        --frames-upbound "${FRAMES_UPBOUND}" \
        "${prepare_overwrite_flag[@]}"
}

prepare_ov_scannet_pooled_coords() {
    require_path "${PREPARED_DATA_YAML}" "prepared OV training YAML"
    require_path "${PREPARED_FRAME_ROOT}" "prepared OV frame directory"
    require_path "${THREE_D_CONFIG}" "ScanNet3D data-path configuration"
    require_path "${TRAIN_SCRIPT_DIR}/precompute_train_pooled_coords.py" "OV coordinate-cache precompute script"

    local overwrite_flag=()
    if [[ "${COORD_PRECOMPUTE_OVERWRITE:-0}" == "1" ]]; then
        overwrite_flag=(--overwrite)
    fi
    local limit_flag=()
    if [[ -n "${COORD_PRECOMPUTE_LIMIT:-}" ]]; then
        limit_flag=(--limit "${COORD_PRECOMPUTE_LIMIT}")
    fi

    echo "[train] precompute pooled coords: ${SCANNET3D_COORDS_CACHE_ROOT}"
    python "${TRAIN_SCRIPT_DIR}/precompute_train_pooled_coords.py" \
        --input-yaml "${PREPARED_DATA_YAML}" \
        --frame-root "${PREPARED_FRAME_ROOT}" \
        --output-root "${SCANNET3D_COORDS_CACHE_ROOT}" \
        --three-d-config "${THREE_D_CONFIG}" \
        --frame-shape "${SCANNET_POOLED_COORDS_FRAME_SHAPE:-14,14}" \
        --crop-size "${SCANNET_POOLED_COORDS_CROP_SIZE:-384}" \
        --depth-scale "${SCANNET_POOLED_COORDS_DEPTH_SCALE:-1000.0}" \
        "${limit_flag[@]}" \
        "${overwrite_flag[@]}"
}

run_ov_train() {
    local input_data_yaml="${1:-${DEFAULT_DATA_YAML}}"
    local run_name_prefix="${2:-llava_ov_scanqa_sqa3d_proj_llm_ft}"
    local compressor_type="${3:-}"
    local compressor_config="${4:-}"
    local llm_compressor_type="${5:-}"
    local llm_compressor_config="${6:-}"

    prepare_ov_scannet_data "${input_data_yaml}"

    require_path "${LLAVA_NEXT_ROOT}/llava/train/train_mem.py" "LLaVA-OV training entry point"
    require_path "${LLAVA_NEXT_ROOT}/scripts/zero3.json" "DeepSpeed ZeRO-3 configuration"
    require_path "${MODEL_PATH}" "initial OV checkpoint"
    require_path "${VISION_TOWER}" "SigLip vision tower"

    local num_gpus
    num_gpus="$(count_csv_items "${GPU_IDS}")"
    local per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"
    require_divisible_global_batch "${GLOBAL_BATCH_SIZE}" "${num_gpus}" "${per_device_train_batch_size}"
    local gradient_accumulation_steps=$((GLOBAL_BATCH_SIZE / (num_gpus * per_device_train_batch_size)))
    local master_port
    master_port="$(resolve_master_port "${MASTER_PORT}")"
    if [[ "${master_port}" != "${MASTER_PORT}" ]]; then
        echo "[train] MASTER_PORT=${MASTER_PORT} is in use; selected ${master_port}."
    fi

    local prompt_version="${PROMPT_VERSION:-qwen_1_5}"
    local timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    local run_name
    if [[ -n "${RUN_NAME:-}" ]]; then
        run_name="${RUN_NAME}"
    elif [[ "${ADD_TIMESTAMP:-0}" == "1" ]]; then
        run_name="${run_name_prefix}_${timestamp}"
    else
        run_name="${run_name_prefix}"
    fi
    local output_dir="${OUTPUT_ROOT}/${run_name}"
    local logging_dir="${LOGGING_DIR:-${output_dir}/trainer_logs}"
    local shell_log_dir="${SHELL_LOG_DIR:-${output_dir}/shell_logs}"
    local shell_log_file="${SHELL_LOG_FILE:-${shell_log_dir}/train_${timestamp}.log}"
    local save_steps="${SAVE_STEPS:-500}"
    local dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"
    local report_to="${REPORT_TO:-none}"
    local scannet_pooled_coords_root="${SCANNET_POOLED_COORDS_ROOT:-}"
    local scannet_pooled_coords_required="${SCANNET_POOLED_COORDS_REQUIRED:-false}"
    local video_image_aspect_ratio="${VIDEO_IMAGE_ASPECT_RATIO:-}"
    local compressor_needs_coords="false"
    if [[ -n "${compressor_type}" ]] && compressor_requires_scannet_coords "${compressor_type}"; then
        compressor_needs_coords="true"
    fi
    if [[ "${compressor_needs_coords}" == "true" || "${FORCE_SCANNET_COORDS:-0}" == "1" ]]; then
        scannet_pooled_coords_root="${scannet_pooled_coords_root:-${SCANNET3D_COORDS_CACHE_ROOT}}"
        scannet_pooled_coords_required="${SCANNET_POOLED_COORDS_REQUIRED:-true}"
        video_image_aspect_ratio="${video_image_aspect_ratio:-pad}"
        if [[ "${compressor_needs_coords}" == "true" && "${PRECOMPUTE_SCANNET_COORDS:-1}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
            prepare_ov_scannet_pooled_coords
            if [[ "${PRECOMPUTE_ONLY:-0}" == "1" ]]; then
                echo "[train] PRECOMPUTE_ONLY=1; generated the coordinate cache without starting training."
                return 0
            fi
        fi
    fi

    cd "${LLAVA_NEXT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

    local -a cmd=(
        torchrun
        --nnodes=1
        --nproc_per_node="${num_gpus}"
        --master_port "${master_port}"
        llava/train/train_mem.py
        --deepspeed scripts/zero3.json
        --model_name_or_path "${MODEL_PATH}"
        --model_class_name "${MODEL_CLASS_NAME}"
        --cache_dir "${TRANSFORMERS_CACHE}"
        --version "${prompt_version}"
        --data_path "${PREPARED_DATA_YAML}"
        --image_folder "${DATA_ROOT}"
        --video_folder "${PREPARED_FRAME_ROOT}"
        --mm_tunable_parts "${MM_TUNABLE_PARTS}"
        --vision_tower "${VISION_TOWER}"
        --mm_projector_type mlp2x_gelu
        --mm_vision_select_layer -2
        --mm_use_im_start_end False
        --mm_use_im_patch_token False
        --group_by_modality_length True
        --image_aspect_ratio anyres_max_9
        --image_grid_pinpoints "(1x1),...,(6x6)"
        --mm_patch_merge_type spatial_unpad
        --mm_newline_position "${MM_NEWLINE_POSITION}"
        --mm_spatial_pool_stride "${MM_SPATIAL_POOL_STRIDE}"
        --mm_spatial_pool_mode "${MM_SPATIAL_POOL_MODE}"
        --bf16 True
        --run_name "${run_name}"
        --output_dir "${output_dir}"
        --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
        --per_device_train_batch_size "${per_device_train_batch_size}"
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
        --gradient_accumulation_steps "${gradient_accumulation_steps}"
        --evaluation_strategy no
        --save_strategy steps
        --save_steps "${save_steps}"
        --save_total_limit "${SAVE_TOTAL_LIMIT:-1}"
        --learning_rate "${LEARNING_RATE:-1e-5}"
        --weight_decay "${WEIGHT_DECAY:-0.0}"
        --warmup_ratio "${WARMUP_RATIO:-0.03}"
        --lr_scheduler_type cosine
        --logging_steps "${LOGGING_STEPS:-1}"
        --logging_dir "${logging_dir}"
        --tf32 True
        --model_max_length "${MODEL_MAX_LENGTH:-32768}"
        --gradient_checkpointing True
        --dataloader_num_workers "${dataloader_num_workers}"
        --lazy_preprocess True
        --report_to "${report_to}"
        --dataloader_drop_last True
        --frames_upbound "${FRAMES_UPBOUND}"
        --force_sample True
        --attn_implementation "${ATTN_IMPLEMENTATION}"
    )

    if [[ -n "${video_image_aspect_ratio}" ]]; then
        cmd+=(
            --video_image_aspect_ratio "${video_image_aspect_ratio}"
        )
    fi

    if [[ -n "${compressor_type}" ]]; then
        cmd+=(
            --mm_projector_compressor_type "${compressor_type}"
            --mm_projector_compressor_config "${compressor_config}"
        )
    fi

    if [[ -n "${llm_compressor_type}" ]]; then
        cmd+=(
            --mm_llm_compressor_type "${llm_compressor_type}"
            --mm_llm_compressor_config "${llm_compressor_config}"
        )
    fi

    # Optional strict LLM adaptation boundary. Unset keeps every existing
    # training script on the original full-module trainability behavior.
    if [[ -n "${LLM_TRAINABLE_LAYER_RANGE:-}" ]]; then
        cmd+=(--llm_trainable_layer_range "${LLM_TRAINABLE_LAYER_RANGE}")
    fi
    if [[ "${LORA_ENABLE:-false}" == "true" ]]; then
        cmd+=(
            --lora_enable True
            --lora_r "${LORA_R:-16}"
            --lora_alpha "${LORA_ALPHA:-32}"
            --lora_dropout "${LORA_DROPOUT:-0.05}"
            --lora_bias "${LORA_BIAS:-none}"
        )
    fi

    if [[ -n "${scannet_pooled_coords_root}" ]]; then
        cmd+=(
            --scannet_pooled_coords_root "${scannet_pooled_coords_root}"
            --scannet_pooled_coords_required "${scannet_pooled_coords_required}"
            --scannet_pooled_coords_field "${SCANNET_POOLED_COORDS_FIELD:-ov_pad_avg14}"
            --scannet_pooled_coords_frame_shape "${SCANNET_POOLED_COORDS_FRAME_SHAPE:-14,14}"
            --scannet_pooled_coords_crop_size "${SCANNET_POOLED_COORDS_CROP_SIZE:-384}"
            --scannet_pooled_coords_depth_scale "${SCANNET_POOLED_COORDS_DEPTH_SCALE:-1000.0}"
        )
    fi

    if [[ "${TORCH_COMPILE:-0}" == "1" ]]; then
        cmd+=(--torch_compile True --torch_compile_backend "${TORCH_COMPILE_BACKEND:-inductor}")
    fi
    if [[ -n "${RESUME_FROM_CHECKPOINT:-}" && "${RESUME_FROM_CHECKPOINT}" != "auto" ]]; then
        cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    fi

    local latest_checkpoint=""
    if compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
        latest_checkpoint="$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    fi

    echo "[train] run_name=${run_name}"
    echo "[train] llava_next_root=${LLAVA_NEXT_ROOT}"
    echo "[train] model_path=${MODEL_PATH}"
    echo "[train] model_class_name=${MODEL_CLASS_NAME}"
    echo "[train] vision_tower=${VISION_TOWER}"
    echo "[train] input_data_yaml=${input_data_yaml}"
    echo "[train] prepared_data_yaml=${PREPARED_DATA_YAML}"
    echo "[train] video_folder=${PREPARED_FRAME_ROOT}"
    echo "[train] output_dir=${output_dir}"
    echo "[train] logging_dir=${logging_dir}"
    echo "[train] shell_log_file=${shell_log_file}"
    echo "[train] auto_resume_checkpoint=${latest_checkpoint:-none}"
    echo "[train] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "[train] num_gpus=${num_gpus}"
    echo "[train] master_port=${master_port}"
    echo "[train] global_batch_size=${GLOBAL_BATCH_SIZE}"
    echo "[train] per_device_train_batch_size=${per_device_train_batch_size}"
    echo "[train] gradient_accumulation_steps=${gradient_accumulation_steps}"
    echo "[train] frames_upbound=${FRAMES_UPBOUND}"
    echo "[train] video_image_aspect_ratio=${video_image_aspect_ratio:-default}"
    echo "[train] compressor_type=${compressor_type:-none}"
    echo "[train] llm_compressor_type=${llm_compressor_type:-none}"
    echo "[train] mm_tunable_parts=${MM_TUNABLE_PARTS}"
    echo "[train] llm_trainable_layer_range=${LLM_TRAINABLE_LAYER_RANGE:-all}"
    echo "[train] lora_enable=${LORA_ENABLE:-false}"
    if [[ -n "${compressor_type}" ]]; then
        echo "[train] compressor_config=${compressor_config}"
        echo "[train] scannet_pooled_coords_root=${scannet_pooled_coords_root}"
        echo "[train] scannet_pooled_coords_required=${scannet_pooled_coords_required}"
    fi
    if [[ -n "${llm_compressor_type}" ]]; then
        echo "[train] llm_compressor_config=${llm_compressor_config}"
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[train] CMD='
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    # Keep dry-runs side-effect free: checkpoint/log directories are created
    # only after the command has passed all validation and is about to start.
    mkdir -p "${output_dir}" "${logging_dir}" "${shell_log_dir}"

    set +e
    {
        echo "[train] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
        printf '[train] CMD='
        printf '%q ' "${cmd[@]}"
        printf '\n'
        "${cmd[@]}"
        train_exit_code=$?
        echo "[train] finished_at=$(date '+%Y-%m-%d %H:%M:%S') exit_code=${train_exit_code}"
        exit "${train_exit_code}"
    } 2>&1 | tee -a "${shell_log_file}"
    local train_status=${PIPESTATUS[0]}
    set -e
    return "${train_status}"
}

run_ov_no_compression_train() {
    local input_data_yaml="${1:-${DEFAULT_DATA_YAML}}"
    local run_name_prefix="${2:-llava_ov_scanqa_sqa3d_proj_llm_ft}"
    run_ov_train "${input_data_yaml}" "${run_name_prefix}" "" ""
}
