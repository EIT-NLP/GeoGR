#!/usr/bin/env bash
set -euo pipefail

# Replace xxx with your actual checkpoint path before running a launcher.
#   export MODEL_PATH=xxx
# For local vision weights, replace xxx with your actual SigLIP checkpoint path.
#   export SIGLIP_MODEL_PATH=xxx


# Shared Video3D-LLM evaluation launcher.
# Model, dataset, cache, GPU, task, and limit settings are environment-overridable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_BASE="${CONDA_BASE:-}"
ENV_NAME="${ENV_NAME:-compress3d}"
LMMS_ROOT="${LMMS_ROOT:-${PROJECT_ROOT}/lmms-eval}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/algorithm/Video-3D-LLM}"
# Set MODEL_PATH or VIDEO3D_MODEL_PATH to a local Video3D checkpoint.
MODEL_PATH="${MODEL_PATH:-${VIDEO3D_MODEL_PATH:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] Set MODEL_PATH or VIDEO3D_MODEL_PATH to a local checkpoint." >&2
    return 1 2>/dev/null || exit 1
fi


THREE_D_CONFIG="${THREE_D_CONFIG:-${LMMS_ROOT}/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml}"
VIDEO_FOLDER="${VIDEO_FOLDER:-${PROJECT_ROOT}/data}"
EMBODIEDSCAN_FOLDER="${EMBODIEDSCAN_FOLDER:-${VIDEO_FOLDER}/embodiedscan}"
SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-}"
# Hugging Face and Transformers caches.
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"
SCANNET3D_COORDS_CACHE_ROOT="${SCANNET3D_COORDS_CACHE_ROOT:-${PROJECT_ROOT}/cache/scannet3d_coords}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
FRAME_SAMPLING_STRATEGY="${FRAME_SAMPLING_STRATEGY:-uniform}"
MAX_FRAME_NUM="${MAX_FRAME_NUM:-32}"
# Default five-benchmark order: scan2cap, scanqa, sqa3d, multi3drefer, scanrefer.
TASKS="${TASKS:-scan2cap_val,scanqa_val,sqa3d_test,multi3drefer_val,scanrefer_val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/video3d}"
LOG_SAMPLES="true"
LIMIT="${LIMIT:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"
VERBOSITY="${VERBOSITY:-INFO}"
TEXT_MAX_NEW_TOKENS="${TEXT_MAX_NEW_TOKENS:-512}"
CAPTION_MAX_NEW_TOKENS="${CAPTION_MAX_NEW_TOKENS:-512}"
DEFAULT_EXTRA_PROMPT=$'The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n'
EXTRA_PROMPT="${EXTRA_PROMPT-${DEFAULT_EXTRA_PROMPT}}"
USE_DP="${USE_DP:-true}"
SHOW_ALL_RANK_PROGRESS="${SHOW_ALL_RANK_PROGRESS:-false}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29501}"

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
        scanrefer_*|multi3drefer_*)
            # Grounding tasks do not use text-generation length settings.
            echo ""
            ;;
        scan2cap_*)
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

task_marker_path() {
    local run_root="$1"
    local task="$2"
    printf '%s/.task_%s.completed\n' "${run_root}" "${task}"
}

# Validate one benchmark's persisted output before allowing it to be skipped.
# A successful lmms-eval return code alone is insufficient because an external
# signal can arrive after aggregation but while samples are being serialized.
validate_task_outputs() {
    local task_out="$1"
    local task="$2"
    local requested_limit="${3:-}"

    python - "${task_out}" "${task}" "${requested_limit}" <<'PY'
import json
import sys
from pathlib import Path

task_out = Path(sys.argv[1])
task = sys.argv[2]
requested_limit = sys.argv[3]
if not task_out.is_dir():
    raise SystemExit(f"{task}: output directory is missing: {task_out}")

result_files = list(task_out.rglob("*_results.json"))
sample_files = list(task_out.rglob("*_samples_*.jsonl"))
pairs = []
for result_file in result_files:
    prefix = result_file.name[: -len("_results.json")]
    matching_samples = [
        sample_file
        for sample_file in sample_files
        if sample_file.name.startswith(prefix + "_samples_")
    ]
    for sample_file in matching_samples:
        pairs.append((result_file, sample_file))
if not pairs:
    raise SystemExit(f"{task}: no matching results/sample pair under {task_out}")

result_file, sample_file = max(pairs, key=lambda pair: pair[0].stat().st_mtime_ns)
try:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"{task}: invalid result JSON {result_file}: {exc}")

counts = payload.get("n-samples", {}).get(task, {})
original = counts.get("original")
effective = counts.get("effective")
if not isinstance(original, (int, float)) or not isinstance(effective, (int, float)):
    raise SystemExit(f"{task}: result JSON has no valid n-samples accounting")
if (
    original <= 0
    or effective <= 0
    or int(original) != original
    or int(effective) != effective
    or effective > original
    or (not requested_limit and original != effective)
):
    raise SystemExit(
        f"{task}: incomplete n-samples accounting: original={original}, effective={effective}"
    )
if not payload.get("results", {}).get(task):
    raise SystemExit(f"{task}: result JSON has an empty metric map")

sample_count = 0
line_number = 0
try:
    with sample_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            json.loads(line)
            sample_count += 1
except Exception as exc:
    raise SystemExit(f"{task}: invalid sample JSONL {sample_file} at line {line_number}: {exc}")
if sample_count != int(effective):
    raise SystemExit(
        f"{task}: sample count={sample_count} does not match effective={int(effective)}"
    )

print(f"{task}: validated n={sample_count} result={result_file.name}")
PY
}

task_log_is_clean() {
    local task_log="$1"
    [[ -s "${task_log}" ]] || return 1
    ! grep -q "Traceback (most recent call last)" "${task_log}"
}

write_task_marker() {
    local run_root="$1"
    local task="$2"
    local marker
    marker="$(task_marker_path "${run_root}" "${task}")"
    local temporary_marker="${marker}.tmp.$$"
    {
        printf 'task=%s\n' "${task}"
        printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    } > "${temporary_marker}"
    mv -f -- "${temporary_marker}" "${marker}"
}

# Return success only when both the log and the structured outputs are sound.
# This also upgrades old partial runs that predate task markers: a valid task
# is marked on first resume and can then be skipped deterministically.
task_is_complete() {
    local run_root="$1"
    local task="$2"
    local task_out="${run_root}/${task}"
    local task_log="${run_root}/${task}.log"

    if task_log_is_clean "${task_log}" && validate_task_outputs "${task_out}" "${task}" "${LIMIT:-}" >/dev/null; then
        if [[ ! -f "$(task_marker_path "${run_root}" "${task}")" ]]; then
            write_task_marker "${run_root}" "${task}"
        fi
        return 0
    fi
    return 1
}

archive_incomplete_task() {
    local run_root="$1"
    local task="$2"
    local archive_root="${run_root}/.incomplete_attempts/${task}_$(date +%Y%m%d_%H%M%S)_$$"
    local path
    local has_files=0
    local -a task_paths=(
        "${run_root}/${task}" \
        "${run_root}/${task}.log" \
        "${run_root}/${task}_ranks" \
        "${run_root}/${task}_worker.sh" \
        "$(task_marker_path "${run_root}" "${task}")"
    )

    for path in "${task_paths[@]}"; do
        if [[ -e "${path}" ]]; then
            has_files=1
            break
        fi
    done
    (( has_files == 1 )) || return 0

    mkdir -p "${archive_root}"
    for path in "${task_paths[@]}"; do
        if [[ -e "${path}" ]]; then
            mv -- "${path}" "${archive_root}/"
        fi
    done
}

validate_resume_protocol() {
    local run_root="$1"
    local current_model_args="$2"

    python - "${run_root}" "${current_model_args}" "${TASKS}" "${LIMIT:-}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
current_model_args = sys.argv[2]
requested_tasks = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
requested_limit = sys.argv[4]

def normalize_limit(value):
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(number) if number.is_integer() else number

expected_limit = normalize_limit(requested_limit)
existing_task_dirs = set()
for result_file in run_root.glob("*/**/*_results.json"):
    relative_parts = result_file.relative_to(run_root).parts
    if relative_parts:
        existing_task_dirs.add(relative_parts[0])
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid resume result JSON {result_file}: {exc}")
    config = payload.get("config", {})
    saved_model_args = config.get("model_args")
    if saved_model_args != current_model_args:
        raise SystemExit(
            f"resume protocol mismatch in {result_file}: model_args differ"
        )
    saved_limit = normalize_limit(config.get("limit"))
    if saved_limit != expected_limit:
        raise SystemExit(
            f"resume protocol mismatch in {result_file}: limit={saved_limit!r}, "
            f"current={expected_limit!r}"
        )

unknown_tasks = sorted(existing_task_dirs.difference(requested_tasks))
if unknown_tasks:
    raise SystemExit(
        "resume directory contains results for tasks outside current TASKS: "
        + ", ".join(unknown_tasks)
    )
print("resume protocol matches existing result metadata")
PY
}

write_run_manifest() {
    local run_root="$1"
    local method_name="$2"
    local model_args="$3"
    local manifest="${run_root}/run_manifest.json"
    local temporary_manifest="${manifest}.tmp.$$"

    RUN_MANIFEST_ROOT="${run_root}" \
    RUN_MANIFEST_METHOD="${method_name}" \
    RUN_MANIFEST_MODEL_ARGS="${model_args}" \
    RUN_MANIFEST_TASKS="${TASKS}" \
    RUN_MANIFEST_LIMIT="${LIMIT:-}" \
    RUN_MANIFEST_GPU_IDS="${GPU_IDS}" \
    RUN_MANIFEST_NPROC="${NPROC_PER_NODE}" \
    python - "${temporary_manifest}" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "method_name": os.environ["RUN_MANIFEST_METHOD"],
    "model_args": os.environ["RUN_MANIFEST_MODEL_ARGS"],
    "tasks": os.environ["RUN_MANIFEST_TASKS"],
    "limit": os.environ["RUN_MANIFEST_LIMIT"],
    "gpu_ids": os.environ["RUN_MANIFEST_GPU_IDS"],
    "nproc_per_node": int(os.environ["RUN_MANIFEST_NPROC"]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
    mv -f -- "${temporary_manifest}" "${manifest}"
}

run_video3d_eval() {
    local method_name="$1"
    local compressor_args="$2"

    activate_conda_env

    export TOKENIZERS_PARALLELISM=false
    export PYTHONWARNINGS=ignore
    export LLAVA_SAFE_TOKENIZER_LOCAL_ONLY="${LLAVA_SAFE_TOKENIZER_LOCAL_ONLY:-1}"
    mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
    export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
    export SCANNET3D_COORDS_CACHE_ROOT
    if [[ "${OFFLINE_MODE}" == "1" ]]; then
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    fi
    if [[ -n "${GPU_IDS}" ]]; then
        export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    fi
    if [[ "${USE_DP}" == "true" && "${NPROC_PER_NODE}" -gt 1 ]]; then
        if [[ -z "${GPU_IDS}" ]]; then
            echo "[ERROR] GPU_IDS is required when USE_DP=true and NPROC_PER_NODE>1." >&2
            exit 1
        fi
        if [[ "${NPROC_PER_NODE}" -gt "${#GPU_ARR[@]}" ]]; then
            echo "[ERROR] NPROC_PER_NODE (${NPROC_PER_NODE}) exceeds the number of GPU_IDS (${#GPU_ARR[@]})." >&2
            exit 1
        fi
    fi

    mkdir -p "${OUTPUT_ROOT}"
    local run_root="${VIDEO3D_RESUME_RUN_ROOT:-}"
    if [[ -n "${run_root}" ]]; then
        if [[ ! -d "${run_root}" ]]; then
            echo "[ERROR] VIDEO3D_RESUME_RUN_ROOT does not exist: ${run_root}" >&2
            return 1
        fi
        if [[ -f "${run_root}/.completed" ]]; then
            echo "[ERROR] refusing to resume an already completed run: ${run_root}" >&2
            return 1
        fi
        if [[ "${run_root##*/}" != "${method_name}_"* ]]; then
            echo "[ERROR] resume run does not match method '${method_name}': ${run_root}" >&2
            return 1
        fi
        echo "[INFO] resuming partial run: ${run_root}"
    else
        local ts
        ts="$(date +%Y%m%d_%H%M%S)"
        run_root="${OUTPUT_ROOT}/${method_name}_${ts}"
        mkdir -p "${run_root}"
    fi
    mkdir -p "${run_root}"
    export VIDEO3D_LAST_RUN_ROOT="${run_root}"

    local model_args
    # Projector compressors receive the spatial-unpad grid. Their explicit
    # metadata controls the final serialization strategy.
    model_args="pretrained=${MODEL_PATH},repo_root=${REPO_ROOT},three_d_config=${THREE_D_CONFIG},video_folder=${VIDEO_FOLDER},embodiedscan_folder=${EMBODIEDSCAN_FOLDER},frame_sampling_strategy=${FRAME_SAMPLING_STRATEGY},max_frame_num=${MAX_FRAME_NUM},overwrite_cfg=true,device=auto,mm_patch_merge_type=spatial_unpad,mm_newline_position=grid"

    if [[ -n "${EXTRA_PROMPT}" ]]; then
        model_args="${model_args},extra_prompt=${EXTRA_PROMPT}"
    fi
    if [[ -d "${SIGLIP_MODEL_PATH}" ]]; then
        model_args="${model_args},mm_vision_tower=${SIGLIP_MODEL_PATH}"
    fi
    if [[ -n "${SCANNET3D_COORDS_CACHE_ROOT}" ]]; then
        model_args="${model_args},coords_cache_root=${SCANNET3D_COORDS_CACHE_ROOT}"
    fi
    if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
        model_args="${model_args},attn_implementation=${ATTN_IMPLEMENTATION}"
    fi
    if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
        model_args="${model_args},${EXTRA_MODEL_ARGS}"
    fi
    if [[ -n "${compressor_args}" ]]; then
        model_args="${model_args},${compressor_args}"
    fi

    if [[ -n "${VIDEO3D_RESUME_RUN_ROOT:-}" ]]; then
        if ! validate_resume_protocol "${run_root}" "${model_args}"; then
            echo "[ERROR] Resume protocol does not match the partial run: ${run_root}" >&2
            return 1
        fi
    fi
    write_run_manifest "${run_root}" "${method_name}" "${model_args}"

    echo "[INFO] method=${method_name}"
    echo "[INFO] env_name=${ENV_NAME}"
    echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all}"
    echo "[INFO] model_path=${MODEL_PATH}"
    echo "[INFO] siglip_model_path=${SIGLIP_MODEL_PATH}"
    echo "[INFO] hf_home=${HF_HOME}"
    echo "[INFO] hf_hub_cache=${HF_HUB_CACHE}"
    echo "[INFO] transformers_cache=${TRANSFORMERS_CACHE}"
    echo "[INFO] offline_mode=${OFFLINE_MODE}"
    echo "[INFO] repo_root=${REPO_ROOT}"
    echo "[INFO] coords_cache_root=${SCANNET3D_COORDS_CACHE_ROOT:-<none>}"
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

        echo "============================================================"
        echo "[INFO] Starting task: ${task}"
        echo "[INFO] Output directory: ${task_out}"
        echo "[INFO] Log file: ${task_log}"
        if [[ -n "${gen_kwargs}" ]]; then
            echo "[INFO] gen_kwargs: ${gen_kwargs}"
        else
            echo "[INFO] gen_kwargs: <none>"
        fi
        echo "============================================================"

        if task_is_complete "${run_root}" "${task}"; then
            echo "[SKIP] Task already completed with valid outputs: ${task}"
            continue
        fi

        # Do not let a truncated result from an interrupted attempt coexist
        # with a new attempt under the same task directory. Keep the old
        # files under .incomplete_attempts for debugging instead of deleting
        # them, then start with a clean output namespace.
        archive_incomplete_task "${run_root}" "${task}"
        mkdir -p "${task_out}" "${task_rank_log_dir}"

        if [[ "${USE_DP}" == "true" && "${NPROC_PER_NODE}" -gt 1 ]]; then
            cat > "${task_worker_script}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IFS=',' read -r -a _GPU_ARR <<< "${DP_GPU_IDS}"
_LOCAL_RANK="${LOCAL_RANK:-0}"
if [[ "${_LOCAL_RANK}" -ge "${#_GPU_ARR[@]}" ]]; then
  echo "[ERROR][rank ${RANK:-?}] LOCAL_RANK ${_LOCAL_RANK} is outside the ${#_GPU_ARR[@]} configured GPUs." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${_GPU_ARR[${_LOCAL_RANK}]}"
# Expose one GPU per worker and reset the rank seen by Accelerate.
export LOCAL_RANK=0

CMD=(
  python -m lmms_eval eval
  --model video_3d
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

            printf '[INFO] DP command: torchrun --standalone --nnodes=1 --nproc-per-node=%q --master-port=%q --no-python --tee 3 --log-dir %q %q\n' \
                "${NPROC_PER_NODE}" "${task_master_port}" "${task_rank_log_dir}" "${task_worker_script}"

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
                --model video_3d
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

            printf '[INFO] Command: '
            printf '%q ' "${cmd[@]}"
            printf '\n'

            (
                cd "${LMMS_ROOT}"
                "${cmd[@]}"
            ) 2>&1 | tee "${task_log}"
        fi

        # Some lmms-eval failures print a traceback but return zero.
        if grep -q "Traceback (most recent call last)" "${task_log}"; then
            echo "[ERROR] Python traceback detected: ${task}, log=${task_log}" >&2
            return 1
        fi

        if ! task_is_complete "${run_root}" "${task}"; then
            echo "[ERROR] Output validation failed: ${task}, log=${task_log}" >&2
            return 1
        fi
        echo "[INFO] Task completed and validated: ${task}"
    done

    # Mark the run complete only after every task passes validation.
    touch "${run_root}/.completed"
    echo "[INFO] All tasks completed: ${run_root}"
}
