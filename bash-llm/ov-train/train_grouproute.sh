#!/usr/bin/env bash
set -euo pipefail

# GroupRoute Stage-II post-training for LLaVA-OneVision.
# Only the Late Entry/Early Exit window is enabled during this stage. RGS is
# applied training-free at inference on the resulting checkpoint.
#
# Example:
# Replace xxx with your actual Stage-I GeoSemZip checkpoint path.
#   MODEL_PATH=xxx \
#   OUTPUT_ROOT=/path/to/checkpoints \
#   GPU_IDS=0,1,2,3 bash "$0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Set MODEL_PATH to a GeoSemZip Stage-I checkpoint containing config.json." >&2
    exit 1
fi
export MODEL_PATH
export MM_TUNABLE_PARTS="mm_language_model"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-43200}"

VOXEL_SIZE="${VOXEL_SIZE:-0.1}"
TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO:-0.30}"
DOMINANT_RATIO="${DOMINANT_RATIO:-0.85}"
ATTENTION_REDUCE="${ATTENTION_REDUCE:-max}"
NEWLINE_STRATEGY="grid_drop"
RESIDUAL_MERGE="${RESIDUAL_MERGE:-true}"
COVERAGE_RULE="${COVERAGE_RULE:-morton}"
RANDOM_SEED="${RANDOM_SEED:-0}"
LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-8}"
# Exclusive boundary: 24 means that visual tokens execute layers 8 through 23.
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-24}"

python - "${MODEL_PATH}/config.json" "${VOXEL_SIZE}" "${TARGET_KEEP_RATIO}" "${DOMINANT_RATIO}" "${ATTENTION_REDUCE}" "${NEWLINE_STRATEGY}" "${RESIDUAL_MERGE}" "${COVERAGE_RULE}" "${RANDOM_SEED}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "voxel_size": float(sys.argv[2]),
    "target_keep_ratio": float(sys.argv[3]),
    "dominant_ratio": float(sys.argv[4]),
    "attention_reduce": sys.argv[5],
    "newline_strategy": sys.argv[6],
    "residual_merge": sys.argv[7].lower() == "true",
    "coverage_rule": sys.argv[8],
    "random_seed": int(sys.argv[9]),
}
raw = config.get("mm_projector_compressor_config")
actual = json.loads(raw) if isinstance(raw, str) else raw
actual = dict(actual or {})
actual.setdefault("residual_merge", True)
actual.setdefault("coverage_rule", "morton")
actual.setdefault("random_seed", 0)
if config.get("mm_projector_compressor_type") != "voxel_vtc_visionzip" or any(
    actual.get(key) != value for key, value in expected.items()
):
    raise SystemExit(
        "[ERROR] MODEL_PATH is not the expected GeoSemZip Stage-I checkpoint:\n"
        f"  type={config.get('mm_projector_compressor_type')!r}\n"
        f"  config={actual!r}\n"
        f"  expected={expected!r}"
    )
print("[CHECK] Stage-I GeoSemZip configuration matches the Stage-II protocol.")
PY

if [[ ! "${LATE_ENTRY_LAYER}" =~ ^[1-9][0-9]*$ || ! "${EARLY_EXIT_LAYER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Layer boundaries must be positive integers." >&2
    exit 1
fi
if (( LATE_ENTRY_LAYER >= EARLY_EXIT_LAYER || EARLY_EXIT_LAYER > 29 )); then
    echo "[ERROR] Require 1 <= LATE_ENTRY_LAYER < EARLY_EXIT_LAYER <= 29." >&2
    exit 1
fi
if [[ "${PER_DEVICE_TRAIN_BATCH_SIZE}" != "1" ]]; then
    echo "[ERROR] GroupRoute Stage-II training requires PER_DEVICE_TRAIN_BATCH_SIZE=1." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/bash-paper/ov-train/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
PROJECTOR_CONFIG="{\"voxel_size\":${VOXEL_SIZE},\"target_keep_ratio\":${TARGET_KEEP_RATIO},\"dominant_ratio\":${DOMINANT_RATIO},\"attention_reduce\":\"${ATTENTION_REDUCE}\",\"newline_strategy\":\"${NEWLINE_STRATEGY}\",\"residual_merge\":${RESIDUAL_MERGE},\"coverage_rule\":\"${COVERAGE_RULE}\",\"random_seed\":${RANDOM_SEED}}"
LLM_CONFIG="{\"late_entry_layer\":${LATE_ENTRY_LAYER},\"early_exit_layer\":${EARLY_EXIT_LAYER}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_geogr_window_l${LATE_ENTRY_LAYER}_e${EARLY_EXIT_LAYER}_llm_ft}"

run_ov_train \
    "${DATA_YAML}" \
    "${RUN_NAME_PREFIX}" \
    voxel_vtc_visionzip \
    "${PROJECTOR_CONFIG}" \
    late_entry_early_exit \
    "${LLM_CONFIG}"
