#!/usr/bin/env bash
set -euo pipefail

# GroupRoute Stage-II post-training for Video3D-LLM.
# Only Late Entry/Early Exit is enabled during training. Recoverable
# Group-wise Skipping is applied without an additional training stage during
# GeoGR evaluation.
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
export MASTER_PORT="${MASTER_PORT:-43210}"

# Video3D's native 3D position embedding and grounding head remain trainable
# by default. Set either variable to 1 for an explicit freeze ablation.
export FREEZE_WORLD_POSITION_EMBEDDING="${FREEZE_WORLD_POSITION_EMBEDDING:-0}"
export FREEZE_GROUND_HEAD="${FREEZE_GROUND_HEAD:-0}"

VOXEL_SIZE="${VOXEL_SIZE:-0.1}"
TARGET_KEEP_RATIO="${TARGET_KEEP_RATIO:-0.30}"
DOMINANT_RATIO="${DOMINANT_RATIO:-0.85}"
ATTENTION_REDUCE="${ATTENTION_REDUCE:-max}"
NEWLINE_STRATEGY="grid_drop"
RESIDUAL_MERGE="${RESIDUAL_MERGE:-true}"
COVERAGE_RULE="${COVERAGE_RULE:-morton}"
RANDOM_SEED="${RANDOM_SEED:-0}"
LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-8}"
# Exclusive boundary: 25 means that visual tokens execute layers 8 through 24.
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-25}"

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
errors = []
if config.get("mm_projector_compressor_type") != "voxel_vtc_visionzip":
    errors.append(
        "projector compressor type is "
        f"{config.get('mm_projector_compressor_type')!r}, expected 'voxel_vtc_visionzip'"
    )
if actual != expected:
    errors.append(f"projector config is {actual!r}, expected {expected!r}")
if config.get("mm_llm_compressor_type") not in (None, "", "none", "None"):
    errors.append(
        "Stage-I checkpoint already contains an LLM compressor: "
        f"{config.get('mm_llm_compressor_type')!r}"
    )
if errors:
    raise SystemExit("[ERROR] MODEL_PATH does not match the GeoSemZip Stage-I protocol:\n" + "\n".join(errors))
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
source "${PROJECT_ROOT}/bash-paper/video-3d-train/_common_video3d_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
LLM_CONFIG="{\"late_entry_layer\":${LATE_ENTRY_LAYER},\"early_exit_layer\":${EARLY_EXIT_LAYER}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-video3d_geogr_window_l${LATE_ENTRY_LAYER}_e${EARLY_EXIT_LAYER}_llm_ft}"

# Loading MODEL_PATH preserves its GeoSemZip projector configuration. The LLM
# compressor below adds only the Stage-II visual-computation window.
run_video3d_training \
    llm \
    late_entry_early_exit \
    "${LLM_CONFIG}" \
    "${DATA_YAML}" \
    "${RUN_NAME_PREFIX}"
