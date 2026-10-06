#!/usr/bin/env bash
set -euo pipefail

# Five-task Video3D no-compression post-training baseline.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Override GPU_IDS at launch time to select another GPU group.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
# Auxiliary 3D modules are trainable unless the following flags are enabled.
# FREEZE_WORLD_POSITION_EMBEDDING=1
# FREEZE_GROUND_HEAD=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"

run_video3d_training \
    none \
    "" \
    "" \
    "${DATA_YAML}" \
    "video3d_no_compression"
