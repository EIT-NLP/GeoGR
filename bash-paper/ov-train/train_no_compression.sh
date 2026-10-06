#!/usr/bin/env bash
# LLaVA-OneVision no-compression post-training on the maintained five-source mix.
# GPU and batch settings are controlled by _common_ov_train.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_train.sh"

DATA_YAML="${DATA_YAML:-${DEFAULT_DATA_YAML}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-llava_ov_scan3d_no_compression}"

run_ov_no_compression_train "${DATA_YAML}" "${RUN_NAME_PREFIX}"
