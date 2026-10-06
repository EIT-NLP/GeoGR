#!/usr/bin/env bash
set -euo pipefail

# LLaVA-OV baseline without projector compression.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENABLE_3D_AUX="${ENABLE_3D_AUX:-false}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_ov_eval.sh"

run_ov_eval "ov_no_compression" ""
