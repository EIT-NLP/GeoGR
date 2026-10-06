#!/usr/bin/env bash
set -euo pipefail

# Video3D baseline evaluated through the same model and benchmark stack as the
# projector-compression methods.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_video3d_eval.sh"

run_video3d_eval "no_compression" ""
