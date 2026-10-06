#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name for GeoSemZip Stage-I post-training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_geosemzip.sh" "$@"
