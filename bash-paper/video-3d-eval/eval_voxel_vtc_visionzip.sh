#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name for the GeoSemZip Stage-I launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/eval_geosemzip.sh" "$@"
