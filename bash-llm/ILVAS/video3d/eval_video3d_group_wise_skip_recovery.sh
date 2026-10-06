#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name for the paper-facing GeoGR launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/eval_geogr.sh" "$@"
