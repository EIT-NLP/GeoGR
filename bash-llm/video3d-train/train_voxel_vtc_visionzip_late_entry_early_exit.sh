#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias. Use train_grouproute.sh in new experiments.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_grouproute.sh" "$@"
