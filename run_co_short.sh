#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$ROOT_DIR/run.sh"

bash "$RUN_SCRIPT" shap-summary "$@"
