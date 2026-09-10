#!/bin/bash -
# This script sets up a tmux session for notebook 04.02

set -euo pipefail

SESSION_NAME="nb04_02_fresh"
NOTEBOOK_NAME="nb04.02-rfdetr_small_large_basketball.ipynb"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

export OBJCTRL_RENKU_VENV="${PROJECT_ROOT}/.venv-renku-rfdetr"
export OBJCTRL_RENKU_KERNEL_NAME=object-ctrl-renku-rfdetr
export NOTEBOOK_KERNEL=object-ctrl-renku-rfdetr
export RFDETR_MODE=fresh
unset RFDETR_SMOKE RFDETR_EPOCHS


cd "${PROJECT_ROOT}"
${SCRIPT_DIR}/tmux_notebook.sh run \
    --file "${PROJECT_ROOT}/notebooks/$NOTEBOOK_NAME" \
    --session "$SESSION_NAME" 
