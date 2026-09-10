#!/usr/bin/env bash
# Run notebook 04.02 in evaluation mode in a tmux session.

set -euo pipefail

readonly SESSION_NAME="nb04_02_evaluate"
readonly NOTEBOOK_NAME="nb04.02-rfdetr_small_large_basketball.ipynb"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly RUNS_DIR="${PROJECT_ROOT}/outputs/runs/basketball"
readonly DEFAULT_RUN_DIR="${RUNS_DIR}/rfdetr_small_basketball_large_dataset"
readonly COMPARISON_DIR="${PROJECT_ROOT}/outputs/comparisons/basketball_large_dataset"
readonly DEFAULT_BASELINE_EXPORT_DIR="${COMPARISON_DIR}/baselines"

export OBJCTRL_RENKU_VENV="${PROJECT_ROOT}/.venv-renku-rfdetr"
export OBJCTRL_RENKU_KERNEL_NAME=object-ctrl-renku-rfdetr
export NOTEBOOK_KERNEL=object-ctrl-renku-rfdetr
export RFDETR_MODE=evaluate
export RFDETR_RUN_DIR="${RFDETR_RUN_DIR:-$DEFAULT_RUN_DIR}"
export RFDETR_BASELINE_EXPORT_DIR="${RFDETR_BASELINE_EXPORT_DIR:-\
$DEFAULT_BASELINE_EXPORT_DIR}"
unset RFDETR_SMOKE RFDETR_EPOCHS

cd "${PROJECT_ROOT}"
"${SCRIPT_DIR}/tmux_notebook.sh" run \
    --file "${PROJECT_ROOT}/notebooks/$NOTEBOOK_NAME" \
    --session "$SESSION_NAME"
