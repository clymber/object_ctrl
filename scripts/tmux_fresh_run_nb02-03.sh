#!/usr/bin/env bash
# Run notebooks 02.01 through 03.03 sequentially in tmux sessions.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
export OBJCTRL_RENKU_VENV="${PROJECT_ROOT}/.venv-renku"
export OBJCTRL_RENKU_KERNEL_NAME=object-ctrl-renku
export NOTEBOOK_KERNEL=object-ctrl-renku

readonly SESSION_NAMES=(
    "nb02_01"
    "nb02_02"
    "nb03_01"
    "nb03_02"
    "nb03_03"
)
readonly NOTEBOOK_NAMES=(
    "nb02.01-ultralytics_yolo11n_on_basketball.ipynb"
    "nb02.02-ultra_yolo11n_large_basketball.ipynb"
    "nb03.01-yolox_tiny_on_basketball.ipynb"
    "nb03.02-yolox_tiny_large_basketball.ipynb"
    "nb03.03-yolox_nano_large_basketball.ipynb"
)

clear_experiment_environment() {
    local variable_name

    for variable_name in "${!YOLOX_NANO_@}" "${!YOLOX_TINY_@}" \
        "${!ULTRALYTICS_@}"; do
        unset "$variable_name"
    done
}

wait_for_notebook() {
    local session_name=$1
    local pane_dead

    while true; do
        sleep 10
        pane_dead="$(
            tmux display-message -p -t "=${session_name}:" '#{pane_dead}'
        )"
        "${SCRIPT_DIR}/tmux_notebook.sh" check --session "$session_name"

        if [[ "$pane_dead" == "1" ]]; then
            break
        fi
    done
}

clear_experiment_environment
cd "${PROJECT_ROOT}"

for index in "${!NOTEBOOK_NAMES[@]}"; do
    notebook_name="${NOTEBOOK_NAMES[$index]}"
    session_name="${SESSION_NAMES[$index]}"

    printf "Beginning session %s\n" ${session_name}
    "${SCRIPT_DIR}/tmux_notebook.sh" run \
        --file "${PROJECT_ROOT}/notebooks/$notebook_name" \
        --session "$session_name"
    wait_for_notebook "$session_name"
    tmux kill-session -t "$session_name"
    printf "Finished session %s\n" ${session_name}
done
