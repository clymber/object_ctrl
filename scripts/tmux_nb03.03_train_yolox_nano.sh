#!/usr/bin/env bash
# Run notebook 03.03 as a fresh YOLOX-Nano training session in tmux.

set -euo pipefail

readonly SESSION_NAME="nb03_03_train_yolox_nano"
readonly NOTEBOOK_NAME="nb03.03-yolox_nano_large_basketball.ipynb"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

wait_for_notebook() {
    local pane_status pane_dead pane_exit_status check_status

    while true; do
        sleep 10
        if ! pane_status="$(
            tmux display-message -p -t "=${SESSION_NAME}:" \
                '#{pane_dead} #{pane_dead_status}'
        )"; then
            printf 'Warning: unable to inspect tmux session %s.\n' \
                "$SESSION_NAME" >&2
            return 1
        fi
        read -r pane_dead pane_exit_status <<<"$pane_status"

        case "$pane_dead" in
            0)
                if ! "${SCRIPT_DIR}/tmux_notebook.sh" check \
                    --session "$SESSION_NAME"; then
                    printf 'Warning: unable to check tmux session %s.\n' \
                        "$SESSION_NAME" >&2
                    return 1
                fi
                ;;
            1)
                if "${SCRIPT_DIR}/tmux_notebook.sh" check \
                    --session "$SESSION_NAME"; then
                    tmux kill-session -t "=$SESSION_NAME"
                    printf 'Cleaned successful tmux session %s.\n' \
                        "$SESSION_NAME"
                    return 0
                else
                    check_status=$?
                    printf 'Warning: tmux session %s failed with status %s; ' \
                        "$SESSION_NAME" "$pane_exit_status" >&2
                    printf 'leaving it available for inspection.\n' >&2
                    return "$check_status"
                fi
                ;;
            *)
                printf 'Warning: tmux session %s returned invalid state %s.\n' \
                    "$SESSION_NAME" "$pane_dead" >&2
                return 1
                ;;
        esac
    done
}

export OBJCTRL_RENKU_VENV="${PROJECT_ROOT}/.venv-renku"
export OBJCTRL_RENKU_KERNEL_NAME=object-ctrl-renku
export NOTEBOOK_KERNEL=object-ctrl-renku

# Ensure this launcher starts the notebook's full, fresh 100-epoch run.
unset YOLOX_NANO_SMOKE YOLOX_NANO_EPOCHS YOLOX_NANO_RESUME_RUN

cd "${PROJECT_ROOT}"
"${SCRIPT_DIR}/tmux_notebook.sh" run \
    --file "${PROJECT_ROOT}/notebooks/$NOTEBOOK_NAME" \
    --session "$SESSION_NAME"
wait_for_notebook
