#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'Usage:\n'
    printf '  %s run -f|--file <notebook> ' "$0"
    printf '[-s|--session <session_name>]\n'
    printf '  %s check -s|--session <session_name>\n' "$0"
    printf '\nCommands:\n'
    printf '  run      Start the notebook in a detached tmux session\n'
    printf '  check    Report whether the notebook session is still running\n'
    printf '\nOptions:\n'
    printf '  -f, --file <notebook>          Notebook to run\n'
    printf '  -s, --session <session_name>   tmux session to run or check\n'
    printf '  -h, --help                     Show this help message\n'
}

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 2
    fi

    if [[ $1 == "-h" || $1 == "--help" ]]; then
        usage
        exit 0
    fi

    command_name=$1
    shift

    case "$command_name" in
        run|check)
            ;;
        *)
            printf 'Error: unknown command: %s\n' "$command_name" >&2
            usage >&2
            exit 2
            ;;
    esac

    notebook=""
    session_name=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--file)
                if [[ "$command_name" != "run" ]]; then
                    printf 'Error: %s is only valid with the run command.\n' \
                        "$1" >&2
                    exit 2
                fi
                if [[ $# -lt 2 ]]; then
                    printf 'Error: %s requires a notebook path.\n' "$1" >&2
                    exit 2
                fi
                notebook=$2
                shift 2
                ;;
            -s|--session)
                if [[ $# -lt 2 ]]; then
                    printf 'Error: %s requires a session name.\n' "$1" >&2
                    exit 2
                fi
                session_name=$2
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                printf 'Error: unknown option: %s\n' "$1" >&2
                usage >&2
                exit 2
                ;;
        esac
    done

    case "$command_name" in
        run)
            if [[ -z "$notebook" ]]; then
                printf 'Error: run requires -f or --file.\n' >&2
                usage >&2
                exit 2
            fi
            ;;
        check)
            if [[ -z "$session_name" ]]; then
                printf 'Error: check requires -s or --session.\n' >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
}

resolve_notebook() {
    if [[ ! -f "$notebook" ]]; then
        printf 'Error: notebook not found: %s\n' "$notebook" >&2
        exit 1
    fi

    if [[ "$notebook" != *.ipynb ]]; then
        printf 'Error: notebook must have an .ipynb extension: %s\n' \
            "$notebook" >&2
        exit 1
    fi

    notebook_dir=$(CDPATH= cd -- "$(dirname -- "$notebook")" && pwd -P)
    notebook="$notebook_dir/$(basename -- "$notebook")"

    if [[ -z "$session_name" ]]; then
        session_name=$(basename -- "$notebook" .ipynb)
        session_name=${session_name//[^[:alnum:]_-]/-}
    fi
}

resolve_session() {
    if [[ ! "$session_name" =~ ^[[:alnum:]_-]+$ ]]; then
        printf 'Error: session name may contain only letters, numbers, _ and -.\n' \
            >&2
        exit 2
    fi

    log_file=$repo_dir/outputs/notebook_logs/$session_name.log
}

run_notebook() {
    source "$repo_dir/scripts/activate_renku_env.sh"

    nbconvert_command=$OBJCTRL_RENKU_VENV/bin/jupyter-nbconvert

    if [[ ! -x "$nbconvert_command" ]]; then
        printf 'Error: jupyter-nbconvert not found: %s\n' \
            "$nbconvert_command" >&2
        printf 'Run bash scripts/setup_renku.sh to install the Renku runtime.\n' \
            >&2
        exit 1
    fi

    if tmux has-session -t "=$session_name" 2>/dev/null; then
        printf 'Error: tmux session already exists: %s\n' "$session_name" >&2
        exit 1
    fi

    mkdir -p "$(dirname -- "$log_file")"

    printf -v notebook_command \
        '%q %q --to notebook --execute --inplace %q %s 2>&1 | tee %q' \
        "$nbconvert_command" \
        "$notebook" \
        "--ExecutePreprocessor.kernel_name=$NOTEBOOK_KERNEL" \
        '--ExecutePreprocessor.timeout=-1 --CoalesceStreamsPreprocessor.enabled=True' \
        "$log_file"
    printf -v tmux_command 'bash -o pipefail -c %q' "$notebook_command"

    tmux new-session -d -s "$session_name" -c "$repo_dir" "$tmux_command" \; \
        set-option -t "=$session_name:" remain-on-exit on

    printf 'Started notebook in tmux session %q.\n' "$session_name"
    printf 'Attach with: tmux attach-session -t %q\n' "$session_name"
    printf 'Follow log with: tail -f %q\n' "$log_file"
    printf 'Check with: %q check --session %q\n' "$0" "$session_name"
}

check_notebook() {
    if ! tmux has-session -t "=$session_name" 2>/dev/null; then
        printf 'Error: tmux session not found: %s\n' "$session_name" >&2
        exit 1
    fi

    pane_status=$(tmux display-message -p -t "=$session_name:" \
        '#{pane_dead} #{pane_dead_status}')
    read -r pane_dead pane_exit_status <<<"$pane_status"
    check_status=0

    case "$pane_dead" in
        0)
            printf 'Session %q is running.\n' "$session_name"
            ;;
        1)
            if [[ ! "$pane_exit_status" =~ ^[0-9]+$ ]]; then
                printf 'Error: tmux returned an invalid pane exit status: %s\n' \
                    "$pane_exit_status" >&2
                exit 1
            fi
            if [[ "$pane_exit_status" -eq 0 ]]; then
                printf 'Session %q finished successfully.\n' "$session_name"
            else
                printf 'Session %q failed with exit status %s.\n' \
                    "$session_name" "$pane_exit_status"
                check_status=$pane_exit_status
            fi
            ;;
        *)
            printf 'Error: tmux returned an invalid pane state: %s\n' \
                "$pane_dead" >&2
            exit 1
            ;;
    esac

    printf 'Log: %s\n' "$log_file"
    return "$check_status"
}

parse_arguments "$@"

repo_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ "$command_name" == "run" ]]; then
    resolve_notebook
fi
resolve_session

if ! command -v tmux >/dev/null 2>&1; then
    printf 'Error: tmux is not installed or is not on PATH.\n' >&2
    exit 1
fi

case "$command_name" in
    run)
        run_notebook
        ;;
    check)
        check_notebook
        ;;
esac
