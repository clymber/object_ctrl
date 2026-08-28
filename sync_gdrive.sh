#!/bin/bash -

function usage() {
    printf 'Usage: %s [options]\n' "$0"
    printf 'Options:\n'
    printf '  -h, --help      Show this help message and exit\n'
    printf '  -c, --copy      Copy files to remote (default)\n'
    printf '  -s, --sync      Sync files to remote (delete extra files in remote)\n'
}

function build_exclude_options() {
    exclude_options=()

    local exclude_patterns=(
        ".git/**"
        "datasets/**"
        "datasets/composed/**/labels/*.cache"
        "outputs/**"
        "__pycache__/**"
        ".pytest_cache/**"
        ".ruff_cache/**"
        ".vscode/**"
        ".DS_Store"
        "PLAN.md"

        # Local instructions and credentials
        "/AGENTS.md"
        ".env"
        ".envrc"
        ".pypirc"

        # Generated Python/Jupyter artifacts
        "*.egg-info/**"
        ".notebook-stamps/**"
        ".ipynb_checkpoints/**"
        "*.py[cod]"

        # Local environments and build outputs
        ".venv/**"
        "venv/**"
        "env/**"
        "__pypackages__/**"
        "build/**"
        "dist/**"

        # Test and analysis artifacts
        ".mypy_cache/**"
        ".hypothesis/**"
        ".tox/**"
        ".nox/**"
        "htmlcov/**"
        ".coverage*"

        # Accidental local training and temporary files
        "runs/**"
        "~$*"
        "*.sw[op]"
        "*~"
    )

    local pattern
    for pattern in "${exclude_patterns[@]}"; do
        exclude_options+=("--exclude" "$pattern")
    done
}

function parse_arguments() {
    rclone_cmd=""

    while [[ $# -gt 0 ]]; do
        case "${1}" in
            -h|--help)
                usage
                exit 0
                ;;
            -c|--copy)
                if [[ "${rclone_cmd}" == "sync" ]]; then
                    printf 'Error: cannot specify both --copy and --sync\n' >&2
                    exit 1
                fi
                rclone_cmd="copy"
                shift
                ;;
            -s|--sync)
                if [[ "${rclone_cmd}" == "copy" ]]; then
                    printf 'Error: cannot specify both --copy and --sync\n' >&2
                    exit 1
                fi
                rclone_cmd="sync"
                shift
                ;;
            --)
                shift
                break
                ;;
            *)
                printf 'Error: unknown option: %s\n' "${1}" >&2
                usage >&2
                exit 2
                ;;
        esac
    done

    if [[ $# -gt 0 ]]; then
        printf 'Error: unexpected positional argument: %s\n' "${1}" >&2
        usage >&2
        exit 2
    fi

    if [[ -z "${rclone_cmd}" ]]; then
        rclone_cmd="copy"
    fi
}

################################### main ######################################

remote="gdrive"

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || {
    printf 'Error: cannot determine repository directory\n' >&2
    exit 1
}

repo_name=$(basename "${repo_dir}")

parse_arguments "$@"

build_exclude_options

rclone "${rclone_cmd}" "${repo_dir}" "${remote}":"${repo_name}" \
    "${exclude_options[@]}" \
    -P
