# Runtime environment setup script for YOLO development
# Usage: source scripts/env_setup.sh
#

export CONDA_ENV="${CONDA_ENV:-objctrl}"
export NOTEBOOK_KERNEL="${NOTEBOOK_KERNEL:-${CONDA_ENV}}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

if command -v conda >/dev/null 2>&1; then
    if [ -n "${BASH_VERSION:-}" ]; then
        __conda_shell="bash"
    elif [ -n "${ZSH_VERSION:-}" ]; then
        __conda_shell="zsh"
    else
        __conda_shell="posix"
    fi

    eval "$(conda "shell.${__conda_shell}" hook)"
    conda activate "${CONDA_ENV}"
    unset __conda_shell
else
    echo \
        "conda command not found. Install Conda before sourcing scripts/env_setup.sh." \
        >&2
    return 1 2>/dev/null || exit 1
fi
