#!/usr/bin/env bash
# Activate the isolated object_ctrl environment in the current Renku shell.
# Usage: source scripts/activate_renku_env.sh

# Resolve the repository independently of the caller's working directory.
if [[ -n "${BASH_VERSION:-}" ]]; then
    _objctrl_source_path="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _objctrl_source_path="${(%):-%N}"
else
    printf 'scripts/activate_renku_env.sh requires Bash or Zsh.\n' >&2
    return 1 2>/dev/null || exit 1
fi

_objctrl_script_dir="$(
    CDPATH= cd -- "$(dirname -- "${_objctrl_source_path}")" && pwd -P
)"

# Absolute repository path shared by project scripts and interactive commands.
export OBJCTRL_PROJECT_ROOT="$(
    CDPATH= cd -- "${_objctrl_script_dir}/.." && pwd -P
)"

# Location of the isolated Python environment created by setup_renku.sh.
# Define this before sourcing the script to use a non-default environment path.
export OBJCTRL_RENKU_VENV="${OBJCTRL_RENKU_VENV:-.venv-renku}"
if [[ "${OBJCTRL_RENKU_VENV}" != /* ]]; then
    export OBJCTRL_RENKU_VENV="${OBJCTRL_PROJECT_ROOT}/${OBJCTRL_RENKU_VENV}"
fi

# Internal Jupyter kernel identifier registered by setup_renku.sh. This is not
# the human-readable kernel display name shown in the Jupyter interface.
export OBJCTRL_RENKU_KERNEL_NAME="${OBJCTRL_RENKU_KERNEL_NAME:-object-ctrl-renku}"

# Platform-neutral kernel variable consumed by Makefile notebook commands.
# An explicit NOTEBOOK_KERNEL value takes precedence over the Renku default.
export NOTEBOOK_KERNEL="${NOTEBOOK_KERNEL:-${OBJCTRL_RENKU_KERNEL_NAME}}"

# Prevent packages under the user's site-packages directory from leaking into
# the isolated project environment.
export PYTHONNOUSERSITE=1

if [[ ! -f "${OBJCTRL_RENKU_VENV}/bin/activate" ]]; then
    printf 'Renku environment not found: %s\n' "${OBJCTRL_RENKU_VENV}" >&2
    printf 'Run bash scripts/setup_renku.sh first.\n' >&2
    unset _objctrl_script_dir _objctrl_source_path
    return 1 2>/dev/null || exit 1
fi

# Activate first because reactivating an existing virtual environment restores
# the PATH captured by its previous activation.
source "${OBJCTRL_RENKU_VENV}/bin/activate"

# Put project utilities immediately after the virtual environment's commands.
# Reconstructing the prefix here keeps the virtualenv first and avoids duplicate
# scripts entries when this file is sourced repeatedly.
_objctrl_venv_bin="${OBJCTRL_RENKU_VENV}/bin"
_objctrl_scripts_dir="${OBJCTRL_PROJECT_ROOT}/scripts"
case ":${PATH}:" in
    *":${_objctrl_scripts_dir}:"*)
        ;;
    *)
        _objctrl_path_tail="${PATH#"${_objctrl_venv_bin}:"}"
        export PATH="${_objctrl_venv_bin}:${_objctrl_scripts_dir}:${_objctrl_path_tail}"
        ;;
esac

unset _objctrl_script_dir _objctrl_scripts_dir _objctrl_path_tail
unset _objctrl_source_path _objctrl_venv_bin
