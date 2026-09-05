#!/usr/bin/env bash
# Bootstrap an isolated project environment inside a Renku session.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly RENKU_SETUP_DIR="${SCRIPT_DIR}/renku"
readonly REQUIREMENTS_FILE="${RENKU_SETUP_DIR}/requirements.txt"
readonly CONSTRAINTS_FILE="${RENKU_SETUP_DIR}/constraints.txt"
readonly RENKU_HOST_VENV_PATH="${VIRTUAL_ENV:-${HOME}/work/.venv}"
readonly PROJECT_VENV_PATH="${OBJCTRL_RENKU_VENV:-${PROJECT_ROOT}/.venv-renku}"
readonly HOST_PYTHON_BIN="${RENKU_HOST_VENV_PATH}/bin/python"
readonly PYTHON_BIN="${PROJECT_VENV_PATH}/bin/python"
readonly KERNEL_NAME="${OBJCTRL_RENKU_KERNEL_NAME:-object-ctrl-renku}"
readonly KERNEL_DISPLAY_NAME="Python (object_ctrl Renku)"
readonly DEFAULT_TORCH_INDEX="https://download.pytorch.org/whl/cu130"
readonly TORCH_WHEEL_INDEX="${OBJCTRL_TORCH_INDEX_URL:-${DEFAULT_TORCH_INDEX}}"
readonly YOLOX_COMMIT="6ddff4824372906469a7fae2dc3206c7aa4bbaee"
readonly YOLOX_REPOSITORY="https://github.com/Megvii-BaseDetection/YOLOX.git"
YOLOX_BUILD_DIR=""

log() {
    printf '\n[Renku setup] %s\n' "$*"
}

die() {
    printf '\n[Renku setup] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    local command_name="$1"
    command -v "${command_name}" >/dev/null 2>&1 || \
        die "Required command not found: ${command_name}"
}

cleanup_yolox_build() {
    if [[ "${YOLOX_BUILD_DIR}" == /tmp/object-ctrl-yolox.* \
        && -d "${YOLOX_BUILD_DIR}" ]]; then
        rm -rf -- "${YOLOX_BUILD_DIR}"
    fi
}

trap cleanup_yolox_build EXIT

validate_torch_stack() {
	"${PYTHON_BIN}" - <<-'PY'
	import sys

	try:
	    import torch
	    import torchvision
	    from torchvision.ops import nms

	    if not torch.version.cuda:
	        raise RuntimeError("PyTorch is not a CUDA build")
	    if not torch.cuda.is_available():
	        raise RuntimeError("PyTorch cannot access the Renku GPU")

	    device = torch.device("cuda:0")
	    tensor = torch.tensor([1.0, 2.0], device=device)
	    if tensor.square().sum().item() != 5.0:
	        raise RuntimeError("CUDA tensor verification returned an invalid result")

	    boxes = torch.tensor(
	        [[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.5, 2.5]],
	        device=device,
	    )
	    scores = torch.tensor([0.9, 0.8], device=device)
	    kept = nms(boxes, scores, 0.5)
	    if kept.numel() == 0:
	        raise RuntimeError("Torchvision CUDA NMS returned no indices")

	    print(f"torch={torch.__version__}")
	    print(f"torchvision={torchvision.__version__}")
	    print(f"torch CUDA runtime={torch.version.cuda}")
	    print(f"GPU={torch.cuda.get_device_name(0)}")
	except Exception as exc:
	    print(f"CUDA PyTorch check failed: {exc}", file=sys.stderr)
	    raise SystemExit(1) from exc
	PY
}

install_torch_stack() {
    log "Using PyTorch wheel index ${TORCH_WHEEL_INDEX}"
    "${PYTHON_BIN}" -m pip install \
        --upgrade \
        --force-reinstall \
        --constraint "${CONSTRAINTS_FILE}" \
        --index-url "${TORCH_WHEEL_INDEX}" \
        torch \
        torchvision
}

validate_headless_opencv() {
	"${PYTHON_BIN}" - <<-'PY'
	import sys
	from importlib.metadata import PackageNotFoundError, version

	try:
	    opencv_distributions = (
	        "opencv-python",
	        "opencv-python-headless",
	        "opencv-contrib-python",
	        "opencv-contrib-python-headless",
	    )
	    installed = {}
	    for distribution in opencv_distributions:
	        try:
	            installed[distribution] = version(distribution)
	        except PackageNotFoundError:
	            pass

	    if set(installed) != {"opencv-python-headless"}:
	        found = ", ".join(
	            f"{name}={installed_version}"
	            for name, installed_version in installed.items()
	        ) or "none"
	        raise RuntimeError(
	            "expected only opencv-python-headless; "
	            f"found OpenCV distributions: {found}"
	        )

	    import cv2

	    gui_lines = [
	        line.split(":", maxsplit=1)[1].strip()
	        for line in cv2.getBuildInformation().splitlines()
	        if line.strip().startswith("GUI:")
	    ]
	    if gui_lines != ["NONE"]:
	        gui_value = ", ".join(gui_lines) if gui_lines else "unknown"
	        raise RuntimeError(f"expected the headless build, found GUI={gui_value}")

	    print(f"OpenCV={cv2.__version__} (headless)")
	except Exception as exc:
	    print(f"Headless OpenCV check failed: {exc}", file=sys.stderr)
	    raise SystemExit(1) from exc
	PY
}

yolox_is_expected_commit() {
	"${PYTHON_BIN}" - "${YOLOX_COMMIT}" <<-'PY'
	import sys
	from importlib.metadata import requires

	from packaging.requirements import Requirement
	from packaging.utils import canonicalize_name

	expected_commit = sys.argv[1].lower()

	try:
	    import yolox  # noqa: F401
	    import yolox.layers.fast_cocoeval  # noqa: F401
	    from yolox._object_ctrl_build import CXX_STANDARD, SOURCE_COMMIT

	    dependency_names = {
	        canonicalize_name(Requirement(requirement).name)
	        for requirement in requires("yolox") or []
	    }
	    expected_dependencies = {
	        "onnx",
	        "onnx-simplifier",
	        "opencv-python-headless",
	    }
	    missing_dependencies = expected_dependencies - dependency_names
	    if missing_dependencies:
	        missing = ", ".join(sorted(missing_dependencies))
	        raise RuntimeError(f"installed metadata is missing: {missing}")
	    if "opencv-python" in dependency_names:
	        raise RuntimeError("installed metadata still requires opencv-python")

	    installed_commit = SOURCE_COMMIT.lower()
	    if installed_commit != expected_commit:
	        raise RuntimeError(
	            f"installed commit is {installed_commit or 'unknown'}, "
	            f"expected {expected_commit}"
	        )
	    if CXX_STANDARD != "c++17":
	        raise RuntimeError(f"installed C++ standard is {CXX_STANDARD}")
	except Exception as exc:
	    print(f"YOLOX check failed: {exc}", file=sys.stderr)
	    raise SystemExit(1) from exc
	PY
}

log "Checking the active Renku environment"

[[ "$(uname -s)" == "Linux" ]] || \
    die "This setup script must run inside a Linux Renku session."
[[ -f "${REQUIREMENTS_FILE}" ]] || \
    die "Renku requirements file not found: ${REQUIREMENTS_FILE}"
[[ -f "${CONSTRAINTS_FILE}" ]] || \
    die "Renku constraints file not found: ${CONSTRAINTS_FILE}"
[[ -n "${RENKU_HOST_VENV_PATH}" ]] || \
    die "Activate the Renku .venv before running this script."

[[ -x "${HOST_PYTHON_BIN}" ]] || \
    die "Python was not found in the active virtual environment."
"${HOST_PYTHON_BIN}" -m pip --version >/dev/null 2>&1 || \
    die "pip is required in the active Renku virtual environment."

"${HOST_PYTHON_BIN}" - "${RENKU_HOST_VENV_PATH}" <<'PY'
import sys
from pathlib import Path

expected_prefix = Path(sys.argv[1]).resolve()
active_prefix = Path(sys.prefix).resolve()

if sys.platform != "linux":
    raise SystemExit(f"Linux is required; found {sys.platform}")
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required; found {sys.version.split()[0]}")
if active_prefix != expected_prefix:
    raise SystemExit(
        f"Active Python prefix {active_prefix} does not match {expected_prefix}"
    )

print(f"Python={sys.version.split()[0]}")
print(f"Virtual environment={active_prefix}")
PY

if ! "${HOST_PYTHON_BIN}" - <<-'PY'
	import IPython
	import ipykernel

	print(f"IPython={IPython.__version__}")
	print(f"ipykernel={ipykernel.__version__}")
	PY
then
    die "Renku's existing IPython and ipykernel packages are required."
fi

require_command nvidia-smi
GPU_LISTING="$(nvidia-smi -L)" || \
    die "No NVIDIA GPU is allocated. Start or restart Renku with a GPU."
[[ -n "${GPU_LISTING}" ]] || \
    die "No NVIDIA GPU is allocated. Start or restart Renku with a GPU."
printf '%s\n' "${GPU_LISTING}"

cd "${PROJECT_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "Creating isolated project environment at ${PROJECT_VENV_PATH}"
    env -u PYTHONHOME -u PYTHONPATH \
        "${HOST_PYTHON_BIN}" -m venv "${PROJECT_VENV_PATH}"
else
    log "Reusing isolated project environment at ${PROJECT_VENV_PATH}"
fi

unset PYTHONHOME PYTHONPATH || true
export PYTHONNOUSERSITE=1

"${PYTHON_BIN}" - "${PROJECT_VENV_PATH}" <<'PY'
import sys
from pathlib import Path

expected_prefix = Path(sys.argv[1]).resolve()
active_prefix = Path(sys.prefix).resolve()
shared_package_paths = [
    path
    for path in sys.path
    if path.startswith("/layers/") and "site-packages" in Path(path).parts
]

if active_prefix != expected_prefix:
    raise SystemExit(
        f"Project Python prefix {active_prefix} does not match {expected_prefix}"
    )
if shared_package_paths:
    raise SystemExit(
        "Renku shared package paths leaked into the project environment: "
        + ", ".join(shared_package_paths)
    )

print(f"Project environment={active_prefix}")
PY

log "Checking PyTorch and Torchvision CUDA compatibility"
if validate_torch_stack; then
    log "Keeping the compatible PyTorch installation"
else
    log "Installing a compatible PyTorch and Torchvision pair"
    install_torch_stack
    validate_torch_stack || \
        die "Installed PyTorch packages cannot use the allocated GPU."
fi

log "Removing packages that conflict with the headless OpenCV runtime"
"${PYTHON_BIN}" -m pip uninstall --yes \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    opencv-contrib-python-headless \
    ultralytics \
    ultralytics-opencv-headless \
    supervision
log "Installing notebook runtime packages"
"${PYTHON_BIN}" -m pip install \
    --constraint "${CONSTRAINTS_FILE}" \
    --requirement "${REQUIREMENTS_FILE}"

log "Checking the headless OpenCV runtime"
validate_headless_opencv || \
    die "Exactly one working headless OpenCV distribution is required."

if ! validate_torch_stack; then
    log "Repairing PyTorch after runtime dependency installation"
    install_torch_stack
    validate_torch_stack || \
        die "Runtime packages are incompatible with the CUDA PyTorch stack."
fi

log "Installing object_ctrl in editable mode"
"${PYTHON_BIN}" -m pip install \
    --no-build-isolation \
    --no-deps \
    --editable "${PROJECT_ROOT}"

log "Checking the YOLOX source revision"
if yolox_is_expected_commit; then
    log "Keeping YOLOX at ${YOLOX_COMMIT}"
else
    require_command git
    if ! command -v c++ >/dev/null 2>&1 \
        && ! command -v g++ >/dev/null 2>&1 \
        && ! command -v clang++ >/dev/null 2>&1; then
        die "A C++ compiler is required to build YOLOX."
    fi
    log "Installing YOLOX at ${YOLOX_COMMIT}"
    YOLOX_BUILD_DIR="$(mktemp -d /tmp/object-ctrl-yolox.XXXXXX)"
    git -C "${YOLOX_BUILD_DIR}" init --quiet
    git -C "${YOLOX_BUILD_DIR}" remote add origin "${YOLOX_REPOSITORY}"
    git -C "${YOLOX_BUILD_DIR}" fetch --depth 1 origin "${YOLOX_COMMIT}"
    git -C "${YOLOX_BUILD_DIR}" checkout --quiet --detach FETCH_HEAD

    installed_source_commit="$(git -C "${YOLOX_BUILD_DIR}" rev-parse HEAD)"
    [[ "${installed_source_commit}" == "${YOLOX_COMMIT}" ]] || \
        die "Fetched YOLOX revision does not match the requested commit."

    "${PYTHON_BIN}" - "${YOLOX_BUILD_DIR}" "${YOLOX_COMMIT}" <<-'PY'
	import sys
	from pathlib import Path

	checkout = Path(sys.argv[1])
	source_commit = sys.argv[2]
	jit_ops_path = checkout / "yolox" / "layers" / "jit_ops.py"
	requirements_path = checkout / "requirements.txt"
	jit_ops = jit_ops_path.read_text(encoding="utf-8")
	requirements = requirements_path.read_text(encoding="utf-8")

	if jit_ops.count("-std=c++14") != 2:
	    raise RuntimeError("Unexpected YOLOX compiler-flag layout")
	if requirements.splitlines().count("opencv_python") != 1:
	    raise RuntimeError("Unexpected YOLOX OpenCV requirement layout")
	if requirements.splitlines().count("onnx-simplifier==0.4.10") != 1:
	    raise RuntimeError("Unexpected YOLOX ONNX Simplifier requirement layout")

	jit_ops_path.write_text(
	    jit_ops.replace("-std=c++14", "-std=c++17"),
	    encoding="utf-8",
	)
	requirements_path.write_text(
	    requirements.replace("opencv_python", "opencv-python-headless").replace(
	        "onnx-simplifier==0.4.10",
	        "onnx-simplifier",
	    ),
	    encoding="utf-8",
	)
	(checkout / "yolox" / "_object_ctrl_build.py").write_text(
	    f'SOURCE_COMMIT = "{source_commit}"\nCXX_STANDARD = "c++17"\n',
	    encoding="utf-8",
	)
	PY
    "${PYTHON_BIN}" -m pip install \
        --no-build-isolation \
        --no-deps \
        --force-reinstall \
        "${YOLOX_BUILD_DIR}"
    yolox_is_expected_commit || \
        die "YOLOX was not installed at the required revision."
fi

log "Checking installed dependency compatibility"
"${PYTHON_BIN}" -m pip check

log "Verifying the complete notebook runtime"
"${PYTHON_BIN}" - "${YOLOX_COMMIT}" <<'PY'
import importlib.metadata
import sys

import cv2
import datumaro
import dotenv
import IPython
import ipykernel
import ipywidgets
import jupytext
import loguru
import matplotlib
import nbconvert
import ninja
import numpy
import object_ctrl
import onnx
import onnxsim
import pandas
import PIL
import polars
import psutil
import pycocotools
import requests
import roboflow
import scipy
import sklearn
import tabulate
import tensorboard
import thop
import tqdm
import ultralytics
import yaml
import yolox
from object_ctrl.platforms import yolox as object_ctrl_yolox
from yolox._object_ctrl_build import CXX_STANDARD, SOURCE_COMMIT
import yolox.layers.fast_cocoeval  # noqa: F401

input_info = onnx.helper.make_tensor_value_info(
    "input",
    onnx.TensorProto.FLOAT,
    [1],
)
output_info = onnx.helper.make_tensor_value_info(
    "output",
    onnx.TensorProto.FLOAT,
    [1],
)
onnx_model = onnx.helper.make_model(
    onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "object_ctrl_setup_check",
        [input_info],
        [output_info],
    )
)
onnx.checker.check_model(onnx_model)
_, simplification_ok = onnxsim.simplify(onnx_model)
if not simplification_ok:
    raise RuntimeError("ONNX Simplifier validation failed")

gui_lines = [
    line.split(":", maxsplit=1)[1].strip()
    for line in cv2.getBuildInformation().splitlines()
    if line.strip().startswith("GUI:")
]

expected_commit = sys.argv[1].lower()
installed_commit = SOURCE_COMMIT.lower()
if installed_commit != expected_commit:
    raise RuntimeError(
        f"Expected YOLOX commit {expected_commit}, found {installed_commit}"
    )
if CXX_STANDARD != "c++17":
    raise RuntimeError(f"Expected YOLOX C++17 build, found {CXX_STANDARD}")

distribution_names = (
    "datumaro",
    "ipywidgets",
    "jupytext",
    "nbconvert",
    "numpy",
    "object_ctrl",
    "onnx",
    "onnx-simplifier",
    "onnxsim",
    "opencv-python-headless",
    "pandas",
    "roboflow",
    "scikit-learn",
    "scipy",
    "tensorboard",
    "ultralytics-opencv-headless",
    "ultralytics-thop",
    "yolox",
)
for name in distribution_names:
    print(f"{name}={importlib.metadata.version(name)}")

print(f"object_ctrl root={object_ctrl.PROJECT_ROOT}")
print(f"YOLOX helper={object_ctrl_yolox.__name__}")
PY
[[ -x "${PROJECT_VENV_PATH}/bin/jupyter-nbconvert" ]] || \
    die "The project environment does not provide jupyter-nbconvert."
validate_torch_stack

log "Registering the isolated environment as a Jupyter kernel"
"${PYTHON_BIN}" -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}" \
    --env PYTHONPATH "" \
    --env PYTHONNOUSERSITE "1"

log "Synchronizing project Jupytext notebooks"
shopt -s nullglob
project_notebooks=("${PROJECT_ROOT}"/notebooks/*.py)
shopt -u nullglob
(( ${#project_notebooks[@]} > 0 )) || \
    die "No Jupytext sources were found under notebooks."
"${PYTHON_BIN}" -m jupytext --sync "${project_notebooks[@]}"

log "Renku environment setup completed successfully"
