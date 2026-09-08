#!/usr/bin/env bash
# Set up RF-DETR in its own Linux/CUDA environment; retain existing YOLO packages.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly REQUIREMENTS_FILE="${SCRIPT_DIR}/renku/rfdetr-requirements.txt"
readonly CONSTRAINTS_FILE="${SCRIPT_DIR}/renku/rfdetr-constraints.txt"
PROJECT_VENV="${OBJCTRL_RFDETR_VENV:-${PROJECT_ROOT}/.venv-renku-rfdetr}"
readonly KERNEL_NAME="${OBJCTRL_RFDETR_KERNEL_NAME:-object-ctrl-renku-rfdetr}"
readonly TORCH_INDEX="${OBJCTRL_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
readonly PACKAGE_INDEX="${OBJCTRL_RFDETR_PACKAGE_INDEX_URL:-https://pypi.org/simple}"
readonly AUDIT_DIR="${PROJECT_ROOT}/outputs/environment/rfdetr/setup_$(date -u +%Y%m%dT%H%M%SZ)_$$"
readonly NOTEBOOK="${PROJECT_ROOT}/notebooks/nb04.02-rfdetr_small_large_basketball.py"
readonly YOLO_PYTHON="${PROJECT_ROOT}/.venv-renku/bin/python"

log() {
    printf '\n[RF-DETR setup] %s\n' "$*"
}

die() {
    printf '\n[RF-DETR setup] ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "${KERNEL_NAME}" != object-ctrl-renku ]] || \
    die "The object-ctrl-renku kernel is reserved for the existing YOLO environment."
[[ "$(uname -s)" == Linux ]] || die "Run this script in a Linux Renku GPU session."
command -v nvidia-smi >/dev/null 2>&1 || die "Allocate an NVIDIA GPU on Renku first."
GPU_LISTING="$(nvidia-smi -L)" || die "nvidia-smi cannot access an allocated GPU."
[[ -n "${GPU_LISTING}" ]] || die "No NVIDIA GPU is visible in this session."
printf '%s\n' "${GPU_LISTING}"

# An active YOLO environment is a valid interpreter provider, never an install target.
if [[ -n "${OBJCTRL_RFDETR_PYTHON:-}" ]]; then
    HOST_PYTHON="${OBJCTRL_RFDETR_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${PROJECT_VENV}" ]]; then
    HOST_PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${HOME}/work/.venv/bin/python" ]]; then
    HOST_PYTHON="${HOME}/work/.venv/bin/python"
else
    HOST_PYTHON="$(command -v python3)" || die "Set OBJCTRL_RFDETR_PYTHON to Renku Python."
fi
[[ -x "${HOST_PYTHON}" ]] || die "Interpreter not found: ${HOST_PYTHON}"

# Clear environment variables that can redirect a venv install or leak host imports.
unset PYTHONHOME PYTHONPATH PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONNOUSERSITE=1
# Resolve overrides against the caller's directory before changing to the project.
# Preserve the interpreter's venv symlink: resolving it would select base Python.
HOST_PYTHON="$("${HOST_PYTHON}" -I -c \
    'import os, sys; print(os.path.abspath(sys.argv[1]))' "${HOST_PYTHON}")"
PROJECT_VENV="$("${HOST_PYTHON}" -I -c \
    'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
    "${PROJECT_VENV}")"
readonly HOST_PYTHON PROJECT_VENV
readonly PYTHON_BIN="${PROJECT_VENV}/bin/python"
export MPLBACKEND=Agg
export RF_HOME="${PROJECT_ROOT}/models/pretrained/rfdetr"
export HF_HOME="${PROJECT_ROOT}/models/cache/rfdetr/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/models/cache/rfdetr/torch"
export XDG_CACHE_HOME="${PROJECT_ROOT}/models/cache/rfdetr/xdg"
export MPLCONFIGDIR="${PROJECT_ROOT}/models/cache/rfdetr/matplotlib"
export HF_HUB_DISABLE_TELEMETRY=1

"${HOST_PYTHON}" -I - "${PROJECT_VENV}" "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path

target, project = (Path(value).resolve() for value in sys.argv[1:])
if sys.platform != "linux" or not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit("Use a Linux Python 3.11, 3.12, or 3.13 interpreter (3.11 preferred).")
if target in {Path(sys.prefix).resolve(), (project / ".venv-renku").resolve()}:
    raise SystemExit("RF-DETR must not target the host or the existing YOLO environment.")
if target.exists() and not (target / ".object-ctrl-rfdetr").is_file():
    raise SystemExit(f"Refusing to modify an unrecognized environment: {target}")
print(f"Interpreter provider: {sys.executable} ({sys.version.split()[0]})")
print(f"Isolated RF-DETR environment: {target}")
PY

cd "${PROJECT_ROOT}"
mkdir -p "${AUDIT_DIR}" "${RF_HOME}" "${HF_HOME}" "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
    > "${AUDIT_DIR}/gpu.csv"

snapshot_packages() {
    local interpreter="$1"
    local destination="$2"
    "${interpreter}" -I - "${destination}" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

packages = sorted(
    (
        {"name": distribution.metadata["Name"], "version": distribution.version}
        for distribution in importlib.metadata.distributions()
    ),
    key=lambda entry: entry["name"].lower(),
)
Path(sys.argv[1]).write_text(json.dumps(packages, indent=2) + "\n", encoding="utf-8")
PY
}

snapshot_packages "${HOST_PYTHON}" "${AUDIT_DIR}/host.before.json"
if [[ -x "${YOLO_PYTHON}" ]]; then
    snapshot_packages "${YOLO_PYTHON}" "${AUDIT_DIR}/yolo.before.json"
fi

verify_untouched_environments() {
    local status="$?"
    trap - EXIT
    snapshot_packages "${HOST_PYTHON}" "${AUDIT_DIR}/host.after.json" || status=1
    if ! cmp -s "${AUDIT_DIR}/host.before.json" "${AUDIT_DIR}/host.after.json"; then
        printf 'Host package inventory changed; inspect %s\n' "${AUDIT_DIR}" >&2
        status=1
    fi
    if [[ -f "${AUDIT_DIR}/yolo.before.json" ]]; then
        snapshot_packages "${YOLO_PYTHON}" "${AUDIT_DIR}/yolo.after.json" || status=1
        if ! cmp -s "${AUDIT_DIR}/yolo.before.json" "${AUDIT_DIR}/yolo.after.json"; then
            printf 'YOLO package inventory changed; inspect %s\n' "${AUDIT_DIR}" >&2
            status=1
        fi
    fi
    printf '\nEnvironment audit: %s\n' "${AUDIT_DIR}"
    exit "${status}"
}
trap verify_untouched_environments EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "Creating ${PROJECT_VENV}"
    "${HOST_PYTHON}" -I -m venv "${PROJECT_VENV}"
    printf 'object_ctrl RF-DETR isolated runtime\n' > "${PROJECT_VENV}/.object-ctrl-rfdetr"
fi
"${PYTHON_BIN}" -I - "${PROJECT_VENV}" <<'PY'
import site
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != target or sys.prefix == sys.base_prefix:
    raise SystemExit("The target interpreter is not the requested isolated venv.")
if site.ENABLE_USER_SITE:
    raise SystemExit("User site-packages must be disabled.")
for item in sys.path:
    path = Path(item).resolve()
    if "site-packages" in path.parts and not path.is_relative_to(target):
        raise SystemExit(f"External site-packages leaked into RF-DETR: {path}")
if "include-system-site-packages = true" in (target / "pyvenv.cfg").read_text():
    raise SystemExit("RF-DETR venv must not include system site-packages.")
PY

pip_install() {
    "${PYTHON_BIN}" -I -m pip --isolated \
        --cache-dir "${PROJECT_ROOT}/models/cache/rfdetr/pip" install "$@"
}

log "Installing the pinned CUDA torch/torchvision pair from ${TORCH_INDEX}"
if "${PYTHON_BIN}" -I - "${TORCH_INDEX}" <<'PY'
import importlib.metadata
import sys

expected_cuda = sys.argv[1].rstrip("/").rsplit("/", 1)[-1]
try:
    expected = {"torch": "2.10.0", "torchvision": "0.25.0"}
    for name, version in expected.items():
        if importlib.metadata.version(name) != f"{version}+{expected_cuda}":
            raise SystemExit(1)
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1) from None
PY
then
    log "The requested CUDA wheel versions are already installed"
else
    pip_install --index-url "${TORCH_INDEX}" --constraint "${CONSTRAINTS_FILE}" \
        --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0
fi
log "Installing RF-DETR and notebook dependencies"
pip_install --index-url "${PACKAGE_INDEX}" --constraint "${CONSTRAINTS_FILE}" \
    --requirement "${REQUIREMENTS_FILE}"
pip_install --index-url "${PACKAGE_INDEX}" --constraint "${CONSTRAINTS_FILE}" \
    --no-build-isolation --editable "${PROJECT_ROOT}"
"${PYTHON_BIN}" -I -m pip check

log "Checking headless imports, CUDA kernels, and RF-DETR Small at 640 pixels"
"${PYTHON_BIN}" - "${AUDIT_DIR}" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

import cv2
import ipykernel
import jupytext
import matplotlib
import nbconvert
import object_ctrl
import torch
import torchvision
from PIL import Image
from rfdetr import RFDETRSmall
from torchvision.ops import nms

from object_ctrl.dataset.rfdetr import prepare_coco_dataset

opencv_names = {
    "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless",
    "opencv-python-headless",
}
installed = {
    distribution.metadata["Name"].lower()
    for distribution in importlib.metadata.distributions()
}
if installed & opencv_names != {"opencv-python-headless"}:
    raise SystemExit(f"Require exactly one headless cv2 package: {installed & opencv_names}")
gui = [line.split(":", 1)[1].strip() for line in cv2.getBuildInformation().splitlines()
       if line.strip().startswith("GUI:")]
if gui != ["NONE"]:
    raise SystemExit(f"OpenCV is not headless: GUI={gui}")
if not torch.version.cuda or not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Check the driver and set OBJCTRL_TORCH_INDEX_URL "
        "to a compatible CUDA wheel index, then rerun setup."
    )
tensor = torch.tensor([1., 2.], device="cuda")
assert tensor.square().sum().item() == 5.
boxes = torch.tensor([[0., 0., 4., 4.]], device="cuda")
assert nms(boxes, torch.tensor([0.9], device="cuda"), 0.5).numel() == 1
# Download/cache the official Small checkpoint, then exercise its actual custom
# resolution path. This predicts one synthetic image; it does not train a model.
model = RFDETRSmall(device="cuda", resolution=640, positional_encoding_size=40)
predictions = model.predict(Image.new("RGB", (640, 640)), threshold=0.99)
torch.cuda.synchronize()
if predictions.xyxy.ndim != 2 or predictions.xyxy.shape[1] != 4:
    raise SystemExit("RF-DETR returned unexpected prediction geometry.")
report = {
    "python": sys.version,
    "interpreter": sys.executable,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "rfdetr": importlib.metadata.version("rfdetr"),
    "pycocotools": importlib.metadata.version("pycocotools"),
    "resolution": 640,
    "forward_pass": "passed",
}
Path(sys.argv[1], "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

log "Registering the distinct RF-DETR notebook kernel"
"${PYTHON_BIN}" - "${KERNEL_NAME}" "${PYTHON_BIN}" <<'PY'
import os
import sys

from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

try:
    existing = KernelSpecManager().get_kernel_spec(sys.argv[1])
except NoSuchKernel:
    pass
else:
    if os.path.abspath(existing.argv[0]) != os.path.abspath(sys.argv[2]):
        raise SystemExit(
            f"Kernel {sys.argv[1]!r} already belongs to {existing.argv[0]}; "
            "choose another OBJCTRL_RFDETR_KERNEL_NAME."
        )
PY
"${PYTHON_BIN}" -m ipykernel install --user --name "${KERNEL_NAME}" \
    --display-name "Python (object_ctrl Renku RF-DETR)" \
    --env PYTHONPATH "" --env PYTHONNOUSERSITE 1 --env MPLBACKEND Agg
"${PYTHON_BIN}" - "${KERNEL_NAME}" "${PYTHON_BIN}" "${AUDIT_DIR}" <<'PY'
import json
import os
import sys
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager

kernel = KernelSpecManager().get_kernel_spec(sys.argv[1])
# Compare the venv path without resolving its python symlink to the base binary.
if os.path.abspath(kernel.argv[0]) != os.path.abspath(sys.argv[2]):
    raise SystemExit(f"Wrong RF-DETR kernel interpreter: {kernel.argv[0]}")
Path(sys.argv[3], "kernel.json").write_text(json.dumps(kernel.to_dict(), indent=2) + "\n")
print(f"Kernel {sys.argv[1]} uses {kernel.argv[0]}")
PY

[[ -f "${NOTEBOOK}" ]] || die "RF-DETR Jupytext notebook is missing: ${NOTEBOOK}"
"${PYTHON_BIN}" -m jupytext --sync "${NOTEBOOK}"
snapshot_packages "${PYTHON_BIN}" "${AUDIT_DIR}/rfdetr.packages.json"
"${PYTHON_BIN}" -I -m pip freeze --all > "${AUDIT_DIR}/rfdetr.freeze.txt"
log "RF-DETR setup completed; select kernel ${KERNEL_NAME}."
