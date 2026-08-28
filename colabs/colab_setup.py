# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %%
"""
Reusable setup utilities for running this project in Google Colab.

Google Drive must be mounted before importing this module from Drive. Add the
repository root to ``sys.path``, import ``setup_project``, and call it before
importing ``object_ctrl``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

DEFAULT_EXTRAS = ("data", "yolo")
DEFAULT_PACKAGES = (
    "datumaro",
    "roboflow",
    "pycocotools",
    "supervision",
)
DEFAULT_PRESERVED_PACKAGES = (
    "cachetools",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "pandas",
    "pillow",
    "scikit-learn",
    "scipy",
    "torch",
    "torchaudio",
    "torchvision",
)


def is_google_colab() -> bool:
    """
    Return whether the current Python kernel is running in Google Colab.
    """
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def resolve_project_dir(project_dir: Path | str | None = None) -> Path:
    """
    Resolve and validate the project directory and required source markers.
    """
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[1]

    resolved_dir = Path(project_dir).expanduser().resolve()
    required_paths = (
        resolved_dir / "pyproject.toml",
        resolved_dir / "src" / "object_ctrl",
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Project directory is missing required paths: {missing_text}"
        )
    return resolved_dir


def configure_project(project_dir: Path | str | None = None) -> Path:
    """
    Enter the project directory and expose its source tree to the active kernel.
    """
    resolved_dir = resolve_project_dir(project_dir)
    os.chdir(resolved_dir)

    source_dir = str(resolved_dir / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    return resolved_dir


def editable_requirement(
    project_dir: Path,
    extras: Sequence[str] = DEFAULT_EXTRAS,
) -> str:
    """
    Build the editable-install requirement for the project and selected extras.
    """
    extras_spec = ",".join(extras)
    if not extras_spec:
        return str(project_dir)
    return f"{project_dir}[{extras_spec}]"


def installed_version_requirements(packages: Sequence[str]) -> tuple[str, ...]:
    """
    Pin packages already supplied by the active notebook runtime.
    """
    requirements = []
    for package in packages:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
        requirements.append(f"{package}=={version}")
    return tuple(requirements)


def install_project_dependencies(
    project_dir: Path,
    *,
    extras: Sequence[str] = DEFAULT_EXTRAS,
    packages: Sequence[str] = DEFAULT_PACKAGES,
    preserve_packages: Sequence[str] = DEFAULT_PRESERVED_PACKAGES,
    quiet: bool = True,
) -> None:
    """
    Install dependencies while preserving Colab's core runtime package versions.
    """
    command = [sys.executable, "-m", "pip", "install"]
    if quiet:
        command.append("-q")
    preserved_requirements = installed_version_requirements(preserve_packages)
    command.extend(
        [
            "-e",
            editable_requirement(project_dir, extras),
            *packages,
            *preserved_requirements,
        ]
    )
    subprocess.run(command, cwd=project_dir, check=True)


def setup_project(
    project_dir: Path | str | None = None,
    *,
    install_dependencies: bool | None = None,
    extras: Sequence[str] = DEFAULT_EXTRAS,
    packages: Sequence[str] = DEFAULT_PACKAGES,
    preserve_packages: Sequence[str] = DEFAULT_PRESERVED_PACKAGES,
    quiet: bool = True,
) -> Path:
    """
    Enter the project directory and install dependencies when running in Colab.

    When ``install_dependencies`` is not specified, installation runs in Google
    Colab and is skipped for local kernels.
    """
    resolved_dir = configure_project(project_dir)

    if install_dependencies is None:
        install_dependencies = is_google_colab()
    if install_dependencies:
        install_project_dependencies(
            resolved_dir,
            extras=extras,
            packages=packages,
            preserve_packages=preserve_packages,
            quiet=quiet,
        )

    return resolved_dir


# %% [markdown]
# ## Import from another notebook
#
# Mount Drive before importing a utility stored on Drive:
#
# ```python
# from pathlib import Path
# import sys
#
# from google.colab import drive
#
# drive.mount("/content/drive")
# project_dir = Path("/content/drive/MyDrive/object_ctrl")
# sys.path.insert(0, str(project_dir))
#
# from colabs.colab_setup import setup_project
#
# setup_project(project_dir)
# ```
#
# With the VS Code Colab extension, use **Colab: Mount Google Drive to
# Server...** first, then run the code above starting at `project_dir`.
