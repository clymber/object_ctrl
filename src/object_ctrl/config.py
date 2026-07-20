"""
Runtime configuration for YOLO experiments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .utils.text_stream import set_text_stream_filter


def find_project_root(start: Path | None = None) -> Path:
    """
    Find the project root by looking for pyproject.toml.
    """
    current = (start or Path.cwd()).resolve()

    for path in [current, *current.parents]:
        if (path / "pyproject.toml").is_file():
            return path
        project_child = path / "object_ctrl"
        if (project_child / "pyproject.toml").is_file():
            return project_child

    raise RuntimeError("Project root not found")


# Basic runtime directory configuration
PROJECT_ROOT = find_project_root()


def configure_stdio_relative_path() -> None:
    """
    Display paths beneath a base path relatively for consistent standard streams.
    """
    substitution = {
        f"{Path(PROJECT_ROOT).resolve()}{os.sep}": "",
        f"{Path.home().resolve()}{os.sep}": f"~{os.sep}",
    }
    sys.stdout = set_text_stream_filter(sys.stdout, map=substitution)
    sys.stderr = set_text_stream_filter(sys.stderr, map=substitution)


def get_device() -> str:
    """
    Select the best available device for local YOLO experiments.

    Priority:
    1. Apple Silicon MPS
    2. CPU fallback
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
