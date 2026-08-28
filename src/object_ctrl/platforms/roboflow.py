"""
Utilities for downloading datasets from Roboflow.
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import PROJECT_ROOT

if TYPE_CHECKING:
    from roboflow.core.dataset import Dataset


class RoboflowFormat(StrEnum):
    """
    Commonly used dataset formats for Roboflow datasets.
    """

    COCO = "coco"
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    YOLOV11 = "yolov11"


def dataset_path(
    project_id: str,
    version: int,
) -> Path:
    """
    Return the default local path for a Roboflow dataset version.
    """
    return (
        Path(PROJECT_ROOT)
        / "datasets"
        / "sources"
        / "roboflow"
        / project_id
        / str(version)
    )


def roboflow_api_key() -> str:
    """
    Get the Roboflow API key from the environment or a local .env file.
    """
    return "dZMHwNntFqoQkztTTL6C"
    # api_key = os.getenv("ROBOFLOW_API_KEY")
    # if not api_key:
    #     raise ValueError("ROBOFLOW_API_KEY must be set.")
    # return api_key


def download(
    workspace_id: str,
    project_id: str,
    version: int,
    dataset_format: RoboflowFormat,
) -> Dataset:
    """
    Download a Roboflow project version into the local datasets directory.
    """
    api_key = roboflow_api_key()

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ImportError("Install the roboflow package missing.") from exc

    dataset_location = str(dataset_path(project_id, version))
    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace(workspace_id).project(project_id)
    project_version = project.version(version)
    return project_version.download(
        dataset_format.value,
        location=dataset_location,
        overwrite=False,
    )


def download_by_url(
    roboflow_url: str,
    dataset_format: RoboflowFormat,
) -> Dataset:
    """
    Download a Roboflow dataset using workspace metadata from its URL.
    """
    url_format = "https://.../<workspace>/<project>/dataset/<version>"
    match = re.search(r"/([^/]+)/([^/]+)/dataset/(\d+)(?:[/?#]|$)", roboflow_url)
    if not match:
        raise ValueError("URL error. Format: " + url_format)
    workspace_id, project_id, version = match.groups()

    return download(workspace_id, project_id, int(version), dataset_format)
