"""
Tests for the reusable Google Colab setup utilities.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
colab_setup = importlib.import_module("colabs.colab_setup")


def test_resolve_project_dir_defaults_to_repository() -> None:
    """
    Resolve the repository containing the Colab utility by default.
    """
    expected_root = Path(__file__).resolve().parents[1]

    assert colab_setup.resolve_project_dir() == expected_root


def test_resolve_project_dir_requires_pyproject(tmp_path: Path) -> None:
    """
    Reject a directory that is not an installable project root.
    """
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        colab_setup.resolve_project_dir(tmp_path)


def test_editable_requirement_includes_selected_extras() -> None:
    """
    Add comma-separated extras to the editable project requirement.
    """
    project_dir = Path("/tmp/object ctrl")

    requirement = colab_setup.editable_requirement(
        project_dir,
        extras=("data", "yolo"),
    )

    assert requirement == "/tmp/object ctrl[data,yolo]"


def test_installed_version_requirements_skips_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Pin installed runtime packages and omit packages that are not installed.
    """
    versions = {"numpy": "2.0.2", "pillow": "11.3.0"}

    def fake_version(package: str) -> str:
        """
        Return fixture versions and report unknown distributions as missing.
        """
        try:
            return versions[package]
        except KeyError as error:
            raise colab_setup.metadata.PackageNotFoundError(package) from error

    monkeypatch.setattr(colab_setup.metadata, "version", fake_version)

    requirements = colab_setup.installed_version_requirements(
        ("numpy", "missing", "pillow")
    )

    assert requirements == ("numpy==2.0.2", "pillow==11.3.0")


def test_install_project_dependencies_uses_active_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Install through the Python executable backing the active notebook kernel.
    """
    run = Mock()
    monkeypatch.setattr(colab_setup.subprocess, "run", run)
    project_dir = Path("/tmp/object ctrl")

    colab_setup.install_project_dependencies(
        project_dir,
        extras=("yolo",),
        packages=("datumaro",),
        preserve_packages=("numpy", "pillow"),
    )

    run.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            "/tmp/object ctrl[yolo]",
            "datumaro",
            f"numpy=={colab_setup.metadata.version('numpy')}",
            f"pillow=={colab_setup.metadata.version('pillow')}",
        ],
        cwd=project_dir,
        check=True,
    )


def test_configure_project_adds_source_path_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Expose the source directory without duplicating it in the kernel path.
    """
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "src" / "object_ctrl").mkdir(parents=True)
    kernel_path = ["existing-path"]
    change_directory = Mock()
    monkeypatch.setattr(colab_setup.sys, "path", kernel_path)
    monkeypatch.setattr(colab_setup.os, "chdir", change_directory)

    colab_setup.configure_project(tmp_path)
    colab_setup.configure_project(tmp_path)

    assert kernel_path == [str(tmp_path.resolve() / "src"), "existing-path"]
    assert change_directory.call_count == 2


def test_setup_project_installs_automatically_in_colab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Change directory and install dependencies automatically in Colab.
    """
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "src" / "object_ctrl").mkdir(parents=True)
    kernel_path = ["existing-path"]
    change_directory = Mock()
    install_dependencies = Mock()
    monkeypatch.setattr(colab_setup.sys, "path", kernel_path)
    monkeypatch.setattr(colab_setup.os, "chdir", change_directory)
    monkeypatch.setattr(colab_setup, "is_google_colab", Mock(return_value=True))
    monkeypatch.setattr(
        colab_setup,
        "install_project_dependencies",
        install_dependencies,
    )

    project_dir = colab_setup.setup_project(tmp_path)

    assert project_dir == tmp_path.resolve()
    assert kernel_path[0] == str(tmp_path.resolve() / "src")
    change_directory.assert_called_once_with(tmp_path.resolve())
    install_dependencies.assert_called_once_with(
        tmp_path.resolve(),
        extras=colab_setup.DEFAULT_EXTRAS,
        packages=colab_setup.DEFAULT_PACKAGES,
        preserve_packages=colab_setup.DEFAULT_PRESERVED_PACKAGES,
        quiet=True,
    )


def test_setup_project_skips_installation_for_local_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Avoid changing the local Conda environment during ordinary notebook runs.
    """
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "src" / "object_ctrl").mkdir(parents=True)
    install_dependencies = Mock()
    monkeypatch.setattr(colab_setup.sys, "path", ["existing-path"])
    monkeypatch.setattr(colab_setup.os, "chdir", Mock())
    monkeypatch.setattr(colab_setup, "is_google_colab", Mock(return_value=False))
    monkeypatch.setattr(
        colab_setup,
        "install_project_dependencies",
        install_dependencies,
    )

    colab_setup.setup_project(tmp_path)

    install_dependencies.assert_not_called()
