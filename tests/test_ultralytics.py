from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

from object_ctrl.platforms import ultralytics


def test_ultralytics_dataset_path_uses_project_source_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Build the default local dataset path from the project root.
    """
    monkeypatch.setattr(ultralytics, "PROJECT_ROOT", tmp_path)

    path = ultralytics.dataset_path("coco128")

    assert path == tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"


def test_download_dataset_writes_local_yaml_and_extracts_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Stage an Ultralytics dataset locally without keeping the global path config.
    """
    calls: list[tuple[Path, str]] = []
    dataset_download_url = "https://example.test/coco128.zip"

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Create local fixtures for the official YAML and dataset archive.
        """
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        calls.append((destination, url))

        if url == ultralytics.official_dataset_yaml_url("coco128"):
            destination.write_text(
                yaml.safe_dump(
                    {
                        "path": "coco128",
                        "train": "images/train2017",
                        "val": "images/train2017",
                        "names": {0: "person"},
                        "download": dataset_download_url,
                    },
                    sort_keys=False,
                )
            )
        elif url == dataset_download_url:
            with ZipFile(destination, "w") as archive:
                archive.writestr("coco128/images/train2017/image.jpg", "")
                archive.writestr("coco128/labels/train2017/image.txt", "")
        else:
            raise AssertionError(f"Unexpected URL: {url}")

        return destination

    monkeypatch.setattr(ultralytics, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    data_yaml = ultralytics.download("coco128")
    dataset_dir = ultralytics.dataset_path("coco128")
    local_config = yaml.safe_load(data_yaml.read_text())

    assert data_yaml == dataset_dir / "data.yaml"
    assert (dataset_dir / "images" / "train2017" / "image.jpg").exists()
    assert local_config["path"] == str(dataset_dir)
    assert local_config["train"] == "images/train2017"
    assert "download" not in local_config
    assert calls == [
        (
            dataset_dir / "coco128.official.yaml",
            ultralytics.official_dataset_yaml_url("coco128"),
        ),
        (dataset_dir.parent / "coco128.zip", dataset_download_url),
    ]


def test_download_dataset_reuses_existing_local_data_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Return the local data YAML without fetching metadata when it is usable.
    """
    dataset_dir = tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"
    (dataset_dir / "images" / "train2017").mkdir(parents=True)
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train2017",
                "val": "images/train2017",
                "names": {0: "person"},
            },
            sort_keys=False,
        )
    )

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Fail if the reusable local data YAML path does not short-circuit.
        """
        raise AssertionError(f"Unexpected download call for {url} to {cache_path}")

    monkeypatch.setattr(ultralytics, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    assert ultralytics.download("coco128") == data_yaml


def test_download_dataset_regenerates_stale_local_data_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Download metadata and payload when local data YAML points to missing data.
    """
    calls: list[tuple[Path, str]] = []
    dataset_download_url = "https://example.test/coco128.zip"
    dataset_dir = tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"
    dataset_dir.mkdir(parents=True)
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train2017",
                "val": "images/train2017",
                "names": {0: "person"},
            },
            sort_keys=False,
        )
    )

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Create local fixtures for repairing a stale local data YAML.
        """
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        calls.append((destination, url))

        if url == ultralytics.official_dataset_yaml_url("coco128"):
            destination.write_text(
                yaml.safe_dump(
                    {
                        "path": "coco128",
                        "train": "images/train2017",
                        "val": "images/train2017",
                        "names": {0: "person"},
                        "download": dataset_download_url,
                    },
                    sort_keys=False,
                )
            )
        elif url == dataset_download_url:
            with ZipFile(destination, "w") as archive:
                archive.writestr("coco128/images/train2017/image.jpg", "")
        else:
            raise AssertionError(f"Unexpected URL: {url}")

        return destination

    monkeypatch.setattr(ultralytics, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    assert ultralytics.download("coco128") == data_yaml
    assert (dataset_dir / "images" / "train2017" / "image.jpg").exists()
    assert calls == [
        (
            dataset_dir / "coco128.official.yaml",
            ultralytics.official_dataset_yaml_url("coco128"),
        ),
        (dataset_dir.parent / "coco128.zip", dataset_download_url),
    ]


def test_download_dataset_reuses_existing_dataset_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Avoid downloading the archive again when the expected train split exists.
    """
    calls: list[str] = []
    dataset_dir = tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"
    (dataset_dir / "images" / "train2017").mkdir(parents=True)

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Create only the official YAML fixture for an already-staged dataset.
        """
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(
                {
                    "path": "coco128",
                    "train": "images/train2017",
                    "val": "images/train2017",
                    "names": {0: "person"},
                    "download": "https://example.test/coco128.zip",
                },
                sort_keys=False,
            )
        )
        calls.append(url)
        return destination

    monkeypatch.setattr(ultralytics, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    ultralytics.download("coco128")

    assert calls == [ultralytics.official_dataset_yaml_url("coco128")]
