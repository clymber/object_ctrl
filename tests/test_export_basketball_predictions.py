"""
Check baseline export orchestration without training, loading weights, or a GPU.
"""

import argparse
import importlib.util
from pathlib import Path

import pytest
import yaml
from PIL import Image

from object_ctrl.evaluation import read_prediction_artifact
from object_ctrl.utils.json_io import read_json, write_json


@pytest.fixture
def exporter():
    """
    Import the CLI module without importing either detector framework.
    """
    path = Path(__file__).parents[1] / "scripts" / "export_basketball_predictions.py"
    spec = importlib.util.spec_from_file_location("basketball_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path: Path) -> argparse.Namespace:
    """
    Provide explicit checkpoint/run selection and tiny val/test splits with negatives.
    """
    dataset = tmp_path / "coco_basketball_2_2_2"
    (dataset / "annotations").mkdir(parents=True)
    for offset, split in enumerate(("train", "val", "test")):
        image_dir = dataset / "images" / split
        image_dir.mkdir(parents=True)
        images = []
        for index in range(2):
            image_id = offset * 10 + index + 1
            name = f"{image_id}.png"
            Image.new("RGB", (64, 32), (offset * 40, index * 80, 30)).save(
                image_dir / name
            )
            images.append(
                {"id": image_id, "file_name": name, "width": 64, "height": 32}
            )
        write_json(
            dataset / "annotations" / f"instances_{split}.json",
            {
                "images": images,
                "categories": [{"id": 7, "name": "basketball"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": images[0]["id"],
                        "category_id": 7,
                        "bbox": [2, 3, 10, 12],
                        "area": 120,
                        "iscrowd": 0,
                    }
                ],
            },
        )
    run_dir = tmp_path / "selected_run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best_ckpt.pth").write_bytes(b"fixture")
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": str(dataset),
                "classes": ["basketball"],
                "epochs": 100,
                "smoke_run": False,
                "train_batch_limit": None,
                "nms_threshold": 0.65,
            }
        )
    )
    (run_dir / "results.csv").write_text("epoch,loss\n1,3.0\n2,2.0\n")
    return argparse.Namespace(
        model="yolox",
        run_dir=run_dir,
        dataset_dir=dataset,
        output_dir=tmp_path / "exports",
        split="both",
        device="cpu",
        resolution=640,
        benchmark=False,
    )


def fake_predictor(args, run_settings: dict, category_id: int):
    """
    Return a box on every image so exports exercise positive and negative examples.
    """
    assert run_settings["nms_threshold"] == 0.65
    assert category_id == 7

    def predict_one(image: Image.Image) -> list[dict]:
        """
        Receive the unchanged original image and return original-pixel coordinates.
        """
        assert image.mode == "RGB"
        assert image.size == (64, 32)
        return [{"category_id": 7, "bbox": [2, 3, 10, 12], "score": 0.8}]

    return predict_one, {
        "model": "yolox_tiny",
        "checkpoint": str(args.run_dir / "weights" / "best_ckpt.pth"),
        "postprocessing": {"score_floor": 0.001, "nms_iou": 0.65},
    }


def test_ultralytics_version_prefers_standard_distribution(
    exporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Use the standard distribution metadata when it is installed.
    """
    requested = []

    def fake_version(distribution: str) -> str:
        """
        Record the first successful distribution lookup.
        """
        requested.append(distribution)
        return "8.3.0"

    monkeypatch.setattr(exporter, "version", fake_version)
    assert exporter.ultralytics_version() == "8.3.0"
    assert requested == ["ultralytics"]


def test_ultralytics_version_falls_back_to_headless_distribution(
    exporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve the Renku headless fork when standard metadata is unavailable.
    """
    requested = []

    def fake_version(distribution: str) -> str:
        """
        Fail the standard lookup and resolve the headless distribution.
        """
        requested.append(distribution)
        if distribution == "ultralytics":
            raise exporter.PackageNotFoundError(distribution)
        return "8.3.0-headless"

    monkeypatch.setattr(exporter, "version", fake_version)
    assert exporter.ultralytics_version() == "8.3.0-headless"
    assert requested == ["ultralytics", "ultralytics-opencv-headless"]


def test_ultralytics_version_reports_both_missing_distributions(
    exporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Name both supported distributions when neither has package metadata.
    """

    def missing_version(distribution: str) -> str:
        """
        Simulate an environment without either supported distribution.
        """
        raise exporter.PackageNotFoundError(distribution)

    monkeypatch.setattr(exporter, "version", missing_version)
    with pytest.raises(
        exporter.PackageNotFoundError,
        match="ultralytics or ultralytics-opencv-headless",
    ):
        exporter.ultralytics_version()


def test_export_covers_both_splits_and_preserves_native_metadata(
    exporter,
    args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Write portable original-ID predictions and common metrics for every split image.
    """
    monkeypatch.setattr(exporter, "build_predictor", fake_predictor)
    paths = exporter.export(args)
    assert [path.name for path in paths] == [
        "yolox_tiny_val_predictions.json",
        "yolox_tiny_test_predictions.json",
    ]
    for split, path in zip(("val", "test"), paths, strict=True):
        annotation_path = args.dataset_dir / "annotations" / f"instances_{split}.json"
        artifact = read_prediction_artifact(path, annotation_path)
        metadata = artifact["metadata"]
        assert metadata["split"] == split
        assert metadata["epochs_completed"] == 2
        assert metadata["resolution"] == 640
        assert metadata["device"] == "cpu"
        assert metadata["postprocessing"]["nms_iou"] == 0.65
        assert "no recorded content hash" in metadata["dataset_provenance"]
        assert {row["image_id"] for row in artifact["predictions"]} == set(
            artifact["image_ids"]
        )
        metrics = read_json(args.output_dir / f"yolox_tiny_{split}_metrics.json")
        assert metrics["negative_image_count"] == 1
        assert metrics["negative_false_positives"] == 1


@pytest.mark.parametrize("kind", ["predictions", "metrics"])
def test_existing_export_fails_before_model_loading(
    exporter,
    args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """
    Detect either split's output collision before spending time on inference.
    """
    args.output_dir.mkdir()
    existing = args.output_dir / f"yolox_tiny_test_{kind}.json"
    existing.write_text("preserve me")
    called = []
    monkeypatch.setattr(exporter, "build_predictor", lambda *args: called.append(True))
    with pytest.raises(FileExistsError, match="Choose a new output directory"):
        exporter.export(args)
    assert not called
    assert existing.read_text() == "preserve me"
    assert not (args.output_dir / "yolox_tiny_val_predictions.json").exists()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("smoke_run", True),
        ("train_batch_limit", 2),
        ("fraction", 0.5),
        ("dataset", "datasets/composed/coco_other_dataset"),
    ],
)
def test_incompatible_training_runs_fail_before_model_loading(
    exporter,
    args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value,
) -> None:
    """
    Exclude smoke/truncated or differently named datasets from full comparisons.
    """
    settings_path = args.run_dir / "args.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    settings[setting] = value
    settings_path.write_text(yaml.safe_dump(settings))
    called = []
    monkeypatch.setattr(exporter, "build_predictor", lambda *args: called.append(True))
    with pytest.raises(ValueError):
        exporter.export(args)
    assert not called


def test_split_category_mismatch_fails_before_model_loading(
    exporter,
    args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Avoid silently mapping test predictions to a category from a different split.
    """
    path = args.dataset_dir / "annotations" / "instances_test.json"
    annotations = read_json(path)
    annotations["categories"][0]["id"] = 42
    write_json(path, annotations)
    called = []
    monkeypatch.setattr(exporter, "build_predictor", lambda *args: called.append(True))
    with pytest.raises(ValueError, match="Category mapping differs"):
        exporter.export(args)
    assert not called
