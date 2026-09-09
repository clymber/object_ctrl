#!/usr/bin/env python3
"""
Export existing YOLO checkpoints to common COCO artifacts in the YOLO environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from object_ctrl.evaluation import (
    benchmark_predict,
    evaluate_predictions,
    write_prediction_artifact,
)
from object_ctrl.utils.json_io import read_json, write_json


def ultralytics_version() -> str:
    """
    Return the installed standard or headless Ultralytics distribution version.
    """
    distributions = ("ultralytics", "ultralytics-opencv-headless")
    missing = None
    for distribution in distributions:
        try:
            return version(distribution)
        except PackageNotFoundError as error:
            missing = error
    expected = " or ".join(distributions)
    raise PackageNotFoundError(expected) from missing


def parse_args() -> argparse.Namespace:
    """
    Require explicit baseline selection and keep detector-specific imports lazy.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ultralytics", "yolox"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Frozen composed COCO dataset directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def build_predictor(
    args: argparse.Namespace,
    run_settings: dict,
    category_id: int,
) -> tuple[Callable[[Image.Image], list[dict]], dict[str, Any]]:
    """
    Load a selected best checkpoint and return PIL-to-COCO prediction conversion.
    """
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; allocate a Renku GPU")
    if args.model == "ultralytics":
        from object_ctrl.platforms.ultralytics import configure_privacy

        configure_privacy()
        from ultralytics import YOLO

        framework_version = ultralytics_version()
        checkpoint = args.run_dir / "weights" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(checkpoint)
        if list(model.names.values()) != ["basketball"]:
            raise ValueError(f"Unexpected checkpoint classes: {model.names}")
        nms_iou = float(run_settings.get("iou", 0.7))
        max_det = int(run_settings.get("max_det", 300))
        agnostic_nms = bool(run_settings.get("agnostic_nms", False))

        def predict_one(image: Image.Image) -> list[dict]:
            """
            Predict at a low score floor while preserving native Ultralytics NMS.
            """
            result = model.predict(
                image,
                imgsz=args.resolution,
                device=args.device,
                conf=0.001,
                iou=nms_iou,
                max_det=max_det,
                agnostic_nms=agnostic_nms,
                quantize="fp32",
                verbose=False,
                rect=True,
            )[0]
            rows = []
            for box, score, label in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                if int(label) != 0:
                    raise ValueError("Unexpected Ultralytics class label")
                x1, y1, x2, y2 = box
                rows.append(
                    {
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
            return rows

        metadata = {
            "model": "yolo11n",
            "framework_version": framework_version,
            "parameters": sum(p.numel() for p in model.model.parameters()),
            "checkpoint": str(checkpoint),
            "postprocessing": {
                "score_floor": 0.001,
                "nms_iou": nms_iou,
                "max_det": max_det,
                "agnostic_nms": agnostic_nms,
                "rect": True,
                "precision": "float32",
                "resize": "Ultralytics native letterbox",
            },
        }
    else:
        from object_ctrl.platforms import yolox

        checkpoint = args.run_dir / "weights" / "best_ckpt.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if run_settings.get("classes") != ["basketball"]:
            raise ValueError("YOLOX run must use one basketball class")
        exp = yolox.BasketballTinyExp(
            dataset_dir=args.dataset_dir,
            output_dir=args.run_dir.parent,
            max_epoch=int(run_settings["epochs"]),
            image_size=args.resolution,
            project_name=args.run_dir.name,
        )
        exp.test_conf = 0.001
        exp.nmsthre = float(run_settings.get("nms_threshold", exp.nmsthre))
        device = torch.device(args.device)
        model = yolox.load_trained_model(exp, checkpoint, device)
        categories = SimpleNamespace(class_ids=[category_id])

        def predict_one(image: Image.Image) -> list[dict]:
            """
            Use the existing YOLOX FP32 forward/CPU-NMS path at the AP score floor.
            """
            bgr = np.asarray(image)[:, :, ::-1].copy()
            return yolox.predict_image(
                model, exp, bgr, device, categories, conf_threshold=0.001
            )

        metadata = {
            "model": "yolox_tiny",
            "framework_version": version("yolox"),
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint": str(checkpoint),
            "postprocessing": {
                "score_floor": 0.001,
                "nms_iou": exp.nmsthre,
                "precision": "float32",
                "nms_device": "cpu",
                "resize": "YOLOX native ValTransform letterbox",
            },
        }
    return predict_one, metadata


def export(args: argparse.Namespace) -> list[Path]:
    """
    Export both held-out splits without retraining or modifying the selected run.
    """
    args.run_dir = args.run_dir.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    with (args.run_dir / "args.yaml").open() as stream:
        run_settings = yaml.safe_load(stream)
    if not isinstance(run_settings, dict):
        raise ValueError("Run args.yaml must contain an object")
    if (
        run_settings.get("smoke_run")
        or run_settings.get("train_batch_limit") is not None
        or float(run_settings.get("fraction", 1.0)) != 1.0
    ):
        raise ValueError(
            "Smoke/truncated training cannot enter the baseline comparison"
        )
    if args.resolution <= 0 or args.resolution % 32:
        raise ValueError("YOLO comparison resolution must be a positive multiple of 32")
    trained_path = Path(str(run_settings.get("data", run_settings.get("dataset", ""))))
    trained_name = (
        trained_path.parent.name
        if trained_path.suffix == ".yaml"
        else trained_path.name
    )
    dataset_name = args.dataset_dir.name.removeprefix("coco_")
    if trained_name.removeprefix("yolo_").removeprefix("coco_") != dataset_name:
        raise ValueError(
            "Selected run's dataset name does not match the comparison dataset"
        )
    splits = ("val", "test") if args.split == "both" else (args.split,)
    train_annotations = read_json(
        args.dataset_dir / "annotations" / "instances_train.json"
    )
    if (
        len(train_annotations["categories"]) != 1
        or train_annotations["categories"][0]["name"] != "basketball"
    ):
        raise ValueError("Expected a one-class basketball dataset")
    category_id = train_annotations["categories"][0]["id"]
    model_name = "yolo11n" if args.model == "ultralytics" else "yolox_tiny"
    split_annotations = {}
    for split in splits:
        for kind in ("predictions", "metrics"):
            path = args.output_dir / f"{model_name}_{split}_{kind}.json"
            if path.exists():
                raise FileExistsError(
                    f"Choose a new output directory; export exists: {path}"
                )
        annotation_path = args.dataset_dir / "annotations" / f"instances_{split}.json"
        annotations = read_json(annotation_path)
        categories = annotations["categories"]
        if (
            len(categories) != 1
            or categories[0]["id"] != category_id
            or categories[0]["name"] != "basketball"
        ):
            raise ValueError(f"Category mapping differs in {split} annotations")
        if not annotations["images"]:
            raise ValueError(f"The {split} split is empty")
        for image in annotations["images"]:
            image_path = args.dataset_dir / "images" / split / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
        split_annotations[split] = annotations
    history = pd.read_csv(args.run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    if history.empty or "epoch" not in history:
        raise ValueError("Selected run requires results.csv with completed epochs")
    predict_one, metadata = build_predictor(args, run_settings, category_id)
    metadata.update(
        {
            "run_dir": str(args.run_dir),
            "resolution": args.resolution,
            "device": args.device,
            "smoke_run": False,
            "training_settings": run_settings,
            "epochs_completed": int(history["epoch"].max()),
            "dataset_provenance": (
                "Original run matches dataset name; annotation hashes are "
                "recorded now. "
                "Historical training data has no recorded content hash."
            ),
        }
    )
    paths = []
    for split in splits:
        annotation_path = args.dataset_dir / "annotations" / f"instances_{split}.json"
        ground_truth = split_annotations[split]
        predictions = []
        image_paths = []
        for entry in ground_truth["images"]:
            image_path = args.dataset_dir / "images" / split / entry["file_name"]
            image_paths.append(image_path)
            with Image.open(image_path) as image:
                predictions.extend(
                    {"image_id": entry["id"], **row}
                    for row in predict_one(image.convert("RGB"))
                )
        split_metadata: dict[str, Any] = {**metadata, "split": split}
        if args.benchmark:
            split_metadata["benchmark"] = benchmark_predict(
                predict_one, image_paths, device=args.device
            )
        artifact_path = (
            args.output_dir / f"{metadata['model']}_{split}_predictions.json"
        )
        paths.append(
            write_prediction_artifact(
                artifact_path,
                annotation_path,
                predictions,
                metadata=split_metadata,
            )
        )
        write_json(
            args.output_dir / f"{metadata['model']}_{split}_metrics.json",
            evaluate_predictions(annotation_path, predictions),
        )
    return paths


def main() -> None:
    """
    Run an explicitly selected export and print the resulting portable artifact paths.
    """
    for path in export(parse_args()):
        print(path)


if __name__ == "__main__":
    main()
