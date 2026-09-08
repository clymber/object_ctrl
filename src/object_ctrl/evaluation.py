"""
Shared COCO evaluation and portable prediction artifacts for detector comparisons.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import platform
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .utils.json_io import read_json, write_json


def file_sha256(path: Path | str) -> str:
    """
    Hash a file without loading weights or images into memory all at once.
    """
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _predictions(
    ground_truth: dict, predictions: Sequence[dict], floor: float, maximum: int
) -> list[dict]:
    """
    Validate COCO predictions and apply a common per-image score/top-k policy.
    """
    image_ids = {image["id"] for image in ground_truth["images"]}
    category_ids = {category["id"] for category in ground_truth["categories"]}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for prediction in predictions:
        if not isinstance(prediction, dict) or not {
            "image_id",
            "category_id",
            "bbox",
            "score",
        }.issubset(prediction):
            raise ValueError(
                "Prediction requires image_id, category_id, bbox and score"
            )
        if prediction["image_id"] not in image_ids:
            raise ValueError("Prediction references an unknown image ID")
        if prediction["category_id"] not in category_ids:
            raise ValueError("Prediction references an unknown category ID")
        try:
            box = [float(value) for value in prediction["bbox"]]
            score = float(prediction["score"])
        except (TypeError, ValueError) as error:
            raise ValueError("Prediction box and score must be numeric") from error
        if (
            len(box) != 4
            or not all(math.isfinite(value) for value in box)
            or box[2] < 0
            or box[3] < 0
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Prediction contains an invalid box or confidence")
        if score >= floor:
            grouped[prediction["image_id"]].append(
                {
                    "image_id": prediction["image_id"],
                    "category_id": prediction["category_id"],
                    "bbox": box,
                    "score": score,
                }
            )
    return [
        prediction
        for image_id in sorted(grouped)
        for prediction in sorted(grouped[image_id], key=lambda item: -item["score"])[
            :maximum
        ]
    ]


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    """
    Compute intersection-over-union for two COCO xywh boxes.
    """
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[0] + first[2], second[0] + second[2])
    y2 = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union else 0.0


def evaluate_predictions(
    annotation_path: Path | str,
    predictions: Sequence[dict],
    *,
    score_floor: float = 0.001,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    max_dets: tuple[int, int, int] = (1, 10, 100),
) -> dict[str, Any]:
    """
    Compute COCO AP/AR and score-ordered, one-to-one matching at a fixed threshold.

    All source images, including negatives, enter COCO evaluation. Both metric
    policies retain at most 100 detections per image. Undefined COCO metrics
    (for example AP on an all-negative split) are returned as None.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not 0 <= score_floor <= confidence_threshold <= 1:
        raise ValueError("Require 0 <= score floor <= confidence threshold <= 1")
    if not 0 < iou_threshold <= 1 or max_dets != (1, 10, 100):
        raise ValueError("Use IoU in (0, 1] and standard maxDets=(1, 10, 100)")
    ground_truth = read_json(annotation_path)
    if not ground_truth["images"]:
        raise ValueError("Evaluation requires at least one image")
    # The fixed-threshold policy is designed for this project's non-crowd boxes.
    if any(
        a.get("iscrowd", 0) or a.get("ignore", 0) for a in ground_truth["annotations"]
    ):
        raise ValueError(
            "Fixed-threshold comparison does not support crowd/ignore boxes"
        )
    filtered = _predictions(ground_truth, predictions, score_floor, max_dets[-1])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        coco = COCO(str(annotation_path))
        if filtered:
            detections = coco.loadRes(filtered)
        else:
            # COCO.loadRes([]) indexes its first element; build an empty API instead.
            detections = COCO()
            detections.dataset = {
                "images": ground_truth["images"],
                "categories": ground_truth["categories"],
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(coco, detections, "bbox")
        evaluator.params.imgIds = sorted(
            image["id"] for image in ground_truth["images"]
        )
        evaluator.params.maxDets = list(max_dets)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    actual: dict[int, list[dict]] = defaultdict(list)
    predicted: dict[int, list[dict]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        actual[annotation["image_id"]].append(annotation)
    for detection in filtered:
        if detection["score"] >= confidence_threshold:
            predicted[detection["image_id"]].append(detection)
    true_positives = false_positives = false_negatives = 0
    negatives = negative_false_positives = negative_with_detections = 0
    for image in ground_truth["images"]:
        targets = actual[image["id"]]
        matched: set[int] = set()
        candidates = predicted[image["id"]]
        if not targets:
            negatives += 1
            negative_false_positives += len(candidates)
            negative_with_detections += bool(candidates)
        for detection in candidates:
            overlaps = [
                (_iou(detection["bbox"], target["bbox"]), index)
                for index, target in enumerate(targets)
                if index not in matched
                and detection["category_id"] == target["category_id"]
            ]
            overlap, index = max(overlaps, default=(0.0, -1))
            if overlap >= iou_threshold and index >= 0:
                matched.add(index)
                true_positives += 1
            else:
                false_positives += 1
        false_negatives += len(targets) - len(matched)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    stats = [float(value) if value >= 0 else None for value in evaluator.stats]
    return {
        "ap50": stats[1],
        "ap50_95": stats[0],
        "ar100": stats[8],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0,
        "image_count": len(ground_truth["images"]),
        "prediction_count": len(filtered),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "negative_image_count": negatives,
        "negative_false_positives": negative_false_positives,
        "negative_images_with_detections": negative_with_detections,
        "false_positives_per_negative_image": (
            negative_false_positives / negatives if negatives else None
        ),
        "negative_image_false_positive_fraction": (
            negative_with_detections / negatives if negatives else None
        ),
        "evaluator": {
            "name": "pycocotools",
            "version": version("pycocotools"),
            "score_floor": score_floor,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "max_dets": list(max_dets),
            "ap_ious": evaluator.params.iouThrs.tolist(),
            "area_ranges": evaluator.params.areaRng,
            "matching": "score-descending greedy, same category, unmatched maximum IoU",
        },
        "coco_summary": output.getvalue(),
    }


def _validate_metadata(metadata: dict[str, Any]) -> None:
    """
    Require explicit model, split, checkpoint, and postprocessing provenance.
    """
    required = {
        "model",
        "split",
        "run_dir",
        "checkpoint",
        "resolution",
        "smoke_run",
        "postprocessing",
    }
    if not isinstance(metadata, dict):
        raise ValueError("Prediction metadata must be an object")
    if missing := required - metadata.keys():
        raise ValueError(f"Missing prediction metadata: {sorted(missing)}")
    for name in ("model", "run_dir", "checkpoint"):
        if not isinstance(metadata[name], (str, Path)) or not str(metadata[name]):
            raise ValueError(f"Prediction metadata requires a nonempty {name}")
    if metadata["split"] not in {"val", "test"}:
        raise ValueError("Prediction split must be val or test")
    if type(metadata["smoke_run"]) is not bool:
        raise ValueError("Prediction smoke_run must be an explicit boolean")
    if type(metadata["resolution"]) is not int or metadata["resolution"] <= 0:
        raise ValueError("Prediction resolution must be a positive integer")
    if not isinstance(metadata["postprocessing"], dict):
        raise ValueError("Prediction postprocessing must be an object")
    floor = metadata["postprocessing"].get("score_floor")
    if not isinstance(floor, (int, float)) or not 0 <= floor <= 1:
        raise ValueError("Prediction postprocessing must record a valid score_floor")


def _valid_sha256(value: Any) -> bool:
    """
    Return whether a value is a lowercase SHA256 digest.
    """
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def write_prediction_artifact(
    path: Path | str,
    annotation_path: Path | str,
    predictions: Sequence[dict],
    *,
    metadata: dict[str, Any],
) -> Path:
    """
    Store predictions with the source annotation identity and complete image coverage.
    """
    _validate_metadata(metadata)
    ground_truth = read_json(annotation_path)
    normalized = _predictions(ground_truth, predictions, 0, max(100, len(predictions)))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(metadata["checkpoint"])
    checkpoint_sha256 = (
        file_sha256(checkpoint)
        if checkpoint.is_file()
        else metadata.get("checkpoint_sha256")
    )
    if not _valid_sha256(checkpoint_sha256):
        raise ValueError("Prediction metadata requires the checkpoint SHA256")
    document = {
        "schema_version": 1,
        "annotation_sha256": file_sha256(annotation_path),
        "image_ids": sorted(image["id"] for image in ground_truth["images"]),
        "metadata": {
            **metadata,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "predictions": normalized,
    }
    write_json(destination, document)
    return destination


def read_prediction_artifact(
    path: Path | str,
    annotation_path: Path | str | None = None,
    *,
    allow_smoke: bool = False,
) -> dict:
    """
    Read a detector export, rejecting smoke data and incompatible annotation identities.
    """
    artifact = read_json(path)
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported prediction artifact schema")
    _validate_metadata(artifact.get("metadata", {}))
    if not _valid_sha256(artifact["metadata"].get("checkpoint_sha256")):
        raise ValueError("Prediction artifact requires the checkpoint SHA256")
    if not isinstance(artifact.get("predictions"), list):
        raise ValueError("Prediction artifact requires a predictions list")
    image_ids = artifact.get("image_ids")
    if (
        not isinstance(image_ids, list)
        or not image_ids
        or any(type(image_id) is not int for image_id in image_ids)
        or image_ids != sorted(set(image_ids))
    ):
        raise ValueError("Prediction artifact requires sorted, unique image IDs")
    digest = artifact.get("annotation_sha256")
    if not _valid_sha256(digest):
        raise ValueError("Prediction artifact requires an annotation SHA256")
    if artifact["metadata"]["smoke_run"] and not allow_smoke:
        raise ValueError("Smoke predictions cannot enter a full comparison")
    if annotation_path is not None:
        ground_truth = read_json(annotation_path)
        ids = sorted(image["id"] for image in ground_truth["images"])
        if (
            artifact["annotation_sha256"] != file_sha256(annotation_path)
            or artifact["image_ids"] != ids
        ):
            raise ValueError("Prediction artifact belongs to a different dataset/split")
        _predictions(
            ground_truth,
            artifact["predictions"],
            0,
            max(100, len(artifact["predictions"])),
        )
    return artifact


def write_comparison(
    artifacts: Mapping[str, Path | str | None],
    annotation_path: Path | str,
    output_dir: Path | str,
    *,
    split: str,
) -> pd.DataFrame:
    """
    Re-evaluate all available exports with the same policy and save comparison reports.
    """
    rows = []
    reports = {}
    for model, path in artifacts.items():
        if path is None or not Path(path).is_file():
            rows.append({"model": model, "split": split, "status": "unavailable"})
            continue
        artifact = read_prediction_artifact(path, annotation_path)
        metadata = artifact["metadata"]
        if metadata["model"] != model or metadata["split"] != split:
            raise ValueError(f"Incorrect model/split in {path}")
        postprocess = metadata["postprocessing"]
        if postprocess.get("score_floor") != 0.001:
            raise ValueError(f"Export {path} must use score_floor=0.001")
        metrics = evaluate_predictions(annotation_path, artifact["predictions"])
        reports[model] = {"metadata": metadata, "metrics": metrics}
        rows.append(
            {
                "model": model,
                "split": split,
                "status": "available",
                "run_dir": metadata["run_dir"],
                "checkpoint": metadata["checkpoint"],
                "resolution": metadata["resolution"],
                "parameters": metadata.get("parameters"),
                "epochs_completed": metadata.get("epochs_completed"),
                **{
                    key: metrics[key]
                    for key in (
                        "ap50",
                        "ap50_95",
                        "ar100",
                        "precision",
                        "recall",
                        "f1",
                        "false_positives_per_negative_image",
                        "negative_image_false_positive_fraction",
                    )
                },
            }
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / f"comparison_{split}.csv", index=False)
    write_json(
        destination / f"comparison_{split}.json",
        {
            "annotation_sha256": file_sha256(annotation_path),
            "split": split,
            "models": reports,
            "unavailable": [
                row["model"] for row in rows if row["status"] == "unavailable"
            ],
        },
    )
    lines = [
        f"# Basketball comparison: {split}",
        "",
        "| Model | Status | AP50 | AP50:95 | Precision | Recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        values = [
            f"{row[key]:.4f}" if row.get(key) is not None else "—"
            for key in ("ap50", "ap50_95", "precision", "recall")
        ]
        lines.append(
            f"| {row['model']} | {row['status']} | " + " | ".join(values) + " |"
        )
    lines.extend(
        [
            "",
            "AP uses a score floor of 0.001 and at most 100 detections per image.",
            "Precision and recall use confidence 0.25 and IoU 0.50.",
            "Training budgets and preprocessing may differ between models.",
            "Native training metrics and historical timings are separate measurements.",
        ]
    )
    (destination / f"comparison_{split}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return frame


def benchmark_predict(
    predict_one: Callable[[Image.Image], Any],
    image_paths: Sequence[Path],
    *,
    device: str = "cuda",
    warmup: int = 5,
    sample_count: int = 32,
) -> dict:
    """
    Time batch-one PIL-to-predictions inference with file reads outside the timer.
    """
    import torch

    if not image_paths or warmup < 1 or sample_count < 1:
        raise ValueError("Benchmark requires images and positive sample/warmup counts")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Benchmark supports CPU or synchronized CUDA execution")
    images = []
    for path in image_paths[:sample_count]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    for index in range(warmup):
        predict_one(images[index % len(images)])
    milliseconds = []
    for image in images:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        predict_one(image)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        milliseconds.append((time.perf_counter() - start) * 1000)
    return {
        "protocol": "pil-to-cpu-coco-predictions-v1",
        "batch_size": 1,
        "precision": "float32",
        "warmup": warmup,
        "samples": len(images),
        "end_to_end_ms_mean": float(np.mean(milliseconds)),
        "end_to_end_ms_median": float(np.median(milliseconds)),
        "device": device,
        "host": platform.node(),
        "torch": version("torch"),
        "cuda_runtime": torch.version.cuda,
        "image_sha256": [file_sha256(path) for path in image_paths[:sample_count]],
        "gpu": torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else None,
        "includes": (
            "resize/normalize, transfer, forward, native postprocess, CPU boxes"
        ),
        "excludes": "file I/O, drawing, model loading",
    }
