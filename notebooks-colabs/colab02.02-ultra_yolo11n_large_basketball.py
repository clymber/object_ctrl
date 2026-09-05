# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""
Performance evaluation of pretrained YOLO11 on customized basketball dataset.
"""


# %%
import sys
from pathlib import Path

from google.colab import drive  # pyright: ignore[reportMissingImports]

drive_dir = Path("/content/drive")
if not (drive_dir / "MyDrive").is_dir():
    drive.mount(str(drive_dir))

project_dir = drive_dir / "MyDrive" / "object_ctrl"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# %%
from colabs.colab_setup import setup_project  # noqa: E402

resolved_project_dir = setup_project(project_dir)

from object_ctrl import PROJECT_ROOT, configure_stdio_relative_path  # noqa: E402

# Display project paths relatively for consistent output across environments.
configure_stdio_relative_path()

# %%
from pathlib import Path
from typing import cast

# %%
from object_ctrl import (
    Device,
    aligned_print,
    ensure_dir,
)
from object_ctrl.platforms import ultralytics as ultralitics_platform
from object_ctrl.utils.image import display as display_img

# Must be called before importing ultralytics.
ultralitics_platform.configure_privacy()
from ultralytics import YOLO  # noqa: E402

# %%
PRETRAINED_DIR = ensure_dir(PROJECT_ROOT / "models" / "pretrained" / "ultralytics")
SRC_DATASET_DIR = ensure_dir(PROJECT_ROOT / "datasets/")
DATASET_DIR = ensure_dir(Path("/content") / "datasets/")
DATA_YAML = DATASET_DIR / "composed" / "yolo_basketball_11501_1156_1395" / "data.yaml"

if not DATA_YAML.exists():
    SRC_DATASET = SRC_DATASET_DIR / "composed" / "yolo_basketball_11501_1156_1395"
    if not SRC_DATASET.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_DATASET}.")

    # Copy the dataset to the Colab environment.
    import shutil

    shutil.copytree(SRC_DATASET, DATA_YAML.parent, dirs_exist_ok=True)

# %% [markdown]
# ## Fine-tune Ultrlytics YOLO11 on a custom dataset.

# %%
project_space = PROJECT_ROOT / "outputs" / "runs" / "basketball"
project_name_base = "yolo11n_basketball_large_dataset"
# resume_checkpoint: Path | None = None
resume_checkpoint: Path | None = (
    project_space
    / "yolo11n_basketball_large_dataset-4"
    / "weights"
    / "last.pt"
)

BATCH_SIZE = 32
N_WORKERS = 8
CACHE_DATA = False

if resume_checkpoint is None:
    basketball_model = YOLO(PRETRAINED_DIR / "yolo11n.pt")
    results = basketball_model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        device=Device.auto_choose(),
        project=str(project_space),
        name=project_name_base,
        patience=25,
        batch=BATCH_SIZE,
        workers=N_WORKERS,
        cache=CACHE_DATA,
    )
else:
    if not resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    basketball_model = YOLO(resume_checkpoint)
    results = basketball_model.train(
        resume=True,
        device=Device.auto_choose(),
        batch=BATCH_SIZE,
        workers=N_WORKERS,
        cache=CACHE_DATA,
    )
results = cast(ultralitics_platform.TrainingResult, results)
run_dir = Path(results.save_dir)
print(f"Training run directory: {run_dir}")


# %% [markdown]
# Downstream cells use the actual Ultralytics save directory reported by
# `results.save_dir`. This keeps plots, metrics, and checkpoint evaluation aligned if
# Ultralytics increments the run name.

# %% [markdown]
# To continue an interrupted run, set `resume_checkpoint` to that run's
# `weights/last.pt` path and rerun the training cell. Ultralytics restores the run
# configuration, optimizer state, and completed epoch from the checkpoint. Leave it
# as `None` to start a new run from the pretrained YOLO11n weights.

# %% [markdown]
# ## Plot the training history

# %%
import matplotlib.pyplot as plt
import pandas as pd


def read_training_history(results_csv):
    """
    Read an Ultralytics YOLO training results.csv file.
    """
    history = pd.read_csv(results_csv)
    history.columns = history.columns.str.strip()
    return history


def plot_train_val_losses(history):
    """
    Plot train and validation YOLO losses.
    """
    loss_names = ["box_loss", "cls_loss", "dfl_loss"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

    for axis, loss_name in zip(axes, loss_names, strict=True):
        axis.plot(history["epoch"], history[f"train/{loss_name}"], label="train")
        axis.plot(history["epoch"], history[f"val/{loss_name}"], label="validation")
        axis.set_title(loss_name)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.tight_layout()
    return fig, axes


def plot_detection_metrics(history):
    """
    Plot YOLO validation detection metrics.
    """
    metric_columns = [
        ("precision", "metrics/precision(B)"),
        ("recall", "metrics/recall(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("mAP50-95", "metrics/mAP50-95(B)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    for axis, (metric_name, column) in zip(axes.ravel(), metric_columns, strict=True):
        axis.plot(history["epoch"], history[column])
        axis.set_title(metric_name)
        axis.set_xlabel("epoch")
        axis.set_ylabel("metric")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)

    fig.tight_layout()
    return fig, axes

# %%
print(f"Using run directory: {run_dir}")
history = read_training_history(run_dir / "results.csv")

fig, axes = plot_train_val_losses(history)
display_img(fig, close=True)

# %% [markdown]
# Compare the training and validation curves after the run. Falling losses in both
# sets indicate continued learning. A widening gap, with training loss improving while
# validation loss stalls or rises, indicates overfitting.

# %%
fig, axes = plot_detection_metrics(history)
display_img(fig, close=True)

# %%
selection_metric = "metrics/mAP50-95(B)"
best_history_row = cast(
    pd.Series,
    history.loc[history[selection_metric].idxmax()],
)

print(f"Best validation epoch by {selection_metric}:")
aligned_print({
    "Epoch": int(best_history_row["epoch"]),
    "Precision": best_history_row["metrics/precision(B)"],
    "Recall": best_history_row["metrics/recall(B)"],
    "mAP50": best_history_row["metrics/mAP50(B)"],
    "mAP50-95": best_history_row[selection_metric],
})

# %% [markdown]
# ### Training summary
#
# Use the reported best validation epoch and the plots above to characterize the run.
# Compare precision and recall to identify whether false positives or missed
# basketballs are the larger issue, then confirm the behavior on the held-out test set
# below.


# %% [markdown]
# ## Validation and Test Metrics

# %%
BEST_MODEL_PATH = run_dir / "weights" / "best.pt"
eval_model = YOLO(BEST_MODEL_PATH)

validation_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=Device.auto_choose(),
    split="val",
    plots=True,
    project=str(project_space),
    name=f"{run_dir.name}_val",
)
test_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=Device.auto_choose(),
    split="test",
    plots=True,
    project=str(project_space),
    name=f"{run_dir.name}_test",
)

# %% [markdown]
# ### Validation metrics summary
# %%
print("Validation metrics:")
aligned_print({
    "Precision": validation_metrics.box.mp,
    "Recall": validation_metrics.box.mr,
    "mAP50": validation_metrics.box.map50,
    "mAP50-95": validation_metrics.box.map,
})
print("\nSpeed ms/image:")
aligned_print(validation_metrics.speed)

print("\nTest metrics:")
aligned_print({
    "Precision": test_metrics.box.mp,
    "Recall": test_metrics.box.mr,
    "mAP50": test_metrics.box.map50,
    "mAP50-95": test_metrics.box.map,
})
print("\nSpeed ms/image:")
aligned_print(test_metrics.speed)

# %% [markdown]
# ### Sample prediction VS ground truth

# %%
validation_output_dir = Path(validation_metrics.save_dir)
test_output_dir = Path(test_metrics.save_dir)

print("Validation ground truth sample images:")
display_img(validation_output_dir / "val_batch0_labels.jpg", width=640)
print("Validation predicted sample images:")
display_img(validation_output_dir / "val_batch0_pred.jpg", width=640)

print("Test ground truth sample images:")
display_img(test_output_dir / "val_batch0_labels.jpg", width=640)
print("Test predicted sample images:")
display_img(test_output_dir / "val_batch0_pred.jpg", width=640)

# %%
import importlib.metadata as metadata
import sys

print("Python:", sys.version)

for package in (
    "ultralytics",
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")

try:
    from ultralytics import YOLO
except Exception as error:
    print(f"\nFINAL ERROR: {type(error).__name__}: {error!r}")

# %%
# %pip check
