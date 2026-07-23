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
#     display_name: objctrl
#     language: python
#     name: python3
# ---

# %%
"""
Performance evaluation of pretrained YOLO11 on customized basketball dataset.
"""
# %load_ext autoreload
# %autoreload 2
# %aimport -ultralytics

from pathlib import Path

from object_ctrl import PROJECT_ROOT, configure_stdio_relative_path

# Display project paths relative to project root directory for consistency.
configure_stdio_relative_path()

# %%
from object_ctrl import (
    aligned_print,
    ensure_dir,
    get_device,
)
from object_ctrl.platforms import ultralytics as ultralitics_platform
from object_ctrl.utils.image import display as display_img

# Must be called before importing ultralytics.
ultralitics_platform.configure_privacy()
from ultralytics import YOLO  # noqa: E402

# %%
PRETRAINED_DIR = ensure_dir(PROJECT_ROOT / "models" / "pretrained" / "ultralytics")
DATASET_DIR = ensure_dir(PROJECT_ROOT / "datasets/")
DATA_YAML = DATASET_DIR / "composed" / "yolo_basketball_105_22_23" / "data.yaml"

# %% [markdown]
# ## Fine-tune Ultrlytics YOLO11 on a custom dataset.

# %%
project_space = PROJECT_ROOT / "outputs" / "runs" / "basketball"
project_name_base = "yolo11n_basketball"
basketball_model = YOLO(PRETRAINED_DIR / "yolo11n.pt")
results = basketball_model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    device=get_device(),
    project=str(project_space),
    name=project_name_base,
)
run_dir = Path(results.save_dir)
print(f"Training run directory: {run_dir}")


# %% [markdown]
# Run completed successfully. Downstream cells use the actual Ultralytics save
# directory reported by `results.save_dir`, so plots, metrics, and checkpoint
# evaluation stay aligned even when Ultralytics increments the run name.

# %% [markdown]
# ## Plot the training history

# %%
import matplotlib.pyplot as plt
import pandas as pd


def latest_training_run_dir(project_space, project_name_base):
    """
    Return the newest matching Ultralytics training run directory.
    """
    candidates = [
        path
        for path in project_space.glob(f"{project_name_base}*")
        if path.is_dir() and (path / "results.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No training runs with results.csv found for {project_name_base}."
        )
    return max(candidates, key=lambda path: (path / "results.csv").stat().st_mtime)


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
if "run_dir" not in globals():
    run_dir = latest_training_run_dir(project_space, project_name_base)

print(f"Using run directory: {run_dir}")
history = read_training_history(run_dir / "results.csv")

fig, axes = plot_train_val_losses(history)
display_img(fig, close=True)

# %% [markdown]
# The model is learning: all three training losses (box_loss, cls_loss,
# dfl_loss) decrease steadily through all epochs. That means the network is
# fitting the training data and improving localization/classification on the
# train set.  
#
# Validation behavior is noisier than training. It's as expected, because the
# validation set is tiny currently. Validation losses remain clearly higher
# than training losses. This suggests a generalization gap: the model fits the
# training set better than the validation set.

# %%
fig, axes = plot_detection_metrics(history)
display_img(fig, close=True)

# %% [markdown]
# The `precision` ends high, while the `recall` remains modest. So the current
# checkpoint of the model usually predicts a basketball object correctly, but
# missing too many objects.
#
# The best checkpoint is likely near the late plateau, around epoch ~90 rather
# than the final epoch being dramatically better. From the CSV we inspected
# earlier, best mAP50-95 was around epoch 92.

# %% [markdown]
# ### Training summary
#
# The YOLO11n fine-tune is working, but performance is recall-limited. The model
# has learned useful detections, but it is missing too many objects. Next step
# focus on improving recall: inspect missed validation predictions, check label
# quality, add more varied examples, and tune inference confidence/NMS thresholds.


# %% [markdown]
# ## Validation and Test Metrics

# %%
BEST_MODEL_PATH = run_dir / "weights" / "best.pt"
eval_model = YOLO(BEST_MODEL_PATH)

validation_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=get_device(),
    split="val",
    plots=True,
    project=str(project_space),
    name=f"{run_dir.name}_val",
)
test_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=get_device(),
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
