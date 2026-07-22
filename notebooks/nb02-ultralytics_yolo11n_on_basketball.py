# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
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
project_name = "yolo11n_basketball"
basketball_model = YOLO(PRETRAINED_DIR / "yolo11n.pt")
results = basketball_model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    device=get_device(),
    project=str(project_space),
    name=project_name
)


# %% [markdown]
# Run completed successfully. YOLO11n trained for 100 epochs on MPS in about 60
# minutes. Dataset is small: 105 train images with 35 backgrounds, and 22
# validation images with 7 backgrounds / 18 labeled instances.

# %% [markdown]
# ## Plot the training history

# %%
from pathlib import Path

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
history = read_training_history(project_space / project_name / "results.csv")

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
# ## Inference on test images

# %%
BEST_MODEL_PATH = project_space / project_name / "weights" / "best.pt"
test_model = YOLO(BEST_MODEL_PATH)

test_metrics = test_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=get_device(),
    plots=True,
    project=str(project_space),
    name=f"{project_name}_test",
)

# %% [markdown]
# ### Test metrics summary
# %%
aligned_print({
    "Precision": test_metrics.box.mp,
    "Recall": test_metrics.box.mr,
    "mAP50": test_metrics.box.map50,
    "mAP50-95": test_metrics.box.map,
})
print("\nSpeed ms/image:")
aligned_print(test_metrics.speed)

# %% [markdown]
# ### Sample test prediction VS ground truth

# %%
test_output_dir = Path(test_metrics.save_dir)

print("Ground truth sample images:")
display_img(test_output_dir / "val_batch0_labels.jpg", width=640)
print("Predicted sample images:")
display_img(test_output_dir / "val_batch0_pred.jpg", width=640)
