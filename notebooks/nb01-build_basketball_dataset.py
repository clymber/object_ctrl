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
#     display_name: yolo-dev
#     language: python
#     name: python3
# ---

# %%
"""
Builds separate datasets for object detection tasks in COCO and YOLO formats.
"""

# Reloads all modules every time before executing code, except explicitly excluded using
# # %aimport -<package>, like %aimport -numpy.
# %load_ext autoreload
# %autoreload 2
# %aimport -csv
# %aimport -textwrap
# %aimport -functools
# %aimport -IPython
# %aimport -ultralytics
# %aimport -pandas
# %aimport -datumaro
# %aimport -random

from object_ctrl import configure_stdio_relative_path

# Display project paths relatively for consistent output across environments.
configure_stdio_relative_path()

# %%
import random

import datumaro as dm
import pandas as pd
from IPython.display import display

from object_ctrl import PROJECT_ROOT
from object_ctrl.dataset import (
    prefer_hardlinked_datumaro_media,
    summarize_coco_dataset,
    summarize_datumaro_label_counts,
)
from object_ctrl.platforms import roboflow as roboflow_platform
from object_ctrl.platforms.roboflow import RoboflowFormat as DatasetFormat
from object_ctrl.utils.filepath import dir_tree as dir_tree
from object_ctrl.utils.filepath import ensure_dir

# %% [markdown]
# ## 1. Object detection targets
#
# The object detection targets are defined in the `Object_Control_List.csv` file. This
# file contains a list of objects to be detected. The code below reads and displays the
# targets from the CSV file.
# %%
OBJ_CTL_LIST_CSV = PROJECT_ROOT / "documents" / "Object_Control_List.csv"
DATASET_ROOT = PROJECT_ROOT / "datasets"

oc_df = pd.read_csv(OBJ_CTL_LIST_CSV)
display(oc_df.fillna(""))


# %% [markdown]
# ## 2. Build basketball dataset


# %% [markdown]
# ### 2.1 Download source dataset
# %%
# Source dataset downloaded from Roboflow
url = "https://universe.roboflow.com/cricket-qnb5l/basketball-xil7x/dataset/1"
src_ds_basketball = roboflow_platform.download_by_url(url, DatasetFormat.COCO)

# %% [markdown]
# ### 2.2 Summary of source dataset

# %%
display(summarize_coco_dataset(src_ds_basketball.location))
print(dir_tree(src_ds_basketball.location, max_children=3))


# %% [markdown]
# ### 2.3 Small batch dataset
#
# Compose a compact basketball-only dataset from the downloaded source data.
# Keep 100 images with basketball annotations and add 50 background images with
# no remaining target annotations, using an approximate 70/15/15 split.
#
# | Final split | Images containing `ball` | Images with no `ball` |   Total |
# | ----------- | -----------------------: | --------------------: | ------: |
# | Train       |                       70 |                    35 |     105 |
# | Validation  |                       15 |                     7 |      22 |
# | Test        |                       15 |                     8 |      23 |
# | **Total**   |                  **100** |                **50** | **150** |
#
#
# #### 2.3.1 Remap labels
# Remap labels to keep only `ball` and remove all other labels. The `ball` label is
# remapped to `basketball` to match the target detection category.
# %%
dataset = dm.Dataset.import_from(src_ds_basketball.location, "roboflow_coco")
display(summarize_datumaro_label_counts(dataset))

dataset = dataset.transform(
    "remap_labels",
    mapping={
        "ball": "basketball",
        "human": "",
        "rim": "",
        "ball-person-rim": "",
    },
    default = "delete"
)

# %% [markdown]
# #### 2.3.2 Compose a new dataset
# Randomly sample a compact basketball-only dataset from the downloaded source data.
# %%
split_plan = {
    "train": {"positive": 70, "negative": 35,},
    "valid": {"positive": 15, "negative": 7,},
    "test":  {"positive": 15, "negative": 8,},
}

rng = random.Random(42)
selected_items = []

for split_name, quota in split_plan.items():
    subset = dataset.get_subset(split_name)
    split_name = "val" if split_name == "valid" else split_name

    positives = [
        item.wrap(subset=split_name) for item in subset if len(item.annotations) > 0
    ]
    negatives = [
        item.wrap(subset=split_name) for item in subset if len(item.annotations) == 0
    ]

    selected_items.extend(rng.sample(positives, quota["positive"]))
    selected_items.extend(rng.sample(negatives, quota["negative"]))

composed_dataset = dm.Dataset.from_iterable(
    selected_items,
    categories=dataset.categories(),
)

print(composed_dataset)


# %% [markdown]
# #### 2.3.3 Export composed dataset
# Export the sampled dataset into both COCO and YOLO formats, using hardlinks for media
# files when possible to save disk space.
# %%
coco_dataset_dir = ensure_dir(DATASET_ROOT / "composed" / "coco_basketball_105_22_23")
yolo_dataset_dir = ensure_dir(DATASET_ROOT / "composed" / "yolo_basketball_105_22_23")
with prefer_hardlinked_datumaro_media():
    composed_dataset.export(
        str(coco_dataset_dir),
        format="coco_instances",
        save_media=True,
        reindex=True,
    )
    composed_dataset.export(
        str(yolo_dataset_dir),
        format="yolo_ultralytics",
        save_media=True,
    )

# %%
print(dir_tree(coco_dataset_dir, max_children=3))

# %%
print(dir_tree(yolo_dataset_dir, max_children=3))
