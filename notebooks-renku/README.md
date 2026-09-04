# Object Control Notebooks on Renku

The DCU Renku platform is available at
[soc-gpu.computing.dcu.ie](https://soc-gpu.computing.dcu.ie/).

## Notebook location

Renku and macOS use the same Jupytext notebook sources in `notebooks/`.
On the local MacBook, this directory is:

```text
~/studio/object_ctrl/notebooks
```

In a typical Renku session, it is:

```text
~/work/object_ctrl/notebooks
```

Do not create or maintain Renku-specific notebook copies under
`notebooks-renku/`. The setup script generates paired, ignored `.ipynb` files
beside the existing `notebooks/*.py` sources.

## Set up a Renku session

1. Start a Renku Linux session with an NVIDIA GPU allocation.

2. Open a terminal and confirm that the Renku host environment and GPU are
   available:

   ```bash
   which python
   nvidia-smi
   ```

3. From the repository root, run the setup script:

   ```bash
   cd ~/work/object_ctrl
   bash notebooks-renku/setup_renku.sh 2>&1 | tee setup_renku.log
   ```

The script:

- creates or reuses the isolated `.venv-renku` project environment;
- installs the CUDA-enabled PyTorch stack and notebook dependencies;
- installs `object_ctrl` and the compatible YOLOX revision;
- verifies CUDA, OpenCV, ONNX, and the installed dependencies;
- registers the `Python (object_ctrl Renku)` Jupyter kernel; and
- synchronizes every `notebooks/*.py` source to a paired `.ipynb` file.

The setup is safe to run again after pulling dependency or notebook changes.

## Run a notebook

1. In Renku's file browser, open a generated notebook under `notebooks/`, for
   example `notebooks/nb02.02-ultra_yolo11n_large_basketball.ipynb`.
2. Select the `Python (object_ctrl Renku)` kernel.
3. Run the notebook cells normally.

Confirm the selected kernel from a notebook cell when needed:

```python
import sys

import torch

print(sys.executable)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
```

The Python path should end in `.venv-renku/bin/python`, and CUDA availability
should be `True`.

## Notes

- Keep edits in the Jupytext `.py` sources. Generated `.ipynb` files are
  ignored by Git.
- Datasets remain under `datasets/`, models under `models/`, and training
  results under `outputs/`.
- The large Ultralytics notebook defaults to zero dataloader subprocesses to
  avoid exhausting Renku's limited `/dev/shm` allocation.
- If imports come from Renku's host `.venv`, reselect the
  `Python (object_ctrl Renku)` kernel and restart it.
- If CUDA is unavailable, stop the session and start one with a GPU allocation
  before rerunning the setup script.
