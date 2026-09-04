# Renku Migration Plan

## Goal

Make every notebook in `notebooks/` runnable on the Renku Linux/CUDA platform.
Use the same Jupytext `.py` sources on Renku and macOS, and generate ignored
`.ipynb` files beside them.

## Agreed Approach

- Setup entry point: `notebooks-renku/setup_renku.sh`
- Environment: isolated project `.venv-renku` created from Renku's Python
- Jupyter: reuse Renku's server and register the isolated project kernel
- Dependencies: notebook runtime only; exclude lint and test tools
- GPU stack: a compatible CUDA-enabled PyTorch and Torchvision pair
- Datasets: regenerate in Renku using `ROBOFLOW_API_KEY`
- Notebooks: use `notebooks/*.py` directly on both Renku and macOS
- Notebook storage: commit `.py`; generate and ignore `.ipynb`

## Todo

### 1. Build the Renku setup script

- [x] Resolve the repository root without depending on the current directory.
- [x] Require Linux, Python 3.11 or newer, and an available NVIDIA GPU.
- [x] Keep an existing CUDA-enabled PyTorch installation when it passes the
  compatibility checks; otherwise install PyTorch and Torchvision together
  from the official CUDA wheel index.
- [x] Do not install `torchaudio`, because the migrated notebooks do not use it.
- [x] Keep Renku's existing Jupyter server and install `ipykernel` in the
  isolated project environment.
- [x] Install only the missing notebook helpers, including Jupytext and
  `ipywidgets`; do not install JupyterLab, Notebook, Server, or nbconvert.
- [x] Store the ordinary runtime dependencies in
  `notebooks-renku/requirements.txt` and install them with pip. This includes
  scientific libraries, Datumaro, Roboflow, ONNX, headless Ultralytics, and
  headless OpenCV.
- [x] Apply Renku compatibility ranges from
  `notebooks-renku/constraints-renku.txt`.
- [x] Install this project in editable mode with `--no-deps`.
- [x] Install YOLOX from commit
  `6ddff4824372906469a7fae2dc3206c7aa4bbaee` with
  `--no-build-isolation --no-deps` and its compiler flag updated to C++17 for
  current PyTorch headers; update its package metadata for modern ONNX
  Simplifier and headless OpenCV.
- [x] Verify package versions, imports, CUDA availability, the GPU name, and a
  small CUDA tensor operation.
- [x] Sync every `notebooks/*.py` file to `.ipynb` with Jupytext.
- [x] Make the script safe to run more than once.

### 2. Secure Roboflow access and regenerate datasets

- [ ] Remove the hard-coded Roboflow API key from the source code.
- [ ] Read `ROBOFLOW_API_KEY` from the environment or a local ignored `.env`.
- [ ] Raise a clear error when the key is missing.
- [ ] Rotate the exposed existing key before using Renku.
- [ ] Document enough persistent storage for the regenerated datasets and
  training outputs.

### 3. Prepare the existing notebooks

- [x] Use the existing `notebooks/*.py` sources instead of Renku-specific
  copies.
- [x] Keep project-relative dataset, model, and output paths.
- [x] Let the notebooks select the available accelerator so they also run on
  macOS; let the Renku setup fail early when no NVIDIA GPU is allocated.
- [x] Preserve existing experiment defaults and YOLOX smoke/resume controls.

### 4. Document the workflow

- [ ] Update `notebooks-renku/README.md` with the setup command:

  ```bash
  bash notebooks-renku/setup_renku.sh
  ```

- [ ] Explain how to configure the `ROBOFLOW_API_KEY` Renku secret.
- [ ] List notebook execution order and expected dataset/output directories.
- [ ] Explain that `.ipynb` files are generated beside `notebooks/*.py` and
  are not committed.
- [ ] Add troubleshooting for missing GPU access and stale notebook kernels.

### 5. Verify on Renku

- [x] Validate the setup script with `bash -n`.
- [ ] Run the setup twice to test idempotence.
- [ ] Confirm the `Python (object_ctrl Renku)` kernel sees all imports, ONNX,
  and CUDA checks.
- [ ] Confirm Jupytext generates all six `.ipynb` files.
- [ ] Run both dataset builders with the Renku secret configured.
- [ ] Run both YOLOX notebooks with `YOLOX_TINY_SMOKE=1`.
- [ ] Confirm both Ultralytics notebooks select CUDA and create run outputs.
- [ ] Confirm missing-secret and missing-GPU failures are actionable.

## Completion Criteria

- All six generated notebooks open with Renku's Python kernel.
- Dataset notebooks recreate the required ignored datasets.
- Training notebooks use the NVIDIA GPU and write under `outputs/`.
- No credentials, datasets, generated notebooks, weights, or run artifacts are
  added to Git.

## Assumptions

- Renku allows outbound access to PyPI, PyTorch, GitHub, and Roboflow.
- The active Renku `.venv` can create and register the isolated project kernel;
  Conda and privileged system package installation are not required.
- The session has enough persistent storage for datasets and training results.
