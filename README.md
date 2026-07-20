# Object Control in Computer Vision

## Development and Runtime Environment

### Conda (Anaconda, Miniconda or Miniforge)

This project is currently intended for a local Conda-based workflow, especially
for Apple Silicon MPS environment checks and local Jupytext notebook
development. The Conda environment is defined in `environment.yml`.

Command to create the Conda environment:

```bash
conda env create -f environment.yml
```

When `environment.yml` is edited, update the Conda environment using command:

```bash
conda env update -f environment.yml --prune
```

### Python package configuration

Python package metadata, editable-install configuration, optional dependency groups,
and developer tool settings are defined in `pyproject.toml`.

Install this project as an editable local package:

```bash
python -m pip install --no-deps -e .
```

The `--no-deps` flag is intentional for the Conda workflow. `environment.yml`
already installs the main runtime packages, including PyTorch, torchvision,
torchaudio, Jupyter, Ultralytics, and common scientific Python tools. Installing
the local package in editable mode makes imports such as this work from notebooks
and scripts:

```python
from object_ctrl import PROJECT_ROOT
```

Optional dependency groups are also declared in `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[notebook]"
python -m pip install -e ".[data,vision]"
python -m pip install -e ".[yolo]"
```

After editing package metadata or dependencies in `pyproject.toml`, refresh the
editable install:

```bash
python -m pip install --no-deps -e .
```

For the recommended Conda workflow, prefer `environment.yml` for heavy packages
such as PyTorch and OpenCV. The extras are most useful for pip-only environments
or quick tool installation.

### Shell environment setup

There's also a shell environment setup script, `env_setup.sh`, available for the
convenience of application runtime. Activate it and load the project runtime environment
variables:

```bash
source env_setup.sh
```

Google Colab is not a convenient target at this stage. The repo keeps notebooks
as Jupytext `.py` files, ignores generated `.ipynb` files, and relies on the
project Conda environment. Colab support would need a separate setup path.

### Jupytext over Jupyter notebooks

This repo stores notebooks as Jupytext `.py` files using the percent format This keeps
notebooks easier to review in Git while still allowing them to be opened and run in
Jupyter.

After creating and activating the Conda environment, register the Jupyter kernel
once:

```bash
python -m ipykernel install --user --name objctrl --display-name "Python (objctrl)"
```

Sync the Jupytext notebooks to Jupyter `.ipynb` notebooks with Make:

```bash
make sync-notebooks
```

The Makefile runs Jupytext in the `objctrl` Conda environment and uses
`jupytext.toml`, which pairs each notebook as `ipynb,py:percent`. After
syncing, each notebook has both a Git-friendly `.py` file and a Jupyter
`.ipynb` file. The generated `.ipynb` files are ignored by Git, so commit
changes to the `.py` notebook files.

Start JupyterLab from the activated environment:

```bash
jupyter lab
```

Open the `.ipynb` files from the `notebooks/` directory and select the
`Python (objctrl)` kernel if Jupyter does not select it automatically. When
you save a paired notebook in Jupyter, Jupytext updates the `.py` file too.

If you edit the `.py` notebook directly, run the sync command again before
opening it in Jupyter:

```bash
make sync-notebooks
```

To execute all synced notebooks from the command line:

```bash
make run-notebooks
```
