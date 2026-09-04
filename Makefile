## Jupytext/Jupyter notebook workflow for this project.

SHELL := /bin/bash

NOTEBOOK_DIR := notebooks
NOTEBOOK_STAMP_DIR := .notebook-stamps
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

PY_NOTEBOOKS := $(wildcard $(NOTEBOOK_DIR)/*.py)
PY_STEMS := $(basename $(notdir $(PY_NOTEBOOKS)))

GENERATED_NOTEBOOKS := $(addprefix $(NOTEBOOK_DIR)/,$(addsuffix .ipynb,$(PY_STEMS)))
NOTEBOOK_RUN_STAMPS := \
	$(addprefix $(NOTEBOOK_STAMP_DIR)/,$(addsuffix .executed,$(PY_STEMS)))

NOTEBOOK_EXECUTE_FLAGS ?= --ExecutePreprocessor.timeout=-1
NOTEBOOK_STREAM_FLAGS ?= --CoalesceStreamsPreprocessor.enabled=True

.PHONY: sync-notebooks run-notebooks clean distclean

sync-notebooks: $(GENERATED_NOTEBOOKS)

run-notebooks: $(NOTEBOOK_RUN_STAMPS)

$(NOTEBOOK_STAMP_DIR):
	mkdir -p $@

$(NOTEBOOK_DIR)/%.ipynb: $(NOTEBOOK_DIR)/%.py jupytext.toml
	source ./scripts/env_setup.sh && jupytext --sync $<

$(NOTEBOOK_STAMP_DIR)/%.executed: $(NOTEBOOK_DIR)/%.ipynb \
  $(THIS_MAKEFILE) | $(NOTEBOOK_STAMP_DIR)
	source ./scripts/env_setup.sh && jupyter nbconvert $< \
		--to notebook \
		--execute \
		--inplace \
		--ExecutePreprocessor.kernel_name="$${NOTEBOOK_KERNEL}" \
		$(NOTEBOOK_EXECUTE_FLAGS) \
		$(NOTEBOOK_STREAM_FLAGS)
	touch $@

clean:
	rm -rf $(NOTEBOOK_STAMP_DIR)
	rm -rf $(NOTEBOOK_DIR)/.ipynb_checkpoints

distclean: clean
	rm -f $(GENERATED_NOTEBOOKS)
