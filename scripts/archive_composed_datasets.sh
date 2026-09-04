#!/usr/bin/env bash

set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

COPYFILE_DISABLE=1 tar \
  --exclude='*.cache' \
  --exclude='._*' \
  --exclude='.DS_Store' \
  -czf "$project_dir/datasets/composed_datasets.tgz" \
  -C "$project_dir" \
  datasets/composed
