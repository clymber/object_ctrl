#!/usr/bin/env bash
# Export the latest complete nb02.02, nb03.02, and nb03.03 training runs.

set -euo pipefail

readonly SCRIPT_DIR="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
readonly PROJECT_ROOT="$(realpath -- "$SCRIPT_DIR/..")"
readonly PYTHON_BIN="${OBJCTRL_RENKU_PYTHON:-${PROJECT_ROOT}/.venv-renku/bin/python}"
readonly RUNS_DIR="${BASKETBALL_RUNS_DIR:-${PROJECT_ROOT}/outputs/runs/basketball}"
readonly DATASET_DIR="${BASKETBALL_DATASET_DIR:-${PROJECT_ROOT}/datasets/composed/coco_basketball_11501_1156_1395}"
readonly OUTPUT_DIR="${BASKETBALL_EXPORT_DIR:-${PROJECT_ROOT}/outputs/comparisons/basketball_large_dataset/baselines}"
readonly DEVICE="${BASKETBALL_EXPORT_DEVICE:-cuda:0}"
readonly RESOLUTION="${BASKETBALL_EXPORT_RESOLUTION:-640}"
readonly MODELS=("ultralytics" "yolox" "yolox-nano")

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python environment not found or not executable: %s\n' "$PYTHON_BIN" >&2
    printf 'Run bash scripts/setup_renku.sh first.\n' >&2
    exit 1
fi

cd "$PROJECT_ROOT"
for model in "${MODELS[@]}"; do
    printf 'Exporting %s predictions.\n' "$model"
    "$PYTHON_BIN" "$SCRIPT_DIR/export_basketball_predictions.py" \
        --model "$model" \
        --runs-dir "$RUNS_DIR" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --resolution "$RESOLUTION" \
        "$@"
done
