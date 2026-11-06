#!/usr/bin/env bash
# CPU test suite for the four GRPO baselines.
#
# Runs on the read-only conda env (Python 3.9.21, no pytest, no torch/trl/datasets) --
# which is exactly why the suite is plain ``unittest``.  Every test here is CPU-only:
# nothing loads a model, generates a token, or touches a GPU.
#
# Usage:
#   bash scripts/test_cpu.sh                 # whole suite
#   bash scripts/test_cpu.sh -p test_j4r     # one module (unittest pattern)
#   bash scripts/test_cpu.sh -v              # verbose
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH=${CONDA_SH:-/data1/yanghao/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-grpo-trl-py311}
LOG_DIR=${LOG_DIR:-/tmp/yh_grpo/logs}

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_cpu.log"

cd "${PROJECT_ROOT}" || exit 1

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_NAME}" || { echo "cannot activate ${ENV_NAME}"; exit 1; }

{
  echo "=== $(date -Iseconds) ==="
  echo "pwd:  $(pwd)"
  echo "python: $(python -V 2>&1)"
  echo "executable: $(command -v python)"
  echo
  python -m unittest discover -s tests -t . "$@"
  status=$?
  echo "EXIT=${status}"
  exit ${status}
} 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
