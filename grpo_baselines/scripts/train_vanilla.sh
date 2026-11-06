#!/usr/bin/env bash
# Vanilla GRPO (one state per preference pair, terminal correctness reward).
#
# GPU Smoke is optional. Use all GPUs exposed by CUDA_VISIBLE_DEVICES.
#
# Requires: a Python>=3.11 env with torch + transformers + trl + datasets
#           (see requirements/trl-py311.txt), and one free GPU.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH=${CONDA_SH:-/data1/yanghao/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-grpo-trl-py311}
CONFIG="${PROJECT_ROOT}/configs/vanilla.yaml"
LOG_DIR=${LOG_DIR:-/tmp/yh_grpo/logs}
OUTPUT_DIR="$(sed -n 's/^output_dir:[[:space:]]*//p' "${CONFIG}" | head -1)"
if [ -z "${OUTPUT_DIR}" ]; then
    echo "error: ${CONFIG} has no output_dir" >&2
    exit 2
fi
case "${OUTPUT_DIR}" in
    /*) ;;
    *) OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}" ;;
esac

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_vanilla.log"

# --- argument validation ----------------------------------------------------
# Optional: --resume-from-checkpoint <dir>.  The trainer already refuses a path that is
# not a real checkpoint (see trainers/grpo_base.py:validate_args); this repeats the check
# here so the failure happens before the conda env is activated and before any model load.
RESUME_ARGS=()
if [ "${1:-}" = "--skip-smoke" ]; then
    shift
fi
if [ "$#" -gt 0 ]; then
    if [ "$1" != "--resume-from-checkpoint" ] || [ "$#" -lt 2 ]; then
        echo "usage: $(basename "$0") [--resume-from-checkpoint <checkpoint-dir>]" >&2
        exit 2
    fi
    RESUME_DIR="$2"
    if [ ! -d "${RESUME_DIR}" ]; then
        echo "error: --resume-from-checkpoint ${RESUME_DIR} is not a directory" >&2
        exit 2
    fi
    if [ ! -f "${RESUME_DIR}/optimizer.pt" ]; then
    echo "error: ${RESUME_DIR} has no optimizer.pt; model-only checkpoints cannot resume optimization" >&2
    exit 2
fi
if [ ! -f "${RESUME_DIR}/trainer_state.json" ]; then
        echo "error: ${RESUME_DIR} has no trainer_state.json;" \
             "the optimizer step and sampler position cannot be restored" >&2
        exit 2
    fi
    RESUME_ARGS=(--resume-from-checkpoint "${RESUME_DIR}")
fi

printf 'config:   %s\n' "${CONFIG}"
printf 'output:   %s\n' "${OUTPUT_DIR}"
printf 'resume:   %s\n' "${RESUME_DIR:-<none, starting from the fixed base model>}"

# --- destructive-action gate -------------------------------------------------
if [ "${GRPO_CONFIRM_TRAIN:-}" != "yes" ]; then
    echo "refusing to start training: set GRPO_CONFIRM_TRAIN=yes to confirm." >&2
    exit 4
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"
if ! conda activate "${ENV_NAME}" 2>/dev/null; then
    echo "error: conda env '${ENV_NAME}' does not exist." \
         "Create it from requirements/trl-py311.txt first; this script" \
         "does not install anything." >&2
    exit 3
fi

cd "${PROJECT_ROOT}"



# --dry-run prints the resolved plan (budget, sampler layout, TRL kwargs) without loading
# a model.  Run it first; it is the cheapest way to catch a config mistake.
python -m trainers.vanilla.trl_trainer --config "${CONFIG}" --dry-run

# The real run.  Absolute paths only: the config's output_dir is relative to the project
# root by design, and the trainer refuses to write anywhere outside it.
# Single-process run: the policy is logical cuda:0 and the frozen reference cuda:1;
# CUDA_VISIBLE_DEVICES decides which physical cards back those two logical devices.
export GRPO_POLICY_DEVICE=${GRPO_POLICY_DEVICE:-cuda:0}
export GRPO_REFERENCE_DEVICE=${GRPO_REFERENCE_DEVICE:-cuda:1}
export GRPO_REFERENCE_LOGPS_BATCH_SIZE=${GRPO_REFERENCE_LOGPS_BATCH_SIZE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python -m trainers.vanilla.trl_trainer --config "${CONFIG}" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" 2>&1 | tee "${LOG_FILE}"
