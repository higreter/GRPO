#!/usr/bin/env bash
# J1 (arXiv:2505.10320): verdict correctness + verdict consistency on paired orderings.
#
# *** NEVER EXECUTED. ***  Launch draft only.  The environment it needs
# (requirements/trl-py311.txt) was never created and no training was run.
#
# What is different from vanilla, and why the shell script has to care:
#   * every preference pair becomes TWO dataset rows (original + swapped), so the
#     dataloader holds 2x22190 rows and one optimiser step consumes 2x the sequences;
#   * the consistency term is only defined when both orderings of a pair sit in the same
#     generation batch, so the batch layout is fixed by data/paired_sampler.py and
#     trainers/j1/trl_trainer.py raises if a counterpart is ever missing;
#   * that means the run is NOT the same shape as vanilla -- configs/j1.yaml matches the
#     controlled rollout/token budget, not the step count.  See configs/controlled_budget.yaml.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH=${CONDA_SH:-/data1/yanghao/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-grpo-trl-py311}
CONFIG="${PROJECT_ROOT}/configs/j1.yaml"
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
LOG_FILE="${LOG_DIR}/train_j1.log"

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



python -m trainers.j1.trl_trainer --config "${CONFIG}" --dry-run

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GRPO_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    GRPO_NUM_GPUS=${#GRPO_GPU_LIST[@]}
else
    GRPO_NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
fi
if [ "$GRPO_NUM_GPUS" -lt 1 ]; then
    echo "No CUDA GPU is available." >&2
    exit 5
fi
export GRPO_POLICY_DEVICE=${GRPO_POLICY_DEVICE:-cuda:0}
if [ "$GRPO_NUM_GPUS" -gt 1 ]; then
    export GRPO_REFERENCE_DEVICE=${GRPO_REFERENCE_DEVICE:-cuda:1}
else
    export GRPO_REFERENCE_DEVICE=${GRPO_REFERENCE_DEVICE:-cuda:0}
fi
export GRPO_REFERENCE_LOGPS_BATCH_SIZE=${GRPO_REFERENCE_LOGPS_BATCH_SIZE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
torchrun --standalone --nproc-per-node="$GRPO_NUM_GPUS" \
    -m trainers.j1.trl_trainer --config "${CONFIG}" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" 2>&1 | tee "${LOG_FILE}"
