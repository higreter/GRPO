#!/usr/bin/env bash
# J4R (arXiv:2505.13346): EIS advantages over two equivalent states + judgement/format reward.
#
# *** NEVER EXECUTED. ***  Launch draft only.  No training was run and the GPUs on this
# host were fully occupied for the whole task.
#
# What is different from vanilla:
#   * two states per preference pair again (original + swapped), same 2x row count as J1;
#   * the reward is NOT the terminal correctness: it is
#         R = 1[parsed verdict == ground truth]  +  (+0.5 if the layout is well formed
#                                                     else -0.5)
#     so a malformed completion scores -0.5, not 0 (see trainers/j4r/reward.py);
#   * the advantage is EIS, not TRL's per-group normalisation:
#         A = (R - mean_all)/std_all  +  (R - mean_state)/std_state
#     computed over the whole generation batch BEFORE it is split into gradient-accumulation
#     slices (trainers/j4r/advantage.py documents why the split point matters);
#   * the config uses the same KL coefficient as the other three methods (0.01), not J4R's
#     paper value of 1e-4 -- a controlled-comparison deviation recorded in
#     configs/controlled_budget.yaml.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH=${CONDA_SH:-/data1/yanghao/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-grpo-trl-py311}
CONFIG="${PROJECT_ROOT}/configs/j4r.yaml"
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
LOG_FILE="${LOG_DIR}/train_j4r.log"

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



# The dry run prints the plan including advantage.description; for j4r it must contain
# "EIS" and "mean_all"/"mean_state".  If it does not, the wrong trainer class got wired.
python -m trainers.j4r.trl_trainer --config "${CONFIG}" --dry-run

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
    -m trainers.j4r.trl_trainer --config "${CONFIG}" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" 2>&1 | tee "${LOG_FILE}"
