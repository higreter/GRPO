#!/usr/bin/env bash
# RewardBench evaluation of a trained checkpoint, on BOTH orderings.
#
# Run only against a validated checkpoint. The orchestrator supplies an explicitly
# selected non-zero physical GPU; the default below is retained for manual inspection
# but GPU0 is excluded by every automated run on this host.
#
# The output is one JSONL row per (qid, ordering) -- 2985 qids x 2 = 5970 rows -- even for
# completions that fail to parse.  Parse failures are kept, with their reason code, because
# dropping them silently inflates every accuracy in the summary.
#
# Usage:
#   bash scripts/eval_rewardbench.sh <method> <checkpoint-dir> [output.jsonl]
#     method in {vanilla, j1, j4r, jwc}
set -euo pipefail

METHOD="${1:?usage: eval_rewardbench.sh <vanilla|j1|j4r|jwc> <checkpoint-dir> [output.jsonl]}"
CHECKPOINT="${2:?missing checkpoint directory}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${3:-${PROJECT_ROOT}/runs/${METHOD}/rewardbench_predictions.jsonl}"

case "${METHOD}" in
    vanilla|j1|j4r|jwc) ;;
    *)
        echo "error: METHOD must be one of vanilla, j1, j4r, jwc; got '${METHOD}'" >&2
        exit 2
        ;;
esac

if [ ! -d "${CHECKPOINT}" ]; then
    echo "error: checkpoint directory '${CHECKPOINT}' does not exist" >&2
    exit 2
fi
if [ ! -f "${CHECKPOINT}/config.json" ]; then
    echo "error: '${CHECKPOINT}' has no config.json; it is not a saved checkpoint" >&2
    exit 2
fi

CONDA_SH=${CONDA_SH:-/data1/yanghao/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-grpo-trl-py311}
LOG_DIR=${LOG_DIR:-/tmp/yh_grpo/logs}

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT}")"
LOG_FILE="${LOG_DIR}/eval_${METHOD}.log"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"

# The source file is fixed by the project (never a --source override in a real run): the
# OOD RewardBench split is the only sanctioned test set.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?explicitly select a physical GPU; never default to the faulty GPU0} \
python -m evaluation.rewardbench \
    --method "${METHOD}" \
    --model-path "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --views original,swapped \
    --max-new-tokens 512 \
    --temperature 0.0 \
    2>&1 | tee "${LOG_FILE}"

# Re-aggregate from the written file (never from memory), so the numbers in the report can
# be reproduced from the artifact alone.
python -c "import json,sys; from evaluation import rewardbench as rb; r = rb.load_predictions(sys.argv[1]); print(json.dumps(rb.summarise_predictions(r, method=r[0]['method']), indent=2, sort_keys=True))" "${OUTPUT}"
