#!/usr/bin/env bash
# Gated, serial formal training and fixed-source RewardBench evaluation.
# Waiting never allocates a GPU. Existing incomplete outputs stop the chain.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PYTHON_BIN:-python}
POLL_SECONDS=${POLL_SECONDS:-600}
EXCLUDED_PHYSICAL_GPUS=${EXCLUDED_PHYSICAL_GPUS:-}
RUN_SMOKE=${RUN_SMOKE:-0}
if [ "${1:-}" = "--skip-smoke" ]; then RUN_SMOKE=0; fi
METHODS=(vanilla j1 jwc j4r)
CONFIGS=(vanilla j1 jwc_rl_brier j4r)
cd "$ROOT"
mkdir -p runs/orchestration
is_excluded_gpu() { case ",${EXCLUDED_PHYSICAL_GPUS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
select_policy() {
    policy=""
    local table; table=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    printf '%s\n%s\n' "$(date -Is)" "$table" >> runs/orchestration/gpu_allocation.log
    while IFS=, read -r i used free util; do
        i=${i//[[:space:]]/}; used=${used//[[:space:]]/}; util=${util//[[:space:]]/}
        if ! is_excluded_gpu "$i" && [ -z "$policy" ] && [ "$used" -le 1000 ] && [ "$util" -le 10 ]; then policy=$i; fi
    done <<< "$table"
}
select_reference() {
    reference=""
    local table; table=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    while IFS=, read -r i used free util; do
        i=${i//[[:space:]]/}; free=${free//[[:space:]]/}
        if ! is_excluded_gpu "$i" && [ "$i" != "$policy" ] && [ "$free" -ge 16000 ]; then reference=$i; break; fi
    done <<< "$table"
}
monitor() {
    local pid=$1 dir=$2 policy=$3 reference=$4
    while kill -0 "$pid" 2>/dev/null; do
        printf '%s\n' "$(date -Is)" >> "$dir/gpu_memory.csv"
        nvidia-smi -i "$policy,$reference" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$dir/gpu_memory.csv" 2>&1 || true
        sleep 1
    done
}
if [ "$RUN_SMOKE" = "1" ]; then
  bash scripts/watch_and_launch_smokes.sh
fi
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path.cwd()
for method in ('vanilla','j1','jwc','j4r'):
    d=root/'runs/smoke'/method
    result=json.loads((d/'validation.json').read_text())
    assert result['passed'] is True and result['global_step']==1
    assert (d/'logs/checkpoint_reload.json').is_file()
print('All four independently saved smoke validations passed; formal chain may proceed.')
PY
for position in 0 1 2 3; do
    method=${METHODS[$position]}; config="$ROOT/configs/${CONFIGS[$position]}.yaml"
    run="$ROOT/runs/$method"; dir="$ROOT/runs/orchestration/$method"
    mkdir -p "$dir"
    if [ -f "$dir/training_validation.json" ] && [ -f "$dir/checkpoint_reload.json" ] && [ -f "$run/final/config.json" ]; then
        echo "$(date -Is) $method formal training already validated" | tee -a "$dir/orchestrator.log"
    else
        if [ -e "$run" ]; then
            echo "$(date -Is) $method has an unvalidated run directory; refusing to overwrite" | tee -a "$dir/orchestrator.log"
            exit 20
        fi
        while true; do select_policy; [ -n "$policy" ] && select_reference; [ -n "${reference:-}" ] && break; sleep "$POLL_SECONDS"; done
        echo "$(date -Is) selected formal $method policy=$policy reference=$reference" | tee -a "$dir/orchestrator.log"
        { echo "physical_policy_gpu=$policy"; echo "physical_reference_gpu=$reference"; echo "config=$config"; echo "selected_at=$(date -Is)"; } > "$dir/gpu_allocation.env"
        nvidia-smi > "$dir/preflight_nvidia_smi.txt" 2>&1 || true
        set +e
        CUDA_VISIBLE_DEVICES="$policy,$reference" GRPO_POLICY_DEVICE=cuda:0 GRPO_REFERENCE_DEVICE=cuda:1 \
          GRPO_REFERENCE_LOGPS_BATCH_SIZE=1 GRPO_CONFIRM_TRAIN=yes HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          bash "scripts/train_$method.sh" --skip-smoke > >(tee "$dir/train.log") 2>&1 &
        train_pid=$!
        monitor "$train_pid" "$dir" "$policy" "$reference" & monitor_pid=$!
        wait "$train_pid"; status=$?
        wait "$monitor_pid" 2>/dev/null
        set -e
        echo "exit_code=$status" > "$dir/train_exit_status.txt"
        nvidia-smi > "$dir/postflight_nvidia_smi.txt" 2>&1 || true
        [ "$status" -eq 0 ] || exit "$status"
        "$PY" - "$method" "$dir" <<'PY'
import json, sys
from pathlib import Path
from common.config import load_config
method, directory=sys.argv[1:]
root=Path.cwd(); directory=Path(directory)
config=load_config(str(root/'configs'/({'jwc':'jwc_rl_brier'}.get(method,method))+'.yaml'))
state_path=root/'runs'/method/f'checkpoint-{config.training_steps}'/'trainer_state.json'
state=json.loads(state_path.read_text())
assert state['global_step']==config.training_steps, state['global_step']
manifest=json.loads((root/'runs'/method/'run_manifest.json').read_text())
topology=manifest['extra']['dual_device_reference']
assert topology['reference_model_device']=='cuda:1' and topology['reference_all_frozen'] is True
assert topology['optimizer_reference_parameter_overlap']==0
assert (root/'runs'/method/'final'/'config.json').is_file()
rows=[]
for line in (directory/'gpu_memory.csv').read_text().splitlines():
    parts=line.split(',')
    if len(parts)==4 and parts[0].strip().isdigit(): rows.append((int(parts[0]),int(parts[1])))
peaks={str(i):max(m for j,m in rows if j==i) for i in {i for i,m in rows}}
result={'passed':True,'method':method,'global_step':state['global_step'],'max_steps':state['max_steps'],'total_rollouts':config.total_rollouts_derived,'topology':topology,'physical_gpu_peak_mib':peaks,'log_history':state['log_history']}
(directory/'training_validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY
        CUDA_VISIBLE_DEVICES="" "$PY" scripts/validate_checkpoint.py "$run/final" --output "$dir/checkpoint_reload.json"
    fi
    if [ -f "$dir/evaluation_validation.json" ] && [ -f "$run/rewardbench_summary.json" ]; then
        echo "$(date -Is) $method RewardBench already complete" | tee -a "$dir/orchestrator.log"
        continue
    fi
    output="$run/rewardbench_predictions.jsonl"
    if [ -f "$run/rewardbench_summary.json" ]; then
        echo "$(date -Is) summary exists without evaluation validation; refusing overwrite" | tee -a "$dir/orchestrator.log"
        exit 21
    fi
    while true; do select_policy; [ -n "$policy" ] && break; sleep "$POLL_SECONDS"; done
    set +e
    CUDA_VISIBLE_DEVICES="$policy" GRPO_EVAL_DEVICE=cuda:0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash scripts/eval_rewardbench.sh "$method" "$run/final" "$output" > >(tee "$dir/evaluation.log") 2>&1
    status=$?
    set -e
    echo "exit_code=$status" > "$dir/evaluation_exit_status.txt"
    [ "$status" -eq 0 ] || exit "$status"
    "$PY" - "$method" "$dir" <<'PY'
import json, sys
from pathlib import Path
method,directory=sys.argv[1:]; directory=Path(directory)
summary=json.loads((Path.cwd()/'runs'/method/'rewardbench_summary.json').read_text())
assert summary['qids']==2985, summary
assert summary['samples']==5970, summary
assert len(summary['per_subset'])==23, summary
result={'passed':True,'method':method,'pairs':summary['qids'],'rows':summary['samples'],'subsets':len(summary['per_subset']),'source_sha256':summary['source_sha256'],'rewardbench_scope':'fixed custom original+swapped views; not official RewardBench leaderboard protocol'}
(directory/'evaluation_validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY
done
"$PY" scripts/build_final_report.py > runs/orchestration/final_report.log 2>&1
printf 'completed_at=%s\n' "$(date -Is)" > runs/orchestration/all_complete.txt
