#!/usr/bin/env bash
# Serial longest-row smoke tests. No training GPU is allocated while waiting.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PYTHON_BIN:-python}
POLL_SECONDS=${POLL_SECONDS:-600}
EXCLUDED_PHYSICAL_GPUS=${EXCLUDED_PHYSICAL_GPUS:-}
METHODS=(vanilla j1 jwc j4r)
CONFIGS=(vanilla j1 jwc_rl_brier j4r)
STARTED=$(date -Is)
cd "$ROOT"
if [ "${1:-}" = "--skip" ] || [ "${RUN_SMOKE:-1}" = "0" ]; then
    echo "Smoke skipped by explicit request."
    exit 0
fi
is_excluded_gpu() {
    case ",${EXCLUDED_PHYSICAL_GPUS}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}
for position in 0 1 2 3; do
    method=${METHODS[$position]}
    dir="$ROOT/runs/smoke/$method"
    config="$dir/smoke_config.yaml"
    mkdir -p "$dir/logs"
    if [ ! -f "$config" ]; then
        "$PY" scripts/create_smoke_config.py --formal-config "$ROOT/configs/${CONFIGS[$position]}.yaml" --run-output "$dir/run" --output-config "$config" --output-metadata "$dir/smoke_config_metadata.json"
    fi
    if [ -f "$dir/validation.json" ] && [ -f "$dir/logs/checkpoint_reload.json" ] && [ -f "$dir/run/final/config.json" ]; then
        echo "$(date -Is) $method already validated; not rerunning" | tee -a "$dir/logs/orchestrator.log"
        continue
    fi
    if [ -e "$dir/run" ]; then
        echo "$(date -Is) $method has an unvalidated output; stop for diagnosis rather than overwrite" | tee -a "$dir/logs/orchestrator.log"
        exit 20
    fi
    while true; do
        table=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
        printf '%s\n%s\n' "$(date -Is)" "$table" >> "$ROOT/runs/smoke/gpu_allocation.log"
        policy=""; reference=""
        while IFS=, read -r i used free util; do
            i=${i//[[:space:]]/}; used=${used//[[:space:]]/}; util=${util//[[:space:]]/}
            is_excluded_gpu "$i" && continue
            if [ -z "$policy" ] && [ "$used" -le 1000 ] && [ "$util" -le 10 ]; then policy=$i; fi
        done <<< "$table"
        if [ -n "$policy" ]; then
            while IFS=, read -r i used free util; do
                i=${i//[[:space:]]/}; free=${free//[[:space:]]/}
                if ! is_excluded_gpu "$i" && [ "$i" != "$policy" ] && [ "$free" -ge 16000 ]; then reference=$i; break; fi
            done <<< "$table"
        fi
        [ -n "$reference" ] && break
        sleep "$POLL_SECONDS"
    done
    echo "$(date -Is) selected $method policy=$policy reference=$reference" | tee -a "$dir/logs/orchestrator.log"
    {
      echo "physical_policy_gpu=$policy"; echo "physical_reference_gpu=$reference"
      echo "logical_policy_device=cuda:0"; echo "logical_reference_device=cuda:1"
      echo "source_row_index=17585"; echo "config=$config"; echo "selected_at=$(date -Is)"
    } > "$dir/gpu_allocation.env"
    nvidia-smi > "$dir/logs/preflight_nvidia_smi.txt" 2>&1 || true
    monitor() {
        while kill -0 "$1" 2>/dev/null; do
            printf '%s\n' "$(date -Is)" >> "$dir/logs/gpu_memory.csv"
            nvidia-smi -i "$policy,$reference" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$dir/logs/gpu_memory.csv" 2>&1 || true
            sleep 1
        done
    }
    set +e
    CUDA_VISIBLE_DEVICES="$policy,$reference" GRPO_POLICY_DEVICE=cuda:0 GRPO_REFERENCE_DEVICE=cuda:1 \
      GRPO_REFERENCE_LOGPS_BATCH_SIZE=1 GRPO_CONFIRM_TRAIN=yes HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" -m "trainers.$method.trl_trainer" --config "$config" --source-row-index 17585 \
      > >(tee "$dir/logs/train.log") 2>&1 &
    train_pid=$!
    monitor "$train_pid" & monitor_pid=$!
    wait "$train_pid"; status=$?
    wait "$monitor_pid" 2>/dev/null
    set -e
    echo "exit_code=$status" > "$dir/exit_status.txt"
    echo "ended_at=$(date -Is)" >> "$dir/gpu_allocation.env"
    nvidia-smi > "$dir/logs/postflight_nvidia_smi.txt" 2>&1 || true
    [ "$status" -eq 0 ] || exit "$status"
    "$PY" - "$dir" "$STARTED" <<'PY' | tee "$dir/logs/validation_stdout.log"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); state=json.loads((root/'run/checkpoint-1/trainer_state.json').read_text())
assert state['global_step']==1, state['global_step']
manifest=json.loads((root/'run/run_manifest.json').read_text())
topology=manifest['extra']['dual_device_reference']
assert topology['policy_model_device']=='cuda:0'
assert topology['reference_model_device']=='cuda:1'
assert topology['reference_all_frozen'] is True
assert topology['optimizer_reference_parameter_overlap']==0
assert (root/'run/final/config.json').is_file()
rows=[]
for line in (root/'logs/gpu_memory.csv').read_text().splitlines():
    parts=line.split(',')
    if len(parts)==4 and parts[0].strip().isdigit(): rows.append((int(parts[0]),int(parts[1])))
peaks={str(i):max(m for j,m in rows if j==i) for i in {i for i,m in rows}}
result={'passed':True,'global_step':state['global_step'],'max_steps':state['max_steps'],'topology':topology,'physical_gpu_peak_mib':peaks,'log_history':state['log_history'],'started_chain':sys.argv[2]}
(root/'validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
PY
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/validate_checkpoint.py "$dir/run/checkpoint-1" --output "$dir/logs/checkpoint_reload.json"
done
echo "$(date -Is) all four smoke tests validated" | tee "$ROOT/runs/smoke/all_passed.txt"
