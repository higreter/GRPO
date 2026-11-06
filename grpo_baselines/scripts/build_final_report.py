"""Build the final report from completed artifacts; never infer training results."""
import csv
import json
from pathlib import Path

from common import paths

METHODS = ("vanilla", "j1", "jwc", "j4r")
ALIASES = {"jwc": "JwC (RL-Brier)", "j4r": "J4R (EIS)", "j1": "J1", "vanilla": "Vanilla GRPO"}


def main():
    root = paths.PROJECT_ROOT
    rows = []
    for method in METHODS:
        directory = root / "runs/orchestration" / method
        validation_path = directory / "training_validation.json"
        evaluation_path = directory / "evaluation_validation.json"
        if not validation_path.is_file() or not evaluation_path.is_file():
            raise FileNotFoundError(f"{method}: no completed, independently validated training and evaluation")
        training = json.loads(validation_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        summary_path = root / "runs" / method / "rewardbench_predictions.jsonl.summary.json"
        summary = json.loads(summary_path.read_text())
        reload = json.loads((directory / "checkpoint_reload.json").read_text())
        assert reload["reload_passed"] and training["global_step"] == training["max_steps"]
        assert evaluation["pairs"] == 2985 and evaluation["rows"] == 5970
        row = {
            "method": method,
            "global_step": training["global_step"],
            "total_rollouts": training["total_rollouts"],
            "policy_peak_mib": training["physical_gpu_peak_mib"],
            "checkpoint_reload": "CPU passed; model-only, not optimizer-resumable",
            "original_accuracy": summary["original"]["accuracy_all_treated_wrong"],
            "swapped_accuracy": summary["swapped"]["accuracy_all_treated_wrong"],
            "both_order_accuracy": summary["both_order"]["both_order_accuracy"],
            "position_consistency": summary["position_consistency"]["position_consistency"],
            "summary": str(summary_path),
        }
        rows.append(row)
    target = root / "runs" / "FINAL_RESULTS.md"
    lines = [
        "# GRPO 四方法正式训练与 OOD 评测结果", "",
        "四组均先通过真实 GPU 最长样本一步 Smoke，再按 Vanilla → J1 → JwC → J4R 串行训练。",
        "所有训练使用 Qwen2.5-3B-Instruct、全参数、11264 prompt + 1024 completion tokens、16384 rollouts。",
        "Reference 固定在第二张逻辑卡，不参与优化；物理 GPU 按集群实时资源动态选择。", "",
        "## 结果", "",
        "| 方法 | 完成步数 | Original accuracy | Swapped accuracy | 双序均正确 | 位置一致率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {name} | {step} | {orig:.4f} | {swap:.4f} | {both:.4f} | {cons:.4f} |".format(
            name=ALIASES[row["method"]], step=row["global_step"], orig=row["original_accuracy"],
            swap=row["swapped_accuracy"], both=row["both_order_accuracy"], cons=row["position_consistency"]))
    lines += ["", "评测固定使用 2985 对输入，original+swapped 共 5970 条预测，覆盖 23 个 subset。",
              "解析失败按错误计入 accuracy；同时保留 parsed-only 指标。上述双序指标是项目自定义聚合，",
              "**不是官方 RewardBench leaderboard 协议或官方分数**。每组最终模型均已在 CPU 实际重载；",
              "保存的是 model-only checkpoint，可用于推理，但不宣称可恢复优化器训练。", "",
              "## 每组产物", ""]
    for method in METHODS:
        lines.append(f"- `{method}`: `runs/{method}/final/`、`rewardbench_predictions.jsonl`、"
                     "`rewardbench_predictions.jsonl.summary.json`、`rewardbench_predictions.jsonl.subsets.csv`；"
                     f"验证位于 `runs/orchestration/{method}/`。")
    target.write_text("\n".join(lines) + "\n")
    with (root / "runs/final_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(target)


if __name__ == "__main__":
    main()
