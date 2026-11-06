# GRPO Baselines for LLM-as-a-Judge

四组固定预算实验：Vanilla GRPO、J1、JwC（RL-Brier）和 J4R（EIS-GRPO）。统一使用
Qwen2.5-3B-Instruct、HelpSteer3 Scheme-C Train，以及 RewardBench v1 OOD-clean。

## 当前状态

- 代码、数据适配、多 GPU 训练、frozen Reference 和评测代码已完成。
- 428 项 CPU 测试通过。
- 真实 GPU Smoke、正式训练和 RewardBench 推理**尚未执行**。
- 仓库中没有正式 checkpoint 或结果，不能把现有 dry-run 当成实验结果。

## 文档

- [RUNBOOK.md](RUNBOOK.md)：唯一执行入口，包含数据、超参数、依赖、GPU 拓扑、Smoke、
  正式训练、评测和恢复。
- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)：方法定义、奖励、EIS、数据隔离和测试证据。
- [database/README.md](database/README.md)：五份数据的来源、字段、清洗口径与 SHA-256。
- [grpo_baselines/requirements/README.md](grpo_baselines/requirements/README.md)：环境与 ABI 风险。

## Clone 与依赖

数据库由 Git LFS 管理，四个固定上游源码由 submodule 管理：

```bash
git clone --recurse-submodules <repository-url>
cd GRPO
git lfs pull
python migration/verify_bundle.py
sha256sum -c migration/ESSENTIAL_SHA256SUMS
```

训练主环境使用 Python 3.11、PyTorch 2.8.0+cu128、Transformers 4.57.6，以及仓库固定
commit 的 TRL。精确版本位于 `grpo_baselines/requirements/locks/`。

基座模型和论文不进入仓库。运行服务器需要单独准备 Qwen2.5-3B-Instruct：

```bash
export GRPO_BASE_MODEL=/path/to/Qwen2.5-3B-Instruct
```

## GitHub 存储规则

- `database/*.jsonl*` 必须通过 Git LFS 提交。
- `external/{trl,verl,reasoning-judge-bench,deepseek-math}` 是固定 commit 的 submodule。
- `papers/`、`models/`、密钥、checkpoint 和 optimizer state 均被 `.gitignore` 排除。
- 正式模型应放 Release、对象存储或独立模型仓库，不进入普通 Git 历史。

GPU Smoke 为可选检查（`RUN_SMOKE=1`）；默认直接执行正式训练，训练脚本支持 `--skip-smoke`。
