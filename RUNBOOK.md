# GRPO 四组 Baseline：训练与评测 Runbook

更新日期：2026-11-06（Asia/Shanghai）

本文件是训练服务器上的**唯一执行入口**。算法与实现细节见
`docs/IMPLEMENTATION.md`；环境的精确包版本见 `grpo_baselines/requirements/locks/`。
执行时以本文件、四份正式 YAML 和当前代码为准。

## 0. 当前结论

| 项目 | 状态 |
|---|---|
| 四种方法代码 | 已完成 |
| HelpSteer3 / RewardBench 数据 | 已放入 `database/` |
| TRL / verl / 方法参考源码 | 已放入 `external/`，固定 commit |
| CPU 单元测试 | 428 项通过 |
| 两套原服务器环境 | 已安装并审计；迁移包只带依赖锁，不带 15GB conda 环境实体 |
| 真实 GPU Smoke Test | **尚未完成** |
| 正式训练 | **尚未开始** |
| RewardBench 推理 | **尚未开始** |
| Checkpoint / 正式预测 | **不存在** |

GPU Smoke Test 是可选的前置验证；默认执行正式训练。CPU 测试通过不等于训练已经完成。

## 1. 仓库包含与不包含的内容

### 已包含

- `grpo_baselines/`：四种方法、配置、训练器、奖励、评测、测试和启动脚本。
- `database/`：2 份原始 HelpSteer3、2 份清洗后 HelpSteer3、1 份 RewardBench。
- `external/`：固定版本的 TRL、VERL、ReasoningJudgeBench、DeepSeek-Math 仓库。
- `migration/`：离线完整性检查、数据哈希、上游源码 commit。
- 环境安装报告、实现报告和本总说明。
- 已有的 CPU 测试日志及尚未实际训练的 Smoke 配置/等待日志。

### 明确不包含

- **论文**：`papers/` 不进入 GitHub 仓库。
- **基座模型**：Qwen2.5-3B-Instruct 不进入 GitHub 仓库。
- 原服务器的 conda 环境二进制目录不进入 GitHub 仓库。
- 没有任何训练 checkpoint，因为训练尚未开始。

基座模型由新服务器单独准备，运行前设置：

```bash
export GRPO_BASE_MODEL=/新服务器上的路径/Qwen2.5-3B-Instruct
```

代码不会联网下载模型。模型目录至少必须有：

```text
config.json
generation_config.json
merges.txt
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
vocab.json
```

原模型约 5.8GB，仅作为型号参考：`Qwen2.5-3B-Instruct`。

## 2. 实验目标与四种方法

统一基座：`Qwen2.5-3B-Instruct`。

| 顺序 | 方法 | 训练状态 | 核心区别 |
|---:|---|---|---|
| 1 | Vanilla GRPO | 未训练 | A/B verdict correctness，标准 GRPO 组归一化 |
| 2 | J1 | 未训练 | Original/Swapped 两状态；正确性 + 两顺序都正确的一致性奖励 |
| 3 | JwC RL-Brier | 未训练 | 输出偏好概率，奖励为 `1-(p-y)^2` |
| 4 | J4R EIS-GRPO | 未训练 | Original/Swapped 两状态；EIS global + local advantage |

正式顺序固定为：

```text
Vanilla -> J1 -> JwC -> J4R
```

一次只运行一种方法，不允许四种训练并行。

## 3. 数据

所有正式数据均在 `database/`，迁移后不需要重新下载或重新清洗。

| 文件 | 行数 | 用途 | SHA-256 |
|---|---:|---|---|
| `helpsteer3_train_clean.jsonl` | 22,190 | 唯一正式训练集 | `67784a938e36ed2375faada34b67406ab4befff8cc8b09bcff4613a24dd7af78` |
| `helpsteer3_test_clean.jsonl` | 1,866 | HelpSteer3 ID 测试参考 | `8ffb55e29fa1b73f0ab98f7c441f38bceed807b26a6faf5dff17fdcb2715f703` |
| `rewardbench_v1_ood_clean.jsonl` | 2,985 | 正式 OOD 评测，23 个 subset | `9dc7b9e26fad89a0ac4483ed41df35747da6a576fa78d17330d164c0d7d7199c` |
| `helpsteer3_train_raw.jsonl.gz` | 38,459 | 数据溯源，不用于正式训练 | `32b52e1d378f8dab1e4c9ae549da49a5d6fc0875aeafe3f9139e6053beb906bb` |
| `helpsteer3_test_raw.jsonl.gz` | 2,017 | 数据溯源，不用于正式评测 | `cd0d8b6efd7869a44c1b5ff91701232062555829b3c838abfd770d7dab6c7861` |

训练集协议：Scheme C、context 级隔离、去 Tie、去重复、最大清洗阈值 12,288、无 Dev。
不能换成 Swap 扩增数据，不能把 RewardBench 混入训练。

RewardBench 是官方 filtered split 转换后的 A/B 数据，和 HelpSteer3 Train 的 prompt、
有序三元组、无序回答对精确重叠均为 0，因此 OOD-clean 仍是完整 2,985 条。

详细字段与清洗审计见 `database/README.md`。完整性检查：

```bash
cd /path/to/GRPO
python migration/verify_bundle.py
sha256sum -c migration/ESSENTIAL_SHA256SUMS
```

## 4. 固定训练预算和超参数

四种方法统一：

- seed：42
- 最大 Prompt：11,264 token
- 最大 Completion：1,024 token
- 总上下文预算：12,288 token，不能缩短
- 学习率：`1e-6`
- Scheduler：constant
- warmup：0
- 全局 source-pair batch：8
- micro batch：2
- 每状态 rollout：8
- 每种方法总 rollout：16,384
- 每种方法估计 token 预算：46,268,416
- temperature：1.0
- top-p：1.0
- clip epsilon：0.2
- KL beta：0.01
- Reference：冻结初始 Policy
- bf16、gradient checkpointing
- 全参数微调；不改 LoRA、QLoRA 或量化训练

方法几何：

| 方法 | states/sample | num_generations | grad accumulation | steps | rollouts/update |
|---|---:|---:|---:|---:|---:|
| Vanilla | 1 | 8 | 4 | 256 | 64 |
| J1 | 2 | 16 | 8 | 128 | 128 |
| JwC | 1 | 8 | 4 | 256 | 64 |
| J4R | 2 | 16 | 8 | 128 | 128 |

四份正式配置：

```text
grpo_baselines/configs/vanilla.yaml
grpo_baselines/configs/j1.yaml
grpo_baselines/configs/jwc_rl_brier.yaml
grpo_baselines/configs/j4r.yaml
```

配置中的训练数据和输出目录已改成相对路径；基座模型可由 `GRPO_BASE_MODEL` 覆盖，因此
迁移后不要求继续使用 `/data1/yanghao`。

## 5. 依赖与环境

### 正式 TRL 训练环境（四种方法的主路径）

原服务器验证过的组合：

| 组件 | 版本 |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.6 |
| Tokenizers | 0.22.2 |
| Accelerate | 1.15.0 |
| Datasets | 5.0.1 |
| TRL | 1.18.0.dev0，来自本地固定 commit |
| CUDA runtime | 12.8 |

上游 TRL commit：

```text
3885117d6a1dd647d43befeddf0da482c505cb47
```

推荐在新服务器重建：

```bash
conda create -n grpo-trl-py311 python=3.11 -y
conda activate grpo-trl-py311

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r grpo_baselines/requirements/trl-py311.txt
pip install --no-deps -e external/trl
pip check
```

精确环境快照：

```text
grpo_baselines/requirements/locks/grpo-trl-py311.lock.txt
grpo_baselines/requirements/locks/grpo-trl-py311.conda-list.txt
```

如果集群驱动不支持 CUDA 12.8，应选择与驱动兼容的 PyTorch CUDA wheel，并先完成真实
Smoke；不要混装两个环境的 torch/transformers。

### VERL 环境（可选，不是当前四个 TRL 启动脚本的必需条件）

J4R 同时提供 VERL advantage estimator 注册路径。只有决定改用 VERL 后端时才安装：

| 组件 | 版本 |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 5.16.1 |
| VERL | 0.10.0.dev |
| Ray | 2.60.0 |

固定 commit：`745432653d15af14954af1e055c8aeed72afa064`。

依赖文件：

```text
grpo_baselines/requirements/verl-py310.txt
grpo_baselines/requirements/locks/grpo-verl-py310.lock.txt
grpo_baselines/requirements/locks/grpo-verl-py310.conda-list.txt
```

当前正式计划使用 TRL/Hugging Face generate，不依赖 vLLM、FlashAttention 或 SGLang。
不要为了“加速”临时安装不匹配的 vLLM/FlashAttention，它们和 torch/CUDA 存在 ABI 风险。

系统工具需要：NVIDIA Driver、CUDA 兼容驱动、conda、git、tmux、bash、GNU coreutils。

## 6. 固定上游源码

| 目录 | Commit | 用途 |
|---|---|---|
| `external/trl` | `3885117d6a1dd647d43befeddf0da482c505cb47` | 四种方法的主要 Trainer |
| `external/verl` | `745432653d15af14954af1e055c8aeed72afa064` | J4R 可选 estimator 注册路径 |
| `external/reasoning-judge-bench` | `e5a6c64fc97dd65f2b1b868664fb12c2d12b0606` | J4R prompt/verdict 规则参考 |
| `external/deepseek-math` | `b8b0f8ce093d80bf8e9a641e44142f06d092c305` | Vanilla/GRPO 溯源；训练代码不直接 import |

迁移后运行 `python migration/verify_bundle.py` 会校验这些 HEAD。

## 7. GPU 使用

训练是单进程：可训练的 Policy 在逻辑 `cuda:0`，冻结的 Reference 在逻辑 `cuda:1`（只有一张可见 GPU 时设 `GRPO_REFERENCE_DEVICE=cuda:0`）。`CUDA_VISIBLE_DEVICES` 决定这两个逻辑设备对应哪些物理卡，A100 集群按调度器分配即可。Reference 始终是冻结的初始 Policy，处于 eval 模式，不参与梯度反传，也不进入 optimizer。

```bash
export CUDA_VISIBLE_DEVICES=<分配的两张物理卡>
export GRPO_POLICY_DEVICE=cuda:0
export GRPO_REFERENCE_DEVICE=cuda:1
export GRPO_REFERENCE_LOGPS_BATCH_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Reference 的设备只决定前向位置，不改变冻结状态或 KL 目标。不要为凑卡数额外占用未分配的 GPU。
## 8. 新服务器第一次检查

```bash
cd /path/to/GRPO

export GRPO_BASE_MODEL=/path/to/Qwen2.5-3B-Instruct
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export ENV_NAME=grpo-trl-py311
source "$CONDA_SH"
conda activate "$ENV_NAME"

python migration/verify_bundle.py --base-model "$GRPO_BASE_MODEL" --runtime
cd grpo_baselines
python -m unittest discover -s tests -t .

python -m trainers.vanilla.trl_trainer --config configs/vanilla.yaml --dry-run
python -m trainers.j1.trl_trainer --config configs/j1.yaml --dry-run
python -m trainers.jwc.trl_trainer --config configs/jwc_rl_brier.yaml --dry-run
python -m trainers.j4r.trl_trainer --config configs/j4r.yaml --dry-run
```

预期 CPU 测试为 428 项通过。`--dry-run` 不加载模型、不使用 GPU。

如果测试数量因后续新增测试而改变，以“全部通过”为准，但必须同步更新实现报告里的固定测试数。

## 9. GPU Smoke Test

最长真实训练源样本：

```text
source_row_index = 17585
group_id = train_017585
source token_len = 10496
```

Smoke 顺序：

```text
Vanilla -> J1 -> JwC -> J4R
```

自动串行 Smoke：

```bash
cd /path/to/GRPO/grpo_baselines
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate grpo-trl-py311

export GRPO_BASE_MODEL=/path/to/Qwen2.5-3B-Instruct
export PYTHON_BIN="$(command -v python)"
export POLL_SECONDS=300
export RUN_SMOKE=1

tmux new-session -d -s grpo_smokes \
  "bash scripts/watch_and_launch_smokes.sh 2>&1 | tee runs/smoke/orchestrator.log"
```

如果集群某些卡不可用，可加：

```bash
export EXCLUDED_PHYSICAL_GPUS=0,3
```

每种 Smoke 必须真实完成：

1. 加载真实 3B Policy 和 Reference。
2. 真实 rollout、reward、advantage。
3. 至少一次 backward 和 optimizer update。
4. Reference 位于指定逻辑设备，全部冻结。
5. Reference 参数与 optimizer overlap 为 0。
6. 保存 `checkpoint-1` 和 final。
7. checkpoint 在 CPU/新进程中重新加载成功。
8. 记录所用物理卡的峰值显存。
9. 最长输入和监督目标没有静默截断。

Smoke 通过后可启动正式训练。不需要 Smoke 时，使用正式脚本的 `--skip-smoke` 参数跳过。

## 10. 正式训练

以 Vanilla 为例，使用四张已分配的 GPU：

```bash
cd /path/to/GRPO/grpo_baselines
export GRPO_BASE_MODEL=/path/to/Qwen2.5-3B-Instruct
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export ENV_NAME=grpo-trl-py311
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GRPO_POLICY_DEVICE=cuda:0
export GRPO_REFERENCE_DEVICE=cuda:1
export GRPO_REFERENCE_LOGPS_BATCH_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRPO_CONFIRM_TRAIN=yes

tmux new-session -d -s grpo_vanilla \
  "bash scripts/train_vanilla.sh --skip-smoke 2>&1 | tee logs/formal_vanilla.log"
```

其他方法脚本：

```text
scripts/train_j1.sh
scripts/train_jwc.sh
scripts/train_j4r.sh
```

`scripts/watch_and_launch_formal.sh` 会串行执行四种正式训练及 RewardBench；默认跳过 Smoke，可用
`RUN_SMOKE=1` 开启四组前置 Smoke。它会在全部训练和评测验证通过后调用
`scripts/build_final_report.py` 生成 `runs/FINAL_RESULTS.md` 和 `runs/final_results.csv`。
启动自动链前仍应先审阅集群调度规则；不希望全自动运行时使用上面的逐方法命令。

默认直接正式训练；`--skip-smoke` 仅显式确认跳过可选 Smoke。不要同时启动四组训练，每个方法结束后验证 checkpoint，再进入下一个。

断点恢复示例：

```bash
GRPO_CONFIRM_TRAIN=yes bash scripts/train_vanilla.sh \
  --resume-from-checkpoint runs/vanilla/checkpoint-128
```

恢复前必须先检查 `trainer_state.json`，不要覆盖有效进度。当前配置
`save_optimizer_state: false` 是已有实验设计的一部分；如果实际要求跨进程完整恢复 optimizer，
必须先解决这一配置与恢复目标的矛盾并记录变更，不能假装已经完整恢复。

## 11. RewardBench 评测

四种正式训练全部完成后，在相同的 2,985 条 OOD-clean 上评测。每条执行 Original 和
Swapped 两次推理，保留解析失败，不能丢样本。

```bash
cd /path/to/GRPO/grpo_baselines
bash scripts/eval_rewardbench.sh vanilla runs/vanilla/final
bash scripts/eval_rewardbench.sh j1      runs/j1/final
bash scripts/eval_rewardbench.sh jwc     runs/jwc/final
bash scripts/eval_rewardbench.sh j4r     runs/j4r/final
```

应保存：

- 每种方法逐样本预测 JSONL。
- Original/Swapped 预测、解析状态和原始 completion。
- 整体 Accuracy、Both-order Accuracy、速度。
- RewardBench 23 个 subset 的 Accuracy。
- JwC 的 Brier/ECE/MCE；硬 verdict 方法不能伪造概率校准指标。
- GPU 型号、推理耗时、异常与失败率。

这里的 Both-order Accuracy 定义是：两个顺序都预测正确才为 1；一对一错或都错均为 0。
不要把“答案字母是否翻转”当成 Both-order Accuracy。

## 12. 输出与最终交付

最终应产生：

```text
grpo_baselines/runs/vanilla/
grpo_baselines/runs/j1/
grpo_baselines/runs/jwc/
grpo_baselines/runs/j4r/
grpo_baselines/runs/smoke/
```

并补充总报告，至少包含：

- 代码 commit / 工作区状态。
- 数据哈希和行数。
- 基座模型路径及模型文件校验。
- 四种完整超参数。
- Policy/Reference 物理与逻辑 GPU 映射。
- 每种 Smoke 的 optimizer step、峰值显存、checkpoint reload 证据。
- 正式训练时长、最终 checkpoint 和失败恢复记录。
- RewardBench 整体与 23 个 subset 结果。
- 所有预测、CSV、JSON、日志和复现命令的绝对路径。

## 13. 已知风险

1. 全参数训练仍可能在 12K 上下文和长生成阶段遇到显存不足，应按实际资源调整 micro batch，而不是缩短样本或改用 LoRA。
2. 当前配置保存 model-only checkpoint，可用于评测；不包含 optimizer，不能当作完整训练断点恢复。
3. 外部源码固定 commit；升级 TRL 可能改变内部 hook/API，应重新运行测试。
4. J1/J4R 的成对状态和每组 rollout 布局不能被任意 batching 或 sampler 替换破坏。

## 14. 运行记录

记录实际设备、吞吐和耗时，不使用硬件型号推测固定完成时间。
## 15. 迁移方法

本项目通过 GitHub 仓库迁移。

特别注意：两份 HelpSteer3 大文件超过普通 GitHub 单文件限制，必须使用 Git LFS；`external/`
中的四个固定源码仓库应注册为 submodule。顶层 `.gitignore` 已排除论文、模型、checkpoint、
凭据和缓存，`.gitattributes` 已标记数据库文件为 LFS。

如果先通过 rsync 传到集群再建 GitHub 仓库，可使用：

```bash
rsync -a --info=progress2 \
  --exclude '/papers/' \
  --exclude '/models/' \
  /path/to/GRPO/ user@cluster:/target/GRPO/
```

目录迁移或 clone（含 submodule + LFS）完成后先运行：

```bash
cd /target/GRPO
python migration/verify_bundle.py
sha256sum -c migration/ESSENTIAL_SHA256SUMS
```

然后准备外部基座模型、创建 TRL 环境、运行 428 项 CPU 测试，最后才进入四个 Smoke。
首次 clone 使用 `git clone --recurse-submodules`，随后执行 `git lfs pull`。

