# Database — 五份数据集说明

本目录汇总 5 份用于 JudgeLRM 实验的数据：

1. HelpSteer3 原始训练集（未清洗）
2. HelpSteer3 原始测试集（未清洗）
3. HelpSteer3 清洗后训练集
4. HelpSteer3 清洗后测试集
5. OOD 评测集 RewardBench v1

生成日期：2026-11-04  
清洗版本：Scheme C（按 context 去泄漏）、**删 Tie**、`max_length = 12288`、无 Dev。

---

## 1. 文件清单

| # | 文件 | 行数 | 大小 | 状态 |
|---|---|---:|---:|---|
| 1 | `helpsteer3_train_raw.jsonl.gz` | 38,459 | 102 MB | 原始，未改动 |
| 2 | `helpsteer3_test_raw.jsonl.gz` | 2,017 | 5.3 MB | 原始，未改动 |
| 3 | `helpsteer3_train_clean.jsonl` | 22,190 | 149 MB | 已清洗（删 Tie） |
| 4 | `helpsteer3_test_clean.jsonl` | 1,866 | 12 MB | 已清洗（部分规则，见 §4） |
| 5 | `rewardbench_v1_ood_clean.jsonl` | 2,985 | 11 MB | 官方转换 + 重叠审计，0 条被删 |

SHA-256：

```
32b52e1d378f8dab1e4c9ae549da49a5d6fc0875aeafe3f9139e6053beb906bb  helpsteer3_train_raw.jsonl.gz
cd0d8b6efd7869a44c1b5ff91701232062555829b3c838abfd770d7dab6c7861  helpsteer3_test_raw.jsonl.gz
67784a938e36ed2375faada34b67406ab4befff8cc8b09bcff4613a24dd7af78  helpsteer3_train_clean.jsonl
8ffb55e29fa1b73f0ab98f7c441f38bceed807b26a6faf5dff17fdcb2715f703  helpsteer3_test_clean.jsonl
9dc7b9e26fad89a0ac4483ed41df35747da6a576fa78d17330d164c0d7d7199c  rewardbench_v1_ood_clean.jsonl
```

---

## 2. HelpSteer3 原始数据（文件 1、2）

- 来源：`nvidia/HelpSteer3` 的 preference 子集
- 官方 split：`preference_train` / `heldout`（validation）
- 格式：gzip 压缩的 JSONL，**保持原始发布状态，未做任何修改**
- 字段：`domain`、`language`、`context`、`response1`、`response2`、`overall_preference`、`individual_preference`
- `overall_preference ∈ {-3,-2,-1,0,1,2,3}`，其中 **0 = Tie**

原始偏好分布：

| split | -3 | -2 | -1 | 0 (Tie) | 1 | 2 | 3 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 3,768 | 8,387 | 5,942 | **2,160** | 6,398 | 7,973 | 3,831 | 38,459 |
| Test | 195 | 441 | 309 | **97** | 323 | 453 | 199 | 2,017 |

原始数据的已知问题：

- 官方 Train / validation 之间本身存在大量 context 重叠（旧流程测得评测集约 58.9% 与训练集重叠）
- Train 内部有大量完全重复的三元组
- 存在两个回答完全相同的退化样本
- 存在答案顺序交换后标签自相矛盾的三元组

这些问题由文件 3、4 的清洗流程处理。

---

## 3. HelpSteer3 清洗后训练集（文件 3）

### 3.1 清洗口径

按顺序执行，每条原始行只按**第一条命中**的规则计一次：

| 顺序 | 规则 | 判定方式 | 删除行数 |
|---:|---|---|---:|
| 1 | 偏好值非法 | `overall_preference ∉ {-3,-2,-1,0,1,2,3}` | 0 |
| 2 | 字段缺失 | `context` / `response1` / `response2` 任一为空 | 35 |
| 3 | Tie | `overall_preference == 0` | 2,160 |
| 4 | 无 reasoning | 训练集要求回答中含推理内容，缺失则丢 | 0 |
| 5 | 超长 | Qwen tokenizer 计数 > 12,288（source + target） | 0 |
| 6 | 跨 split 泄漏 | `md5(question_body)` 命中 Test 的问题集合 | 1,965 |
| 7 | 内部三元组重复 | `md5(q, a1, a2)` 重复，只保留首次出现 | 12,105 |
| 8 | 标签自相矛盾 | 同一三元组出现多个 label | 0（见 §5 说明） |
| 9 | 退化样本 | `answer1.strip() == answer2.strip()` | 2 |
| 10 | Swap 自相矛盾 | 三元组 `t` 与交换 `sw(t)` 的标签集合大小 > 1，迭代到不动点 | 2 |
| 11 | Swap 撞进 Test | `sw(t)` ∈ Test 三元组集合 | 0 |

数字链：

```
38,459
  − 35      (字段缺失)
  − 2,160   (Tie)
= 36,264
  − 1,965   (跨 split 泄漏)
  − 12,105  (内部三元组重复)
  − 2       (退化)
  − 2       (Swap 自相矛盾)
= 22,190
```

### 3.2 保留了什么

- 22,190 条有效 A/B 二分类样本
- 标签分布：`A = 11,068`，`B = 11,122`
- 全部为 `view = original`
- 每个三元组唯一；与 Test 的 context 无重叠；两回答不相同

### 3.3 字段

`question_body`、`answer1_body`、`answer2_body`、`text`、`label`（A/B）、`group_id`、`view`、`domain`、`language`、`source_overall_preference`、`source_line`、`token_len`

> 对应的 Swap 视图（Original + Swap 交错，44,380 行）不在本目录，仍在  
> `/data1/yanghao/paper/JudgeLRM/data/helpsteer3_clean/schemeC_context/train_ab_original_swap.jsonl`。

---

## 4. HelpSteer3 清洗后测试集（文件 4）

### 4.1 实际执行了什么

| 规则 | 是否执行 | 删除行数 |
|---|---|---:|
| 偏好值非法 | ✅ | 0 |
| 字段缺失 | ✅ | 3 |
| Tie | ✅ | 97 |
| 超长（按 source 计，> 12,288） | ✅ | 0 |
| 三元组去重（保留首次） | ✅ | 51 |
| 退化样本 `a1 == a2` | ❌ | 未检查 |
| 标签自相矛盾 | ❌ | 未检查 |
| Swap 自相矛盾 | ❌ | 未检查 |
| Swap 撞进 Test | ❌ | 不适用 |

数字链：

```
2,017
  − 97   (Tie)
  − 3    (字段缺失)
= 1,917
  − 51   (三元组重复)
= 1,866
```

### 4.2 保留了什么

- 1,866 条 A/B 二分类样本
- `gt_class` 分布：`A = 913`，`B = 953`
- 每条记录带 `qid`（0–1865）、`group_id`（`test_XXXXXX`）、`source_line`

### 4.3 字段

`question_body`、`answer1_body`、`answer2_body`、`overall_preference`、`gt_class`、`domain`、`language`、`source_line`、`token_len`、`group_id`、`view`、`qid`

### 4.4 已知遗留问题

Test 路径没有经过退化样本与 Swap 自洽检查，因此保留了一条两回答完全相同的样本：

- `source_line = 432`，`qid = 412`
- `overall_preference = -1`，`gt_class = A`
- 两个回答均为 `You're welcome! If you have any more questions or need further assistance, feel free to ask.`

这条样本只在 Test，不在 Train。若后续评测使用 Original + Swap 双视图，该样本会因为“同一输入、标签翻转”被强制锁在 50%。

---

## 5. 关于“标签自相矛盾 = 0”的说明

这条规则本身已实现，但由于执行顺序，实际命中数为 0：

- 规则 7（三元组去重）先执行，每个三元组只保留首次出现的一行
- 规则 8 再按三元组查多标签时，已经不存在重复三元组

原始 Train 中确实存在一组冲突标注（同一三元组 7 行，4 行标 A、3 行标 B），但规则 7 已保留首行（label A）并删除其余 6 行。因此该规则在当前顺序下不可能命中。

---

## 6. OOD RewardBench v1（文件 5）

### 6.1 来源与转换

- 源数据集：`allenai/reward-bench`，官方 `filtered` split
- dataset revision：`168d848cdbbea9764fae4a544dc9ca1e6cca4931`
- 官方源码 commit：`59f6ff7005a9f9484727ca27f5383420875e5dbf`
- 源 Parquet SHA-256：`65473c20ed0627e02503557ae102c25c6b0e66b5ed69ee327e224bb95e70ca29`
- 规模：2,985 条，23 个 subset
- A/B 放置：`random.Random(42).getrandbits(1)`，按官方 Parquet 原始行序；`chosen_A = 1495`，`chosen_B = 1490`

### 6.2 “清洗”做了什么

只做了与 HelpSteer3 训练集的**精确归一化重叠审计**：

归一化：Unicode NFC → CRLF/CR 转 LF → 去除首尾空白。

| 重叠范围 | 命中 |
|---|---:|
| Prompt | 0 |
| 有序三元组 `(prompt, chosen, rejected)` | 0 |
| 无序 `prompt + answer pair` | 0 |
| 任意声明范围 | 0 |

因为全部为 0，`ood_clean` 没有删除任何记录，与完整集完全相同（2,985 条）。

**没有执行**的操作：

- 去重（RewardBench 官方 filtered split 本身未做额外去重）
- 去 Tie（该 split 是 chosen/rejected 二分类，没有 Tie 标签）
- 退化样本、Swap 自洽检查（不适用于该基准）

### 6.3 字段

`qid`、`source_dataset`、`source_revision`、`source_row_index`、`source_id`、`source_subset`、`chosen_model`、`rejected_model`、`question_body`、`answer1_body`、`answer2_body`、`gt_class`、`label`（A/B）、`view`（`chosen_A` / `chosen_B`）、`source_chosen`、`source_rejected`、`overlap_prompt_exact_normalized`、`overlap_ordered_triple_exact_normalized`、`overlap_unordered_pair_exact_normalized`、`overlap_any_exact_normalized`

---

## 7. 五个文件之间的关系

```
HelpSteer3 原始 Train (38,459)
        │  清洗：去 Tie / 去泄漏 / 去重 / 去退化 / Swap 自洽
        ▼
HelpSteer3 清洗后 Train (22,190)  ──┐
                                     ├── 训练与同分布测试
HelpSteer3 原始 Test (2,017)         │
        │  清洗：去 Tie / 去重（未做退化与 Swap 检查）
        ▼                            │
HelpSteer3 清洗后 Test (1,866)  ─────┘

RewardBench v1 官方 filtered (2,985)
        │  仅做与 HelpSteer3 Train 的重叠审计（0 条重叠）
        ▼
RewardBench v1 OOD (2,985)  ── 独立 OOD 测试
```

---

## 8. 复现与验证

清洗脚本与报告：

```
/data1/yanghao/paper/JudgeLRM/data/helpsteer3_clean/clean_helpsteer3.py
/data1/yanghao/paper/JudgeLRM/data/helpsteer3_clean/report_schemeC_context.json
/data1/yanghao/paper/JudgeLRM/data/helpsteer3_clean/verify_clean.py
```

RewardBench 转换与审计：

```
/data1/yanghao/paper/JudgeLRM/baselines_ab/data/ood_rewardbench_v1/rewardbench_v1_manifest.json
/data1/yanghao/paper/JudgeLRM/baselines_ab/data/ood_rewardbench_v1/rewardbench_v1_overlap_audit.csv
/data1/yanghao/paper/JudgeLRM/baselines_ab/OOD_REWARDBENCH_REPORT.md
```

行数校验：

```bash
wc -l helpsteer3_train_clean.jsonl helpsteer3_test_clean.jsonl rewardbench_v1_ood_clean.jsonl
gzip -cd helpsteer3_train_raw.jsonl.gz | wc -l   # 38459
gzip -cd helpsteer3_test_raw.jsonl.gz  | wc -l   # 2017
```

完整性校验：

```bash
sha256sum *
```

结果应与 §1 中的 SHA-256 完全一致。
