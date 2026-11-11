# 后训练研究实验设计 —— Pilot 阶段（2026-11-11 → 12-08）

目标：3 周内用 pilot 结果决定投 **ARR 1/4（→ACL 2027）** 还是 **ICML 2027（~1 月下旬）**。

硬件：RTX PRO 5000 Blackwell 72GB（sm120）。API 预算可用（judge/翻译/教师）。

---

## 方向 A：多语言 RLVR（首选，→ ACL）

### 核心研究问题

- **RQ1（迁移）**：单语言 RL（如只用英文数学数据训 GRPO）能否把推理提升迁移到其他语言？反过来，非英文 RL 对英文任务有何影响？
- **RQ2（思考语言动力学）**：模型在非英文任务上用什么语言生成 CoT？RL 训练如何改变"思考语言"的分布？（预期现象：RL 可能把 CoT 压向奖励信号更强的语言 —— 即"思考语言塌缩"）
- **RQ3（reward 设计）**：显式的 thinking-language reward / 语言一致性格式奖励能否控制思考语言、提升目标语言表现？代价是什么？
- **RQ4（LoRA vs 全参）**：LoRA-RL 与全参 RL 在多语言行为上是否一致？（学术界都用 LoRA，若行为不同本身是重要发现）
- **RQ5（低资源）**：低资源语言（sw/th）上的 RLVR 是否因基座能力不足出现零优势组（全对/全错）主导？hint-based 数据能否救？

### Pilot 设计（Week 2-3，目标：看现象是否存在）

| Run | 模型 | 训练数据 | 规模 | 目的 |
|---|---|---|---|---|
| A1 | Qwen3-1.7B (+LoRA r32) | GSM8K-train **英文** | ~300 steps | 英文 RL 基线 |
| A2 | Qwen3-1.7B (+LoRA r32) | GSM8K-train **中文翻译版** | ~300 steps | 非英文 RL |
| A3 | Qwen3-1.7B (+LoRA r32) | 中英 1:1 混合 | ~300 steps | 混合语言 |
| A4（可选） | Llama-3.2-3B (+LoRA) | 英文 | ~200 steps | 家族对照 |

每个 run 评估：
- MGSM 各语言 pass@1（en/zh/fr/es/de，各 250 题 → 用 vLLM 离线批量推理，半小时内）
- **思考语言分布**：对每个 eval prompt 的生成 CoT 做语言检测（`langdetect`），统计 en/zh/other 占比随训练的变化曲线 ← 这是 RQ2 的关键图
- CoT 长度、pass@k（k=8）

**冷启动注意（CPU 冒烟实测发现）**：base 模型在 GSM8K 上初期 `frac_reward_zero_std` 会接近 1（整组全错 → 优势为零 → 无梯度）。对策：① 前 50 步盯 `rewards/math_accuracy` 和 `frac_reward_zero_std`，若 100 步后仍 ≈1 就换更简单的任务（Countdown）或混入简单样本；② 或用 -Instruct 版本起步（但注意：用 Instruct 会引入"已有多语言 RLHF"的混淆变量，论文里需要额外论证——base 模型对 RQ2 思考语言分析更干净）；③ `completions/clipped_ratio` 高说明 `max_completion_length` 太短，按需上调。

**pilot 成功判据**（满足其一即可继续）：
- 英文 RL 后非英文任务 CoT 语言分布显著偏移（塌缩/英文化）→ 反直觉现象成立
- 中文 RL vs 英文 RL 在非英文任务上出现不对称迁移 → 机制问题成立
- A2/A3 在低资源语言上出现零优势主导 → 结合 RQ5 展开

### 完整实验矩阵（12 月，若 pilot 通过）

- **语言**：en, zh, fr, es + 低资源 sw/th（共 5-6）
- **模型**：Qwen3-1.7B / 4B（多语家族）、Llama-3.2-3B（英文中心家族）；有条件加 Gemma-3-4B
- **Arm**：单语 RL × 5 语言 + 混合 RL + 英文→零样本迁移 + thinking-language reward 开/关 + LoRA vs 全参（3B 档）
- **种子**：关键结论 ≥3 seeds
- **奖励变体**：纯答案正确 / +语言一致性格式奖励 / +thinking-language reward（目标语言思考加分）

### 数据清单

| 数据 | 来源 | 用途 |
|---|---|---|
| GSM8K train (7.4k) | openai/gsm8k | 英文训练集 |
| GSM8K 各语言翻译 | 自翻译（API, ~$20-50）或 HF 现有翻译 | 非英文训练集 |
| MGSM | google-research-datasets/mgsm（11 语言 × 250） | 跨语言评测集（不训练，防污染） |
| 多语言 IFEval 式约束 | 自构造 rule-checkable 指令 | 第二类可验证任务 |
| Countdown（TinyZero 式） | 自生成 | 语言无关的 RL 对照沙盒 |

---

## 方向 B：Rubric/Judge 奖励失效模式（次选，→ ICML）

### Pilot A（Day 1，零 GPU）：judge 可 hack 性探测

用 API 对 rubric judge 做对抗测试（`scripts/judge_probe.py`）：
- 对同一 prompt 生成 6 类回答：正常好答案 / 空回答 / 关键词填充 / 谄媚 judge 话术 / 格式伪装（伪 checklist）/ 超长重复
- 看 DeepSeek/GPT judge 按 rubric 打分是否被 hack
- **30 分钟出信号**：若空回答/关键词填充能拿高分 → 方向 B 的现象坐实

### Pilot B（Week 2-3，与方向 A 排队共用 GPU）

| Run | 模型 | 奖励 | 目的 |
|---|---|---|---|
| B1 | Qwen3-1.7B LoRA | rubric judge（API） | 追踪 judge 分 vs 真实质量发散 |
| B2 | Qwen3-1.7B LoRA | rubric judge + checklist 分解 | 分解能否缓解 hacking |

数据：HealthBench 风格 rubric 标注 prompts（健康域），或自动生成 rubric 的开放式指令集（RaR 方法）。
评估：judge 分数曲线 + 每 50 步抽样人工/强 judge 独立评估（不与训练 judge 同模型）→ 画"judge 分↑真实分↓"的发散图。

**pilot 成功判据**：训练 100-200 步内出现 judge 分与真实质量显著发散，且能归因到具体 hack 类型。

---

## 时间线

```
11/11-11/17  环境搭建 + 冒烟 run（Qwen3-1.7B GSM8K GRPO-LoRA 100 步）+ judge probe（零GPU）
11/18-12/01  A1/A2/A3 pilot 顺序跑（每 run ~15-25h）；B1 若排队允许穿插
12/01-12/08  决策点：A 现象成立 → 全力补 A 完整矩阵，冲 ARR 1/4
                    B 发散明显 → 冲 ICML ~1月下旬
                    都不理想 → 转 COLM 3月底/ARR 5月，扩大规模
12/08-12/28  主实验 + 写作
12/29-01/04  ARR 投稿缓冲（ACL 格式 8 页）
```

## 避坑提醒

- 不要再做"GRPO 改一处归一化"——评审疲劳
- 评测集严格隔离：MGSM 只测不训；翻译数据注意数字格式保持
- 关键发现一律 ≥3 seeds + 第二模型家族验证，否则审稿人会以"单家族偶然"拒稿
- 现在就注册 OpenReview 账号（ARR/ICML 都用）
- ARR 投稿需指定有资格的审稿服务人（designated contributor），提前确认
