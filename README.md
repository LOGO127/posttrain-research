# posttrain-research

单卡（RTX PRO 5000 72GB）多语言 RLVR / rubric-reward 研究 pipeline。

目标：冲 **ARR 2027-01-04（→ACL 2027）** 或 **ICML 2027（~1 月下旬）**。
实验设计与决策规则见 `docs/experiment_matrix.md`。

## 结构

```
docs/experiment_matrix.md   # 方向A/B 的 RQ、pilot 设计、成功判据、时间线
train/grpo_lora.py          # GRPO + LoRA + vLLM colocate 训练脚本（TRL）
train/rewards.py            # reward：数学答案/格式/语言/rubric-judge/惩罚项
data/prepare_data.py        # GSM8K 构建、API 翻译、MGSM 评测集、Countdown
analysis/eval_mgsm.py       # 多语言评测 + 思考语言分布分析（vLLM 离线推理）
scripts/judge_probe.py      # 方向B零GPU探测：rubric judge 能否被 hack
scripts/pilot_a.sh          # 方向A pilot 一键三连跑
requirements.txt
```

## 服务器环境搭建（Blackwell sm120 注意）

```bash
# Python 3.11+，建议新建 venv/conda 环境
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 验证 GPU 识别（应显示 sm_120 / Blackwell）
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

# flash-attn 若编译失败可跳过（attention_implementation 默认 sdpa 也能跑）
```

## 快速开始

```bash
# 1) 数据（英文 GSM8K 训练集 + MGSM 评测集）
python data/prepare_data.py build_gsm8k --out data/gsm8k_en_train.jsonl
python data/prepare_data.py build_mgsm --langs en zh fr es --out data/mgsm_eval/

# 2) 翻译训练集（需要 API；DeepSeek 约 $3-5 / 7.4k 条）
export JUDGE_API_KEY=sk-...          # 或 OPENAI_API_KEY
export JUDGE_BASE_URL=https://api.deepseek.com   # 按需
python data/prepare_data.py translate --in data/gsm8k_en_train.jsonl \
    --lang zh --out data/gsm8k_zh_train.jsonl
# 混合集：cat 后 shuffle
cat data/gsm8k_en_train.jsonl data/gsm8k_zh_train.jsonl | shuf \
    > data/gsm8k_zhen_mix_train.jsonl

# 3) 零GPU探测（方向B，先跑这个拿信号）
python scripts/judge_probe.py --judge_model deepseek-chat

# 4) 冒烟测试（先确认 pipeline 通）
python train/grpo_lora.py --model Qwen/Qwen3-1.7B \
    --dataset data/gsm8k_en_train.jsonl \
    --output_dir runs/smoke --max_steps 50 --save_generations

# 5) pilot 三连跑（顺序执行，~2-3 天）
bash scripts/pilot_a.sh
```

## 显存/速度参考（72GB）

| 配置 | 预期 |
|---|---|
| 1.7B + LoRA-GRPO | 很舒适，可加大 num_generations/prompts_per_step |
| 3-4B + 全参 GRPO | 可行（`--full_finetune`，开 8bit optim 更稳） |
| 7-8B + LoRA-GRPO | 可行，vllm_gpu_mem 降到 ~0.2 |
| 14B + QLoRA | 需加量化加载逻辑，未默认开启 |

## 已验证（2026-11-11 本机 CPU 冒烟通过）

- `build_gsm8k` → 7473 行；`build_mgsm` → 4 语言 × 250 行
- reward 函数单测通过（数学提取/格式/语言检测；`langdetect` 返回 `zh-cn` 已归一化为 `zh`）
- `grpo_lora.py` 在 trl 1.17.1 + transformers 5.x 下 2 步端到端跑通、LoRA adapter 正常保存
- 本机无 GPU 也可冒烟：加 `--no_vllm`（HF generate 回退，极慢，仅供验证）
- TRL 1.17 已移除 `max_prompt_length` 字段（脚本已适配）

## 注意
- MGSM 只做评测，**绝不进入训练数据**（防污染，审稿人必查）
- 翻译数据做了数字一致性校验，数字被翻坏的样本自动回退英文并打 `translation_ok=false` 标记
- judge probe / rubric reward 走 OpenAI 兼容 API；DeepSeek 用 `JUDGE_BASE_URL=https://api.deepseek.com`，模型 `deepseek-chat`
