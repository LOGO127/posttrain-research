#!/usr/bin/env bash
# Direction A pilot: multilingual RLVR runs (sequential on single GPU)
# Expected: ~15-25h per run at 1.7B + LoRA on RTX PRO 5000 72GB
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-1.7B}

# ---------- 0. smoke test (run first, ~2h) ----------
# python train/grpo_lora.py \
#     --model $MODEL \
#     --dataset data/gsm8k_en_train.jsonl \
#     --output_dir runs/smoke \
#     --max_steps 100 --save_generations

# ---------- A1: English RL baseline ----------
python train/grpo_lora.py \
    --model $MODEL \
    --dataset data/gsm8k_en_train.jsonl \
    --output_dir runs/A1_en \
    --reward math format \
    --max_steps 300 --save_generations

# ---------- A2: Chinese RL ----------
python train/grpo_lora.py \
    --model $MODEL \
    --dataset data/gsm8k_zh_train.jsonl \
    --output_dir runs/A2_zh \
    --reward math format \
    --max_steps 300 --save_generations

# ---------- A3: zh+en mixed ----------
python train/grpo_lora.py \
    --model $MODEL \
    --dataset data/gsm8k_zhen_mix_train.jsonl \
    --output_dir runs/A3_mix \
    --reward math format \
    --max_steps 300 --save_generations

# ---------- eval each run on all MGSM langs ----------
for run in A1_en A2_zh A3_mix; do
    python analysis/eval_mgsm.py \
        --model runs/$run/final --base_model $MODEL \
        --eval_dir data/mgsm_eval --langs en zh fr es \
        --out results/${run}_eval.json
done
