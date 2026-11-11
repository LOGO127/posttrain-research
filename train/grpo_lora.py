"""GRPO training with LoRA + vLLM colocate on a single GPU.

Smoke test (Qwen3-1.7B, GSM8K, ~100 steps):

    python train/grpo_lora.py \
        --model Qwen/Qwen3-1.7B \
        --dataset data/gsm8k_en_train.jsonl \
        --output_dir runs/smoke_qwen3_1p7b_en \
        --max_steps 100

Multilingual pilot runs: see scripts/pilot_a.sh
Rubric judge runs (Direction B): --reward rubric --judge_model deepseek-chat

Notes:
- TRL API moves fast; if a config field is missing, check
  `python -c "from trl import GRPOConfig; help(GRPOConfig)"` against your version.
- Colocate mode shares the GPU between vLLM (generation) and training.
  On 72GB, vllm_gpu_memory_utilization=0.25 leaves plenty for 1.7B-8B LoRA training.
"""

import argparse
import json
import os
import random

from datasets import Dataset


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def parse_args():
    p = argparse.ArgumentParser()
    # data / model
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", required=True, help="jsonl with 'prompt' (+ 'answer'/'rubric'/'lang')")
    p.add_argument("--eval_dataset", default=None)
    p.add_argument("--output_dir", required=True)
    # reward mix
    p.add_argument("--reward", nargs="+", default=["math"],
                   choices=["math", "format", "lang", "rubric"],
                   help="reward components to enable")
    p.add_argument("--target_lang", default=None, help="for --reward lang, e.g. zh/fr/es")
    p.add_argument("--lang_reward_weight", type=float, default=0.1)
    p.add_argument("--judge_model", default="deepseek-chat")
    p.add_argument("--judge_base_url", default=None)
    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--full_finetune", action="store_true", help="disable LoRA (needs <=4B on 72GB)")
    # GRPO hyper-params
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--prompts_per_step", type=int, default=16,
                   help="effective prompts per update = this value; completions = prompts x num_generations")
    p.add_argument("--max_completion_length", type=int, default=768)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="KL to reference model; 0 = disabled (GRPO standard)")
    p.add_argument("--vllm_gpu_mem", type=float, default=0.25)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_generations", action="store_true",
                   help="dump sampled generations each log step for thinking-language analysis")
    p.add_argument("--no_vllm", action="store_true",
                   help="use HF generate instead of vLLM (CPU smoke tests only, very slow)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    import rewards as R
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    import torch

    # ---------------- dataset ----------------
    train_ds = load_jsonl(args.dataset)
    eval_ds = load_jsonl(args.eval_dataset) if args.eval_dataset else None

    # ---------------- reward functions ----------------
    reward_fns = []
    if "math" in args.reward:
        reward_fns.append(R.math_accuracy)
    if "format" in args.reward:
        reward_fns.append(R.format_reward)
    if "lang" in args.reward:
        assert args.target_lang, "--target_lang required for lang reward"
        reward_fns.append(R.make_language_reward(args.target_lang, args.lang_reward_weight))
    if "rubric" in args.reward:
        reward_fns.append(R.make_rubric_judge_reward(
            model=args.judge_model, base_url=args.judge_base_url))

    # ---------------- LoRA ----------------
    peft_config = None
    if not args.full_finetune:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    # ---------------- GRPO config ----------------
    # In TRL, generation_batch_size = per_device_train_batch_size * grad_accum
    # must be divisible by num_generations.
    completions_per_step = args.prompts_per_step * args.num_generations
    use_cuda = torch.cuda.is_available()
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        per_device_train_batch_size=completions_per_step,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        # vLLM colocate on the single training GPU; --no_vllm falls back to
        # transformers.generate for CPU smoke tests.
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        bf16=use_cuda,
        use_cpu=not use_cuda,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to=[],
        seed=args.seed,
        log_completions=args.save_generations,
        mask_truncated_completions=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fns,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print("done ->", args.output_dir)


if __name__ == "__main__":
    main()
