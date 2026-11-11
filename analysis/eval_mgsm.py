"""Evaluate a checkpoint on multilingual math eval + thinking-language analysis.

Runs vLLM offline generation over per-language eval jsonl files, then reports:
- pass@1 per language
- thinking-language distribution per eval language (RQ2 关键指标)
- mean CoT length

Usage:
    python analysis/eval_mgsm.py \
        --model runs/A2_zh/final \
        --base_model Qwen/Qwen3-1.7B \
        --eval_dir data/mgsm_eval --langs en zh fr es \
        --out results/A2_zh_eval.json
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from rewards import extract_answer, _num_eq, _to_text, norm_lang  # noqa: E402

from langdetect import detect, DetectorFactory  # noqa: E402
DetectorFactory.seed = 0

ANSWER_RE = re.compile(r"####\s*-?[\d,]+(?:\.\d+)?")


def detect_lang(text: str) -> str:
    body = ANSWER_RE.split(text)[0].strip()
    if len(body) < 20:
        return "too_short"
    try:
        return norm_lang(detect(body))
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir (LoRA adapter ok) or base model")
    ap.add_argument("--base_model", default=None, help="required when --model is a LoRA adapter dir")
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--langs", nargs="+", default=["en", "zh"])
    ap.add_argument("--n_samples", type=int, default=1, help="generations per prompt (pass@k if >1)")
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    model_path = args.model
    enable_lora = False
    if args.base_model:
        model_path = args.base_model
        enable_lora = True

    llm = LLM(model=model_path, enable_lora=enable_lora,
              max_lora_rank=64, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048)
    lora_req = None
    if enable_lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, args.model)

    sp = SamplingParams(temperature=args.temperature, top_p=1.0,
                        max_tokens=args.max_tokens, n=args.n_samples, seed=0)
    tok = llm.get_tokenizer()

    results = {}
    for lang in args.langs:
        path = os.path.join(args.eval_dir, f"mgsm_{lang}.jsonl")
        rows = [json.loads(l) for l in open(path) if l.strip()]
        prompts = [
            tok.apply_chat_template(r["prompt"], tokenize=False,
                                    add_generation_prompt=True)
            for r in rows
        ]
        outs = llm.generate(prompts, sp, lora_request=lora_req)

        correct = lang_hist = 0
        cot_lens = []
        think_dist = {}
        samples_dump = []
        for r, o in zip(rows, outs):
            texts = [c.text for c in o.outputs]
            hit = any(
                (p := extract_answer(t)) is not None and _num_eq(p, r["answer"])
                for t in texts
            )
            correct += hit
            dl = detect_lang(texts[0])
            think_dist[dl] = think_dist.get(dl, 0) + 1
            cot_lens.append(len(texts[0].split()))
            samples_dump.append({"q": r["question"], "gold": r["answer"],
                                 "gen": texts[0], "think_lang": dl, "correct": hit})

        n = len(rows)
        results[lang] = {
            "pass": correct / n,
            "think_lang_dist": {k: v / n for k, v in think_dist.items()},
            "mean_cot_words": sum(cot_lens) / n,
        }
        dump_path = args.out.replace(".json", f"_{lang}_samples.jsonl")
        with open(dump_path, "w") as f:
            for s in samples_dump:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[{lang}] pass@1={correct/n:.3f} think_lang={think_dist} "
              f"cot_words={sum(cot_lens)/n:.0f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
