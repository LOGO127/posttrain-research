"""Dataset preparation for multilingual RLVR experiments.

Subcommands:
    build_gsm8k     GSM8K train -> jsonl {prompt, answer, lang: en}
    translate       translate a gsm8k jsonl into target language via API
    build_mgsm      MGSM -> per-language eval jsonl {prompt, answer, lang}
    build_countdown language-agnostic Countdown task (TinyZero-style)

Usage:
    python data/prepare_data.py build_gsm8k --out data/gsm8k_en_train.jsonl
    python data/prepare_data.py translate --in data/gsm8k_en_train.jsonl \
        --lang zh --out data/gsm8k_zh_train.jsonl --model deepseek-chat
    python data/prepare_data.py build_mgsm --langs en zh fr es --out data/mgsm_eval/
    python data/prepare_data.py build_countdown --n 2000 --out data/countdown_train.jsonl
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

# Per-language instruction suffix appended to the translated question.
# Answer marker "#### N" is kept language-agnostic so the reward parser is shared.
INSTRUCTION = {
    "en": "Solve the problem step by step, then give the final answer as '#### <number>'.",
    "zh": "请一步一步推理，最后以 '#### <数字>' 的格式给出最终答案。",
    "fr": "Résolvez le problème étape par étape, puis donnez la réponse finale sous la forme '#### <nombre>'.",
    "es": "Resuelve el problema paso a paso y da la respuesta final como '#### <número>'.",
    "de": "Löse das Problem Schritt für Schritt und gib die Antwort als '#### <Zahl>' an.",
    "sw": "Tatua tatizo hatua kwa hatua, kisha toa jibu la mwisho kama '#### <namba>'.",
    "th": "แก้โจทย์ทีละขั้นตอน แล้วตอบคำตอบสุดท้ายในรูปแบบ '#### <ตัวเลข>'",
    "ru": "Решите задачу шаг за шагом и дайте окончательный ответ в формате '#### <число>'.",
}


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_prompt(question: str, lang: str):
    instr = INSTRUCTION.get(lang, INSTRUCTION["en"])
    # chat-template format; TRL applies the model's chat template
    return [{"role": "user", "content": f"{question}\n\n{instr}"}]


# ---------------------------------------------------------------- gsm8k

def cmd_build_gsm8k(args):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for ex in ds:
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({
            "prompt": make_prompt(ex["question"], "en"),
            "question": ex["question"],
            "answer": gold,
            "lang": "en",
        })
    write_jsonl(rows, args.out)


# ---------------------------------------------------------------- translate

TRANSLATE_SYS = (
    "Translate the following grade-school math word problem into {lang_name}. "
    "Keep all numbers exactly as numerals (do not translate digits into words). "
    "Keep currency/units unchanged. Output ONLY the translated problem text."
)
LANG_NAME = {"zh": "Simplified Chinese", "fr": "French", "es": "Spanish",
             "de": "German", "sw": "Swahili", "th": "Thai", "ru": "Russian"}


def cmd_translate(args):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=args.base_url or os.environ.get("JUDGE_BASE_URL"),
    )
    rows = load_jsonl(getattr(args, "in"))
    lang = args.lang

    def work(row):
        src_q = row["question"]
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": TRANSLATE_SYS.format(lang_name=LANG_NAME[lang])},
                    {"role": "user", "content": src_q},
                ],
                temperature=0.0,
            )
            q = r.choices[0].message.content.strip()
            nums_src = re.findall(r"\d[\d,]*\.?\d*", src_q)
            nums_tgt = re.findall(r"\d[\d,]*\.?\d*", q)
            if nums_src != nums_tgt:
                q = src_q  # keep English if numbers were mangled
                ok = False
            else:
                ok = True
        except Exception:
            q, ok = src_q, False
        return {
            "prompt": make_prompt(q, lang),
            "question": q,
            "answer": row["answer"],
            "lang": lang,
            "translation_ok": ok,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(work, rows))
    bad = sum(1 for r in out if not r["translation_ok"])
    print(f"translation fallback (numbers mismatch/API error): {bad}/{len(out)}")
    write_jsonl(out, args.out)


# ---------------------------------------------------------------- mgsm

def cmd_build_mgsm(args):
    """MGSM eval set. HF dataset 'juletxara/mgsm' has per-language configs."""
    from datasets import load_dataset
    os.makedirs(args.out, exist_ok=True)
    for lang in args.langs:
        try:
            ds = load_dataset("juletxara/mgsm", lang, split="test")
        except Exception as e:
            print(f"[warn] mgsm/{lang}: {e}")
            continue
        rows = []
        for ex in ds:
            ans = str(ex.get("answer_number", ex.get("answer", ""))).replace(",", "").strip()
            q = ex["question"]
            rows.append({"prompt": make_prompt(q, lang), "question": q,
                         "answer": ans, "lang": lang})
        write_jsonl(rows, os.path.join(args.out, f"mgsm_{lang}.jsonl"))


# ---------------------------------------------------------------- countdown

def cmd_build_countdown(args):
    """Countdown/24-game task: reach target using given numbers (TinyZero-style)."""
    import random
    rng = random.Random(args.seed)
    rows = []
    while len(rows) < args.n:
        nums = [rng.randint(1, 12) for _ in range(3)]
        ops = rng.choices("+-*", k=2)
        # construct a reachable target: ((a op1 b) op2 c) evaluated left to right
        try:
            v = eval(f"({nums[0]}{ops[0]}{nums[1]}){ops[1]}{nums[2]}")
        except ZeroDivisionError:
            continue
        if not (1 <= v <= 100):
            continue
        q = (f"Using the numbers {nums}, create an equation that equals {v}. "
             "You may use +, -, *, and each number at most once. "
             "Think step by step, then give your equation as '#### <equation>'.")
        rows.append({
            "prompt": [{"role": "user", "content": q}],
            "question": q,
            "answer": str(v),
            "numbers": nums,
            "lang": "agnostic",
        })
    write_jsonl(rows, args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    s = sub.add_parser("build_gsm8k")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_gsm8k)

    s = sub.add_parser("translate")
    s.add_argument("--in", dest="in", required=True)
    s.add_argument("--lang", required=True, choices=list(LANG_NAME))
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="deepseek-chat")
    s.add_argument("--base_url", default=None)
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(fn=cmd_translate)

    s = sub.add_parser("build_mgsm")
    s.add_argument("--langs", nargs="+", default=["en", "zh", "fr", "es"])
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_mgsm)

    s = sub.add_parser("build_countdown")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_build_countdown)

    args = ap.parse_args()
    args.fn(args)
