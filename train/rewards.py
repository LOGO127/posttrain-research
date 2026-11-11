"""Reward functions for GRPO training.

TRL GRPOTrainer reward function signature:
    fn(prompts, completions, completion_ids=None, **dataset_columns) -> list[float]

`completions` are strings for standard prompts, or [{"role": "assistant", "content": ...}]
when the dataset "prompt" column is a chat-template list. We normalize both cases.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- helpers

def _to_text(completion) -> str:
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion)
    return str(completion)


ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str):
    """Extract final numeric answer: #### N > \\boxed{N} > last number."""
    m = ANSWER_RE.findall(text)
    if m:
        return m[-1].replace(",", "")
    m = BOXED_RE.findall(text)
    if m:
        nums = NUMBER_RE.findall(m[-1])
        if nums:
            return nums[0].replace(",", "")
    nums = NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _num_eq(a, b, tol=1e-4) -> bool:
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- rewards

def math_accuracy(prompts, completions, answer=None, **kwargs):
    """1.0 if extracted numeric answer matches gold `answer` column, else 0."""
    out = []
    for comp, gold in zip(completions, answer or [None] * len(completions)):
        pred = extract_answer(_to_text(comp))
        out.append(1.0 if pred is not None and _num_eq(pred, gold) else 0.0)
    return out


def format_reward(prompts, completions, weight: float = 0.2, **kwargs):
    """Small bonus for emitting an explicit final-answer marker (#### or \\boxed)."""
    out = []
    for comp in completions:
        t = _to_text(comp)
        out.append(weight if (ANSWER_RE.search(t) or BOXED_RE.search(t)) else 0.0)
    return out


# langdetect emits "zh-cn"/"zh-tw" for Chinese; normalize to ISO-639-1 style.
LANG_ALIASES = {"zh-cn": "zh", "zh-tw": "zh", "zh": "zh"}


def norm_lang(code: str) -> str:
    return LANG_ALIASES.get(code, code)


def make_language_reward(target_lang: str, weight: float = 0.1, only_body: bool = True):
    """Reward completions whose *reasoning body* is in `target_lang`.

    For RQ3 (thinking-language reward). `only_body` strips the final-answer
    line so digits/markers don't bias detection.
    """
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0

    def reward(prompts, completions, **kwargs):
        out = []
        for comp in completions:
            t = _to_text(comp)
            if only_body:
                t = ANSWER_RE.split(t)[0]
            try:
                out.append(weight if norm_lang(detect(t)) == target_lang else 0.0)
            except Exception:
                out.append(0.0)
        return out

    reward.__name__ = f"lang_reward_{target_lang}"
    return reward


def make_rubric_judge_reward(
    model: str = "deepseek-chat",
    base_url: str = None,
    weight: float = 1.0,
    max_workers: int = 16,
):
    """LLM-judge rubric reward (Direction B).

    Dataset must provide a `rubric` column (checklist text). The judge scores
    each completion 0-10 against the rubric; score/10 * weight is the reward.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=base_url or os.environ.get("JUDGE_BASE_URL"),
    )

    def _score(prompt, completion, rubric):
        ptxt = prompt if isinstance(prompt, str) else "\n".join(
            m.get("content", "") for m in prompt)
        msg = (
            "You are a strict grader. Score the response 0-10 against each "
            "rubric item, then output only the TOTAL as an integer.\n\n"
            f"TASK:\n{ptxt}\n\nRUBRIC:\n{rubric}\n\n"
            f"RESPONSE:\n{_to_text(completion)}\n\nTOTAL (0-10):"
        )
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": msg}],
                temperature=0.0,
                max_tokens=8,
            )
            nums = NUMBER_RE.findall(r.choices[0].message.content or "")
            return min(max(float(nums[0]), 0.0), 10.0) / 10.0 * weight if nums else 0.0
        except Exception:
            return 0.0

    def reward(prompts, completions, rubric=None, **kwargs):
        rubrics = rubric or [""] * len(completions)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_score, prompts, completions, rubrics))

    reward.__name__ = f"rubric_judge_{model}"
    return reward


def make_penalty_reward(patterns, penalty: float = -0.5):
    """Negative reward for hack patterns (Direction B analysis).

    patterns: list of compiled regex or plain strings matched case-insensitively.
    """
    compiled = [p if hasattr(p, "search") else re.compile(p, re.I) for p in patterns]

    def reward(prompts, completions, **kwargs):
        return [
            penalty if any(p.search(_to_text(c)) for p in compiled) else 0.0
            for c in completions
        ]

    return reward
