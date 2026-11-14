"""
Sample reasoning paths from one model for the self-consistency baseline and
the sample-budget sweep (paper Section 6.5, Appendix G.5).

For each question the model generates `--num_samples` step-by-step answers at
temperature `--temperature`; the final letter of each is parsed and stored.
evaluation/baselines.py then scores any prefix of the samples
(budgets 5, 10, 20, 40, 100).

Usage:
    python experiments/self_consistency/sample_answers.py \
        --model /path/to/Qwen2.5-3B-Instruct --split test --num_samples 100 \
        --out outputs/self_consistency/qwen3b_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ddd"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "static_disagreement"))

from config import load_config, resolve_path  # noqa: E402
from run_heterogeneous import format_options, load_questions  # noqa: E402
from split_guard import select_records  # noqa: E402

ANSWER_RE = re.compile(r"answer\s*(?:is|:)?\s*\(?([A-J])\)?", re.IGNORECASE)


def parse_letter(text: str, k: int) -> str:
    matches = ANSWER_RE.findall(text)
    if matches and matches[-1].upper() in "ABCDEFGHIJ"[:k]:
        return matches[-1].upper()
    letters = re.findall(r"\b([A-J])\b", text)
    return letters[-1] if letters and letters[-1] in "ABCDEFGHIJ"[:k] else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--chunk", type=int, default=10, help="Samples generated per forward call")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config()
    qs = load_questions(str(resolve_path(cfg["data"]["questions"])), None)
    qs, _ = select_records(qs, resolve_path(cfg["data"]["split_manifest"]), {args.split},
                           expected_sha256=cfg["data"]["split_manifest_sha256"])
    qs = qs[: args.limit] if args.limit else qs

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto",
                                                 trust_remote_code=True).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["question_id"] for line in out.read_text().splitlines() if line.strip()}

    with out.open("a", encoding="utf-8") as f:
        for q in qs:
            if q["question_id"] in done:
                continue
            msg = [{"role": "user", "content": (
                f"Question:\n{q['question']}\n\nOptions:\n{format_options(q['options'])}\n\n"
                "Think step by step, then finish with 'The answer is X' where X is the option letter.")}]
            prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            samples = []
            while len(samples) < args.num_samples:
                n = min(args.chunk, args.num_samples - len(samples))
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                         max_new_tokens=args.max_new_tokens, num_return_sequences=n,
                                         pad_token_id=tok.eos_token_id)
                for seq in gen:
                    text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    samples.append(parse_letter(text, len(q["options"])))
            f.write(json.dumps({"question_id": q["question_id"], "samples": samples,
                                "correct_answer": q["answer"]}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
