"""Reason-then-answer agent scoring (method change M1 candidate).

Each agent (LoRA adapter on its base) first generates a short chain of thought for the question, then the option
distribution is read from the next-token logits over the option letters after the prompt
"...<reasoning>\nFinal answer: ". With --samples > 1 each agent draws several reasoning paths (temperature
--temp) and its distribution is the mean of the per-path letter distributions.

Outputs one JSONL row per question with each agent's distribution, written in the same `per_agent.all_probs` layout
as ddd/coevolution.predictions_table, plus metrics from ddd/coevolution.evaluate.

Only the train / validation splits should be used for design decisions; test is scored once per frozen design.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ddd"))
sys.path.insert(0, str(ROOT / "experiments" / "coevolution"))
sys.path.insert(0, str(ROOT / "experiments" / "static_disagreement"))
sys.path.insert(0, str(ROOT / "repro"))

from coevolution import evaluate, predictions_table, score_questions  # noqa: E402
from config import load_config  # noqa: E402
from run_heterogeneous import AGENT_SYSTEMS, format_options, letter_token_ids  # noqa: E402
from score_agents import load_split, parse_agents  # noqa: E402

LETTERS = "ABCDEFGHIJ"


def cot_messages(agent: str, q: dict) -> list[dict]:
    user = (
        "Answer this multiple-choice question.\n\n"
        f"Question:\n{q['question']}\n\n"
        f"Options:\n{format_options(q['options'])}\n\n"
        "Think step by step briefly (at most a few sentences or short calculations), "
        "then finish with a line of the form 'Final answer: <letter>'."
    )
    return [{"role": "system", "content": AGENT_SYSTEMS[agent]}, {"role": "user", "content": user}]


@torch.no_grad()
def cot_probs(model, tok, agent: str, questions: list[dict], temperature: float, samples: int, max_new: int,
              batch: int, max_length: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for start in range(0, len(questions), batch):
        chunk = questions[start:start + batch]
        prompts = [tok.apply_chat_template(cot_messages(agent, q), tokenize=False, add_generation_prompt=True)
                   for q in chunk]
        acc = [np.zeros(len(q["options"])) for q in chunk]
        for s in range(samples):
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            # cut each trace before any answer statement, then ask for the letter
            follow = []
            for p, t in zip(prompts, texts):
                t = t.split("Final answer")[0].rstrip()
                follow.append(p + t + "\nFinal answer: ")
            enc2 = tok(follow, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_length + max_new).to(model.device)
            logits = model(**enc2, logits_to_keep=1).logits[:, -1, :].float()
            k = max(len(q["options"]) for q in chunk)
            ids = letter_token_ids(tok, k)
            choice = logits[:, ids]
            for i, q in enumerate(chunk):
                acc[i] += F.softmax(choice[i, : len(q["options"])], dim=-1).cpu().numpy()
        out.extend(a / samples for a in acc)
        print(f"[cot] {agent}: {min(start + batch, len(questions))}/{len(questions)}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_model", required=True)
    p.add_argument("--agent", action="append", required=True)
    p.add_argument("--agent_base", action="append", default=[])
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    config = load_config(None, {})
    specs = parse_agents(a.agent, a.base_model, dict(x.split("=", 1) for x in a.agent_base))
    from hf_backend import HFAgentBackend
    backend = HFAgentBackend(specs, seed=a.seed)
    qs = load_split(config, a.split)
    if a.n and a.n < len(qs):
        rng = np.random.default_rng(a.seed)
        qs = [qs[i] for i in np.sort(rng.choice(len(qs), a.n, replace=False))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    probs = {}
    for ag in backend.agents:
        model, tok = backend._activate(ag, train=False)
        probs[ag] = cot_probs(model, tok, ag, qs, a.temp, a.samples, a.max_new, a.batch, backend.max_length)
        # agent temperature (Agent-C calibration) applied on the log scale, as in predict()
        T = max(backend.temperature[ag], 1e-6)
        if T != 1.0:
            probs[ag] = [np.exp(np.log(np.clip(v, 1e-12, 1)) / T) / np.exp(np.log(np.clip(v, 1e-12, 1)) / T).sum()
                         for v in probs[ag]]
    print(f"[cot] scored {len(qs)} questions in {time.time() - t0:.0f}s", flush=True)
    scored = score_questions(qs, probs, config["score"]["weights"])
    metrics = evaluate(scored, backend.agents, config["eval"]["ece_bins"], config["score"]["hard_percentile"])
    metrics.update(vars(a))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "predictions.jsonl").open("w") as f:
        for row in predictions_table(scored, backend.agents):
            f.write(json.dumps(row) + "\n")
    print("[cot] " + json.dumps({k: v for k, v in metrics.items() if isinstance(v, float)}), flush=True)


if __name__ == "__main__":
    main()
