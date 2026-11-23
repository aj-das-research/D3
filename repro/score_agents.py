"""Score a split with a set of agent adapters (static / Iteration-0 evaluation, batched).

Reuses the repository's own pieces so every stage shares one code path:
  experiments/coevolution/hf_backend.HFAgentBackend.predict   (prompt, letter tokens, temperature: as in run_heterogeneous.py)
  ddd/coevolution.score_questions / evaluate / predictions_table   (D, D_LF, metrics, per-question rows)

Seeds: for n < split size, each seed draws its own n-question subsample of the split (seeded, without replacement);
for the full split all seeds see the same questions and only the bootstrap sd is informative.

Example:
  python repro/score_agents.py --split test --n 50 --seeds 42 1337 2024 \
     --base_model $QWEN --agent R=$AR --agent D=$AD --agent C=$AC --out $D3_OUT/static/same_family_n50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ddd"))
sys.path.insert(0, str(ROOT / "experiments" / "coevolution"))
sys.path.insert(0, str(ROOT / "experiments" / "static_disagreement"))

from coevolution import evaluate, predictions_table, score_questions  # noqa: E402
from config import load_config, resolve_path  # noqa: E402
from split_guard import select_records  # noqa: E402


def load_split(config, split):
    from run_heterogeneous import load_questions

    data = config["data"]
    questions = load_questions(str(resolve_path(data["questions"])), None)
    rows, _ = select_records(questions, resolve_path(data["split_manifest"]), {split},
                             expected_sha256=data.get("split_manifest_sha256"))
    return rows


def parse_agents(items, base_model, overrides):
    specs = {}
    for item in items:
        name, path = item.split("=", 1)
        specs[name] = {"base": overrides.get(name, base_model), "adapter": path}
    return specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--n", type=int, default=None, help="questions per seed (default: whole split)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    p.add_argument("--base_model", required=True)
    p.add_argument("--agent", action="append", required=True, help="NAME=adapter_dir, in agent order (R, D, C)")
    p.add_argument("--agent_base", action="append", default=[], help="NAME=base_model override (different family)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--config", default=None, help="JSON config layered over configs/paper.json (e.g. configs/repro_m1.json)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    config = load_config(a.config, {})
    overrides = dict(x.split("=", 1) for x in a.agent_base)
    specs = parse_agents(a.agent, a.base_model, overrides)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[score] agents: {json.dumps(specs, indent=1)}", flush=True)

    from hf_backend import HFAgentBackend
    t0 = time.time()
    backend = HFAgentBackend(specs, load_in_4bit=not a.bf16, seed=a.seeds[0])
    print(f"[score] loaded backend in {time.time() - t0:.0f}s; temperatures {backend.temperature}", flush=True)

    questions = load_split(config, a.split)
    n = a.n or len(questions)
    if n > len(questions):
        raise SystemExit(f"[score] --n {n} exceeds split size {len(questions)}")

    # Score every question of the split once per agent (inference is deterministic), then subsample per seed.
    t0 = time.time()
    probs = {ag: backend.predict(ag, questions) for ag in backend.agents}
    print(f"[score] scored {len(questions)} {a.split} questions x {len(backend.agents)} agents in {time.time() - t0:.0f}s",
          flush=True)
    weights = config["score"]["weights"]
    pct = config["score"]["hard_percentile"]
    bins = config["eval"]["ece_bins"]
    all_scored = score_questions(questions, probs, weights, config["score"].get("distribution_term", "js"))
    with (out / f"{a.split}_predictions_all.jsonl").open("w") as f:
        for row in predictions_table(all_scored, backend.agents):
            f.write(json.dumps(row) + "\n")

    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(questions), size=n, replace=False)) if n < len(questions) else np.arange(n)
        scored = [all_scored[i] for i in idx]
        metrics = evaluate(scored, backend.agents, bins, pct)
        # bootstrap sd of rho(D, s) over questions (1,000 resamples)
        d = np.asarray([s.d_full for s in scored]); t = np.asarray([float(s.question["difficulty_score"]) for s in scored])
        from scipy.stats import spearmanr
        boots = [spearmanr(d[b], t[b]).statistic for b in (rng.integers(0, n, n) for _ in range(1000))]
        metrics["rho_D_boot_sd"] = float(np.nanstd(boots))
        metrics.update({"seed": seed, "split": a.split, "question_ids": [s.question["question_id"] for s in scored]})
        sd = out / f"seed{seed}"
        sd.mkdir(exist_ok=True)
        (sd / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with (sd / "predictions.jsonl").open("w") as f:
            for row in predictions_table(scored, backend.agents):
                f.write(json.dumps(row) + "\n")
        print(f"[score] seed {seed}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                   for k, v in metrics.items() if k != "question_ids"}), flush=True)


if __name__ == "__main__":
    main()
