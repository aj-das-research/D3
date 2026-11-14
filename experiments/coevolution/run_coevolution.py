"""
Run diversity-regularised co-evolution (paper Section 3.6, Table 2, Figure 2)
and its ablations (Tables 8-9, Appendix G.7) on GPU.

Example (different-family setting, one seed):
    python experiments/coevolution/run_coevolution.py \
        --base_model /path/to/Qwen2.5-3B-Instruct \
        --agent_d_base_model /path/to/Mistral-7B-Instruct-v0.3 \
        --agent_r /path/to/agent_r --agent_d /path/to/agent_d_mistral --agent_c /path/to/agent_c \
        --seed 42 --output_dir outputs/coevolution/diff_family/seed42

Ablations:
    --no_regularizer            lambda_div = 0 (Table 8)
    --agents R,C                drop an agent (Table 9)
    --buffer_size 1000          buffer sweep (Appendix G.7)
    --hard_percentile 90        threshold sweep (Appendix G.7)

Questions are read through the split manifest: buffering and updates use the
training split, the per-iteration metrics use the validation split, and the
test split is scored once at the end only with --eval_test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ddd"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coevolution import evaluate, predictions_table, run_coevolution, score_questions  # noqa: E402
from config import load_config, resolve_path  # noqa: E402
from split_guard import select_records  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="JSON config layered over configs/paper.json")
    p.add_argument("--base_model", required=True, help="Base model for Agents R and C (and D unless overridden)")
    p.add_argument("--agent_d_base_model", default=None, help="Separate base for Agent-D (different-family)")
    p.add_argument("--agent_r", required=True)
    p.add_argument("--agent_d", required=True)
    p.add_argument("--agent_c", required=True)
    p.add_argument("--agents", default="R,D,C", help="Comma-separated subset, e.g. R,C for Table 9")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_regularizer", action="store_true")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--buffer_size", type=int, default=None)
    p.add_argument("--hard_percentile", type=float, default=None)
    p.add_argument("--max_train", type=int, default=None, help="Subsample the training split (debugging)")
    p.add_argument("--max_val", type=int, default=None)
    p.add_argument("--eval_test", action="store_true", help="Score the sealed test split once at the end")
    p.add_argument("--bf16", action="store_true", help="Load base weights in bf16 instead of 4-bit")
    p.add_argument("--save_adapters", action="store_true")
    return p.parse_args()


def load_split(config, split, limit=None):
    from run_heterogeneous import load_questions  # noqa: WPS433

    data = config["data"]
    questions = load_questions(str(resolve_path(data["questions"])), None)
    rows, _ = select_records(questions, resolve_path(data["split_manifest"]), {split},
                             expected_sha256=data.get("split_manifest_sha256"))
    return rows[:limit] if limit else rows


def main():
    args = parse_args()
    overrides = {"coevolution": {}, "buffer": {}, "score": {}}
    if args.iterations is not None:
        overrides["coevolution"]["iterations"] = args.iterations
    if args.buffer_size is not None:
        overrides["buffer"]["max_size"] = args.buffer_size
    if args.hard_percentile is not None:
        overrides["score"]["hard_percentile"] = args.hard_percentile
    config = load_config(args.config, overrides)

    random.seed(args.seed)
    np.random.seed(args.seed)

    from hf_backend import HFAgentBackend  # imported late: needs torch/peft

    paths = {"R": args.agent_r, "D": args.agent_d, "C": args.agent_c}
    specs = {}
    for a in [x.strip() for x in args.agents.split(",") if x.strip()]:
        base = args.agent_d_base_model if (a == "D" and args.agent_d_base_model) else args.base_model
        specs[a] = {"base": base, "adapter": paths[a]}

    co = config["coevolution"]
    backend = HFAgentBackend(
        specs, learning_rate=co["learning_rate"], micro_batch_size=co["micro_batch_size"],
        lambda_ece=co["lambda_ece"], load_in_4bit=not args.bf16, seed=args.seed,
    )
    train = load_split(config, "train", args.max_train)
    val = load_split(config, "validation", args.max_val)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"args": vars(args), "config": config}, indent=2))

    run_coevolution(backend, train, val, config, seed=args.seed, output_dir=out,
                    regularize=not args.no_regularizer)

    if args.save_adapters:
        for a in backend.agents:
            backend.save(a, out / "adapters" / f"agent_{a}")

    if args.eval_test:
        test = load_split(config, "test")
        probs = {a: backend.predict(a, test) for a in backend.agents}
        scored = score_questions(test, probs, config["score"]["weights"])
        metrics = evaluate(scored, backend.agents, config["eval"]["ece_bins"], config["score"]["hard_percentile"])
        (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        with (out / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
            for row in predictions_table(scored, backend.agents):
                f.write(json.dumps(row) + "\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
