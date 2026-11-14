"""
Difficulty scores for the comparison methods of Table 4 and Appendix G.5.

All functions return one score per question; higher = predicted harder.
The command-line tool reads a per-question JSONL and writes a score CSV
(`question_id,score[,seed]`) for evaluation/compare_estimators.py.

Methods (and the JSONL fields they read):
  verbalized   1 - verbalised confidence            {"confidence": float}
  entropy      normalised entropy of one model's    {"all_probs": {letter: p}}
               option distribution
  composite    label-free composite D_LF over a     {"per_agent": {name: {"all_probs": ...}}}
               set of agents (homogeneous debate,
               untrained ensemble, or our agents)
  self_consistency
               composite D_LF over three pseudo-    {"samples": ["A", "C", ...]}
               agents, each the empirical answer
               distribution of one third of the
               samples (Appendix G.5 scores self-
               consistency "with the same composite")

MUSE (Kruse et al., 2025) and EIP (Hu et al., 2026) are run with their
official implementations (see README); write their per-question outputs in
the same score-CSV format to include them in Table 4.

Usage:
    python evaluation/baselines.py --method entropy --input single_model.jsonl --out scores/entropy.csv
    python evaluation/baselines.py --method self_consistency --input sc.jsonl --budget 20 --out scores/sc20.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ddd"))

from coevolution import score_questions  # noqa: E402

LETTERS = "ABCDEFGHIJ"
WEIGHTS = {"entropy": 0.45, "js": 0.35, "confidence": 0.15, "correctness": 0.05}


def _vec(probs: Dict[str, float], k: int) -> np.ndarray:
    v = np.asarray([float(probs.get(LETTERS[i], 0.0)) for i in range(k)])
    return v / max(v.sum(), 1e-12)


def verbalized_score(confidence: float) -> float:
    return 1.0 - float(np.clip(confidence, 0.0, 1.0))


def entropy_score(probs: Dict[str, float], k: int = 10) -> float:
    p = _vec(probs, k)
    h = -float(np.sum(p * np.log(p + 1e-12)))
    return h / math.log(k)


def composite_scores(rows: List[Dict], k: int = 10) -> List[float]:
    agents = list(rows[0]["per_agent"])
    questions = [{"question_id": str(r["question_id"]), "options": [""] * k} for r in rows]
    probs = {a: [_vec(r["per_agent"][a]["all_probs"], k) for r in rows] for a in agents}
    return [s.d_label_free for s in score_questions(questions, probs, WEIGHTS)]


def sample_groups(samples: Sequence[str], k: int = 10, groups: int = 3) -> List[np.ndarray]:
    """Split samples into `groups` pseudo-agents with Laplace-smoothed empirical distributions."""
    parts = np.array_split(np.asarray(list(samples)), groups)
    out = []
    for part in parts:
        counts = np.ones(k) * 0.5
        for s in part:
            if s and s[0] in LETTERS[:k]:
                counts[LETTERS.index(s[0])] += 1
        out.append(counts / counts.sum())
    return out


def self_consistency_scores(rows: List[Dict], budget: int, k: int = 10) -> List[float]:
    pseudo = [sample_groups(r["samples"][:budget], k) for r in rows]
    questions = [{"question_id": str(r["question_id"]), "options": [""] * k} for r in rows]
    probs = {f"s{g}": [p[g] for p in pseudo] for g in range(3)}
    return [s.d_label_free for s in score_questions(questions, probs, WEIGHTS)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["verbalized", "entropy", "composite", "self_consistency"])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--num_options", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    k = args.num_options
    if args.method == "verbalized":
        scores = [verbalized_score(r["confidence"]) for r in rows]
    elif args.method == "entropy":
        scores = [entropy_score(r["all_probs"], k) for r in rows]
    elif args.method == "composite":
        scores = composite_scores(rows, k)
    else:
        scores = self_consistency_scores(rows, args.budget, k)
    df = pd.DataFrame({"question_id": [str(r["question_id"]) for r in rows], "score": scores})
    if args.seed is not None:
        df["seed"] = args.seed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} scores to {args.out}")


if __name__ == "__main__":
    main()
