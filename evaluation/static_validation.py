"""
Score the saved static (pre-co-evolution) outputs against validated difficulty
targets built from a public model pool, in the format of Tables 4 and 5.

Estimators (all from the saved per-question agent outputs, no inference):
  D_LF (static)           label-free composite of Eq. 8
  D (static)              full composite of Eq. 7
  Single-model entropy    normalised entropy of Agent-C's option distribution
  Committee confidence    1 - mean agent confidence
  Structural proxy s(q)   Eq. 10
  Question length         word count
Targets: 1 - pass rate and 2PL IRT difficulty over the pool
(data/pool/difficulty_targets.csv). Downstream: triage rank preservation over
the pool (top 20%) and committee accuracy at 80% coverage, with the committee
answer taken as the confidence-weighted vote stored in each row.

Usage:
    python evaluation/static_validation.py --out results/validated_targets
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ddd"))
sys.path.insert(0, str(ROOT / "evaluation"))

from coevolution import score_questions  # noqa: E402
from downstream import selective_accuracy, triage_rank_preservation  # noqa: E402

LETTERS = "ABCDEFGHIJ"
WEIGHTS = {"entropy": 0.45, "js": 0.35, "confidence": 0.15, "correctness": 0.05}
STATIC_FILES = ("heterogeneous_same_family_n50", "heterogeneous_diff_family_n50", "heterogeneous_diff_family_n300")


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def estimators(rows, subset: pd.DataFrame) -> pd.DataFrame:
    agents = ["R", "D", "C"]
    k = len(LETTERS)
    questions = [{"question_id": str(r["question_id"]), "options": [""] * k, "answer": r["correct_answer"]} for r in rows]
    probs = {a: [np.asarray([r["per_agent"][a]["all_probs"].get(L, 0.0) for L in LETTERS]) for r in rows] for a in agents}
    scored = score_questions(questions, probs, WEIGHTS)
    c_probs = [p / p.sum() for p in probs["C"]]
    meta = subset.set_index("question_id")
    ids = [q["question_id"] for q in questions]
    return pd.DataFrame({
        "question_id": ids,
        "D_LF (static)": [s.d_label_free for s in scored],
        "D (static)": [s.d_full for s in scored],
        "Single-model entropy (Agent-C)": [float(-(p * np.log(p + 1e-12)).sum() / math.log(k)) for p in c_probs],
        "Committee confidence": [1.0 - float(np.mean([r["per_agent"][a]["confidence"] for a in agents])) for r in rows],
        "Structural proxy s(q)": [float(meta.loc[i, "difficulty_score"]) for i in ids],
        "Question length": [len(str(meta.loc[i, "question"]).split()) for i in ids],
        "committee_correct": [float(r.get("ensemble_answer") == r["correct_answer"]) for r in rows],
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default=str(ROOT / "experiments" / "static_disagreement" / "outputs"))
    p.add_argument("--pool", default=str(ROOT / "data" / "pool" / "mmlu_pro_hard_pool_responses.csv"))
    p.add_argument("--targets", default=str(ROOT / "data" / "pool" / "difficulty_targets.csv"))
    p.add_argument("--n_random", type=int, default=200)
    p.add_argument("--out", default=str(ROOT / "results" / "validated_targets"))
    args = p.parse_args()

    subset = pd.DataFrame(json.loads((ROOT / "data" / "mmlu_pro_hard_subset.json").read_text()))
    subset["question_id"] = subset["question_id"].astype(str)
    targets = pd.read_csv(args.targets, dtype={"question_id": str})
    pool = pd.read_csv(args.pool, dtype={"question_id": str}).set_index("question_id")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    corr_rows, down_rows = [], []
    for name in STATIC_FILES:
        rows = load_rows(Path(args.outputs_dir) / f"{name}.jsonl")
        est = estimators(rows, subset).merge(targets, on="question_id")
        pool_m = pool.loc[est["question_id"]].to_numpy(float)
        for col in [c for c in est.columns if c not in ("question_id", "committee_correct", *targets.columns)]:
            corr_rows.append({
                "file": name, "n": len(est), "estimator": col,
                "rho_one_minus_pass_rate": spearmanr(est[col], est["one_minus_pass_rate"]).statistic,
                "rho_irt_b": spearmanr(est[col], est["irt_b"]).statistic,
            })
            s = est[col].to_numpy(float)
            down_rows.append({
                "file": name, "n": len(est), "strategy": col,
                "triage_rho_rank": triage_rank_preservation(s, pool_m, 0.2),
                "selective_acc@0.8": selective_accuracy(s, est["committee_correct"].to_numpy(), 0.8),
            })
        rand = [(triage_rank_preservation(r, pool_m, 0.2), selective_accuracy(r, est["committee_correct"].to_numpy(), 0.8))
                for r in (rng.random(len(est)) for _ in range(args.n_random))]
        down_rows.append({"file": name, "n": len(est), "strategy": "Random (mean of draws)",
                          "triage_rho_rank": float(np.mean([x[0] for x in rand])),
                          "selective_acc@0.8": float(np.mean([x[1] for x in rand]))})

    corr = pd.DataFrame(corr_rows)
    down = pd.DataFrame(down_rows)
    corr.to_csv(out / "static_vs_validated_targets.csv", index=False)
    down.to_csv(out / "static_downstream.csv", index=False)
    pd.set_option("display.width", 200)
    print(corr.round(3).to_string(index=False))
    print()
    print(down.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
