"""
Downstream utility of a difficulty ranking at equal budget (paper Table 5, Appendix I).

Three tasks, each driven only by a per-question score (higher = harder):

triage      Score a model pool on only the top `fraction` of questions by
            score; report the Spearman correlation between the models'
            accuracies on that subset and on the full pool (rho_rank).
selective   Abstain on the highest-scoring (1 - coverage) share of questions;
            report the committee's accuracy on the questions it keeps
            (acc@coverage).
allocation  A fixed budget of extra deliberation units. Concentrated: the top
            `fraction` of questions each get all K units. Uniform: every
            question gets round(fraction * K) units. Report the accuracy gain
            over no extra deliberation (delta_acc). Input is a matrix of
            per-question correctness after k = 0..K extra units.

Usage:
    python evaluation/downstream.py \
        --score "D_LF co-evolved=scores/dlf.csv" --score "Random=random" \
        --pool pool_responses.csv --committee committee_correct.csv \
        --deliberation deliberation.csv --out results/table5.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def top_fraction(scores: np.ndarray, fraction: float) -> np.ndarray:
    n = max(1, int(round(fraction * len(scores))))
    return np.argsort(-scores, kind="stable")[:n]


def triage_rank_preservation(scores: np.ndarray, pool: np.ndarray, fraction: float = 0.2) -> float:
    """pool: (n_questions, n_models) 0/1 correctness."""
    idx = top_fraction(scores, fraction)
    full = np.nanmean(pool, axis=0)
    subset = np.nanmean(pool[idx], axis=0)
    return float(spearmanr(subset, full).statistic)


def selective_accuracy(scores: np.ndarray, committee_correct: np.ndarray, coverage: float = 0.8) -> float:
    n_keep = max(1, int(round(coverage * len(scores))))
    keep = np.argsort(scores, kind="stable")[:n_keep]
    return float(np.mean(committee_correct[keep]))


def allocation_gain(scores: Optional[np.ndarray], deliberation: np.ndarray, fraction: float = 0.2) -> float:
    """deliberation: (n_questions, K+1) correctness after 0..K extra units.
    scores=None means uniform allocation."""
    n, kp1 = deliberation.shape
    k_max = kp1 - 1
    base = deliberation[:, 0].mean()
    if scores is None:
        units = int(round(fraction * k_max))
        return float(deliberation[:, units].mean() - base)
    acc = deliberation[:, 0].copy()
    idx = top_fraction(scores, fraction)
    acc[idx] = deliberation[idx, k_max]
    return float(acc.mean() - base)


def _align(score_df: pd.DataFrame, ids: pd.Series) -> np.ndarray:
    m = dict(zip(score_df["question_id"].astype(str), score_df["score"].astype(float)))
    return np.asarray([m.get(i, np.nan) for i in ids.astype(str)])


def evaluate_all(score_specs: Dict[str, Optional[pd.DataFrame]], pool: pd.DataFrame, committee: pd.DataFrame,
                 deliberation: pd.DataFrame, fraction: float = 0.2, coverage: float = 0.8,
                 n_random: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = pool["question_id"].astype(str)
    pool_m = pool.drop(columns=["question_id"]).to_numpy(float)
    comm = committee.set_index(committee["question_id"].astype(str)).loc[ids, "correct"].to_numpy(float)
    delib = deliberation.set_index(deliberation["question_id"].astype(str)).loc[ids].drop(columns=["question_id"]).to_numpy(float)
    rows = []
    for name, df in score_specs.items():
        if df is None:  # random selection, averaged over draws
            vals = [(triage_rank_preservation(s, pool_m, fraction), selective_accuracy(s, comm, coverage),
                     allocation_gain(s, delib, fraction)) for s in (rng.random(len(ids)) for _ in range(n_random))]
            t, sel, alloc = np.mean(vals, axis=0)
        else:
            s = _align(df, ids)
            s = np.where(np.isnan(s), np.nanmin(s), s)
            t, sel, alloc = (triage_rank_preservation(s, pool_m, fraction), selective_accuracy(s, comm, coverage),
                             allocation_gain(s, delib, fraction))
        rows.append({"strategy": name, "triage_rho_rank": t, f"selective_acc@{coverage}": sel, "allocation_delta_acc": alloc})
    rows.append({"strategy": "Uniform allocation", "triage_rho_rank": np.nan, f"selective_acc@{coverage}": np.nan,
                 "allocation_delta_acc": allocation_gain(None, delib, fraction)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score", action="append", required=True, help='"Name=path.csv" or "Random=random"')
    p.add_argument("--pool", required=True, help="question_id + one 0/1 column per pool model")
    p.add_argument("--committee", required=True, help="question_id,correct for the agent committee")
    p.add_argument("--deliberation", required=True, help="question_id,k0,...,kK correctness after k extra units")
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    specs = {}
    for spec in args.score:
        name, path = spec.split("=", 1)
        specs[name] = None if path == "random" else pd.read_csv(path)
    table = evaluate_all(specs, pd.read_csv(args.pool), pd.read_csv(args.committee), pd.read_csv(args.deliberation),
                         args.fraction, args.coverage)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
