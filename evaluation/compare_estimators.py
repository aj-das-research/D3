"""
Compare difficulty estimators against validated targets (paper Table 4).

Each estimator is a CSV with columns `question_id`, `score` and optionally
`seed`. Higher score = predicted harder. For every estimator and target the
script reports the Spearman correlation on the shared questions, as mean and
one standard deviation over seeds.

Usage:
    python evaluation/compare_estimators.py \
        --targets results/targets.csv \
        --estimator "D_LF co-evolved=results/scores/dlf_coevolved.csv" \
        --estimator "Self-consistency (20)=results/scores/sc20.csv" \
        --out results/table4.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEFAULT_TARGETS = ("one_minus_pass_rate", "irt_b")


def compare(targets: pd.DataFrame, estimators: Dict[str, pd.DataFrame], target_cols=DEFAULT_TARGETS) -> pd.DataFrame:
    targets = targets.assign(question_id=targets["question_id"].astype(str))
    rows = []
    for name, df in estimators.items():
        df = df.assign(question_id=df["question_id"].astype(str))
        if "seed" not in df.columns:
            df = df.assign(seed=0)
        row = {"estimator": name, "n_seeds": df["seed"].nunique()}
        for col in target_cols:
            rhos = []
            for _, part in df.groupby("seed"):
                merged = part.merge(targets[["question_id", col]], on="question_id").dropna()
                if len(merged) > 2:
                    rhos.append(spearmanr(merged["score"], merged[col]).statistic)
            row[f"rho_{col}_mean"] = float(np.mean(rhos)) if rhos else np.nan
            row[f"rho_{col}_sd"] = float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0
            row["n_questions"] = int(len(merged)) if rhos else 0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", required=True)
    p.add_argument("--estimator", action="append", required=True, help='"Name=path/to/scores.csv"')
    p.add_argument("--out", required=True)
    args = p.parse_args()
    estimators = {}
    for spec in args.estimator:
        name, path = spec.split("=", 1)
        estimators[name] = pd.read_csv(path)
    table = compare(pd.read_csv(args.targets), estimators)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
