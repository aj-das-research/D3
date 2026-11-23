"""Correlate candidate difficulty scores from a predictions file with s(q) and the validated pool targets.

Input: JSONL rows in the ddd/coevolution.predictions_table layout (per_agent.<A>.all_probs, difficulty_score, D, D_LF).
Use on train / validation predictions for design decisions; test only for the frozen design.

  python repro/signal_probe.py PRED.jsonl [PRED2.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TARGETS = pd.read_csv(ROOT / "data/pool/difficulty_targets.csv", dtype={"question_id": str}).set_index("question_id")


def H(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def candidates(row):
    pa = row["per_agent"]
    agents = list(pa)
    P = {a: np.array(list(pa[a]["all_probs"].values()), dtype=float) for a in agents}
    k = min(len(v) for v in P.values())
    P = {a: v[:k] / v[:k].sum() for a, v in P.items()}
    m = np.mean([P[a] for a in agents], axis=0)
    conf = np.array([P[a].max() for a in agents])
    votes = [int(np.argmax(P[a])) for a in agents]
    out = {"D": row["D"], "D_LF": row["D_LF"], "D_js": row["D_js"], "D_entropy": row["D_entropy"],
           "1-mean_conf": 1 - conf.mean(), "ens_entropy": H(m), "mean_agent_entropy": np.mean([H(P[a]) for a in agents]),
           "mutual_info": H(m) - np.mean([H(P[a]) for a in agents]), "1-ens_maxprob": 1 - m.max(),
           "vote_split": len(set(votes)) / len(agents)}
    for a in agents:
        out[f"entropy_{a}"] = H(P[a])
    return out


def main(paths):
    for path in paths:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        df = pd.DataFrame([{"qid": str(r["question_id"]), "s": r["difficulty_score"], **candidates(r)} for r in rows])
        df = df.join(TARGETS, on="qid")
        print(f"\n== {path}  (n={len(df)}; rho(s, 1-pass)={spearmanr(df.s, df.one_minus_pass_rate).statistic:+.3f})")
        print(f"{'score':20s} {'rho_s':>7s} {'rho_pass':>9s} {'rho_irt':>8s}")
        for c in [c for c in df.columns if c not in ("qid", "s", "one_minus_pass_rate", "irt_b", "irt_a")]:
            r = [spearmanr(df[c], df[t]).statistic for t in ("s", "one_minus_pass_rate", "irt_b")]
            print(f"{c:20s} {r[0]:+7.3f} {r[1]:+9.3f} {r[2]:+8.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
