"""
Validated difficulty targets from a model-pool response matrix
(paper Section 7, Table 4; Appendix I.5).

Input: a CSV with a `question_id` column and one 0/1 column per model
(1 = the model answered the question correctly).

Targets written:
  one_minus_pass_rate  1 - (fraction of pool models that are correct)
  irt_b                2PL IRT difficulty b_i, fitted jointly with model
                       abilities theta_m and discriminations a_i:
                           P(correct) = sigmoid(a_i * (theta_m - b_i))
                       (maximum likelihood with weak Gaussian priors;
                        abilities standardised to mean 0, sd 1)
  irt_a                2PL discrimination

Usage:
    python evaluation/difficulty_targets.py --responses pool_responses.csv --out targets.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def pass_rate_difficulty(responses: np.ndarray) -> np.ndarray:
    """responses: (n_questions, n_models) in {0, 1}; NaN allowed for missing."""
    return 1.0 - np.nanmean(responses, axis=1)


def fit_2pl(responses: np.ndarray, prior_sd: float = 3.0, max_iter: int = 2000):
    """Joint maximum-a-posteriori 2PL fit. Returns (b, a, theta)."""
    r = np.asarray(responses, dtype=float)
    observed = ~np.isnan(r)
    y = np.nan_to_num(r)
    n_q, n_m = r.shape

    def unpack(x):
        theta = x[:n_m]
        b = x[n_m : n_m + n_q]
        log_a = x[n_m + n_q :]
        return theta, b, log_a

    def objective(x):
        theta, b, log_a = unpack(x)
        a = np.exp(log_a)
        logit = a[:, None] * (theta[None, :] - b[:, None])
        p = expit(logit)
        resid = (p - y) * observed
        nll = -np.sum(observed * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        nll += 0.5 * np.sum(theta**2) + 0.5 * np.sum(b**2) / prior_sd**2 + 0.5 * np.sum(log_a**2)
        g_logit = resid
        g_theta = np.sum(g_logit * a[:, None], axis=0) + theta
        g_b = -np.sum(g_logit * a[:, None], axis=1) + b / prior_sd**2
        g_log_a = np.sum(g_logit * (theta[None, :] - b[:, None]), axis=1) * a + log_a
        return nll, np.concatenate([g_theta, g_b, g_log_a])

    pr = np.clip(np.nanmean(r, axis=1), 0.02, 0.98)
    x0 = np.concatenate([np.zeros(n_m), -np.log(pr / (1 - pr)), np.zeros(n_q)])
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    theta, b, log_a = unpack(res.x)
    mu, sd = theta.mean(), theta.std() or 1.0
    a = np.exp(log_a)
    # standardise abilities; keep the logits unchanged
    return (b - mu) / sd, a * sd, (theta - mu) / sd


def build_targets(responses_csv: str) -> pd.DataFrame:
    df = pd.read_csv(responses_csv)
    ids = df["question_id"].astype(str)
    mat = df.drop(columns=["question_id"]).to_numpy(dtype=float)
    b, a, _ = fit_2pl(mat)
    return pd.DataFrame({
        "question_id": ids,
        "one_minus_pass_rate": pass_rate_difficulty(mat),
        "irt_b": b,
        "irt_a": a,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = build_targets(args.responses)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} targets to {args.out}")


if __name__ == "__main__":
    main()
