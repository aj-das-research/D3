"""Tests for validated targets, estimator comparison, baselines and downstream tasks."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "ddd"))

from baselines import entropy_score, sample_groups, self_consistency_scores, verbalized_score  # noqa: E402
from compare_estimators import compare  # noqa: E402
from difficulty_targets import fit_2pl, pass_rate_difficulty  # noqa: E402
from downstream import allocation_gain, selective_accuracy, triage_rank_preservation  # noqa: E402


def synthetic_pool(n_q=300, n_m=25, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 1, n_q)
    a = np.exp(rng.normal(0, 0.3, n_q))
    theta = rng.normal(0, 1, n_m)
    p = expit(a[:, None] * (theta[None, :] - b[:, None]))
    return (rng.random((n_q, n_m)) < p).astype(float), b, theta


class TargetTests(unittest.TestCase):
    def test_pass_rate(self):
        r = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
        np.testing.assert_allclose(pass_rate_difficulty(r), [0.5, 1.0])

    def test_2pl_recovers_difficulty(self):
        r, b_true, theta_true = synthetic_pool(n_m=100)
        b, a, theta = fit_2pl(r)
        self.assertGreater(spearmanr(b, b_true).statistic, 0.93)
        self.assertGreater(spearmanr(theta, theta_true).statistic, 0.9)
        self.assertAlmostEqual(float(theta.mean()), 0.0, places=6)

    def test_2pl_handles_missing(self):
        r, b_true, _ = synthetic_pool(n_m=100, seed=1)
        r[::7, 3] = np.nan
        b, _, _ = fit_2pl(r)
        self.assertGreater(spearmanr(b, b_true).statistic, 0.9)


class CompareTests(unittest.TestCase):
    def test_seed_mean_and_sd(self):
        ids = [str(i) for i in range(50)]
        target = pd.DataFrame({"question_id": ids, "one_minus_pass_rate": np.arange(50), "irt_b": np.arange(50)})
        perfect = pd.concat([pd.DataFrame({"question_id": ids, "score": np.arange(50), "seed": s}) for s in (1, 2)])
        table = compare(target, {"perfect": perfect})
        self.assertAlmostEqual(table.loc[0, "rho_one_minus_pass_rate_mean"], 1.0)
        self.assertEqual(table.loc[0, "n_seeds"], 2)


class BaselineTests(unittest.TestCase):
    def test_simple_scores(self):
        self.assertAlmostEqual(verbalized_score(0.8), 0.2)
        self.assertAlmostEqual(entropy_score({"A": 1.0}, 4), 0.0, places=6)
        self.assertAlmostEqual(entropy_score({c: 0.25 for c in "ABCD"}, 4), 1.0, places=6)

    def test_self_consistency_orders_agreement(self):
        rows = [
            {"question_id": "easy", "samples": ["A"] * 21},
            {"question_id": "hard", "samples": list("ABCDEFGHIJ" * 3)[:21]},
        ]
        easy, hard = self_consistency_scores(rows, budget=21)
        self.assertGreater(hard, easy)
        groups = sample_groups(["A"] * 9, k=4)
        self.assertEqual(len(groups), 3)
        for g in groups:
            self.assertAlmostEqual(float(g.sum()), 1.0)


class DownstreamTests(unittest.TestCase):
    def test_triage_with_oracle_score(self):
        r, b_true, _ = synthetic_pool(seed=2)
        oracle = triage_rank_preservation(b_true, r, 0.2)
        rand = np.mean([triage_rank_preservation(np.random.default_rng(i).random(len(b_true)), r, 0.2) for i in range(30)])
        self.assertGreater(oracle, 0.5)
        self.assertGreater(oracle, rand)

    def test_selective_abstains_on_high_scores(self):
        correct = np.array([1, 1, 1, 1, 0], dtype=float)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        self.assertEqual(selective_accuracy(scores, correct, coverage=0.8), 1.0)

    def test_allocation_concentrated_vs_uniform(self):
        # Extra deliberation helps only on the 2 hardest of 10 questions.
        delib = np.zeros((10, 6))
        delib[:8, :] = 1
        delib[8:, 5] = 1
        scores = np.arange(10, dtype=float)
        self.assertAlmostEqual(allocation_gain(scores, delib, 0.2), 0.2)
        self.assertAlmostEqual(allocation_gain(None, delib, 0.2), 0.0)


if __name__ == "__main__":
    unittest.main()


class PoolMatrixTests(unittest.TestCase):
    def test_reads_archives_and_drops_low_coverage(self):
        import json as _json
        import tempfile
        import zipfile
        from build_pool_matrix import build_matrix, model_name

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            recs = [{"question_id": "1", "answer": "A", "pred": "A"},
                    {"question_id": "2", "answer": "B", "pred": None},
                    {"question_id": "3", "answer": "C", "pred": "D"}]
            with zipfile.ZipFile(tmp / "model_outputs_good_5shots.zip", "w") as z:
                z.writestr("eval_results/math_result.json", _json.dumps(recs))
                z.writestr("eval_results/summary.json", _json.dumps({"x": 1}))
                z.writestr("__MACOSX/eval_results/._math_result.json", "junk")
            with zipfile.ZipFile(tmp / "model_outputs_sparse_5shots.json.zip", "w") as z:
                z.writestr("m.json", _json.dumps(recs[:1]))
            df = build_matrix(tmp, ["1", "2", "3"], min_coverage=0.9)
            self.assertEqual(list(df.columns), ["question_id", "good_5shots"])
            self.assertEqual(df["good_5shots"].tolist(), [1, 0, 0])
            self.assertEqual(model_name(Path("model_outputs_x_5shots.json.zip")), "x_5shots")
