"""Closed-loop co-evolution on a small NumPy backend (no model downloads)."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ddd"))

from coevolution import (  # noqa: E402
    RollingBuffer,
    run_coevolution,
    sample_slice,
    sampling_weights,
    score_questions,
)
from config import load_config  # noqa: E402

LETTERS = "ABCD"


class LogitTableBackend:
    """Each agent holds a free logit vector per question, updated by gradient descent."""

    agents = ("R", "D", "C")

    def __init__(self, questions, seed=0, lr=0.5):
        rng = np.random.default_rng(seed)
        self.index = {q["question_id"]: i for i, q in enumerate(questions)}
        self.logits = {a: rng.normal(0, 1.5, size=(len(questions), 4)) for a in self.agents}
        self.lr = lr

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict(self, agent, questions):
        return [self._softmax(self.logits[agent][self.index[q["question_id"]]]) for q in questions]

    def _loss(self, z, gold, peers, lambda_div, tau):
        p = self._softmax(z)
        loss = -math.log(p[gold] + 1e-12)
        if lambda_div > 0 and peers:
            dists = [p] + list(peers)
            js = []
            for i in range(len(dists)):
                for j in range(i + 1, len(dists)):
                    m = 0.5 * (dists[i] + dists[j])
                    kl = lambda x: float(np.sum(x * (np.log(x + 1e-12) - np.log(m + 1e-12))))
                    js.append(0.5 * (kl(dists[i]) + kl(dists[j])) / math.log(2))
            loss += lambda_div * max(0.0, tau - float(np.mean(js)))
        return loss

    def update(self, agent, questions, peer_probs, steps, lambda_div, tau_js):
        for _ in range(steps):
            for q, peers in zip(questions, peer_probs):
                i = self.index[q["question_id"]]
                gold = LETTERS.index(q["answer"])
                z = self.logits[agent][i]
                grad = np.zeros_like(z)
                for k in range(len(z)):
                    dz = np.zeros_like(z)
                    dz[k] = 1e-4
                    grad[k] = (
                        self._loss(z + dz, gold, list(peers.values()), lambda_div, tau_js)
                        - self._loss(z - dz, gold, list(peers.values()), lambda_div, tau_js)
                    ) / 2e-4
                self.logits[agent][i] = z - self.lr * grad
        return {"steps": steps, "n": len(questions)}


def make_questions(n, seed=0):
    rng = np.random.default_rng(seed)
    cats = ["math", "biology", "law", "health"]
    return [
        {
            "question_id": f"q{i}",
            "question": "",
            "options": ["a", "b", "c", "d"],
            "answer": LETTERS[rng.integers(4)],
            "category": cats[i % 4],
            "difficulty_score": float(rng.random()),
        }
        for i in range(n)
    ]


def small_config(iterations=6):
    return load_config(overrides={
        "coevolution": {"iterations": iterations, "train_batch_size": 60, "steps_per_agent": 3,
                        "slice_size": 20, "tau_js": 0.30, "lambda_div": 4.0},
        "buffer": {"max_size": 50, "exposed_top_k": 30},
    })


class ClosedLoopTests(unittest.TestCase):
    def test_paper_config_values(self):
        cfg = load_config()
        self.assertEqual(cfg["coevolution"]["iterations"], 30)
        self.assertEqual(cfg["coevolution"]["steps_per_agent"], 50)
        self.assertEqual(cfg["coevolution"]["tau_js"], 0.30)
        self.assertEqual(cfg["coevolution"]["lambda_div"], 1.0)
        self.assertEqual(cfg["buffer"]["max_size"], 500)
        self.assertEqual(cfg["buffer"]["exposed_top_k"], 200)

    def test_buffer_is_fifo_and_exposes_top_k(self):
        qs = make_questions(40)
        backend = LogitTableBackend(qs)
        scored = score_questions(qs, {a: backend.predict(a, qs) for a in backend.agents},
                                 load_config()["score"]["weights"])
        buf = RollingBuffer(max_size=10, exposed_top_k=4)
        buf.add_hard(scored, 50, 1)
        self.assertEqual(len(buf), 10)
        exposed = buf.exposed()
        self.assertEqual(len(exposed), 4)
        self.assertEqual([s.d_full for s in exposed], sorted([s.d_full for s in exposed], reverse=True))

    def test_sampling_weights_never_use_target(self):
        qs = make_questions(20)
        backend = LogitTableBackend(qs)
        cfg = load_config()
        scored = score_questions(qs, {a: backend.predict(a, qs) for a in backend.agents}, cfg["score"]["weights"])
        before = {a: sampling_weights(a, scored, cfg["buffer"]) for a in "RDC"}
        for s in scored:
            s.question["difficulty_score"] = 123.0
        after = {a: sampling_weights(a, scored, cfg["buffer"]) for a in "RDC"}
        for a in "RDC":
            np.testing.assert_array_equal(before[a], after[a])
        chosen = sample_slice(before["R"], 5, np.random.default_rng(0))
        self.assertTrue(all(before["R"][i] > 0 for i in chosen))

    def test_loop_writes_logs(self):
        train, val = make_questions(120, 1), make_questions(40, 2)
        for q in val:
            q["question_id"] = "v" + q["question_id"]
        backend = LogitTableBackend(train + val, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            hist = run_coevolution(backend, train, val, small_config(), seed=0, output_dir=Path(tmp))
            self.assertTrue((Path(tmp) / "iterations.jsonl").exists())
            self.assertTrue((Path(tmp) / "val_predictions.jsonl").exists())
        self.assertEqual([h["iteration"] for h in hist], list(range(7)))
        for key in ("acc_R", "ece_C", "mean_js", "rho_D", "rho_D_LF", "hard_rate"):
            self.assertIn(key, hist[-1])

    def test_regulariser_preserves_divergence(self):
        # Every question is trained every iteration, so shared gold answers
        # push the agents together; the hinge must keep them further apart.
        qs = make_questions(30, 1)
        cfg = load_config(overrides={
            "coevolution": {"iterations": 8, "train_batch_size": 30, "steps_per_agent": 5,
                            "slice_size": 30, "tau_js": 0.30, "lambda_div": 4.0},
            "buffer": {"max_size": 30, "exposed_top_k": 30},
            "score": {"hard_percentile": 0},
        })
        final = {}
        for regularize in (True, False):
            backend = LogitTableBackend(qs, seed=3)
            with tempfile.TemporaryDirectory() as tmp:
                hist = run_coevolution(backend, qs, qs, cfg, seed=0, output_dir=Path(tmp), regularize=regularize)
            self.assertGreater(hist[-1]["acc_R"], hist[0]["acc_R"])
            final[regularize] = hist[-1]["mean_js"]
        self.assertGreater(final[True], 1.5 * final[False])


if __name__ == "__main__":
    unittest.main()
