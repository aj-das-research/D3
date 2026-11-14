import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ddd"))

from diversity_regularizer import DiversityRegularizer, js_divergence, normalize_probs  # noqa: E402


class DiversityRegularizerTests(unittest.TestCase):
    def test_normalize_probs_handles_zero_mass(self):
        self.assertEqual(normalize_probs([0, 0]), [0.5, 0.5])

    def test_js_is_zero_for_identical_distributions(self):
        self.assertAlmostEqual(js_divergence([0.8, 0.2], [0.8, 0.2]), 0.0, places=8)

    def test_penalty_positive_for_collapsed_agents(self):
        reg = DiversityRegularizer(weight=0.1, min_js_divergence=0.03)
        score = reg.score_probs(
            {
                "R": [0.7, 0.2, 0.1],
                "D": [0.7, 0.2, 0.1],
                "C": [0.69, 0.21, 0.10],
            }
        )
        self.assertTrue(score["collapsed"])
        self.assertGreater(score["penalty"], 0.0)

    def test_penalty_zero_for_diverse_agents(self):
        reg = DiversityRegularizer(weight=0.1, min_js_divergence=0.03)
        score = reg.score_probs(
            {
                "R": [0.95, 0.03, 0.02],
                "D": [0.02, 0.95, 0.03],
                "C": [0.03, 0.02, 0.95],
            }
        )
        self.assertFalse(score["collapsed"])
        self.assertEqual(score["penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
