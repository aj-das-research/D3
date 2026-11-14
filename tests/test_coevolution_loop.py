import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ddd"))

from coevolution_loop import build_iteration_batches, rebuild_buffer  # noqa: E402
from diversity_regularizer import DiversityRegularizer  # noqa: E402


class CoevolutionLoopTests(unittest.TestCase):
    def test_rebuild_buffer_and_write_batches(self):
        rows = [
            {
                "question_id": "q1",
                "question": "Pick A.",
                "correct_answer": "A",
                "options": ["alpha", "beta", "gamma"],
                "D_q": 0.8,
                "D_semantic": 0.7,
                "D_confidence": 0.2,
                "is_hard_case": True,
                "per_agent": {
                    "R": {"answer": "A", "option_text": "alpha", "confidence": 0.9, "all_probs": {"A": 0.8, "B": 0.1, "C": 0.1}},
                    "D": {"answer": "B", "option_text": "beta", "confidence": 0.2, "all_probs": {"A": 0.2, "B": 0.7, "C": 0.1}},
                    "C": {"answer": "C", "option_text": "gamma", "confidence": 0.85, "all_probs": {"A": 0.2, "B": 0.2, "C": 0.6}},
                },
            }
        ]
        args = SimpleNamespace(buffer_size=500, sample_size=200, hard_threshold=0.5)
        buffer = rebuild_buffer(rows, args)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(len(buffer.sample_for_agent_R()), 1)
        self.assertEqual(len(buffer.sample_for_agent_C()), 1)

        with tempfile.TemporaryDirectory() as tmp:
            paths = build_iteration_batches(rows, buffer, 0, Path(tmp), DiversityRegularizer())
            self.assertTrue(paths["R"].exists())
            self.assertTrue(paths["D"].exists())
            self.assertTrue(paths["C"].exists())


if __name__ == "__main__":
    unittest.main()
