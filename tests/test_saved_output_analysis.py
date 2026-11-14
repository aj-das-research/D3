"""Tests for canonical saved-output rescoring."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from saved_output_analysis import (
    CanonicalAgentOutput,
    extract_outputs,
    output_distribution,
    score_components,
)


class LocalSavedOutputAnalysisTests(unittest.TestCase):
    def test_extracts_per_agent_schema(self):
        row = {
            "per_agent": {
                "R": {"answer": "A", "confidence": 0.8},
                "D": {"answer": "B", "confidence": 0.7},
                "C": {"answer": "C", "confidence": 0.6},
            }
        }
        outputs = extract_outputs(row)
        self.assertIsNotNone(outputs)
        self.assertEqual([output.name for output in outputs], ["R", "D", "C"])

    def test_extracts_per_role_schema(self):
        row = {
            "per_role": {
                "careful_reasoner": {
                    "agent_letter": "R",
                    "answer": "A",
                    "confidence": 0.8,
                },
                "domain_expert": {
                    "agent_letter": "D",
                    "answer": "B",
                    "confidence": 0.7,
                },
                "uncertain_analyst": {
                    "agent_letter": "C",
                    "answer": "C",
                    "confidence": 0.6,
                },
            }
        }
        outputs = extract_outputs(row)
        self.assertIsNotNone(outputs)
        self.assertEqual([output.answer for output in outputs], ["A", "B", "C"])

    def test_reconstructs_over_ten_mmlu_options(self):
        output = CanonicalAgentOutput("R", "J", 0.55)
        distribution = output_distribution(output)
        self.assertEqual(len(distribution), 10)
        self.assertTrue(math.isclose(float(distribution.sum()), 1.0))
        self.assertTrue(math.isclose(float(distribution[9]), 0.55))

    def test_no_correctness_score_does_not_change_with_gold(self):
        common = {
            "per_agent": {
                "R": {"answer": "A", "confidence": 0.8},
                "D": {"answer": "B", "confidence": 0.7},
                "C": {"answer": "A", "confidence": 0.6},
            }
        }
        first = score_components({**common, "correct_answer": "A"})
        second = score_components({**common, "correct_answer": "C"})
        self.assertNotEqual(first["D_full"], second["D_full"])
        self.assertEqual(
            first["D_no_correctness_paper"],
            second["D_no_correctness_paper"],
        )


if __name__ == "__main__":
    unittest.main()
