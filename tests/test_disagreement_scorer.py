"""Unit tests for ddd/disagreement_scorer.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ddd"))

from disagreement_scorer import AgentOutput, DisagreementScorer  # noqa: E402


def test_identical_outputs_score_zero():
    scorer = DisagreementScorer()
    result = scorer.score("q1", "Question?", [
        AgentOutput("R", "A", 0.9),
        AgentOutput("D", "A", 0.9),
        AgentOutput("C", "A", 0.9),
    ])
    assert result.d_answer == 0.0
    assert result.d_confidence == 0.0
    assert result.d_score == 0.0


def test_partial_answer_disagreement_uses_normalized_entropy():
    scorer = DisagreementScorer()
    result = scorer.score("q2", "Question?", [
        AgentOutput("R", "A", 0.8),
        AgentOutput("D", "B", 0.8),
        AgentOutput("C", "A", 0.8),
    ])
    assert 0.57 < result.d_answer < 0.59
    assert 0.35 < result.d_distribution < 0.36
    assert result.d_confidence == 0.0
    assert 0.38 < result.d_score < 0.39


def test_confidence_variance_component_is_normalized():
    scorer = DisagreementScorer()
    result = scorer.score("q3", "Question?", [
        AgentOutput("R", "C", 0.9),
        AgentOutput("D", "C", 0.5),
        AgentOutput("C", "C", 0.1),
    ])
    assert result.d_answer == 0.0
    assert 0.27 < result.d_distribution < 0.28
    assert 0.31 < result.d_confidence < 0.33
    assert 0.14 < result.d_score < 0.15


def test_combined_answer_and_confidence_disagreement_flags_hard_case():
    scorer = DisagreementScorer(threshold=0.5)
    result = scorer.score("q4", "Question?", [
        AgentOutput("R", "A. Paris", 0.95),
        AgentOutput("D", "B. London", 0.50),
        AgentOutput("C", "C. Rome", 0.05),
    ])
    assert result.d_answer == 1.0
    assert result.d_score > 0.6
    assert result.is_hard_case


if __name__ == "__main__":
    test_identical_outputs_score_zero()
    test_partial_answer_disagreement_uses_normalized_entropy()
    test_confidence_variance_component_is_normalized()
    test_combined_answer_and_confidence_disagreement_flags_hard_case()
    print("4/4 tests passed.")
