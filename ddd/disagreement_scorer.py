"""
Composite disagreement score D(q) (paper Section 3.4, Eqs. 3-8).

Multiple-choice disagreement scorer:
  D_answer       = normalized entropy over chosen answers
  D_distribution = mean pairwise Jensen-Shannon divergence over answer distributions
  D_confidence   = normalized variance(confidence)
  D_correctness  = optional correctness-weighted disagreement signal
  D(q)           = weighted mixture of the above

For agents that only return a selected answer and scalar confidence, we build a
small option distribution by placing confidence mass on the selected answer and
spreading the remaining mass across the other options. If logits/probabilities
are available, pass them through AgentOutput.probs.

The label-free variant D_LF (Eq. 8) is obtained with correctness_weight=0.0;
the remaining weights are renormalised by their sum.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AgentOutput:
    """Output from one agent on one multiple-choice question."""

    agent_name: str
    answer: str
    confidence: float
    probs: Optional[Dict[str, float]] = None


@dataclass
class DisagreementResult:
    """Full scored result for one question."""

    question_id: str
    question: str
    agent_outputs: List[AgentOutput]
    d_answer: float
    d_confidence: float
    d_score: float
    is_hard_case: bool
    d_distribution: float = 0.0
    d_correctness: float = 0.0

    @property
    def d_semantic(self) -> float:
        """Backward-compatible alias for the answer-entropy component."""
        return self.d_answer


class DisagreementScorer:
    """Compute D(q) from answer entropy, JS divergence, confidence, and correctness."""

    def __init__(
        self,
        threshold: float = 0.5,
        adaptive_percentile: Optional[float] = None,
        answer_weight: float = 0.45,
        distribution_weight: float = 0.35,
        confidence_weight: float = 0.15,
        correctness_weight: float = 0.05,
        num_options: int = 4,
        epsilon: float = 1e-12,
    ):
        self.threshold = threshold
        self.adaptive_percentile = adaptive_percentile
        self.answer_weight = answer_weight
        self.distribution_weight = distribution_weight
        self.confidence_weight = confidence_weight
        self.correctness_weight = correctness_weight
        self.num_options = num_options
        self.epsilon = epsilon

        total = (
            answer_weight
            + distribution_weight
            + confidence_weight
            + correctness_weight
        )
        if total <= 0:
            raise ValueError("At least one disagreement weight must be positive.")
        self._weight_total = total

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        text = str(answer).strip().upper()
        if not text:
            return ""
        if text[0].isalpha():
            return text[0]
        return text

    def _answer_disagreement(self, answers: List[str]) -> float:
        """Backward-compatible binary disagreement."""
        normalized = [self._normalize_answer(answer) for answer in answers]
        return 0.0 if len(set(normalized)) == 1 else 1.0

    def _answer_entropy(self, answers: List[str]) -> float:
        normalized = [self._normalize_answer(answer) for answer in answers]
        counts = np.asarray(
            [normalized.count(answer) for answer in sorted(set(normalized))],
            dtype=float,
        )
        probs = counts / max(float(len(normalized)), 1.0)
        entropy = -float(np.sum(probs * np.log(probs + self.epsilon)))
        max_entropy = math.log(min(len(normalized), self.num_options))
        if max_entropy <= 0:
            return 0.0
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def _confidence_variance(self, confidences: List[float]) -> float:
        clipped = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
        var = float(np.var(clipped))
        n = len(clipped)
        # Three agents: the paper's 1/3 normaliser (Eq. 5). Other committee
        # sizes (agent-removal ablation): the largest population variance of
        # n values in [0, 1].
        max_var = 1.0 / 3.0 if n == 3 else (n // 2) * (n - n // 2) / float(n * n)
        return float(np.clip(var / max_var, 0.0, 1.0))

    def _option_labels(self, outputs: List[AgentOutput]) -> List[str]:
        labels = set()
        for output in outputs:
            labels.add(self._normalize_answer(output.answer))
            if output.probs:
                labels.update(self._normalize_answer(option) for option in output.probs)

        canonical = [chr(ord("A") + i) for i in range(self.num_options)]
        labels.update(canonical)
        return sorted(label for label in labels if label)

    def _distribution_for_output(
        self,
        output: AgentOutput,
        labels: List[str],
    ) -> np.ndarray:
        if output.probs:
            raw = np.asarray(
                [
                    max(float(output.probs.get(label, output.probs.get(label.lower(), 0.0))), 0.0)
                    for label in labels
                ],
                dtype=float,
            )
            total = float(np.sum(raw))
            if total > 0:
                return raw / total

        chosen = self._normalize_answer(output.answer)
        confidence = float(np.clip(output.confidence, 0.0, 1.0))
        dist = np.zeros(len(labels), dtype=float)
        if chosen in labels:
            chosen_idx = labels.index(chosen)
            dist[chosen_idx] = confidence
            other_count = max(len(labels) - 1, 1)
            spill = (1.0 - confidence) / other_count
            for idx in range(len(labels)):
                if idx != chosen_idx:
                    dist[idx] = spill
        else:
            dist[:] = 1.0 / len(labels)
        return dist / max(float(np.sum(dist)), self.epsilon)

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log((p + self.epsilon) / (m + self.epsilon))))
        kl_qm = float(np.sum(q * np.log((q + self.epsilon) / (m + self.epsilon))))
        js = 0.5 * (kl_pm + kl_qm)
        return float(np.clip(js / math.log(2), 0.0, 1.0))

    def _distribution_disagreement(self, outputs: List[AgentOutput]) -> float:
        labels = self._option_labels(outputs)
        distributions = [self._distribution_for_output(output, labels) for output in outputs]
        divergences = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                divergences.append(self._js_divergence(distributions[i], distributions[j]))
        if not divergences:
            return 0.0
        return float(np.clip(np.mean(divergences), 0.0, 1.0))

    def _correctness_disagreement(
        self,
        outputs: List[AgentOutput],
        correct_answer: Optional[str],
    ) -> float:
        if correct_answer is None:
            return 0.0
        correct = self._normalize_answer(correct_answer)
        correctness = [
            1.0 if self._normalize_answer(output.answer) == correct else 0.0
            for output in outputs
        ]
        return 0.0 if len(set(correctness)) == 1 else 1.0

    def score(
        self,
        question_id: str,
        question: str,
        outputs: List[AgentOutput],
        correct_answer: Optional[str] = None,
    ) -> DisagreementResult:
        if len(outputs) < 2:
            raise ValueError("Need at least 2 agent outputs.")

        answers = [o.answer for o in outputs]
        confidences = [o.confidence for o in outputs]

        d_answer = self._answer_entropy(answers)
        d_distribution = self._distribution_disagreement(outputs)
        d_conf = self._confidence_variance(confidences)
        d_correctness = self._correctness_disagreement(outputs, correct_answer)
        d_q = round(
            (
                self.answer_weight * d_answer
                + self.distribution_weight * d_distribution
                + self.confidence_weight * d_conf
                + self.correctness_weight * d_correctness
            )
            / self._weight_total,
            4,
        )

        return DisagreementResult(
            question_id=question_id,
            question=question,
            agent_outputs=outputs,
            d_answer=round(d_answer, 4),
            d_confidence=round(d_conf, 4),
            d_score=d_q,
            is_hard_case=d_q > self.threshold,
            d_distribution=round(d_distribution, 4),
            d_correctness=round(d_correctness, 4),
        )

    def score_batch(
        self,
        questions: List[Dict],
        outputs_per_question: List[List[AgentOutput]],
    ) -> List[DisagreementResult]:
        results = [
            self.score(
                q["question_id"],
                q["question"],
                outputs,
                correct_answer=q.get("answer") or q.get("correct_answer"),
            )
            for q, outputs in zip(questions, outputs_per_question)
        ]
        if self.adaptive_percentile is not None and results:
            scores = np.asarray([result.d_score for result in results], dtype=float)
            adaptive_threshold = float(np.percentile(scores, self.adaptive_percentile))
            for result in results:
                result.is_hard_case = result.d_score >= adaptive_threshold
        return results


class HardCaseBuffer:
    """Minimal rolling FIFO buffer for standalone use of the scorer."""

    def __init__(self, max_size: int = 500):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, result: DisagreementResult):
        if result.is_hard_case:
            self.buffer.append(result)

    def top_k(self, k: int = 200) -> List[DisagreementResult]:
        return sorted(self.buffer, key=lambda x: x.d_score, reverse=True)[:k]

    def __len__(self):
        return len(self.buffer)


def run_unit_tests() -> bool:
    scorer = DisagreementScorer(threshold=0.5)
    results = []

    print("=" * 60)
    print("DISAGREEMENT SCORER SELF-TEST")
    print("=" * 60)

    r1 = scorer.score("t1", "same answer, same confidence", [
        AgentOutput("R", "A", 0.9),
        AgentOutput("D", "A", 0.9),
        AgentOutput("C", "A", 0.9),
    ])
    ok1 = r1.d_answer == 0.0 and r1.d_confidence == 0.0 and r1.d_distribution == 0.0 and r1.d_score == 0.0
    print(f"Test 1 identical outputs: D_answer={r1.d_answer}, D_js={r1.d_distribution}, D_conf={r1.d_confidence}, D={r1.d_score}")
    results.append(("Identical outputs", ok1))

    r2 = scorer.score("t2", "answer disagreement", [
        AgentOutput("R", "A", 0.8),
        AgentOutput("D", "B", 0.8),
        AgentOutput("C", "A", 0.8),
    ])
    ok2 = 0.55 < r2.d_answer < 0.60 and r2.d_confidence == 0.0 and 0.25 < r2.d_score < 0.45
    print(f"Test 2 answer disagreement: D_answer={r2.d_answer}, D_js={r2.d_distribution}, D_conf={r2.d_confidence}, D={r2.d_score}")
    results.append(("Answer disagreement", ok2))

    r3 = scorer.score("t3", "confidence disagreement", [
        AgentOutput("R", "C", 0.9),
        AgentOutput("D", "C", 0.5),
        AgentOutput("C", "C", 0.1),
    ])
    ok3 = r3.d_answer == 0.0 and 0.01 < r3.d_score < 0.20
    print(f"Test 3 confidence variance: D_answer={r3.d_answer}, D_js={r3.d_distribution}, D_conf={r3.d_confidence}, D={r3.d_score}")
    results.append(("Confidence variance", ok3))

    r4 = scorer.score("t4", "combined disagreement", [
        AgentOutput("R", "A. Paris", 0.95),
        AgentOutput("D", "B. London", 0.50),
        AgentOutput("C", "C. Rome", 0.05),
    ])
    ok4 = r4.d_answer > 0.99 and r4.d_score > 0.5 and r4.is_hard_case
    print(f"Test 4 combined signal: D_answer={r4.d_answer}, D_js={r4.d_distribution}, D_conf={r4.d_confidence}, D={r4.d_score}")
    results.append(("Combined signal", ok4))

    print("\n" + "=" * 60)
    passed = sum(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} tests passed.")
    return passed == len(results)


if __name__ == "__main__":
    all_passed = run_unit_tests()
    raise SystemExit(0 if all_passed else 1)
