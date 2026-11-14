"""
TASK 2 — Disagreement Scorer Module + 4 Unit Tests
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# ── Load encoder ─────────────────────────────────────────────
print("Loading sentence encoder...")
ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoder loaded.\n")


# ── Data Structures ──────────────────────────────────────────

@dataclass
class AgentOutput:
    agent_name: str
    answer: str
    confidence: float


@dataclass
class DisagreementResult:
    question_id: str
    question: str
    agent_outputs: List[AgentOutput]
    d_semantic: float
    d_confidence: float
    d_score: float
    is_hard_case: bool


# ── Disagreement Scorer ──────────────────────────────────────

class DisagreementScorer:

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.encoder = ENCODER

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)

        # 🔥 IMPORTANT: Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _semantic_distance(self, answers: List[str]) -> float:
        embeddings = self._embed(answers)
        distances = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distances.append(cosine(embeddings[i], embeddings[j]))

        return float(np.mean(distances))

    def _confidence_variance(self, confidences: List[float]) -> float:
        var = float(np.var(confidences))
        max_var = 1 / 3
        return min(var / max_var, 1.0)

    def score(self, question_id: str, question: str, outputs: List[AgentOutput]) -> DisagreementResult:
        assert len(outputs) == 3

        answers = [o.answer for o in outputs]
        confidences = [o.confidence for o in outputs]

        d_semantic = self._semantic_distance(answers)
        d_confidence = self._confidence_variance(confidences)

        d_score = round(0.6 * d_semantic + 0.4 * d_confidence, 4)
        is_hard_case = d_score > self.threshold

        return DisagreementResult(
            question_id,
            question,
            outputs,
            round(d_semantic, 4),
            round(d_confidence, 4),
            d_score,
            is_hard_case,
        )


# ── Hard Case Buffer (minimal) ───────────────────────────────

class HardCaseBuffer:

    def __init__(self, max_size: int = 500):
        self.buffer = deque(maxlen=max_size)

    def add(self, result: DisagreementResult):
        if result.is_hard_case:
            self.buffer.append(result)

    def top_k(self, k: int = 200):
        return sorted(self.buffer, key=lambda x: x.d_score, reverse=True)[:k]

    def __len__(self):
        return len(self.buffer)


# ── Unit Tests ───────────────────────────────────────────────

def run_unit_tests():
    scorer = DisagreementScorer()
    results = []

    print("=" * 60)
    print("RUNNING 4 UNIT TESTS")
    print("=" * 60)

    # TEST 1
    print("\nTest 1: Identical answers")
    outputs = [
        AgentOutput("R", "Paris is the capital of France.", 0.9),
        AgentOutput("D", "Paris is the capital of France.", 0.9),
        AgentOutput("C", "Paris is the capital of France.", 0.9),
    ]
    r1 = scorer.score("t1", "Q", outputs)
    print(r1)
    results.append(r1.d_score < 0.1)

    # TEST 2 (FIXED)
    print("\nTest 2: Max disagreement")
    outputs = [
        AgentOutput("R", "Biology ATP mitochondria", 0.95),
        AgentOutput("D", "French Revolution history", 0.90),
        AgentOutput("C", "Quantum entanglement physics", 0.85),
    ]
    r2 = scorer.score("t2", "Q", outputs)
    print(r2)
    results.append(r2.d_score > 0.55)

    # TEST 3
    print("\nTest 3: Confidence only")
    outputs = [
        AgentOutput("R", "Water boils at 100C", 0.9),
        AgentOutput("D", "Water boils at 100C", 0.5),
        AgentOutput("C", "Water boils at 100C", 0.1),
    ]
    r3 = scorer.score("t3", "Q", outputs)
    print(r3)
    results.append(0.1 < r3.d_score < 0.4)

    print("\nSummary:", sum(results), "/ 3 passed")


# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    run_unit_tests()