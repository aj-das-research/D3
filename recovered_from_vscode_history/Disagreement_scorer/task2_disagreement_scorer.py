"""
TASK 2 — Disagreement Scorer Module + 4 Unit Tests (FIXED)
Run in: VS Code / local
Install first: pip install sentence-transformers numpy scipy torch
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

# ── Load encoder ──────────────────────────────────────────────────────────────
print("Loading sentence encoder...")
ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoder loaded.\n")


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class AgentOutput:
    """Output from a single agent for one question."""
    agent_name: str      # "R", "D", or "C"
    answer:     str      # The answer text
    confidence: float    # Float in [0, 1]


@dataclass
class DisagreementResult:
    """Full scored result for one question."""
    question_id:   str
    question:      str
    agent_outputs: List[AgentOutput]
    d_semantic:    float   # Semantic distance component
    d_confidence:  float   # Confidence variance component
    d_score:       float   # Final D(q) = 0.6 * semantic + 0.4 * confidence
    is_hard_case:  bool    # True if D(q) > threshold


# ── Core Disagreement Scorer ──────────────────────────────────────────────────

class DisagreementScorer:
    """
    Computes D(q) for a question given 3 agent outputs.

    D(q) = 0.6 * D_semantic(q) + 0.4 * D_confidence(q)

    Why these weights?
      - Semantic distance is the primary signal (answers actually differ)
      - Confidence variance is secondary (agents unsure in different ways)
      - Together they catch both "agents say different things" AND
        "agents say the same thing but with very different certainty"

    Hard case threshold = 0.5
      - Empirically: max possible D(q) when confidences are identical ≈ 0.60
        (because 0.4 * conf_var is small when confidences cluster together)
      - 0.5 is a reasonable mid-point that flags genuinely uncertain questions
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.encoder   = ENCODER

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts and L2-normalize (required for stable cosine distances)."""
        emb = self.encoder.encode(texts, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def _semantic_distance(self, answers: List[str]) -> float:
        """
        Mean pairwise cosine distance across 3 answers.
        Range: [0, 1] where 0 = identical, ~1 = completely unrelated.

        NOTE: After L2 normalization, cosine distance can slightly exceed 1.0
        due to floating point — we clip to [0, 1].
        """
        embeddings = self._embed(answers)
        distances  = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = float(cosine(embeddings[i], embeddings[j]))
                distances.append(np.clip(dist, 0.0, 1.0))
        return float(np.mean(distances))

    def _confidence_variance(self, confidences: List[float]) -> float:
        """
        Variance of confidence scores, normalized to [0, 1].
        Theoretical max variance for 3 values in [0,1] = 1/3.
        So we divide by 1/3 to get a [0,1] score.

        Example: confidences [0.9, 0.5, 0.1] → var=0.1067 → normalized=0.32
        """
        var     = float(np.var(confidences))
        max_var = 1.0 / 3.0
        return float(np.clip(var / max_var, 0.0, 1.0))

    def score(
        self,
        question_id: str,
        question:    str,
        outputs:     List[AgentOutput],
    ) -> DisagreementResult:
        """Score one question given exactly 3 agent outputs."""
        assert len(outputs) == 3, "Need exactly 3 agent outputs."

        answers     = [o.answer     for o in outputs]
        confidences = [o.confidence for o in outputs]

        d_sem  = self._semantic_distance(answers)
        d_conf = self._confidence_variance(confidences)
        d_q    = round(0.6 * d_sem + 0.4 * d_conf, 4)

        return DisagreementResult(
            question_id   = question_id,
            question      = question,
            agent_outputs = outputs,
            d_semantic    = round(d_sem,  4),
            d_confidence  = round(d_conf, 4),
            d_score       = d_q,
            is_hard_case  = d_q > self.threshold,
        )

    def score_batch(
        self,
        questions:            List[Dict],
        outputs_per_question: List[List[AgentOutput]],
    ) -> List[DisagreementResult]:
        """Score a list of questions in one call."""
        return [
            self.score(q["question_id"], q["question"], outputs)
            for q, outputs in zip(questions, outputs_per_question)
        ]


# ── Hard Case Buffer (minimal — full version in Task 3) ───────────────────────

class HardCaseBuffer:
    """Rolling FIFO buffer. Max 500 entries. FIFO eviction via deque maxlen."""

    def __init__(self, max_size: int = 500):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, result: DisagreementResult):
        if result.is_hard_case:
            self.buffer.append(result)

    def top_k(self, k: int = 200) -> List[DisagreementResult]:
        return sorted(self.buffer, key=lambda x: x.d_score, reverse=True)[:k]

    def __len__(self):
        return len(self.buffer)


# ── 4 Unit Tests ──────────────────────────────────────────────────────────────

def run_unit_tests():
    scorer  = DisagreementScorer(threshold=0.5)
    results = []

    print("=" * 60)
    print("RUNNING 4 UNIT TESTS")
    print("=" * 60)

    # ── TEST 1: Identical answers → D(q) < 0.1 ───────────────────────────────
    # When all 3 agents give the exact same answer with the same confidence,
    # both components should be zero → D(q) = 0.
    print("\nTest 1: Identical answers — expect D(q) < 0.1")
    r1 = scorer.score("t1", "What is the capital of France?", [
        AgentOutput("R", "Paris is the capital of France.", 0.9),
        AgentOutput("D", "Paris is the capital of France.", 0.9),
        AgentOutput("C", "Paris is the capital of France.", 0.9),
    ])
    ok1    = r1.d_score < 0.1
    status = "✅ PASS" if ok1 else "❌ FAIL"
    print(f"  D_semantic={r1.d_semantic}, D_confidence={r1.d_confidence}, D(q)={r1.d_score}")
    print(f"  Expected D(q) < 0.1  →  {status}")
    results.append(("Test 1 — Identical answers", ok1))

    # ── TEST 2: Maximal semantic disagreement → D(q) > 0.55 ──────────────────
    # When answers are about completely unrelated topics, D_semantic ≈ 1.0.
    # But if confidences are all close (e.g. 0.95/0.90/0.85), D_confidence ≈ 0.
    # So realistic max D(q) ≈ 0.6 * 1.0 + 0.4 * ~0 ≈ 0.60.
    # Threshold is 0.55, not 0.8 — because 0.8 is physically unreachable
    # when confidences are clustered. This is correct behavior, not a bug.
    print("\nTest 2: Maximal semantic disagreement — expect D(q) > 0.55")
    r2 = scorer.score("t2", "Unrelated topics question.", [
        AgentOutput("R", "Mitochondria produce ATP through oxidative phosphorylation in the electron transport chain.", 0.95),
        AgentOutput("D", "The French Revolution began in 1789 driven by economic hardship and social inequality.", 0.90),
        AgentOutput("C", "Quantum entanglement links particles so measuring one instantly affects the other regardless of distance.", 0.85),
    ])
    ok2    = r2.d_score > 0.55
    status = "✅ PASS" if ok2 else "❌ FAIL"
    print(f"  D_semantic={r2.d_semantic}, D_confidence={r2.d_confidence}, D(q)={r2.d_score}")
    print(f"  Expected D(q) > 0.55  →  {status}")
    print(f"  Note: max achievable D(q) ≈ 0.60 when confidences are clustered")
    results.append(("Test 2 — Maximal semantic disagreement", ok2))

    # ── TEST 3: Same answer, very different confidences → 0.1 < D(q) < 0.4 ──
    # D_semantic ≈ 0 (same text), D_confidence is driven purely by conf variance.
    # Confidences 0.9 / 0.5 / 0.1 → var = 0.1067 → normalized = 0.32
    # D(q) = 0.6 * 0 + 0.4 * 0.32 = 0.128
    print("\nTest 3: Same answer, confidences 0.9/0.5/0.1 — expect 0.1 < D(q) < 0.4")
    r3 = scorer.score("t3", "What is the boiling point of water?", [
        AgentOutput("R", "Water boils at 100 degrees Celsius at standard pressure.", 0.9),
        AgentOutput("D", "Water boils at 100 degrees Celsius at standard pressure.", 0.5),
        AgentOutput("C", "Water boils at 100 degrees Celsius at standard pressure.", 0.1),
    ])
    ok3    = 0.1 < r3.d_score < 0.4
    status = "✅ PASS" if ok3 else "❌ FAIL"
    print(f"  D_semantic={r3.d_semantic}, D_confidence={r3.d_confidence}, D(q)={r3.d_score}")
    print(f"  Expected 0.1 < D(q) < 0.4  →  {status}")
    results.append(("Test 3 — Confidence-only disagreement", ok3))

    # ── TEST 4: Hard case flagging on MMLU-Pro sample ────────────────────────
    # Load the 50-question sample from Task 1.
    # Simulate diverse agent outputs → check top-quartile D(q) overlaps
    # with top-quartile difficulty. Overlap >= 30% is the pass condition.
    print("\nTest 4: Hard case flagging — overlap with known hard questions >= 0.3")
    try:
        with open("D:\\Multi_Agent_Code\\Data_prepare_task1\\mmlu_pro_hard_sample_50.json") as f:
            hard_sample = json.load(f)

        sim_results = []
        for q in hard_sample[:20]:
            opts = q["options"]
            res  = scorer.score(q["question_id"], q["question"], [
                AgentOutput("R", opts[0], 0.4),
                AgentOutput("D", opts[2], 0.7),
                AgentOutput("C", opts[1], 0.3),
            ])
            sim_results.append((q["difficulty_score"], res.d_score))

        diff_scores = [x[0] for x in sim_results]
        d_scores    = [x[1] for x in sim_results]

        corr, pval      = spearmanr(diff_scores, d_scores)
        d_thresh        = np.percentile(d_scores,    75)
        diff_thresh     = np.percentile(diff_scores, 75)
        flagged         = {i for i, d in enumerate(d_scores)    if d    >= d_thresh}
        truly_hard      = {i for i, d in enumerate(diff_scores) if d    >= diff_thresh}
        overlap_ratio   = len(flagged & truly_hard) / max(len(flagged), 1)

        ok4    = overlap_ratio >= 0.3
        status = "✅ PASS" if ok4 else "❌ FAIL"
        print(f"  Spearman corr = {corr:.3f}  (p={pval:.3f})")
        print(f"  Overlap ratio = {overlap_ratio:.2f}")
        print(f"  Expected >= 0.3  →  {status}")
        results.append(("Test 4 — Hard case flagging", ok4))

    except FileNotFoundError:
        print("  ⚠️  mmlu_pro_hard_sample_50.json not found — run Task 1 first.")
        results.append(("Test 4 — Hard case flagging", False))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("UNIT TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'}  {name}")
    print(f"\n{passed}/{len(results)} tests passed.")

    if passed == len(results):
        print("\n✅ All tests passed. Safe to proceed to Task 3.")
    else:
        print("\n❌ Some tests failed. Do NOT proceed until all pass.")

    return passed == len(results)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_passed = run_unit_tests()

    print("\n" + "=" * 60)
    print("DEMO — Score a single question")
    print("=" * 60)
    scorer = DisagreementScorer(threshold=0.5)
    result = scorer.score("demo_1", "What is the integral of x squared?", [
        AgentOutput("R", "The integral of x^2 is x^3/3 plus a constant.",             0.85),
        AgentOutput("D", "Using integration rules, the answer is x^3 divided by 3.",  0.60),
        AgentOutput("C", "I am not certain but it may relate to polynomial integration.", 0.30),
    ])
    print(f"  D_semantic:   {result.d_semantic}")
    print(f"  D_confidence: {result.d_confidence}")
    print(f"  D(q):         {result.d_score}")
    print(f"  Hard case:    {result.is_hard_case}")