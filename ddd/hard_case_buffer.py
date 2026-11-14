"""
Adaptive hard-case buffer (paper Section 3.5, Appendix D.6).

Implements the rolling FIFO buffer, per-agent filtering, and sampling logic.
The module is self-contained: it defines minimal AgentOutput and
DisagreementResult dataclasses that are field-compatible with the ones in
disagreement_scorer.py.
"""

import json
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# ── Minimal result dataclasses (field-compatible with disagreement_scorer.py) ──

from dataclasses import dataclass
from typing import List


@dataclass
class AgentOutput:
    agent_name: str      # "R", "D", or "C"
    answer:     str
    confidence: float
    probs:      Optional[Dict[str, float]] = None   # full option distribution, if available


@dataclass
class DisagreementResult:
    question_id:   str
    question:      str
    agent_outputs: List[AgentOutput]
    d_semantic:    float
    d_confidence:  float
    d_score:       float
    is_hard_case:  bool


# ── BufferEntry: what gets stored per question ────────────────────────────────

@dataclass
class BufferEntry:
    """
    Everything the co-evolution loop needs about a hard question.

    Fields beyond DisagreementResult:
      - timestamp:     Unix time when the question entered the buffer.
                       Used to break ties in FIFO eviction.
      - iteration:     Co-evolution loop iteration number when this was added.
                       Used for per-iteration analysis and logging.
      - rd_gap:        |confidence_R - confidence_D|. Used by per-agent
                       filtering to route R/D disagreement cases.
      - c_wrong_conf:  Agent-C's confidence when it was wrong (for calibration
                       error batches). None if C answer is unknown as correct/wrong.
    """
    result:       DisagreementResult
    timestamp:    float
    iteration:    int
    rd_gap:       float         # |conf_R - conf_D|
    c_wrong_conf: Optional[float]  # set externally after label is known


# ── HardCaseBuffer ────────────────────────────────────────────────────────────

class HardCaseBuffer:
    """
    Rolling FIFO buffer for hard questions, with per-agent sampling.

    Spec (paper Section 3.5):
      - Max 500 questions; FIFO eviction via deque maxlen.
      - Top-200 by D(q) score sampled per co-evolution iteration.
      - Agent-R and Agent-D see cases where R and D disagree (high rd_gap).
      - Agent-C sees cases where C confidence was wrong (high c_wrong_conf).

    Design notes:
      - We use a deque(maxlen=500) as the primary store.
        Python deques handle FIFO eviction automatically.
      - All sampling is done at call time (not on insert), so the same
        entry can appear in multiple agents' batches if it qualifies.
      - Thread-safety: single-threaded co-evolution loop assumed; no locks.
    """

    # ── Thresholds & constants ────────────────────────────────────────────────

    MAX_SIZE       = 500      # Hard cap — FIFO eviction beyond this
    SAMPLE_SIZE    = 200      # Top-k by D(q) for co-evolution fine-tuning
    RD_GAP_THRESH  = 0.2     # Minimum |conf_R - conf_D| to count as R/D disagreement
    C_CONF_THRESH  = 0.6     # Agent-C confidence must be >= this to count as overconfident

    def __init__(self, max_size: int = MAX_SIZE, sample_size: int = SAMPLE_SIZE):
        self.max_size    = max_size
        self.sample_size = sample_size
        self.buffer: deque = deque(maxlen=max_size)
        self._total_added = 0    # monotonic count (for diagnostics)
        self._total_evicted = 0  # estimated evictions

    # ── Core insert ───────────────────────────────────────────────────────────

    def add(
        self,
        result:       DisagreementResult,
        iteration:    int,
        c_wrong_conf: Optional[float] = None,
    ) -> bool:
        """
        Add a DisagreementResult to the buffer.
        Only adds if result.is_hard_case is True.

        Args:
            result:       Scored result from DisagreementScorer.
            iteration:    Current co-evolution loop iteration number.
            c_wrong_conf: If Agent-C was wrong on this question and its
                          confidence is known, pass it here. Otherwise None.

        Returns:
            True if the entry was added, False if skipped (not a hard case).
        """
        if not result.is_hard_case:
            return False

        rd_gap = self._compute_rd_gap(result)

        entry = BufferEntry(
            result       = result,
            timestamp    = time.time(),
            iteration    = iteration,
            rd_gap       = rd_gap,
            c_wrong_conf = c_wrong_conf,
        )

        if len(self.buffer) == self.max_size:
            self._total_evicted += 1

        self.buffer.append(entry)
        self._total_added += 1
        return True

    # ── Sampling: top-K by D(q) ───────────────────────────────────────────────

    def top_k(self, k: Optional[int] = None) -> List[BufferEntry]:
        """
        Return top-k entries by D(q), highest first.
        Default k = self.sample_size (200 per spec).

        Used by the co-evolution loop to select fine-tuning data.
        """
        k = k or self.sample_size
        return sorted(self.buffer, key=lambda e: e.result.d_score, reverse=True)[:k]

    # ── Per-agent filtered batches ────────────────────────────────────────────

    def sample_for_agent_R(self, k: Optional[int] = None) -> List[BufferEntry]:
        """
        Cases where R and D disagree (high |conf_R - conf_D|).
        Agent-R should learn from questions where it diverges from Agent-D.

        Filter: rd_gap >= RD_GAP_THRESH, then top-k by D(q).
        """
        k = k or self.sample_size
        rd_cases = [e for e in self.buffer if e.rd_gap >= self.RD_GAP_THRESH]
        return sorted(rd_cases, key=lambda e: e.result.d_score, reverse=True)[:k]

    def sample_for_agent_D(self, k: Optional[int] = None) -> List[BufferEntry]:
        """
        Cases where D fails vs R (same filter as R: high R/D gap).
        Agent-D learns from exactly the same divergence cases as Agent-R,
        but will be trained with its own loss objective.

        Filter: rd_gap >= RD_GAP_THRESH, then top-k by D(q).
        """
        return self.sample_for_agent_R(k)  # same selection, different training objective

    def sample_for_agent_C(self, k: Optional[int] = None) -> List[BufferEntry]:
        """
        Cases where C was overconfident on a wrong answer.
        Agent-C learns calibration from its own miscalibration events.

        Filter: c_wrong_conf >= C_CONF_THRESH (C was confident and wrong),
        then top-k by c_wrong_conf (worst overconfidence first).
        """
        k = k or self.sample_size
        c_cases = [
            e for e in self.buffer
            if e.c_wrong_conf is not None and e.c_wrong_conf >= self.C_CONF_THRESH
        ]
        return sorted(c_cases, key=lambda e: e.c_wrong_conf, reverse=True)[:k]

    # ── Bulk add from a batch ─────────────────────────────────────────────────

    def add_batch(
        self,
        results:      List[DisagreementResult],
        iteration:    int,
        c_wrong_confs: Optional[List[Optional[float]]] = None,
    ) -> Tuple[int, int]:
        """
        Add a list of results. Returns (n_added, n_skipped).

        Args:
            results:       List of DisagreementResult from scorer.score_batch().
            iteration:     Current loop iteration.
            c_wrong_confs: Optional parallel list of c_wrong_conf values.
                           Pass None to default all to None.
        """
        if c_wrong_confs is None:
            c_wrong_confs = [None] * len(results)

        added = skipped = 0
        for result, cwc in zip(results, c_wrong_confs):
            if self.add(result, iteration, cwc):
                added += 1
            else:
                skipped += 1
        return added, skipped

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """
        Summary statistics for logging.
        Returns a dict with counts, score distribution, agent-specific counts.
        """
        if not self.buffer:
            return {"size": 0, "total_added": self._total_added}

        scores    = [e.result.d_score for e in self.buffer]
        rd_gaps   = [e.result.d_score for e in self.buffer if e.rd_gap >= self.RD_GAP_THRESH]
        c_cases   = [e for e in self.buffer
                     if e.c_wrong_conf is not None and e.c_wrong_conf >= self.C_CONF_THRESH]

        return {
            "size":              len(self.buffer),
            "total_added":       self._total_added,
            "total_evicted":     self._total_evicted,
            "d_score_mean":      round(float(np.mean(scores)), 4),
            "d_score_max":       round(float(np.max(scores)),  4),
            "d_score_min":       round(float(np.min(scores)),  4),
            "d_score_p75":       round(float(np.percentile(scores, 75)), 4),
            "n_rd_disagreement": len(rd_gaps),    # eligible for R/D sampling
            "n_c_overconfident": len(c_cases),    # eligible for C sampling
        }

    def iter_stats(self) -> Dict[int, int]:
        """Count entries per co-evolution iteration (useful for monitoring drift)."""
        counts: Dict[int, int] = {}
        for e in self.buffer:
            counts[e.iteration] = counts.get(e.iteration, 0) + 1
        return dict(sorted(counts.items()))

    def export_json(self, path: str):
        """Export all buffer entries to JSON for offline analysis."""
        records = []
        for e in self.buffer:
            records.append({
                "question_id":   e.result.question_id,
                "question":      e.result.question,
                "d_score":       e.result.d_score,
                "d_semantic":    e.result.d_semantic,
                "d_confidence":  e.result.d_confidence,
                "is_hard_case":  e.result.is_hard_case,
                "rd_gap":        e.rd_gap,
                "c_wrong_conf":  e.c_wrong_conf,
                "iteration":     e.iteration,
                "timestamp":     e.timestamp,
                "agents": [
                    {"name": a.agent_name, "answer": a.answer, "confidence": a.confidence}
                    for a in e.result.agent_outputs
                ],
            })
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Exported {len(records)} entries to {path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_rd_gap(self, result: DisagreementResult) -> float:
        """
        Compute |conf_R - conf_D| from agent outputs.
        Agents are identified by agent_name in {"R", "D", "C"}.
        Falls back to 0.0 if names don't match (graceful degradation).
        """
        conf = {o.agent_name: o.confidence for o in result.agent_outputs}
        r = conf.get("R", None)
        d = conf.get("D", None)
        if r is None or d is None:
            return 0.0
        return abs(r - d)

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"HardCaseBuffer(size={len(self.buffer)}/{self.max_size}, added={self._total_added})"


# ── 4 Unit Tests ──────────────────────────────────────────────────────────────

def _make_result(qid, question, answers, confs, d_score, threshold=0.5) -> DisagreementResult:
    """Helper: build a DisagreementResult with minimal boilerplate."""
    outputs = [
        AgentOutput(name, ans, conf)
        for name, ans, conf in zip(["R", "D", "C"], answers, confs)
    ]
    d_sem  = d_score  # simplified: just use d_score as proxy for test purposes
    d_conf = 0.0
    return DisagreementResult(
        question_id   = qid,
        question      = question,
        agent_outputs = outputs,
        d_semantic    = round(d_sem, 4),
        d_confidence  = round(d_conf, 4),
        d_score       = round(d_score, 4),
        is_hard_case  = d_score > threshold,
    )


def run_unit_tests():
    print("=" * 60)
    print("HARD CASE BUFFER SELF-TEST")
    print("=" * 60)

    results = []

    # ── TEST 1: Non-hard cases are not added ─────────────────────────────────
    print("\nTest 1: Non-hard cases (D(q) <= 0.5) must NOT enter the buffer")
    buf = HardCaseBuffer()
    easy = _make_result("easy_1", "What is 2+2?",
                        ["4", "4", "4"], [0.99, 0.99, 0.99], d_score=0.0)
    added = buf.add(easy, iteration=0)
    ok1 = (not added) and (len(buf) == 0)
    print(f"  Added={added}, Buffer size={len(buf)}")
    print(f"  Expected: not added, size=0  →  {'✅ PASS' if ok1 else '❌ FAIL'}")
    results.append(("Test 1 — Non-hard cases not added", ok1))

    # ── TEST 2: FIFO eviction at max capacity ─────────────────────────────────
    print("\nTest 2: FIFO eviction — buffer must not exceed max_size=10 (small test)")
    buf2 = HardCaseBuffer(max_size=10, sample_size=5)
    for i in range(15):
        r = _make_result(f"q{i}", f"Question {i}",
                         [f"ans{i}a", f"ans{i}b", f"ans{i}c"],
                         [0.8, 0.4, 0.3], d_score=0.55 + i * 0.01)
        buf2.add(r, iteration=0)
    ok2 = len(buf2) == 10
    # Confirm FIFO: oldest entries (q0–q4) are gone; newest (q5–q14) remain
    ids_in_buf = {e.result.question_id for e in buf2.buffer}
    oldest_evicted = "q0" not in ids_in_buf and "q4" not in ids_in_buf
    ok2 = ok2 and oldest_evicted
    print(f"  Buffer size={len(buf2)}, IDs in buf: {sorted(ids_in_buf)}")
    print(f"  Expected size=10, q0/q4 evicted  →  {'✅ PASS' if ok2 else '❌ FAIL'}")
    results.append(("Test 2 — FIFO eviction at max_size", ok2))

    # ── TEST 3: top_k returns highest D(q) entries ────────────────────────────
    print("\nTest 3: top_k() must return entries sorted by D(q) descending")
    buf3 = HardCaseBuffer(max_size=500, sample_size=3)
    scores = [0.52, 0.78, 0.61, 0.95, 0.55]
    for i, sc in enumerate(scores):
        r = _make_result(f"q{i}", f"Q{i}",
                         [f"a{i}R", f"a{i}D", f"a{i}C"],
                         [0.7, 0.3, 0.5], d_score=sc)
        buf3.add(r, iteration=1)
    top3 = buf3.top_k(k=3)
    top3_scores = [e.result.d_score for e in top3]
    ok3 = top3_scores == sorted(top3_scores, reverse=True) and top3_scores[0] == 0.95
    print(f"  Top-3 D(q) scores: {top3_scores}")
    print(f"  Expected [0.95, 0.78, 0.61]  →  {'✅ PASS' if ok3 else '❌ FAIL'}")
    results.append(("Test 3 — top_k returns highest D(q)", ok3))

    # ── TEST 4: Per-agent filtering ───────────────────────────────────────────
    print("\nTest 4: Per-agent filtering")
    buf4 = HardCaseBuffer()

    # Entry 1: high rd_gap (R=0.9, D=0.1 → gap=0.8) → eligible for R/D
    r_rd = _make_result("rd_case", "R/D disagree hard",
                        ["math answer", "med answer", "unsure answer"],
                        [0.9, 0.1, 0.5], d_score=0.72)
    buf4.add(r_rd, iteration=1, c_wrong_conf=None)

    # Entry 2: Agent-C overconfident and wrong (c_wrong_conf=0.85)
    r_c = _make_result("c_case", "C overconfident",
                        ["answer A", "answer B", "answer A"],
                        [0.6, 0.4, 0.85], d_score=0.60)
    buf4.add(r_c, iteration=1, c_wrong_conf=0.85)

    # Entry 3: low rd_gap (R=0.55, D=0.50 → gap=0.05) → NOT eligible for R/D
    r_low = _make_result("low_gap", "Low gap case",
                         ["similar A", "similar B", "similar C"],
                         [0.55, 0.50, 0.52], d_score=0.58)
    buf4.add(r_low, iteration=1, c_wrong_conf=None)

    r_batch  = buf4.sample_for_agent_R()
    d_batch  = buf4.sample_for_agent_D()
    c_batch  = buf4.sample_for_agent_C()

    # R/D batches must include high-gap case but NOT low-gap case
    r_ids = {e.result.question_id for e in r_batch}
    c_ids = {e.result.question_id for e in c_batch}

    ok4_r = "rd_case" in r_ids and "low_gap" not in r_ids
    ok4_c = "c_case" in c_ids and "rd_case" not in c_ids
    ok4   = ok4_r and ok4_c

    print(f"  Agent-R sees: {r_ids}  (should contain 'rd_case', not 'low_gap')")
    print(f"  Agent-C sees: {c_ids}  (should contain 'c_case' only)")
    print(f"  Agent-R filter ok={ok4_r}, Agent-C filter ok={ok4_c}")
    print(f"  Expected both ok  →  {'✅ PASS' if ok4 else '❌ FAIL'}")
    results.append(("Test 4 — Per-agent filtering", ok4))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("UNIT TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'}  {name}")
    print(f"\n{passed}/{len(results)} tests passed.")
    if passed == len(results):
        print("\n✅ All tests passed.")
    else:
        print("\n❌ Some tests failed. Fix before proceeding.")
    return passed == len(results)


# ── Integration demo with the scorer ──────────────────────────────────────

def demo_integration():
    """
    Shows how the buffer plugs into the disagreement scorer.
    """
    print("\n" + "=" * 60)
    print("DEMO — Buffer + Scorer integration")
    print("=" * 60)

    try:
        from disagreement_scorer import DisagreementScorer
    except ImportError:
        print("  (Paste DisagreementScorer here if running standalone)")
        return

    scorer = DisagreementScorer(threshold=0.5)
    buf    = HardCaseBuffer()

    questions = [
        ("q1", "What is the integral of x squared?"),
        ("q2", "Should aspirin be given in dengue fever?"),
        ("q3", "What is 7 times 8?"),
    ]
    all_outputs = [
        [AgentOutput("R", "x^3/3 + C, by power rule",        0.85),
         AgentOutput("D", "The antiderivative is x cubed over 3.", 0.60),
         AgentOutput("C", "I'm uncertain — perhaps x^3/3.",  0.30)],
        [AgentOutput("R", "Yes, aspirin reduces fever.",              0.70),
         AgentOutput("D", "No — aspirin is contraindicated in dengue due to bleeding risk.", 0.95),
         AgentOutput("C", "There is strong evidence against aspirin in dengue.",             0.80)],
        [AgentOutput("R", "56", 0.99),
         AgentOutput("D", "56", 0.99),
         AgentOutput("C", "56", 0.99)],
    ]

    for (qid, question), outputs in zip(questions, all_outputs):
        result = scorer.score(qid, question, outputs)
        added  = buf.add(result, iteration=0)
        status = "→ added to buffer" if added else "→ skipped (not hard)"
        print(f"  {qid}: D(q)={result.d_score:.4f}, hard={result.is_hard_case}  {status}")

    print(f"\n  Buffer stats: {buf.stats()}")
    print(f"  Buffer repr:  {buf}")

    top = buf.top_k(k=2)
    print(f"\n  Top-2 by D(q):")
    for e in top:
        print(f"    {e.result.question_id}: D(q)={e.result.d_score}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_passed = run_unit_tests()
    demo_integration()