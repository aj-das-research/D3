"""
Closed-loop co-evolution with a diversity regulariser (paper Sections 3.5-3.6,
Appendix D.6-D.8).

Each iteration:

1. every agent scores a fresh batch of training questions;
2. the composite score D(q) (Eq. 7) is computed and questions at or above the
   batch's 80th percentile enter a rolling FIFO buffer (size 500);
3. the top 200 buffered questions by D(q) are exposed, and each agent draws its
   own slice with the sampling rules of Appendix D.6;
4. each agent takes a fixed number of LoRA gradient steps on its slice with the
   task loss plus the diversity hinge of Eq. 9 (the other agents' current
   distributions act as fixed references);
5. all agents are re-scored on the validation split and the metrics are logged.

The evaluation target s(q) is used only for logging Spearman correlations on
held-out questions; it never enters buffering, sampling or training.

The loop is backend-agnostic. A backend implements `predict` and `update`; the
Hugging Face/PEFT backend lives in experiments/coevolution/hf_backend.py and a
small NumPy backend is used in the unit tests.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
from scipy.stats import spearmanr

from disagreement_scorer import AgentOutput, DisagreementScorer

LETTERS = "ABCDEFGHIJ"
EPS = 1e-12


class AgentBackend(Protocol):
    """Minimal interface a model backend must provide."""

    agents: Sequence[str]

    def predict(self, agent: str, questions: List[Dict]) -> List[np.ndarray]:
        """Return one option-probability vector per question."""

    def update(
        self,
        agent: str,
        questions: List[Dict],
        peer_probs: List[Dict[str, np.ndarray]],
        steps: int,
        lambda_div: float,
        tau_js: float,
    ) -> Dict:
        """Take `steps` gradient steps for `agent`; return a small log dict."""


# --------------------------------------------------------------------------- scoring


def js_normalized(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p + EPS) - np.log(m + EPS))))
    kl_qm = float(np.sum(q * (np.log(q + EPS) - np.log(m + EPS))))
    return float(np.clip(0.5 * (kl_pm + kl_qm) / math.log(2.0), 0.0, 1.0))


def mean_pairwise_js(dists: Sequence[np.ndarray]) -> float:
    vals = [js_normalized(dists[i], dists[j]) for i in range(len(dists)) for j in range(i + 1, len(dists))]
    return float(np.mean(vals)) if vals else 0.0


@dataclass
class ScoredQuestion:
    question: Dict
    probs: Dict[str, np.ndarray]
    answers: Dict[str, str]
    confidences: Dict[str, float]
    d_full: float
    d_label_free: float
    d_entropy: float
    d_js: float
    d_confidence: float
    d_correctness: float


def score_questions(
    questions: List[Dict],
    probs_by_agent: Dict[str, List[np.ndarray]],
    weights: Dict[str, float],
) -> List[ScoredQuestion]:
    """Composite D(q) (Eq. 7) and label-free D_LF (Eq. 8) for each question."""
    agents = list(probs_by_agent)
    w = weights
    scorer = DisagreementScorer(
        answer_weight=1.0, distribution_weight=1.0, confidence_weight=1.0, correctness_weight=1.0,
        num_options=len(LETTERS),
    )
    lf_total = w["entropy"] + w["js"] + w["confidence"]
    full_total = lf_total + w["correctness"]
    out = []
    for i, q in enumerate(questions):
        k = len(q["options"])
        probs = {a: np.asarray(probs_by_agent[a][i], dtype=float)[:k] for a in agents}
        probs = {a: p / max(p.sum(), EPS) for a, p in probs.items()}
        answers = {a: LETTERS[int(np.argmax(p))] for a, p in probs.items()}
        confs = {a: float(np.max(p)) for a, p in probs.items()}
        outputs = [
            AgentOutput(a, answers[a], confs[a], probs={LETTERS[j]: float(probs[a][j]) for j in range(k)})
            for a in agents
        ]
        scorer.num_options = k
        res = scorer.score(q["question_id"], q.get("question", ""), outputs, correct_answer=q.get("answer"))
        ent, js, conf, corr = res.d_answer, res.d_distribution, res.d_confidence, res.d_correctness
        lf = (w["entropy"] * ent + w["js"] * js + w["confidence"] * conf) / lf_total
        full = (w["entropy"] * ent + w["js"] * js + w["confidence"] * conf + w["correctness"] * corr) / full_total
        out.append(ScoredQuestion(q, probs, answers, confs, full, lf, ent, js, conf, corr))
    return out


# --------------------------------------------------------------------------- buffer


@dataclass
class BufferItem:
    scored: ScoredQuestion
    iteration: int


@dataclass
class RollingBuffer:
    """FIFO hard-case buffer; the top-k by D(q) are exposed each iteration."""

    max_size: int = 500
    exposed_top_k: int = 200
    items: deque = field(default_factory=deque)

    def __post_init__(self):
        self.items = deque(self.items, maxlen=self.max_size)

    def add_hard(self, scored: List[ScoredQuestion], percentile: float, iteration: int) -> float:
        scores = np.asarray([s.d_full for s in scored])
        tau = float(np.percentile(scores, percentile)) if len(scores) else 0.0
        for s in scored:
            if s.d_full >= tau:
                self.items.append(BufferItem(s, iteration))
        return tau

    def exposed(self) -> List[ScoredQuestion]:
        ranked = sorted(self.items, key=lambda it: it.scored.d_full, reverse=True)
        return [it.scored for it in ranked[: self.exposed_top_k]]

    def __len__(self) -> int:
        return len(self.items)


def sampling_weights(agent: str, exposed: List[ScoredQuestion], cfg: Dict) -> np.ndarray:
    """Agent-specific sampling weights of Appendix D.6 (the target s(q) is never used)."""
    alpha, beta, gamma = cfg["alpha_r"], cfg["beta_d"], cfg["gamma_c"]
    bio = {c.lower() for c in cfg["biomedical_categories"]}
    w = []
    for s in exposed:
        a, c = s.answers, s.confidences
        if agent == "R":
            val = float(a.get("R") != a.get("D")) + alpha * abs(c.get("R", 0.0) - c.get("D", 0.0))
        elif agent == "D":
            val = float(str(s.question.get("category", "")).lower() in bio) + beta * float(a.get("D") != a.get("C"))
        elif agent == "C":
            correct = float(a.get("C") == str(s.question.get("answer", "")).strip().upper()[:1])
            val = 1.0 if abs(c.get("C", 0.0) - correct) > gamma else 0.0
        else:
            val = 1.0
        w.append(val)
    return np.asarray(w, dtype=float)


def sample_slice(weights: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    nonzero = np.flatnonzero(weights > 0)
    if len(nonzero) == 0:
        return np.asarray([], dtype=int)
    size = min(size, len(nonzero))
    p = weights[nonzero] / weights[nonzero].sum()
    return rng.choice(nonzero, size=size, replace=False, p=p)


# --------------------------------------------------------------------------- evaluation


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def evaluate(scored: List[ScoredQuestion], agents: Sequence[str], ece_bins: int = 10, percentile: float = 80) -> Dict:
    gold = np.asarray([str(s.question.get("answer", "")).strip().upper()[:1] for s in scored])
    target = np.asarray([float(s.question.get("difficulty_score", np.nan)) for s in scored])
    d_full = np.asarray([s.d_full for s in scored])
    d_lf = np.asarray([s.d_label_free for s in scored])
    metrics: Dict = {"n": len(scored)}
    for a in agents:
        ans = np.asarray([s.answers[a] for s in scored])
        conf = np.asarray([s.confidences[a] for s in scored])
        correct = (ans == gold).astype(float)
        metrics[f"acc_{a}"] = float(correct.mean())
        metrics[f"ece_{a}"] = expected_calibration_error(conf, correct, ece_bins)
    metrics["mean_js"] = float(np.mean([mean_pairwise_js([s.probs[a] for a in agents]) for s in scored]))
    metrics["mean_D"] = float(d_full.mean())
    metrics["hard_rate"] = float((d_full >= np.percentile(d_full, percentile)).mean())
    ok = ~np.isnan(target)
    for name, score in (("rho_D", d_full), ("rho_D_LF", d_lf)):
        metrics[name] = float(spearmanr(score[ok], target[ok]).statistic) if ok.sum() > 2 else float("nan")
    return metrics


def predictions_table(scored: List[ScoredQuestion], agents: Sequence[str]) -> List[Dict]:
    rows = []
    for s in scored:
        rows.append({
            "question_id": s.question["question_id"],
            "category": s.question.get("category", ""),
            "difficulty_score": s.question.get("difficulty_score"),
            "correct_answer": s.question.get("answer"),
            "D": s.d_full, "D_LF": s.d_label_free,
            "D_entropy": s.d_entropy, "D_js": s.d_js, "D_confidence": s.d_confidence, "D_correctness": s.d_correctness,
            "per_agent": {
                a: {"answer": s.answers[a], "confidence": s.confidences[a],
                    "all_probs": {LETTERS[j]: float(v) for j, v in enumerate(s.probs[a])}}
                for a in agents
            },
        })
    return rows


# --------------------------------------------------------------------------- loop


def run_coevolution(
    backend: AgentBackend,
    train_questions: List[Dict],
    val_questions: List[Dict],
    config: Dict,
    seed: int,
    output_dir: Path,
    regularize: bool = True,
    save_callback=None,
) -> List[Dict]:
    """Run the loop and write `iterations.jsonl` under `output_dir`."""
    co, buf_cfg, weights = config["coevolution"], config["buffer"], config["score"]["weights"]
    percentile = config["score"]["hard_percentile"]
    ece_bins = config.get("eval", {}).get("ece_bins", 10)
    agents = list(backend.agents)
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "iterations.jsonl"
    log_path.write_text("", encoding="utf-8")

    def score_split(questions):
        probs = {a: backend.predict(a, questions) for a in agents}
        return score_questions(questions, probs, weights)

    buffer = RollingBuffer(buf_cfg["max_size"], buf_cfg["exposed_top_k"])
    lambda_div = co["lambda_div"] if regularize else 0.0
    history = []

    def log(iteration: int, extra: Dict):
        val_scored = score_split(val_questions)
        rec = {"iteration": iteration, "seed": seed, "regularized": regularize, **evaluate(val_scored, agents, ece_bins, percentile), **extra}
        history.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return val_scored

    val_scored = log(0, {"buffer_size": 0})
    for t in range(1, co["iterations"] + 1):
        idx = rng.choice(len(train_questions), size=min(co["train_batch_size"], len(train_questions)), replace=False)
        batch = [train_questions[i] for i in idx]
        scored = score_split(batch)
        tau = buffer.add_hard(scored, percentile, t)
        exposed = buffer.exposed()
        updates = {}
        for a in agents:
            chosen = sample_slice(sampling_weights(a, exposed, buf_cfg), co["slice_size"], rng)
            if len(chosen) == 0:
                updates[a] = {"skipped": True}
                continue
            qs = [exposed[i].question for i in chosen]
            peers = [{b: exposed[i].probs[b] for b in agents if b != a} for i in chosen]
            updates[a] = backend.update(a, qs, peers, co["steps_per_agent"], lambda_div, co["tau_js"])
        val_scored = log(t, {"buffer_size": len(buffer), "train_tau80": tau, "updates": updates})
        if save_callback:
            save_callback(t, val_scored)

    with (output_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions_table(val_scored, agents):
            f.write(json.dumps(row) + "\n")
    return history
