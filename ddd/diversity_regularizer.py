"""
Diversity regulariser (paper Section 3.6, Eq. 9; Appendix D.7).

The regularizer detects and penalizes representation/output collapse between
Agent-R, Agent-D, and Agent-C. It is intentionally lightweight:

1. Pure Python/NumPy metrics for JSONL reports and co-evolution manifests.
2. Optional torch loss for training scripts that pass logits.

Training use:
    from diversity_regularizer import torch_diversity_loss
    loss = task_loss + torch_diversity_loss(
        {"R": r_logits, "D": d_logits, "C": c_logits},
        weight=1.0,             # lambda_div in the paper
        min_js_divergence=0.30, # tau_JS in the paper
    )

Note: the function defaults (weight=0.05, min_js_divergence=0.03) are
development values. Pass the paper's values explicitly, as above.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EPS = 1e-12
AGENTS = ("R", "D", "C")


def normalize_probs(values: Sequence[float]) -> List[float]:
    cleaned = [max(float(v), 0.0) for v in values]
    total = sum(cleaned)
    if total <= EPS:
        return [1.0 / len(cleaned)] * len(cleaned) if cleaned else []
    return [v / total for v in cleaned]


def softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not logits:
        return []
    temp = max(float(temperature), EPS)
    scaled = [float(x) / temp for x in logits]
    shift = max(scaled)
    exps = [math.exp(x - shift) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p_norm = normalize_probs(p)
    q_norm = normalize_probs(q)
    return float(
        sum(pi * math.log((pi + EPS) / (qi + EPS)) for pi, qi in zip(p_norm, q_norm))
    )


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p_norm = normalize_probs(p)
    q_norm = normalize_probs(q)
    midpoint = [(pi + qi) / 2.0 for pi, qi in zip(p_norm, q_norm)]
    return 0.5 * kl_divergence(p_norm, midpoint) + 0.5 * kl_divergence(q_norm, midpoint)


def entropy(p: Sequence[float]) -> float:
    probs = normalize_probs(p)
    return float(-sum(pi * math.log(pi + EPS) for pi in probs))


def pairwise_js(probs_by_agent: Dict[str, Sequence[float]]) -> Dict[str, float]:
    pairs = {}
    names = [name for name in AGENTS if name in probs_by_agent and probs_by_agent[name]]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairs[f"{left}{right}"] = js_divergence(probs_by_agent[left], probs_by_agent[right])
    return pairs


def mean_pairwise_js(probs_by_agent: Dict[str, Sequence[float]]) -> float:
    values = list(pairwise_js(probs_by_agent).values())
    return float(sum(values) / len(values)) if values else 0.0


def extract_probs_from_record(record: Dict) -> Dict[str, List[float]]:
    probs = {}
    for agent_name, agent_row in record.get("per_agent", {}).items():
        all_probs = agent_row.get("all_probs")
        if isinstance(all_probs, dict):
            ordered = [all_probs[k] for k in sorted(all_probs)]
            probs[agent_name] = normalize_probs(ordered)
    return probs


@dataclass
class DiversityRegularizer:
    weight: float = 0.05
    min_js_divergence: float = 0.03

    def penalty(self, probs_by_agent: Dict[str, Sequence[float]]) -> float:
        mean_js = mean_pairwise_js(probs_by_agent)
        return float(self.weight * max(0.0, self.min_js_divergence - mean_js))

    def score_probs(self, probs_by_agent: Dict[str, Sequence[float]]) -> Dict:
        pairs = pairwise_js(probs_by_agent)
        mean_js = float(sum(pairs.values()) / len(pairs)) if pairs else 0.0
        return {
            "pairwise_js": pairs,
            "mean_pairwise_js": mean_js,
            "penalty": self.penalty(probs_by_agent),
            "collapsed": mean_js < self.min_js_divergence if pairs else False,
        }

    def score_record(self, record: Dict) -> Dict:
        return self.score_probs(extract_probs_from_record(record))


def build_diversity_report(records: Iterable[Dict], regularizer: Optional[DiversityRegularizer] = None) -> Dict:
    regularizer = regularizer or DiversityRegularizer()
    rows = []
    for record in records:
        score = regularizer.score_record(record)
        rows.append(
            {
                "question_id": record.get("question_id"),
                "D_q": record.get("D_q"),
                **score,
            }
        )

    penalties = [row["penalty"] for row in rows]
    collapsed = [row for row in rows if row["collapsed"]]
    return {
        "n_records": len(rows),
        "n_collapsed": len(collapsed),
        "collapse_rate": len(collapsed) / max(len(rows), 1),
        "mean_penalty": sum(penalties) / max(len(penalties), 1),
        "rows": rows,
    }


def write_diversity_report(input_jsonl: str, output_json: str, weight: float = 0.05, min_js_divergence: float = 0.03) -> Dict:
    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    report = build_diversity_report(records, DiversityRegularizer(weight, min_js_divergence))
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def torch_diversity_loss(
    logits_by_agent: Dict[str, "object"],
    weight: float = 0.05,
    min_js_divergence: float = 0.03,
    temperature: float = 1.0,
):
    """
    Optional torch loss. Pass a dict like {"R": logits_r, "D": logits_d, "C": logits_c}.
    Logits should share shape [..., num_choices]. Returns a scalar tensor.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("torch_diversity_loss requires torch") from exc

    names = [name for name in AGENTS if name in logits_by_agent and logits_by_agent[name] is not None]
    if len(names) < 2:
        first = next(iter(logits_by_agent.values()))
        return torch.zeros((), dtype=first.dtype, device=first.device)

    probs = {
        name: F.softmax(logits_by_agent[name] / max(float(temperature), EPS), dim=-1)
        for name in names
    }
    penalties = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            p = probs[left]
            q = probs[right]
            midpoint = 0.5 * (p + q)
            js = 0.5 * (
                F.kl_div((p + EPS).log(), midpoint, reduction="none").sum(dim=-1)
                + F.kl_div((q + EPS).log(), midpoint, reduction="none").sum(dim=-1)
            )
            penalties.append(torch.relu(torch.as_tensor(min_js_divergence, device=js.device) - js).mean())
    return weight * torch.stack(penalties).mean()


if __name__ == "__main__":
    demo = {
        "per_agent": {
            "R": {"all_probs": {"A": 0.7, "B": 0.2, "C": 0.1}},
            "D": {"all_probs": {"A": 0.68, "B": 0.22, "C": 0.1}},
            "C": {"all_probs": {"A": 0.34, "B": 0.33, "C": 0.33}},
        }
    }
    print(json.dumps(DiversityRegularizer().score_record(demo), indent=2))
