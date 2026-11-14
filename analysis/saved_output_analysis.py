"""Canonical rescoring, component ablation and significance tests on saved outputs
(paper Sections 3.4 and 6.4, Appendix G.8).

This script supports both repository schemas:

- heterogeneous rows: ``per_agent`` with optional full choice probabilities;
- homogeneous rows: ``per_role`` with answer letters and scalar confidence.

It never performs model inference. For rows without full probabilities it uses
the paper's maximum-entropy reconstruction over the ten MMLU-Pro options.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "experiments" / "static_disagreement" / "outputs"
OUTPUT_DIR = REPO_ROOT / "results" / "saved_output_analysis"
AGENT_NAMES = ("R", "D", "C")
MMLU_OPTION_LABELS = tuple("ABCDEFGHIJ")
EPSILON = 1e-12


@dataclass(frozen=True)
class CanonicalAgentOutput:
    name: str
    answer: str
    confidence: float
    probabilities: dict[str, float] | None = None


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:1] if text and text[:1].isalpha() else text


def bounded_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def normalize_probability_dict(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    cleaned = {}
    for key, raw in value.items():
        label = normalize_answer(key)
        if not label:
            continue
        try:
            number = max(float(raw), 0.0)
        except (TypeError, ValueError):
            continue
        cleaned[label] = number
    total = sum(cleaned.values())
    if total <= EPSILON:
        return None
    return {label: number / total for label, number in cleaned.items()}


def extract_outputs(row: dict[str, Any]) -> list[CanonicalAgentOutput] | None:
    per_agent = row.get("per_agent")
    if isinstance(per_agent, dict):
        outputs = []
        for name in AGENT_NAMES:
            item = per_agent.get(name)
            if not isinstance(item, dict) or item.get("answer") is None:
                return None
            outputs.append(
                CanonicalAgentOutput(
                    name=name,
                    answer=normalize_answer(item.get("answer")),
                    confidence=bounded_confidence(item.get("confidence", item.get("conf"))),
                    probabilities=normalize_probability_dict(
                        item.get("all_probs", item.get("probs", item.get("probabilities")))
                    ),
                )
            )
        return outputs

    per_role = row.get("per_role")
    if isinstance(per_role, dict):
        by_name: dict[str, CanonicalAgentOutput] = {}
        for role_name, item in per_role.items():
            if not isinstance(item, dict):
                continue
            fallback = {
                "careful_reasoner": "R",
                "domain_expert": "D",
                "uncertain_analyst": "C",
            }.get(str(role_name), "")
            name = normalize_answer(item.get("agent_letter", fallback))
            if name not in AGENT_NAMES:
                continue
            by_name[name] = CanonicalAgentOutput(
                name=name,
                answer=normalize_answer(item.get("answer")),
                confidence=bounded_confidence(item.get("confidence", item.get("conf"))),
                probabilities=normalize_probability_dict(
                    item.get("all_probs", item.get("probs", item.get("probabilities")))
                ),
            )
        if all(name in by_name for name in AGENT_NAMES):
            return [by_name[name] for name in AGENT_NAMES]

    raw_outputs = row.get("agent_outputs") or row.get("agents") or row.get("outputs")
    if isinstance(raw_outputs, list) and len(raw_outputs) == 3:
        outputs = []
        for index, item in enumerate(raw_outputs):
            if not isinstance(item, dict):
                return None
            outputs.append(
                CanonicalAgentOutput(
                    name=str(item.get("agent_name", item.get("name", AGENT_NAMES[index]))),
                    answer=normalize_answer(item.get("answer", item.get("prediction"))),
                    confidence=bounded_confidence(item.get("confidence", item.get("conf"))),
                    probabilities=normalize_probability_dict(
                        item.get("all_probs", item.get("probs", item.get("probabilities")))
                    ),
                )
            )
        return outputs
    return None


def answer_entropy(outputs: list[CanonicalAgentOutput]) -> float:
    answers = [normalize_answer(output.answer) for output in outputs]
    counts = np.asarray([answers.count(value) for value in sorted(set(answers))], dtype=float)
    probabilities = counts / len(answers)
    entropy = -float(np.sum(probabilities * np.log(probabilities + EPSILON)))
    maximum = math.log(min(len(outputs), len(MMLU_OPTION_LABELS)))
    return float(np.clip(entropy / maximum, 0.0, 1.0)) if maximum > 0 else 0.0


def output_distribution(
    output: CanonicalAgentOutput,
    labels: tuple[str, ...] = MMLU_OPTION_LABELS,
) -> np.ndarray:
    if output.probabilities:
        values = np.asarray(
            [max(float(output.probabilities.get(label, 0.0)), 0.0) for label in labels],
            dtype=float,
        )
        if float(values.sum()) > EPSILON:
            return values / values.sum()

    confidence = bounded_confidence(output.confidence)
    chosen = normalize_answer(output.answer)
    values = np.zeros(len(labels), dtype=float)
    if chosen not in labels:
        values[:] = 1.0 / len(labels)
        return values
    chosen_index = labels.index(chosen)
    values[chosen_index] = confidence
    spill = (1.0 - confidence) / max(len(labels) - 1, 1)
    for index in range(len(labels)):
        if index != chosen_index:
            values[index] = spill
    return values / values.sum()


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    kl_left = float(
        np.sum(left * np.log((left + EPSILON) / (midpoint + EPSILON)))
    )
    kl_right = float(
        np.sum(right * np.log((right + EPSILON) / (midpoint + EPSILON)))
    )
    return float(np.clip(0.5 * (kl_left + kl_right) / math.log(2), 0.0, 1.0))


def mean_pairwise_js(outputs: list[CanonicalAgentOutput]) -> float:
    distributions = [output_distribution(output) for output in outputs]
    values = [
        js_divergence(distributions[left], distributions[right])
        for left in range(len(distributions))
        for right in range(left + 1, len(distributions))
    ]
    return float(np.mean(values)) if values else 0.0


def confidence_variance(outputs: list[CanonicalAgentOutput]) -> float:
    values = np.asarray([bounded_confidence(output.confidence) for output in outputs])
    return float(np.clip(float(np.var(values)) / (1.0 / 3.0), 0.0, 1.0))


def correctness_diversity(
    outputs: list[CanonicalAgentOutput], correct_answer: Any
) -> float:
    correct = normalize_answer(correct_answer)
    if not correct:
        return 0.0
    statuses = [normalize_answer(output.answer) == correct for output in outputs]
    return 0.0 if len(set(statuses)) == 1 else 1.0


def score_components(row: dict[str, Any]) -> dict[str, float] | None:
    outputs = extract_outputs(row)
    if outputs is None:
        return None
    entropy = answer_entropy(outputs)
    js = mean_pairwise_js(outputs)
    confidence = confidence_variance(outputs)
    correctness = correctness_diversity(outputs, row.get("correct_answer", row.get("answer")))
    full = 0.45 * entropy + 0.35 * js + 0.15 * confidence + 0.05 * correctness
    no_correctness_paper = 0.45 * entropy + 0.35 * js + 0.20 * confidence
    no_correctness_renormalized = (
        0.45 * entropy + 0.35 * js + 0.15 * confidence
    ) / 0.95
    return {
        "D_entropy": entropy,
        "D_js": js,
        "D_confidence": confidence,
        "D_correctness": correctness,
        "D_full": full,
        "D_no_correctness_paper": no_correctness_paper,
        "D_no_correctness_renormalized": no_correctness_renormalized,
    }


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def score_file(path: Path) -> pd.DataFrame:
    records = []
    for row in read_jsonl(path):
        components = score_components(row)
        if components is None:
            continue
        records.append(
            {
                "question_id": str(row.get("question_id", "")),
                "category": str(row.get("category", "")),
                "difficulty": row.get(
                    "difficulty_score", row.get("final_difficulty", row.get("difficulty"))
                ),
                "correct_answer": normalize_answer(
                    row.get("correct_answer", row.get("answer"))
                ),
                **components,
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["difficulty"] = pd.to_numeric(frame["difficulty"], errors="coerce")
    return frame


def correlation(frame: pd.DataFrame, score_column: str) -> tuple[float | None, float | None, int]:
    valid = frame[[score_column, "difficulty"]].dropna()
    if len(valid) < 3 or valid[score_column].nunique() < 2 or valid["difficulty"].nunique() < 2:
        return None, None, len(valid)
    result = spearmanr(valid[score_column], valid["difficulty"])
    return float(result.statistic), float(result.pvalue), len(valid)


def bootstrap_spearman(
    frame: pd.DataFrame,
    score_column: str,
    repetitions: int = 1000,
    seed: int = 2026,
) -> tuple[float | None, float | None]:
    valid = frame[[score_column, "difficulty"]].dropna().to_numpy(dtype=float)
    if len(valid) < 3:
        return None, None
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        indices = generator.integers(0, len(valid), size=len(valid))
        sample = valid[indices]
        if len(np.unique(sample[:, 0])) < 2 or len(np.unique(sample[:, 1])) < 2:
            continue
        estimate = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None, None
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def permutation_p_value(
    frame: pd.DataFrame,
    score_column: str,
    repetitions: int = 10000,
    seed: int = 2026,
) -> float | None:
    valid = frame[[score_column, "difficulty"]].dropna()
    rho, _, count = correlation(valid, score_column)
    if rho is None or count < 3:
        return None
    score_ranks = rankdata(valid[score_column].to_numpy(dtype=float))
    target_ranks = rankdata(valid["difficulty"].to_numpy(dtype=float))
    observed = abs(float(np.corrcoef(score_ranks, target_ranks)[0, 1]))
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repetitions):
        shuffled = generator.permutation(target_ranks)
        permuted = abs(float(np.corrcoef(score_ranks, shuffled)[0, 1]))
        if permuted >= observed:
            exceedances += 1
    return (exceedances + 1) / (repetitions + 1)


def triage_metrics(
    frame: pd.DataFrame,
    score_column: str,
    selection_fraction: float = 0.20,
    hard_fraction: float = 0.20,
) -> dict[str, float | int | None]:
    valid = frame[[score_column, "difficulty"]].dropna().copy()
    if len(valid) < 5:
        return {
            "n": len(valid),
            "selected": 0,
            "hard": 0,
            "precision": None,
            "recall": None,
            "enrichment_over_random": None,
        }
    selected_count = max(1, int(math.ceil(len(valid) * selection_fraction)))
    hard_count = max(1, int(math.ceil(len(valid) * hard_fraction)))
    selected_indices = set(valid.nlargest(selected_count, score_column).index)
    hard_indices = set(valid.nlargest(hard_count, "difficulty").index)
    true_positives = len(selected_indices & hard_indices)
    precision = true_positives / selected_count
    recall = true_positives / hard_count
    baseline = hard_count / len(valid)
    return {
        "n": len(valid),
        "selected": selected_count,
        "hard": hard_count,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "enrichment_over_random": precision / baseline if baseline else None,
    }


def analyze_file(path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    frame = score_file(path)
    correlation_rows = []
    triage_rows = []
    score_columns = [
        "D_full",
        "D_no_correctness_paper",
        "D_no_correctness_renormalized",
        "D_entropy",
        "D_js",
        "D_confidence",
        "D_correctness",
    ]
    for score_column in score_columns:
        rho, asymptotic_p, count = correlation(frame, score_column)
        ci_lower, ci_upper = bootstrap_spearman(frame, score_column)
        permutation_p = permutation_p_value(frame, score_column)
        correlation_rows.append(
            {
                "file": _rel(path),
                "schema": "per_role" if "homogeneous" in path.name else "per_agent",
                "score": score_column,
                "n": count,
                "spearman_rho": rho,
                "asymptotic_p": asymptotic_p,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "permutation_p": permutation_p,
                "target": "surface difficulty heuristic",
            }
        )
        triage_rows.append(
            {
                "file": _rel(path),
                "score": score_column,
                "target": "top 20% by surface difficulty heuristic",
                **triage_metrics(frame, score_column),
            }
        )
    return frame, correlation_rows, triage_rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(INPUT_DIR),
                        help="Folder of per-question JSONL files (static outputs or co-evolution predictions)")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(input_dir.glob("*.jsonl"))

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    all_correlations = []
    all_triage = []
    manifest = []
    for path in input_paths:
        frame, correlations, triage = analyze_file(path)
        output_path = output_dir / f"{path.stem}_canonical_rescored.csv"
        frame.to_csv(output_path, index=False)
        all_correlations.extend(correlations)
        all_triage.extend(triage)
        manifest.append(
            {
                "input": rel(path),
                "output": rel(output_path),
                "input_rows": len(read_jsonl(path)),
                "scored_rows": len(frame),
            }
        )

    correlation_frame = pd.DataFrame(all_correlations)
    triage_frame = pd.DataFrame(all_triage)
    correlation_frame.to_csv(output_dir / "component_ablation.csv", index=False)
    triage_frame.to_csv(output_dir / "surface_proxy_triage.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    key_rows = correlation_frame[
        correlation_frame["score"].isin(
            ["D_full", "D_no_correctness_paper", "D_no_correctness_renormalized"]
        )
    ][
        [
            "file",
            "score",
            "n",
            "spearman_rho",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
            "permutation_p",
        ]
    ]
    print(key_rows.to_string(index=False))


if __name__ == "__main__":
    main()
