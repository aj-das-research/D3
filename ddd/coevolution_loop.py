"""
Co-evolution batch builder (paper Section 3.6).

This script consumes per-question heterogeneous agent outputs for training-split
questions, rebuilds the hard-case buffer, writes per-agent training batches
(Appendix D.6 sampling rules), and optionally runs user-provided commands for
the next Agent-R/D/C update. Each training row carries a `regularizer_hint`
so that update scripts can add the diversity penalty of Eq. 9.

Scope: the driver builds batches from the predictions file it is given and
delegates adapter updates to the commands you pass. Re-scoring with the
updated adapters between iterations is left to those commands.

Fast local smoke test:
    python ddd/coevolution_loop.py --dry_run --max_iterations 1

Kaggle style:
    python coevolution_loop.py \
        --predictions_jsonl /kaggle/working/heterogeneous_train_predictions.jsonl \
        --split_manifest /kaggle/input/splits/mmlu_pro_hard_split_manifest.csv \
        --output_dir /kaggle/working/coevolution \
        --max_iterations 30
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from hard_case_buffer import AgentOutput, DisagreementResult, HardCaseBuffer
except ModuleNotFoundError:
    from collections import deque
    from dataclasses import dataclass

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

    @dataclass
    class BufferEntry:
        result: DisagreementResult
        iteration: int
        rd_gap: float
        c_wrong_conf: Optional[float]

    class HardCaseBuffer:
        RD_GAP_THRESH = 0.2
        C_CONF_THRESH = 0.6

        def __init__(self, max_size: int = 500, sample_size: int = 200):
            self.max_size = max_size
            self.sample_size = sample_size
            self.buffer = deque(maxlen=max_size)
            self._total_added = 0

        def add(self, result: DisagreementResult, iteration: int, c_wrong_conf: Optional[float] = None) -> bool:
            if not result.is_hard_case:
                return False
            conf = {o.agent_name: o.confidence for o in result.agent_outputs}
            rd_gap = abs(conf.get("R", 0.0) - conf.get("D", 0.0))
            self.buffer.append(BufferEntry(result, iteration, rd_gap, c_wrong_conf))
            self._total_added += 1
            return True

        def sample_for_agent_R(self, k: Optional[int] = None):
            k = k or self.sample_size
            rows = [e for e in self.buffer if e.rd_gap >= self.RD_GAP_THRESH]
            return sorted(rows, key=lambda e: e.result.d_score, reverse=True)[:k]

        def sample_for_agent_D(self, k: Optional[int] = None):
            return self.sample_for_agent_R(k)

        def sample_for_agent_C(self, k: Optional[int] = None):
            k = k or self.sample_size
            rows = [
                e for e in self.buffer
                if e.c_wrong_conf is not None and e.c_wrong_conf >= self.C_CONF_THRESH
            ]
            return sorted(rows, key=lambda e: e.c_wrong_conf, reverse=True)[:k]

        def stats(self) -> Dict:
            return {"size": len(self.buffer), "total_added": self._total_added}

        def __len__(self):
            return len(self.buffer)
from diversity_regularizer import DiversityRegularizer
from split_guard import select_records


LETTERS = "ABCDEFGHIJ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_jsonl",
        default=str(
            PROJECT_ROOT
            / "experiments"
            / "static_disagreement"
            / "outputs"
            / "heterogeneous_same_family_n50.jsonl"
        ),
        help="Per-question JSONL written by run_heterogeneous.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "co_evolution"),
    )
    parser.add_argument(
        "--split_manifest",
        default=str(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "mmlu_pro_hard_split_manifest.csv"
        ),
        help="Immutable question_id/split CSV. Co-evolution accepts train rows only.",
    )
    parser.add_argument(
        "--split_manifest_sha256",
        default=os.environ.get("SPLIT_MANIFEST_SHA256"),
        help="Optional expected manifest digest; mismatch aborts before batch creation.",
    )
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--hard_threshold", type=float, default=0.5)
    parser.add_argument("--diversity_weight", type=float, default=0.05)
    parser.add_argument("--min_js_divergence", type=float, default=0.03)
    parser.add_argument("--dry_run", action="store_true", help="Build batches only; do not run update commands.")
    parser.add_argument("--agent_r_cmd", default=os.environ.get("AGENT_R_UPDATE_CMD", ""))
    parser.add_argument("--agent_d_cmd", default=os.environ.get("AGENT_D_UPDATE_CMD", ""))
    parser.add_argument("--agent_c_cmd", default=os.environ.get("AGENT_C_UPDATE_CMD", ""))
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _agent_answer_text(agent_name: str, row: Dict) -> str:
    per_agent = row.get("per_agent", {})
    agent_row = per_agent.get(agent_name, {})
    answer = str(agent_row.get("answer", "")).strip()
    option_text = str(agent_row.get("option_text", "")).strip()
    if option_text and not answer.endswith(option_text):
        return f"{answer}. {option_text}"
    return answer


def record_to_disagreement(row: Dict, threshold: float) -> DisagreementResult:
    outputs = []
    for name in ["R", "D", "C"]:
        agent_row = row.get("per_agent", {}).get(name, {})
        outputs.append(
            AgentOutput(
                agent_name=name,
                answer=_agent_answer_text(name, row),
                confidence=float(agent_row.get("confidence", 0.0) or 0.0),
            )
        )
    d_score = float(row.get("D_q", row.get("d_score", 0.0)) or 0.0)
    return DisagreementResult(
        question_id=str(row.get("question_id")),
        question=str(row.get("question", "")),
        agent_outputs=outputs,
        d_semantic=float(row.get("D_semantic", row.get("d_semantic", 0.0)) or 0.0),
        d_confidence=float(row.get("D_confidence", row.get("d_confidence", 0.0)) or 0.0),
        d_score=d_score,
        is_hard_case=bool(row.get("is_hard_case", d_score > threshold)),
    )


def c_wrong_conf(row: Dict) -> Optional[float]:
    correct = str(row.get("correct_answer", "")).strip().upper()[:1]
    c_row = row.get("per_agent", {}).get("C", {})
    c_answer = str(c_row.get("answer", "")).strip().upper()[:1]
    if correct and c_answer and c_answer != correct:
        return float(c_row.get("confidence", 0.0) or 0.0)
    return None


def rebuild_buffer(rows: List[Dict], args: argparse.Namespace) -> HardCaseBuffer:
    buffer = HardCaseBuffer(max_size=args.buffer_size, sample_size=args.sample_size)
    for row in rows:
        result = record_to_disagreement(row, args.hard_threshold)
        buffer.add(result, iteration=int(row.get("iteration", 0) or 0), c_wrong_conf=c_wrong_conf(row))
    return buffer


def make_training_row(agent_name: str, entry, source_row: Dict, regularizer: DiversityRegularizer) -> Dict:
    correct = str(source_row.get("correct_answer", "")).strip().upper()[:1]
    options = source_row.get("options", [])
    per_agent = source_row.get("per_agent", {})
    diversity = regularizer.score_record(source_row)

    if agent_name == "C":
        c_answer = str(per_agent.get("C", {}).get("answer", "")).strip().upper()[:1]
        target_confidence = 0.2 if correct and c_answer and c_answer != correct else 0.8
        objective = "calibration"
    elif agent_name == "R":
        target_confidence = 0.9
        objective = "reasoning_on_rd_disagreement"
    else:
        target_confidence = 0.9
        objective = "domain_update_on_rd_disagreement"

    return {
        "question_id": entry.result.question_id,
        "question": entry.result.question,
        "options": options,
        "target_answer": correct,
        "target_confidence": target_confidence,
        "agent": agent_name,
        "objective": objective,
        "d_score": entry.result.d_score,
        "rd_gap": entry.rd_gap,
        "c_wrong_conf": entry.c_wrong_conf,
        "diversity": diversity,
        "regularizer_hint": {
            "weight": regularizer.weight,
            "penalty": diversity["penalty"],
            "reason": "increase inter-agent distribution diversity when JS divergence collapses",
        },
    }


def rows_by_id(rows: List[Dict]) -> Dict[str, Dict]:
    return {str(row.get("question_id")): row for row in rows}


def build_iteration_batches(
    rows: List[Dict],
    buffer: HardCaseBuffer,
    iteration: int,
    output_dir: Path,
    regularizer: DiversityRegularizer,
) -> Dict[str, Path]:
    lookup = rows_by_id(rows)
    batches = {
        "R": buffer.sample_for_agent_R(),
        "D": buffer.sample_for_agent_D(),
        "C": buffer.sample_for_agent_C(),
    }

    paths = {}
    for agent_name, entries in batches.items():
        train_rows = []
        for entry in entries:
            source = lookup.get(entry.result.question_id, {})
            train_rows.append(make_training_row(agent_name, entry, source, regularizer))
        path = output_dir / f"iter_{iteration:02d}" / f"agent_{agent_name.lower()}_hard_batch.jsonl"
        write_jsonl(path, train_rows)
        paths[agent_name] = path
    return paths


def run_update(command_template: str, agent_name: str, batch_path: Path, iteration: int, dry_run: bool) -> Dict:
    if not command_template:
        return {"agent": agent_name, "status": "skipped", "reason": "no command configured"}

    command = command_template.format(
        agent=agent_name,
        batch=str(batch_path),
        iteration=iteration,
    )
    if dry_run:
        return {"agent": agent_name, "status": "dry_run", "command": command}

    completed = subprocess.run(shlex.split(command), capture_output=True, text=True, check=False)
    return {
        "agent": agent_name,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def write_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = read_jsonl(Path(args.predictions_jsonl))
    regularizer = DiversityRegularizer(
        weight=args.diversity_weight,
        min_js_divergence=args.min_js_divergence,
    )

    if not rows:
        demo_path = output_dir / "demo_empty_manifest.json"
        write_manifest(
            demo_path,
            {
                "status": "no_prediction_rows",
                "message": "Run run_heterogeneous.py first or pass --predictions_jsonl.",
            },
        )
        print(f"No prediction rows found. Wrote {demo_path}")
        return

    rows, split_report = select_records(
        rows,
        args.split_manifest,
        {"train"},
        expected_sha256=args.split_manifest_sha256,
    )
    print(
        "Strict split: "
        f"{split_report['selected_rows']} training rows selected from "
        f"{split_report['input_rows']} input rows"
    )

    manifests = []
    for iteration in range(args.max_iterations):
        buffer = rebuild_buffer(rows, args)
        batch_paths = build_iteration_batches(rows, buffer, iteration, output_dir, regularizer)
        updates = [
            run_update(args.agent_r_cmd, "R", batch_paths["R"], iteration, args.dry_run),
            run_update(args.agent_d_cmd, "D", batch_paths["D"], iteration, args.dry_run),
            run_update(args.agent_c_cmd, "C", batch_paths["C"], iteration, args.dry_run),
        ]
        manifest = {
            "iteration": iteration,
            "input_rows": len(rows),
            "split_protocol": split_report,
            "buffer_stats": buffer.stats(),
            "batch_paths": {k: str(v) for k, v in batch_paths.items()},
            "updates": updates,
        }
        manifest_path = output_dir / f"iter_{iteration:02d}" / "manifest.json"
        write_manifest(manifest_path, manifest)
        manifests.append(str(manifest_path))
        print(f"Iteration {iteration}: wrote {manifest_path}")

    write_manifest(
        output_dir / "coevolution_summary.json",
        {"split_protocol": split_report, "manifests": manifests},
    )


if __name__ == "__main__":
    main()
