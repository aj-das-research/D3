"""
Static (Iteration 0) disagreement: different-family heterogeneous agents,
Qwen Agent-R/C + Mistral Agent-D (paper Sections 5 and 6.1, Table 1).

Use this when Agent-R and Agent-C remain Qwen2.5-3B LoRA adapters but
Agent-D is trained on Mistral-7B. run_heterogeneous.py loads
all adapters into one base model, which is only valid when every adapter was
trained from the same base model.

Kaggle example:
  !python /kaggle/working/run_heterogeneous_multibackbone.py \
      --qwen_base_model /kaggle/input/qwen2-5-3b-instruct \
      --mistral_base_model /kaggle/input/mistral-7b-instruct-v0-3 \
      --agent_r /kaggle/input/agent-r/agent_r_v0_grpo_adapter.zip \
      --agent_c /kaggle/input/agent-c/results.zip \
      --agent_d /kaggle/input/agent-d-mistral/agent_d_mistral7b_v0.zip \
      --mmlu_path /kaggle/input/mmlu/mmlu_pro_hard_subset.csv \
      --split_manifest /kaggle/input/splits/mmlu_pro_hard_split_manifest.csv \
      --eval_split validation \
      --limit 50

Outputs:
  /kaggle/working/heterogeneous_diff_family_<split>.jsonl
  /kaggle/working/heterogeneous_diff_family_<split>.run_summary.csv
  /kaggle/working/heterogeneous_diff_family_<split>.run_summary.json
"""

import argparse
import glob
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from peft import PeftModel
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "ddd"))

from run_heterogeneous import (  # noqa: E402
    AgentOutput,
    DisagreementScorer,
    HardCaseBuffer,
    find_adapter,
    find_data_file,
    find_qwen_model,
    load_done,
    load_questions,
    load_temperature,
    majority_vote,
    score_one,
)
from split_guard import select_records  # noqa: E402


def existing_dir(path: Optional[str]) -> Optional[str]:
    return path if path and Path(path).is_dir() else None


def find_mistral_model(explicit: Optional[str]) -> str:
    if existing_dir(explicit):
        return explicit

    candidates = [
        os.environ.get("MISTRAL_MODEL_PATH", ""),
        os.environ.get("AGENT_D_BASE_MODEL_PATH", ""),
        "/kaggle/input/mistral-7b-instruct-v0-3",
        "/kaggle/input/mistral-7b-instruct-v0-2",
        "/kaggle/input/mistral-7b-instruct",
        *glob.glob("/kaggle/input/*mistral*7b*", recursive=False),
        *glob.glob("/kaggle/input/**/*mistral*7b*", recursive=True),
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir() and (path / "config.json").exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find Mistral-7B under /kaggle/input. Pass "
        "--mistral_base_model /kaggle/input/... or set MISTRAL_MODEL_PATH."
    )


def unzip_if_needed(path: str, work_dir: Path, label: str) -> Path:
    src = Path(path)
    if src.is_dir():
        return src
    if src.suffix.lower() != ".zip":
        raise ValueError(f"{label} must be an adapter directory or zip: {path}")

    dest = work_dir / f"unzipped_{label}"
    marker = dest / ".done"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(dest)
        marker.write_text("ok", encoding="utf-8")
    return dest


def locate_adapter(path: Path, preferred_name: str) -> Path:
    if (path / "adapter_config.json").exists():
        return path

    preferred = list(path.rglob(preferred_name))
    for candidate in preferred:
        if candidate.is_dir() and (candidate / "adapter_config.json").exists():
            return candidate

    configs = list(path.rglob("adapter_config.json"))
    if configs:
        return configs[0].parent

    raise FileNotFoundError(f"No adapter_config.json found under {path}")


def find_mistral_d_adapter(explicit: Optional[str], work_dir: Path) -> Path:
    if explicit:
        return locate_adapter(unzip_if_needed(explicit, work_dir, "mistral_d"), "agent_d_mistral7b_v0")

    candidates = []
    if os.environ.get("AGENT_D_MISTRAL_PATH"):
        candidates.append(os.environ["AGENT_D_MISTRAL_PATH"])
    candidates.extend(glob.glob("/kaggle/input/**/*agent*d*mistral*.zip", recursive=True))
    candidates.extend(glob.glob("/kaggle/input/**/agent_d_mistral7b_v0", recursive=True))

    for candidate in candidates:
        try:
            return locate_adapter(
                unzip_if_needed(candidate, work_dir, "mistral_d"),
                "agent_d_mistral7b_v0",
            )
        except Exception:
            continue

    raise FileNotFoundError("Could not find Mistral Agent-D adapter.")


def load_quantized_model(
    base_model: str,
    max_memory_gb: int,
    adapter_dir: Optional[Path],
    adapter_name: str,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {
            i: f"{max_memory_gb}GiB"
            for i in range(torch.cuda.device_count())
        }

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        max_memory=max_memory,
    )
    model.config.use_cache = True
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=adapter_name)
    model.eval()
    return model, tokenizer


def table_accuracy(clean: pd.DataFrame, agent_name: str) -> float:
    return float(
        clean.apply(
            lambda row: row["per_agent"][agent_name]["answer"] == row["correct_answer"],
            axis=1,
        ).mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_base_model", default=None)
    parser.add_argument("--mistral_base_model", default=None)
    parser.add_argument("--mmlu_path", default=None)
    parser.add_argument("--agent_r", default=None)
    parser.add_argument("--agent_c", default=None)
    parser.add_argument("--agent_d", default=None)
    parser.add_argument("--agent_d_base_only", action="store_true")
    parser.add_argument("--output_dir", default="/kaggle/working")
    parser.add_argument("--work_dir", default="/kaggle/working/hetero_mistral_d_work")
    parser.add_argument(
        "--split_manifest",
        default=str(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "mmlu_pro_hard_split_manifest.csv"
        ),
        help="Immutable question_id/split CSV. Evaluation is fail-closed.",
    )
    parser.add_argument(
        "--split_manifest_sha256",
        default=os.environ.get("SPLIT_MANIFEST_SHA256"),
    )
    parser.add_argument(
        "--eval_split",
        choices=["validation", "test"],
        default="validation",
        help="Use validation while developing; use test once for final reporting.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--qwen_max_memory_gb", type=int, default=5)
    parser.add_argument("--mistral_max_memory_gb", type=int, default=9)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    qwen_base = find_qwen_model(args.qwen_base_model)
    mistral_base = find_mistral_model(args.mistral_base_model)
    data_path = find_data_file(args.mmlu_path)
    agent_r = find_adapter(
        args.agent_r,
        work_dir,
        "r",
        "agent_r_v0_grpo_adapter",
        [
            "/kaggle/input/**/*agent*r*adapter*.zip",
            "/kaggle/input/**/*grpo*.zip",
            "/kaggle/input/**/agent_r_v0_grpo_adapter",
        ],
    )
    agent_c = find_adapter(
        args.agent_c,
        work_dir,
        "c",
        "agent_c_v0",
        [
            "/kaggle/input/**/*agent*c*results*.zip",
            "/kaggle/input/**/agent_c_v0",
        ],
    )
    agent_d = None if args.agent_d_base_only else find_mistral_d_adapter(args.agent_d, work_dir)

    print("Qwen R/C base :", qwen_base)
    print("Mistral D base:", mistral_base)
    print("Data path     :", data_path)
    print("Agent R       :", agent_r)
    print("Agent C       :", agent_c)
    print("Agent D       :", agent_d if agent_d else "base-only Mistral")

    questions = load_questions(data_path, None)
    questions, split_report = select_records(
        questions,
        args.split_manifest,
        {args.eval_split},
        expected_sha256=args.split_manifest_sha256,
    )
    if args.limit:
        questions = questions[: args.limit]
    print(
        f"Questions     : {len(questions)} ({args.eval_split}; "
        f"manifest {split_report['manifest_sha256'][:12]})"
    )

    qwen_model, qwen_tokenizer = load_quantized_model(
        qwen_base, args.qwen_max_memory_gb, agent_r, "R"
    )
    qwen_model.load_adapter(str(agent_c), adapter_name="C")

    d_model, d_tokenizer = load_quantized_model(
        mistral_base, args.mistral_max_memory_gb, agent_d, "D"
    )

    temperatures = {"R": 1.0, "D": load_temperature(agent_d) if agent_d else 1.0, "C": load_temperature(agent_c)}
    print("Temperatures  :", temperatures)

    output_path = (
        output_dir / f"heterogeneous_diff_family_{args.eval_split}.jsonl"
    )
    done = load_done(output_path)
    if done:
        print(f"Resuming: {len(done)} questions already written.")

    scorer = DisagreementScorer(threshold=args.threshold)
    buffer = HardCaseBuffer(max_size=500, sample_size=200)

    with output_path.open("a", encoding="utf-8") as fout:
        for question in tqdm(questions, desc="R/C Qwen + D Mistral"):
            qid = question["question_id"]
            if qid in done:
                continue

            per_agent = {}
            scorer_outputs = []
            for agent_name in ["R", "C"]:
                qwen_model.set_adapter(agent_name)
                row = score_one(
                    qwen_model,
                    qwen_tokenizer,
                    agent_name,
                    question,
                    temperatures[agent_name],
                )
                per_agent[agent_name] = row

            row_d = score_one(d_model, d_tokenizer, "D", question, temperatures["D"])
            per_agent["D"] = row_d

            ordered = {"R": per_agent["R"], "D": per_agent["D"], "C": per_agent["C"]}
            for agent_name, row in ordered.items():
                scorer_outputs.append(
                    AgentOutput(
                        agent_name=agent_name,
                        answer=f"{row['answer']}. {row['option_text']}",
                        confidence=row["confidence"],
                    )
                )

            disagreement = scorer.score(qid, question["question"], scorer_outputs)
            buffer.add(disagreement, iteration=0, c_wrong_conf=None)
            ensemble_answer, ensemble_conf = majority_vote(ordered)

            record = {
                "question_id": qid,
                "protocol_split": question["protocol_split"],
                "category": question["category"],
                "difficulty_score": question["difficulty_score"],
                "correct_answer": question["answer"],
                "per_agent": ordered,
                "ensemble_answer": ensemble_answer,
                "ensemble_confidence": ensemble_conf,
                "D_answer": disagreement.d_answer,
                "D_semantic": disagreement.d_semantic,
                "D_confidence": disagreement.d_confidence,
                "D_q": disagreement.d_score,
                "is_hard_case": disagreement.is_hard_case,
                "model_family": {
                    "R": "Qwen2.5-3B-Instruct adapter",
                    "D": "Mistral-7B-Instruct adapter" if agent_d else "Mistral-7B-Instruct base",
                    "C": "Qwen2.5-3B-Instruct adapter",
                },
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    df = pd.read_json(output_path, lines=True)
    clean = df.dropna(subset=["D_q", "difficulty_score"]).copy()
    rho, p_value = spearmanr(clean["D_q"], clean["difficulty_score"])

    table = {
        "run": "heterogeneous_qwen_r_c_mistral_d",
        "evaluation_split": args.eval_split,
        "split_manifest_sha256": split_report["manifest_sha256"],
        "n_questions": int(len(df)),
        "n_scored": int(len(clean)),
        "spearman_rho_Dq_vs_difficulty": None if math.isnan(rho) else float(rho),
        "spearman_p_value": None if math.isnan(p_value) else float(p_value),
        "mean_D_q": float(clean["D_q"].mean()),
        "hard_case_rate": float(clean["is_hard_case"].mean()),
        "agent_r_accuracy": table_accuracy(clean, "R"),
        "agent_d_accuracy": table_accuracy(clean, "D"),
        "agent_c_accuracy": table_accuracy(clean, "C"),
        "ensemble_accuracy": float(
            clean.apply(
                lambda row: row["ensemble_answer"] == row["correct_answer"],
                axis=1,
            ).mean()
        ),
    }

    csv_path = (
        output_dir / f"heterogeneous_diff_family_{args.eval_split}.run_summary.csv"
    )
    json_path = (
        output_dir / f"heterogeneous_diff_family_{args.eval_split}.run_summary.json"
    )
    pd.DataFrame([table]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print("\nTABLE 1 HETEROGENEOUS ROW - MISTRAL AGENT-D")
    print(json.dumps(table, indent=2))
    print(f"\nWrote: {output_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Buffer stats: {buffer.stats()}")


if __name__ == "__main__":
    main()
