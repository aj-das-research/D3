"""
Static (Iteration 0) disagreement: same-family heterogeneous Agent-R/D/C
(paper Sections 5 and 6.1, Table 1).

Run this on Kaggle GPU after adding these Kaggle inputs:
  1. Qwen2.5-3B-Instruct base model
  2. Agent-R adapter zip or extracted adapter folder
  3. Agent-D results.zip or extracted agent_d_v0 folder
  4. Agent-C results.zip or extracted agent_c_v0 folder
  5. This project/data folder, or at least mmlu_pro_hard_subset.csv/json

Example Kaggle command:
  !python /kaggle/working/run_heterogeneous.py \
      --split_manifest /kaggle/input/splits/mmlu_pro_hard_split_manifest.csv \
      --eval_split validation --limit 50

Full run:
  !python /kaggle/working/run_heterogeneous.py \
      --split_manifest /kaggle/input/splits/mmlu_pro_hard_split_manifest.csv \
      --eval_split test

Outputs:
  /kaggle/working/heterogeneous_same_family_<split>.jsonl
  /kaggle/working/heterogeneous_same_family_<split>.run_summary.csv
  /kaggle/working/heterogeneous_same_family_<split>.run_summary.json
"""

import argparse
import ast
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
sys.path.insert(0, str(PROJECT_ROOT / "ddd"))

from disagreement_scorer import AgentOutput, DisagreementScorer
from hard_case_buffer import HardCaseBuffer
from split_guard import select_records


LETTERS = "ABCDEFGHIJ"

AGENT_SYSTEMS = {
    "R": (
        "You are Agent-R, a careful mathematical and symbolic reasoner. "
        "Solve deliberately and choose the best option."
    ),
    "D": (
        "You are Agent-D, a domain specialist. Use subject-matter knowledge "
        "and choose the best option."
    ),
    "C": (
        "You are Agent-C, a calibrated analyst. Choose the best option while "
        "keeping uncertainty honest."
    ),
}


def existing_dir(path: Optional[str]) -> Optional[str]:
    return path if path and Path(path).is_dir() else None


def existing_file(path: Optional[str]) -> Optional[str]:
    return path if path and Path(path).is_file() else None


def first_existing_dir(candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return candidate
    return None


def first_existing_file(candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def find_qwen_model(explicit: Optional[str]) -> str:
    if existing_dir(explicit):
        return explicit

    candidates = [
        os.environ.get("BASE_MODEL_PATH", ""),
        "/kaggle/input/qwen2-5-3b-instruct",
        "/kaggle/input/qwen2.5-3b-instruct",
        "/kaggle/input/qwen25-3b-instruct",
        *glob.glob("/kaggle/input/*qwen*", recursive=False),
        *glob.glob("/kaggle/input/**/*qwen*", recursive=True),
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir() and (path / "config.json").exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find Qwen base model. Pass --base_model /kaggle/input/... "
        "or set BASE_MODEL_PATH."
    )


def find_data_file(explicit: Optional[str]) -> str:
    if existing_file(explicit):
        return explicit

    candidates = [
        os.environ.get("MMLU_PATH", ""),
        str(PROJECT_ROOT / "data" / "mmlu_pro_hard_subset.csv"),
        str(PROJECT_ROOT / "data" / "mmlu_pro_hard_subset.json"),
        "/kaggle/working/data/mmlu_pro_hard_subset.csv",
        "/kaggle/working/data/mmlu_pro_hard_subset.json",
        *glob.glob("/kaggle/input/**/mmlu_pro_hard_subset.csv", recursive=True),
        *glob.glob("/kaggle/input/**/mmlu_pro_hard_subset.json", recursive=True),
    ]
    found = first_existing_file(candidates)
    if found:
        return found
    raise FileNotFoundError("Could not find mmlu_pro_hard_subset.csv/json.")


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


def find_adapter(
    explicit: Optional[str],
    work_dir: Path,
    label: str,
    preferred_name: str,
    glob_patterns: List[str],
) -> Path:
    if explicit:
        root = unzip_if_needed(explicit, work_dir, label)
        return locate_adapter(root, preferred_name)

    candidates = []
    env_name = f"AGENT_{label.upper()}_PATH"
    if os.environ.get(env_name):
        candidates.append(os.environ[env_name])
    for pattern in glob_patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    for candidate in candidates:
        try:
            root = unzip_if_needed(candidate, work_dir, label)
            return locate_adapter(root, preferred_name)
        except Exception:
            continue

    raise FileNotFoundError(f"Could not find Agent-{label.upper()} adapter.")


def load_temperature(adapter_dir: Path) -> float:
    temp_path = adapter_dir / "temperature.json"
    if not temp_path.exists():
        return 1.0
    try:
        data = json.loads(temp_path.read_text(encoding="utf-8"))
        for key in ["temperature", "T", "best_temperature"]:
            if key in data:
                value = float(data[key])
                return value if value > 0 else 1.0
    except Exception:
        pass
    return 1.0


def parse_options(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, dict):
        return [str(value[k]) for k in sorted(value)]
    text = str(value)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
            if isinstance(parsed, dict):
                return [str(parsed[k]) for k in sorted(parsed)]
        except Exception:
            pass
    return [text]


def load_questions(data_path: str, limit: Optional[int]) -> List[Dict]:
    path = Path(data_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        rows = df.to_dict(orient="records")
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))

    normalized = []
    for idx, row in enumerate(rows):
        options = parse_options(row.get("options", []))
        if len(options) < 2:
            continue
        normalized.append(
            {
                "question_id": str(row.get("question_id", row.get("id", idx))),
                "question": str(row.get("question", "")),
                "options": options[: len(LETTERS)],
                "answer": str(row.get("answer", "")).strip().upper()[:1],
                "category": row.get("category", ""),
                "difficulty_score": row.get(
                    "difficulty_score", row.get("final_difficulty")
                ),
            }
        )
    return normalized[:limit] if limit else normalized


def format_options(options: List[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(options))


def build_prompt(tokenizer, agent_name: str, question: str, options: List[str]) -> str:
    user = (
        "Answer this multiple-choice question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "Return only the single best answer letter."
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEMS[agent_name]},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def letter_token_ids(tokenizer, num_options: int) -> List[int]:
    ids = []
    for letter in LETTERS[:num_options]:
        candidates = []
        for text in [letter, f" {letter}"]:
            tokenized = tokenizer.encode(text, add_special_tokens=False)
            if tokenized:
                candidates.append(tokenized[-1])
        if not candidates:
            raise ValueError(f"Could not tokenize answer letter {letter}")
        ids.append(candidates[0])
    return ids


@torch.inference_mode()
def score_one(
    model,
    tokenizer,
    agent_name: str,
    question: Dict,
    temperature: float,
) -> Dict:
    prompt = build_prompt(
        tokenizer, agent_name, question["question"], question["options"]
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]

    token_ids = letter_token_ids(tokenizer, len(question["options"]))
    choice_logits = torch.stack([logits[token_id] for token_id in token_ids])
    probs = torch.softmax(choice_logits / max(temperature, 1e-6), dim=0)
    idx = int(torch.argmax(probs).item())
    answer = LETTERS[idx]
    confidence = float(probs[idx].item())

    return {
        "answer": answer,
        "confidence": confidence,
        "option_text": question["options"][idx],
        "all_probs": {
            LETTERS[i]: float(probs[i].item())
            for i in range(len(question["options"]))
        },
    }


def majority_vote(agent_rows: Dict[str, Dict]) -> Tuple[str, float]:
    counts: Dict[str, float] = {}
    for row in agent_rows.values():
        counts[row["answer"]] = counts.get(row["answer"], 0.0) + row["confidence"]
    winner = max(counts, key=counts.get)
    return winner, counts[winner] / max(sum(counts.values()), 1e-9)


def load_done(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--mmlu_path", default=None)
    parser.add_argument("--agent_r", default=None)
    parser.add_argument("--agent_d", default=None)
    parser.add_argument("--agent_c", default=None)
    parser.add_argument("--output_dir", default="/kaggle/working")
    parser.add_argument("--work_dir", default="/kaggle/working/hetero_work")
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
    parser.add_argument("--max_memory_gb", type=int, default=14)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_model = find_qwen_model(args.base_model)
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
    agent_d = find_adapter(
        args.agent_d,
        work_dir,
        "d",
        "agent_d_v0",
        [
            "/kaggle/input/**/*agent*d*results*.zip",
            "/kaggle/input/**/agent_d_v0",
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

    print("Base model:", base_model)
    print("Data path :", data_path)
    print("Agent R   :", agent_r)
    print("Agent D   :", agent_d)
    print("Agent C   :", agent_c)

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
        f"Questions : {len(questions)} ({args.eval_split}; "
        f"manifest {split_report['manifest_sha256'][:12]})"
    )

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
            i: f"{args.max_memory_gb}GiB"
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

    model = PeftModel.from_pretrained(model, str(agent_r), adapter_name="R")
    model.load_adapter(str(agent_d), adapter_name="D")
    model.load_adapter(str(agent_c), adapter_name="C")
    model.eval()

    temperatures = {"R": 1.0, "D": 1.0, "C": load_temperature(agent_c)}
    print("Temperatures:", temperatures)

    output_path = output_dir / f"heterogeneous_same_family_{args.eval_split}.jsonl"
    done = load_done(output_path)
    if done:
        print(f"Resuming: {len(done)} questions already written.")

    scorer = DisagreementScorer(threshold=args.threshold)
    buffer = HardCaseBuffer(max_size=500, sample_size=200)

    with output_path.open("a", encoding="utf-8") as fout:
        for question in tqdm(questions, desc="Agent-R/D/C"):
            qid = question["question_id"]
            if qid in done:
                continue

            per_agent = {}
            scorer_outputs = []
            for agent_name in ["R", "D", "C"]:
                model.set_adapter(agent_name)
                row = score_one(
                    model,
                    tokenizer,
                    agent_name,
                    question,
                    temperatures[agent_name],
                )
                per_agent[agent_name] = row
                scorer_outputs.append(
                    AgentOutput(
                        agent_name=agent_name,
                        answer=f"{row['answer']}. {row['option_text']}",
                        confidence=row["confidence"],
                    )
                )

            disagreement = scorer.score(qid, question["question"], scorer_outputs)
            buffer.add(disagreement, iteration=0, c_wrong_conf=None)
            ensemble_answer, ensemble_conf = majority_vote(per_agent)

            record = {
                "question_id": qid,
                "protocol_split": question["protocol_split"],
                "category": question["category"],
                "difficulty_score": question["difficulty_score"],
                "correct_answer": question["answer"],
                "per_agent": per_agent,
                "ensemble_answer": ensemble_answer,
                "ensemble_confidence": ensemble_conf,
                "D_answer": disagreement.d_answer,
                "D_semantic": disagreement.d_semantic,
                "D_confidence": disagreement.d_confidence,
                "D_q": disagreement.d_score,
                "is_hard_case": disagreement.is_hard_case,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    df = pd.read_json(output_path, lines=True)
    clean = df.dropna(subset=["D_q", "difficulty_score"]).copy()
    rho, p_value = spearmanr(clean["D_q"], clean["difficulty_score"])

    def acc_for(agent_name: str) -> float:
        return float(
            clean.apply(
                lambda row: row["per_agent"][agent_name]["answer"]
                == row["correct_answer"],
                axis=1,
            ).mean()
        )

    table = {
        "run": "heterogeneous_agent_r_d_c",
        "evaluation_split": args.eval_split,
        "split_manifest_sha256": split_report["manifest_sha256"],
        "n_questions": int(len(df)),
        "n_scored": int(len(clean)),
        "spearman_rho_Dq_vs_difficulty": None
        if math.isnan(rho)
        else float(rho),
        "spearman_p_value": None if math.isnan(p_value) else float(p_value),
        "mean_D_q": float(clean["D_q"].mean()),
        "hard_case_rate": float(clean["is_hard_case"].mean()),
        "agent_r_accuracy": acc_for("R"),
        "agent_d_accuracy": acc_for("D"),
        "agent_c_accuracy": acc_for("C"),
        "ensemble_accuracy": float(
            clean.apply(
                lambda row: row["ensemble_answer"] == row["correct_answer"],
                axis=1,
            ).mean()
        ),
    }

    csv_path = output_dir / f"heterogeneous_same_family_{args.eval_split}.run_summary.csv"
    json_path = output_dir / f"heterogeneous_same_family_{args.eval_split}.run_summary.json"
    pd.DataFrame([table]).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(table, indent=2), encoding="utf-8"
    )

    print("\nTABLE 1 HETEROGENEOUS ROW")
    print(json.dumps(table, indent=2))
    print(f"\nWrote: {output_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Buffer stats: {buffer.stats()}")


if __name__ == "__main__":
    main()
