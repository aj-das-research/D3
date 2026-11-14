"""
Prepare the MMLU-Pro hard subset and the structural complexity proxy s(q)
(paper Section 4, Eq. 10; Appendix C).

Outputs are kept separate:
  - MMLU-Pro hard subset: N=3016 by default
  - Optional second dataset: GPQA or BBH

Examples:
  python data/prepare_mmlu_pro.py
  python data/prepare_mmlu_pro.py --secondary gpqa
  python data/prepare_mmlu_pro.py --secondary bbh --bbh-task causal_judgement
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from datasets import load_dataset


OUTPUT_DIR = Path(__file__).resolve().parent
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

HARD_CATEGORIES = {
    "math": 0.20,
    "physics": 0.15,
    "chemistry": 0.12,
    "engineering": 0.10,
    "law": 0.08,
    "biology": 0.05,
}


def compute_surface_difficulty(row: pd.Series) -> float:
    question = str(row["question"])
    options = row["options"]
    q_len = len(question.split())
    opt_lengths = [len(str(o).split()) for o in options]
    opt_len = float(np.mean(opt_lengths))
    math_tokens = sum(1 for c in question if c in "0123456789=+-*/^")
    opt_len_std = float(np.std(opt_lengths))
    similarity_score = 1.0 / (opt_len_std + 1.0)

    difficulty = (
        0.3 * min(q_len / 100, 1.0)
        + 0.2 * min(opt_len / 30, 1.0)
        + 0.3 * min(math_tokens / 20, 1.0)
        + 0.2 * similarity_score
    )
    return round(float(difficulty), 4)


def category_boost(category: str) -> float:
    cat = str(category).lower()
    for key, boost in HARD_CATEGORIES.items():
        if key in cat:
            return boost
    return 0.0


def records_from_mmlu(df: pd.DataFrame) -> List[Dict]:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "question_id": str(row.get("question_id", row.name)),
                "question": row["question"],
                "options": list(row["options"]),
                "answer": row["answer"],
                "answer_index": int(row["answer_index"])
                if "answer_index" in row and not pd.isna(row["answer_index"])
                else -1,
                "category": row["category"],
                "difficulty_score": float(row["difficulty_score"]),
                "final_difficulty": float(row["final_difficulty"]),
                "dataset": "mmlu_pro",
            }
        )
    return records


def prep_mmlu_pro(target_n: int) -> None:
    print("Loading MMLU-Pro test split...")
    df = pd.DataFrame(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
    print(f"Total MMLU-Pro questions: {len(df)}")

    print("Computing difficulty scores...")
    df["difficulty_score"] = df.apply(compute_surface_difficulty, axis=1)
    df["category_boost"] = df["category"].apply(category_boost)
    df["final_difficulty"] = (
        df["difficulty_score"] + df["category_boost"]
    ).clip(0, 1).round(4)

    hard_df = (
        df.sort_values("final_difficulty", ascending=False)
        .head(target_n)
        .reset_index(drop=True)
    )

    full_csv = OUTPUT_DIR / "mmlu_pro_full_with_difficulty.csv"
    hard_csv = OUTPUT_DIR / "mmlu_pro_hard_subset.csv"
    hard_json = OUTPUT_DIR / "mmlu_pro_hard_subset.json"
    sample_json = OUTPUT_DIR / "mmlu_pro_hard_sample_50.json"

    df.to_csv(full_csv, index=False)
    hard_df.to_csv(hard_csv, index=False)
    records = records_from_mmlu(hard_df)
    hard_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    sample_json.write_text(json.dumps(records[:50], indent=2), encoding="utf-8")

    print(f"Saved {full_csv}")
    print(f"Saved {hard_csv} with N={len(hard_df)}")
    print(f"Saved {hard_json}")
    print(f"Saved {sample_json}")


def normalize_gpqa_record(row: Dict, idx: int) -> Dict:
    correct = str(row.get("Correct Answer", row.get("correct_answer", "")))
    distractors = [
        str(row.get("Incorrect Answer 1", row.get("incorrect_answer_1", ""))),
        str(row.get("Incorrect Answer 2", row.get("incorrect_answer_2", ""))),
        str(row.get("Incorrect Answer 3", row.get("incorrect_answer_3", ""))),
    ]
    options = [correct, *[d for d in distractors if d]]
    return {
        "question_id": str(row.get("Record ID", row.get("id", idx))),
        "question": str(row.get("Question", row.get("question", ""))),
        "options": options,
        "answer": "A",
        "answer_index": 0,
        "category": "gpqa",
        "difficulty_score": 1.0,
        "dataset": "gpqa",
    }


def prep_gpqa() -> None:
    print("Loading GPQA diamond split...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    records = [normalize_gpqa_record(row, idx) for idx, row in enumerate(ds)]
    out = OUTPUT_DIR / "gpqa_diamond.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "gpqa_diamond.csv", index=False)
    print(f"Saved separate GPQA dataset with N={len(records)}")


def normalize_bbh_record(row: Dict, idx: int, task_name: str) -> Dict:
    target = str(row.get("target", row.get("answer", ""))).strip()
    return {
        "question_id": f"{task_name}_{idx}",
        "question": str(row.get("input", row.get("question", ""))),
        "options": [],
        "answer": target,
        "answer_index": -1,
        "category": task_name,
        "difficulty_score": 1.0,
        "dataset": "bbh",
    }


def prep_bbh(task_name: str) -> None:
    print(f"Loading BBH task: {task_name}")
    ds = load_dataset("lukaemon/bbh", task_name, split="test")
    records = [
        normalize_bbh_record(row, idx, task_name)
        for idx, row in enumerate(ds)
    ]
    stem = f"bbh_{task_name}"
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    print(f"Saved separate BBH dataset with N={len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu-n", type=int, default=3016)
    parser.add_argument(
        "--secondary",
        choices=["none", "gpqa", "bbh"],
        default="none",
        help="Optional extra dataset to prepare separately.",
    )
    parser.add_argument("--bbh-task", default="causal_judgement")
    args = parser.parse_args()

    prep_mmlu_pro(args.mmlu_n)
    if args.secondary == "gpqa":
        prep_gpqa()
    elif args.secondary == "bbh":
        prep_bbh(args.bbh_task)


if __name__ == "__main__":
    main()
