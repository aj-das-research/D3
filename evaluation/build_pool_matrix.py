"""
Build a model-pool response matrix for the hard subset from the public
MMLU-Pro model predictions (TIGER-AI-Lab/MMLU-Pro, folder `eval_results/`).

Each archive holds one model's predictions on the MMLU-Pro test set
(`question_id`, gold `answer`, predicted `pred`). For every question in
data/mmlu_pro_hard_subset.json the matrix stores 1 if the model's prediction
equals the gold answer, 0 if it does not (including unparsed predictions), and
leaves the cell empty if the model's file has no record for that question.
Models covering less than --min_coverage of the subset are dropped.

The matrix is the input to evaluation/difficulty_targets.py (Table 4 targets)
and to the triage task in evaluation/downstream.py (Table 5).

Usage:
    python evaluation/build_pool_matrix.py --download_dir pool_zips \
        --out data/pool/mmlu_pro_hard_pool_responses.csv
(--download_dir is reused if the archives are already there.)
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LISTING_URL = "https://api.github.com/repos/TIGER-AI-Lab/MMLU-Pro/contents/eval_results"


def download_archives(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    listing = json.loads(urllib.request.urlopen(LISTING_URL, timeout=60).read())
    for item in listing:
        if not item["name"].endswith(".zip"):
            continue
        path = target / item["name"]
        if path.exists() and path.stat().st_size == item["size"]:
            continue
        urllib.request.urlretrieve(item["download_url"], path)


def model_name(archive: Path) -> str:
    name = archive.name
    name = re.sub(r"^model_outputs_", "", name)
    name = re.sub(r"(\.json)?\.zip$", "", name)
    return name


def read_predictions(archive: Path) -> Dict[str, int]:
    correct: Dict[str, int] = {}
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            base = member.rsplit("/", 1)[-1]
            if not base.endswith(".json") or base.startswith("._") or "__MACOSX" in member or base == "summary.json":
                continue
            records = json.loads(z.read(member))
            if isinstance(records, dict):
                records = [records]
            for rec in records:
                if not isinstance(rec, dict) or "question_id" not in rec:
                    continue
                gold = str(rec.get("answer", "")).strip().upper()
                pred = str(rec.get("pred") or "").strip().upper()
                correct[str(rec["question_id"])] = int(bool(pred) and pred == gold)
    return correct


def build_matrix(archive_dir: Path, question_ids, min_coverage: float = 0.9) -> pd.DataFrame:
    ids = [str(q) for q in question_ids]
    columns = {}
    for archive in sorted(archive_dir.glob("*.zip")):
        preds = read_predictions(archive)
        col = [preds.get(q) for q in ids]
        coverage = sum(v is not None for v in col) / len(ids)
        if coverage >= min_coverage:
            columns[model_name(archive)] = col
        else:
            print(f"skip {archive.name}: covers {coverage:.1%} of the subset")
    df = pd.DataFrame(columns, dtype="Int64")
    df.insert(0, "question_id", ids)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download_dir", required=True)
    p.add_argument("--questions", default=str(REPO_ROOT / "data" / "mmlu_pro_hard_subset.json"))
    p.add_argument("--min_coverage", type=float, default=0.9)
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    archive_dir = Path(args.download_dir)
    if not args.no_download:
        download_archives(archive_dir)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    df = build_matrix(archive_dir, [q["question_id"] for q in questions], args.min_coverage)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    acc = df.drop(columns=["question_id"]).mean()
    print(f"{len(df)} questions x {df.shape[1] - 1} models -> {args.out}")
    print(acc.sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
