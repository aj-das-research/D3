"""Create an immutable stratified split manifest for MMLU-Pro hard.

The split is created before any agent specialization, buffer construction,
co-evolution, or hyperparameter selection:

- train: 2,416 questions
- validation: 300 questions
- test: 300 questions

Strata combine category and global difficulty quintile. Allocation uses a
largest-remainder method so validation and test have exact requested sizes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "data" / "mmlu_pro_hard_subset.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "splits"
MANIFEST_PATH = OUTPUT_DIR / "mmlu_pro_hard_split_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "mmlu_pro_hard_split_summary.json"
SEED = 2026
VALIDATION_SIZE = 300
TEST_SIZE = 300


def load_source() -> pd.DataFrame:
    records = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)
    required = {"question_id", "category", "difficulty_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source is missing required columns: {sorted(missing)}")
    frame["question_id"] = frame["question_id"].astype(str)
    frame["category"] = frame["category"].astype(str)
    frame["difficulty_score"] = pd.to_numeric(
        frame["difficulty_score"], errors="raise"
    )
    if frame["question_id"].duplicated().any():
        duplicates = frame.loc[
            frame["question_id"].duplicated(keep=False), "question_id"
        ].tolist()
        raise ValueError(f"Duplicate question IDs: {duplicates[:10]}")
    frame["difficulty_quintile"] = pd.qcut(
        frame["difficulty_score"],
        q=5,
        labels=False,
        duplicates="drop",
    ).astype(int)
    frame["stratum"] = (
        frame["category"].str.strip().str.lower()
        + "::q"
        + frame["difficulty_quintile"].astype(str)
    )
    return frame


def largest_remainder_allocation(
    counts: pd.Series,
    requested_total: int,
) -> dict[str, int]:
    if requested_total < 0 or requested_total > int(counts.sum()):
        raise ValueError("Requested allocation is outside the available range.")
    exact = counts.astype(float) * requested_total / float(counts.sum())
    allocation = np.floor(exact).astype(int)
    remainder = requested_total - int(allocation.sum())
    fractions = (exact - allocation).sort_values(ascending=False)
    for stratum in fractions.index:
        if remainder <= 0:
            break
        if allocation[stratum] < counts[stratum]:
            allocation[stratum] += 1
            remainder -= 1
    if remainder != 0:
        raise RuntimeError("Could not complete largest-remainder allocation.")
    return {stratum: int(value) for stratum, value in allocation.items()}


def build_manifest(
    validation_size: int = VALIDATION_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    frame = load_source()
    counts = frame.groupby("stratum").size().sort_index()
    test_allocation = largest_remainder_allocation(counts, test_size)
    remaining_counts = counts - pd.Series(test_allocation)
    validation_allocation = largest_remainder_allocation(
        remaining_counts,
        validation_size,
    )

    generator = np.random.default_rng(seed)
    split_by_id: dict[str, str] = {}
    for stratum, group in frame.groupby("stratum", sort=True):
        question_ids = group["question_id"].tolist()
        generator.shuffle(question_ids)
        test_count = test_allocation[stratum]
        validation_count = validation_allocation[stratum]
        for question_id in question_ids[:test_count]:
            split_by_id[question_id] = "test"
        for question_id in question_ids[
            test_count : test_count + validation_count
        ]:
            split_by_id[question_id] = "validation"
        for question_id in question_ids[test_count + validation_count :]:
            split_by_id[question_id] = "train"

    manifest = frame[["question_id"]].copy()
    manifest["split"] = manifest["question_id"].map(split_by_id)
    if manifest["split"].isna().any():
        raise RuntimeError("At least one question was not assigned to a split.")
    return manifest.sort_values("question_id", kind="stable").reset_index(drop=True)


def validate_manifest(
    manifest: pd.DataFrame,
    validation_size: int = VALIDATION_SIZE,
    test_size: int = TEST_SIZE,
) -> dict:
    source = load_source()
    expected_counts = {
        "train": len(source) - validation_size - test_size,
        "validation": validation_size,
        "test": test_size,
    }
    observed_counts = manifest["split"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise AssertionError(
            f"Unexpected split counts: {observed_counts}; expected {expected_counts}"
        )
    if manifest["question_id"].duplicated().any():
        raise AssertionError("Manifest contains duplicate question IDs.")
    if set(manifest["question_id"]) != set(source["question_id"]):
        raise AssertionError("Manifest does not cover exactly the source question IDs.")

    audit_frame = manifest.merge(
        source[["question_id", "category", "difficulty_score"]],
        on="question_id",
        how="left",
        validate="one_to_one",
    )
    overall_category_share = source["category"].value_counts(normalize=True)
    maximum_category_share_gap = 0.0
    category_gaps = {}
    for split_name, group in audit_frame.groupby("split"):
        share = group["category"].value_counts(normalize=True)
        aligned = pd.concat(
            [overall_category_share.rename("overall"), share.rename("split")],
            axis=1,
        ).fillna(0.0)
        gaps = (aligned["split"] - aligned["overall"]).abs()
        maximum_category_share_gap = max(
            maximum_category_share_gap,
            float(gaps.max()),
        )
        category_gaps[split_name] = {
            category: float(value) for category, value in gaps.sort_index().items()
        }
    return {
        "source": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "seed": SEED,
        "counts": expected_counts,
        "unique_question_ids": int(manifest["question_id"].nunique()),
        "difficulty_range_by_split": {
            split_name: {
                "min": float(group["difficulty_score"].min()),
                "max": float(group["difficulty_score"].max()),
                "mean": float(group["difficulty_score"].mean()),
                "std": float(group["difficulty_score"].std(ddof=0)),
            }
            for split_name, group in audit_frame.groupby("split")
        },
        "maximum_absolute_category_share_gap": maximum_category_share_gap,
        "category_share_absolute_gaps": category_gaps,
    }


def write_manifest() -> dict:
    manifest = build_manifest()
    summary = validate_manifest(manifest)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    summary["manifest"] = str(MANIFEST_PATH.relative_to(REPO_ROOT))
    summary["manifest_sha256"] = digest
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = write_manifest()
    print(json.dumps(result, indent=2))
