"""Fail-closed helpers for the train/validation/test protocol (paper Section 4, Appendix F.1).

The helpers deliberately depend only on the Python standard library so the
same guard can run in local tests and in Kaggle training/evaluation scripts.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VALID_SPLITS = frozenset({"train", "validation", "test"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_assignments(
    manifest_path: str | Path,
    expected_sha256: str | None = None,
) -> tuple[dict[str, str], str]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    digest = file_sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"Split manifest SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )

    assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question_id", "split"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Split manifest is missing columns: {sorted(missing_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            question_id = str(row.get("question_id", "")).strip()
            split = str(row.get("split", "")).strip().lower()
            if not question_id:
                raise ValueError(f"Blank question_id at {path}:{line_number}")
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"Invalid split {split!r} at {path}:{line_number}; "
                    f"expected one of {sorted(VALID_SPLITS)}"
                )
            if question_id in assignments:
                raise ValueError(f"Duplicate question_id in manifest: {question_id}")
            assignments[question_id] = split

    if not assignments:
        raise ValueError(f"Split manifest is empty: {path}")
    return assignments, digest


def select_records(
    records: Sequence[Mapping],
    manifest_path: str | Path,
    allowed_splits: Iterable[str],
    expected_sha256: str | None = None,
    require_all_records_in_manifest: bool = True,
) -> tuple[list[dict], dict]:
    allowed = {str(value).strip().lower() for value in allowed_splits}
    if not allowed or not allowed <= VALID_SPLITS:
        raise ValueError(
            f"allowed_splits must be a non-empty subset of {sorted(VALID_SPLITS)}"
        )

    assignments, digest = load_split_assignments(
        manifest_path,
        expected_sha256=expected_sha256,
    )
    record_ids = [str(row.get("question_id", "")).strip() for row in records]
    blank_ids = [index for index, question_id in enumerate(record_ids) if not question_id]
    if blank_ids:
        raise ValueError(f"Input records have blank question IDs at indices: {blank_ids[:10]}")
    duplicate_ids = [
        question_id
        for question_id, count in Counter(record_ids).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Input records contain duplicate question IDs: {duplicate_ids[:10]}")

    missing_ids = [question_id for question_id in record_ids if question_id not in assignments]
    if missing_ids and require_all_records_in_manifest:
        raise ValueError(
            f"{len(missing_ids)} input question IDs are absent from the split manifest: "
            f"{missing_ids[:10]}"
        )

    selected = []
    observed_counts: Counter[str] = Counter()
    for row, question_id in zip(records, record_ids):
        split = assignments.get(question_id)
        if split is None:
            continue
        observed_counts[split] += 1
        if split in allowed:
            copied = dict(row)
            copied["protocol_split"] = split
            selected.append(copied)

    if not selected:
        raise ValueError(
            f"No input records belong to requested split(s): {sorted(allowed)}"
        )

    report = {
        "manifest": str(Path(manifest_path)),
        "manifest_sha256": digest,
        "allowed_splits": sorted(allowed),
        "input_rows": len(records),
        "selected_rows": len(selected),
        "excluded_rows": len(records) - len(selected),
        "missing_manifest_rows": len(missing_ids),
        "observed_input_split_counts": dict(sorted(observed_counts.items())),
    }
    return selected, report


def assert_disjoint_question_ids(
    named_record_sets: Mapping[str, Sequence[Mapping]],
) -> None:
    ids_by_name = {
        name: {str(row.get("question_id", "")).strip() for row in rows}
        for name, rows in named_record_sets.items()
    }
    names = sorted(ids_by_name)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = ids_by_name[left_name] & ids_by_name[right_name]
            if overlap:
                raise ValueError(
                    f"Question-ID overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:10]}"
                )
