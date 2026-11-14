import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ddd"))

from split_guard import (  # noqa: E402
    assert_disjoint_question_ids,
    load_split_assignments,
    select_records,
)


class SplitGuardTests(unittest.TestCase):
    def write_manifest(self, directory: str) -> Path:
        path = Path(directory) / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question_id", "split"])
            writer.writeheader()
            writer.writerows(
                [
                    {"question_id": "q1", "split": "train"},
                    {"question_id": "q2", "split": "validation"},
                    {"question_id": "q3", "split": "test"},
                ]
            )
        return path

    def test_selects_only_requested_split_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory)
            selected, report = select_records(
                [{"question_id": "q1"}, {"question_id": "q2"}, {"question_id": "q3"}],
                manifest,
                {"validation"},
            )
            self.assertEqual([row["question_id"] for row in selected], ["q2"])
            self.assertEqual(selected[0]["protocol_split"], "validation")
            self.assertEqual(report["selected_rows"], 1)
            self.assertEqual(len(report["manifest_sha256"]), 64)

    def test_fails_closed_for_unknown_question_id(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory)
            with self.assertRaisesRegex(ValueError, "absent from the split manifest"):
                select_records([{"question_id": "unknown"}], manifest, {"train"})

    def test_rejects_duplicate_manifest_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory)
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write("q1,test\n")
            with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
                load_split_assignments(manifest)

    def test_detects_cross_stage_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_question_ids(
                {
                    "train": [{"question_id": "q1"}],
                    "test": [{"question_id": "q1"}],
                }
            )


if __name__ == "__main__":
    unittest.main()
