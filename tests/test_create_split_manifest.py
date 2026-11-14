"""Tests for the immutable MMLU-Pro split manifest."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from create_split_manifest import build_manifest, validate_manifest


class SplitManifestTests(unittest.TestCase):
    def test_split_is_deterministic(self):
        first = build_manifest()
        second = build_manifest()
        self.assertTrue(first.equals(second))

    def test_split_is_complete_disjoint_and_exact(self):
        manifest = build_manifest()
        summary = validate_manifest(manifest)
        self.assertEqual(
            summary["counts"],
            {"train": 2416, "validation": 300, "test": 300},
        )
        self.assertEqual(summary["unique_question_ids"], 3016)
        self.assertFalse(manifest["question_id"].duplicated().any())

    def test_category_imbalance_is_bounded(self):
        manifest = build_manifest()
        summary = validate_manifest(manifest)
        self.assertLess(summary["maximum_absolute_category_share_gap"], 0.02)


if __name__ == "__main__":
    unittest.main()
