"""All sampling conditions must preserve training-user folds and duplicate-linked groups."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/model_training"))
import pipeline_v2 as pipeline


class TestFrozenFolds(unittest.TestCase):
    def test_fewer_windows_do_not_reassign_users(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "folds.json"
            frozen = {"001": 0, "002": 1, "003": 2, "004": 3, "005": 4}
            path.write_text(json.dumps(frozen))
            prior = pipeline.FOLD_MAP
            try:
                pipeline.FOLD_MAP = path
                table = pd.DataFrame({"user_id": ["001", "002"], "label": ["walk", "bus"], "speed": [1., 5.]})
                self.assertEqual(pipeline.user_folds(table), frozen)
            finally:
                pipeline.FOLD_MAP = prior

    def test_duplicate_linked_users_cannot_split_across_folds(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "folds.json"
            path.write_text(json.dumps({"069": 0, "126": 1, "001": 2, "002": 3, "003": 4}))
            prior = pipeline.FOLD_MAP
            try:
                pipeline.FOLD_MAP = path
                table = pd.DataFrame({"user_id": ["069", "126"], "label": ["walk", "bus"], "speed": [1., 5.]})
                with self.assertRaises(AssertionError):
                    pipeline.user_folds(table)
            finally:
                pipeline.FOLD_MAP = prior


if __name__ == "__main__":
    unittest.main()
