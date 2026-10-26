"""Optional read-only historical check, excluded from generic test_*.py discovery.

From project root:
python -B -m unittest discover -s posture_labeler/tests -p check_legacy_batch.py -v
"""
import json
from pathlib import Path
import sqlite3
import unittest
import numpy as np

from posture_labeler.mine_hard_samples import same_location, geometry, select

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "hard_samples/batch_20261019_v1"


class HardSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queue = json.loads((BATCH / "queue.json").read_text(encoding="utf-8"))
        cls.audit = json.loads((BATCH / "selection_audit.json").read_text(encoding="utf-8"))
        cls.install = json.loads((BATCH / "installation.json").read_text(encoding="utf-8"))
        cls.targets = cls.audit["targets"]

    def test_target_identity_and_count(self):
        keys = [(t["output_frame"], t["person_id"]) for t in self.targets]
        self.assertEqual(len(keys), 96)
        self.assertEqual(len(set(keys)), 96)
        self.assertEqual(keys, [tuple(pair) for pair in self.queue["targets"]])
        self.assertEqual(len({f for f,p in keys}), 67)

    def test_targets_were_not_previously_labeled(self):
        uri = (BATCH / "annotations_before_queue.sqlite3").as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            labeled = set(db.execute("SELECT output_frame,person_id FROM annotations WHERE label IS NOT NULL"))
        self.assertEqual(len(labeled), 255)
        self.assertFalse(labeled & {tuple(pair) for pair in self.queue["targets"]})

    def test_installation_preserved_labels(self):
        # Do not freeze the live database in a regression assertion: legitimate
        # human annotations will continue after this batch is delivered.
        self.assertTrue(self.install["human_annotations_unchanged"])
        self.assertEqual(self.install["human_label_count"], 255)

    def test_temporal_diversity(self):
        for i, a in enumerate(self.targets):
            for b in self.targets[i+1:]:
                dt = abs(a["source_time_sec"]-b["source_time_sec"])
                if same_location(np.array(a["box"]), np.array(b["box"])):
                    self.assertGreaterEqual(dt, 3)
                if dt > 0:
                    self.assertGreaterEqual(dt, 1)

    def test_public_queue_has_no_predictions(self):
        text = json.dumps(self.queue)
        for field in ("standing", "sitting", "confidence", "prediction", "selection_bucket"):
            self.assertNotIn(field, text)
        self.assertEqual(self.queue["purpose"], "development_annotation_not_independent_test")

    def test_nonfinite_or_missing_torso_is_not_mined(self):
        points = [{"index": i, "x": 10+i, "y": 20+i, "confidence": .1} for i in range(17)]
        self.assertIsNone(geometry(points))


if __name__ == "__main__":
    unittest.main()
