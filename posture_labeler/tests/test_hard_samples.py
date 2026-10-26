"""Generic queue validation tests; all mutations use temporary fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from posture_labeler.server import HardSampleQueue
from posture_labeler.mine_hard_samples import geometry, same_location, select


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="labeler-queue-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = SimpleNamespace(identity="dataset-a", frames={
            1: {"person_ids": [1, 2]}, 2: {"person_ids": [1]},
        })
        self.queue = HardSampleQueue(self.dataset, self.root)
        self.manifest = {"version": 1, "dataset_id": "dataset-a", "batch": "sample",
                         "targets": [[1, 2]], "prediction": "MUST_NOT_LEAK"}
        (self.root / "sample").mkdir()
        self.save()

    def save(self):
        (self.root / "sample" / "queue.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_empty_when_unconfigured(self):
        self.assertEqual(self.queue.current()["targets"], [])
        self.assertFalse((self.root / "active.json").exists())

    def test_activate_and_reopen(self):
        result = self.queue.activate("sample")
        self.assertEqual(result["targets"], [[1, 2]])
        self.assertNotIn("prediction", result)
        self.assertEqual(HardSampleQueue(self.dataset, self.root).current(), result)
        pointer = json.loads((self.root / "active.json").read_text())
        self.assertEqual(pointer, {"version": 1, "batch": "sample"})

    def test_reload_single_source(self):
        self.queue.activate("sample")
        self.manifest["targets"] = [[2, 1]]
        self.save()
        self.assertEqual(self.queue.current()["targets"], [[2, 1]])

    def test_wrong_dataset_disables_read_but_blocks_install(self):
        self.queue.activate("sample")
        before = (self.root / "active.json").read_bytes()
        self.manifest["dataset_id"] = "other"
        self.save()
        self.assertEqual(self.queue.current()["status"], "dataset_mismatch")
        self.assertEqual(self.queue.current()["targets"], [])
        with self.assertRaises(ValueError):
            self.queue.activate("sample")
        self.assertEqual((self.root / "active.json").read_bytes(), before)

    def test_invalid_paths(self):
        for name in ("../sample", "/sample", r"..\sample", ".", "", "missing", None):
            with self.assertRaises(ValueError):
                self.queue.activate(name)
        self.assertFalse((self.root / "active.json").exists())

    def test_bad_targets_do_not_activate(self):
        for targets in ([[1,2],[1,2]], [[99,1]], [[1,99]], [[True,1]], [["1",2]], [[1]], None):
            self.manifest["targets"] = targets
            self.save()
            with self.assertRaises(ValueError):
                self.queue.activate("sample")
            self.assertFalse((self.root / "active.json").exists())

    def test_corrupt_manifest_and_wrong_batch(self):
        (self.root / "sample" / "queue.json").write_text("invalid", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.queue.activate("sample")
        self.manifest["batch"] = "different"
        self.save()
        with self.assertRaises(ValueError):
            self.queue.activate("sample")

    def test_read_does_not_modify_files(self):
        self.queue.activate("sample")
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        self.queue.current()
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.json")})


class MiningTests(unittest.TestCase):
    def test_missing_torso_is_not_mined(self):
        points = [{"index": i, "x": i+10, "y": i+20, "confidence": .1} for i in range(17)]
        self.assertIsNone(geometry(points))

    def test_location_matching(self):
        self.assertTrue(same_location(np.array([0,0,100,200]), np.array([2,2,102,202])))
        self.assertFalse(same_location(np.array([0,0,100,200]), np.array([500,500,600,700])))

    def test_selection_excludes_nearby_duplicates(self):
        candidates = [{"descriptor": np.ones(68)*i, "source_time_sec": t,
                       "box": np.array([0,0,100,200]), "segment": 1,
                       "reasons": ["pose_diversity"], "priority": 0}
                      for i,t in enumerate((0,1,4,8))]
        result = select(candidates, 10)
        for i, a in enumerate(result):
            for b in result[i+1:]:
                self.assertGreaterEqual(abs(a["source_time_sec"]-b["source_time_sec"]), 3)


if __name__ == "__main__":
    unittest.main()
