"""All mutations in these tests use temporary data, never real annotations."""
import csv
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from posture_labeler.server import AnnotationStore, AppServer, Conflict, Dataset


class AnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="posture-labeler-tests-")
        cls.root = Path(cls.directory.name)
        cls.video = cls.root / "test.avi"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MJPG writer is unavailable.")
        rows = []
        for frame_id in range(20):
            image = np.full((64, 96, 3), frame_id * 10, np.uint8)
            writer.write(image)
            people = [] if frame_id == 0 else [
                {"person_id": pid, "bbox_xyxy": [10 * pid, 5, 10 * pid + 25, 55],
                 "detection_confidence": 0.9,
                 "posture": {"label": "standing"}, "previous_posture": {"label": "sitting"},
                 "keypoints": [{"index": i, "name": f"point_{i}", "x": i + 10,
                                "y": i + 5, "confidence": 0.85} for i in range(17)]}
                for pid in (1, 2)]
            rows.append({"output_frame": frame_id, "source_frame": frame_id,
                         "source_time_sec": frame_id / 10, "source_time": f"00:00:0{frame_id // 10}.{frame_id % 10}00",
                         "segment": 1 if frame_id < 10 else 2, "people": people})
        writer.release()
        cls.keypoints = cls.root / "test_keypoints.jsonl"
        cls.keypoints.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        cls.dataset = Dataset(cls.video, cls.keypoints)
        cls.original = cls.keypoints.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()
        cls.directory.cleanup()

    def setUp(self):
        self.path = self.root / f"{self.id().split('.')[-1]}.sqlite3"
        self.store = AnnotationStore(self.path, self.dataset)

    def tearDown(self):
        self.store.close()
        self.assertEqual(self.original, self.keypoints.read_bytes())

    def save(self, **overrides):
        return self.store.save({"output_frame": 1, "person_id": 1, "label": "sitting",
                                "occlusion": "severe", "note": "桌面遮挡", "expected_revision": 0,
                                **overrides})

    def test_sampling_is_per_segment_and_skips_empty(self):
        frames = self.dataset.sample(0.5)
        self.assertEqual([r["output_frame"] for r in frames], [1, 5, 10, 15])
        self.assertNotIn("offset", frames[0])

    def test_extra_frames_are_sorted_and_deduplicated(self):
        frames = self.dataset.sample(1, [3, 3, 10, 999])
        self.assertEqual([r["output_frame"] for r in frames], [1, 3, 10])

    def test_invalid_sampling_interval(self):
        for value in (0, -1, float("nan"), float("inf"), 100):
            with self.assertRaises(ValueError): self.dataset.sample(value)

    def test_predictions_are_hidden(self):
        person = self.dataset.frame(1)["people"][0]
        self.assertNotIn("posture", person)
        self.assertNotIn("previous_posture", person)
        self.assertEqual(len(person["keypoints"]), 17)

    def test_image_matches_requested_source_frame(self):
        for frame, expected in ((1, 10), (15, 150), (1, 10)):
            image = cv2.imdecode(np.frombuffer(self.dataset.jpeg(frame), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (64, 96, 3))
            self.assertAlmostEqual(float(image.mean()), expected, delta=3)

    def test_resolve_frame_and_time(self):
        self.assertEqual(self.dataset.resolve("15", "frame"), 15)
        self.assertEqual(self.dataset.resolve("00:01.5", "time"), 15)
        with self.assertRaises(ValueError): self.dataset.resolve("06:00", "time")
        with self.assertRaises(ValueError): self.dataset.resolve("nan", "time")
        with self.assertRaises(ValueError): self.dataset.resolve("300", "frame")

    def test_save_and_reopen_persists(self):
        result = self.save()
        self.assertEqual(result["annotation"]["revision"], 1)
        other = AnnotationStore(self.path, self.dataset)
        try: self.assertEqual(other.all()[0]["label"], "sitting")
        finally: other.close()

    def test_conflict_does_not_overwrite(self):
        self.save()
        with self.assertRaises(Conflict): self.save(label="standing")
        self.assertEqual(self.store.all()[0]["label"], "sitting")

    def test_invalid_labels_and_people_rejected(self):
        for overrides in ({"label": "unknown"}, {"person_id": 99}, {"output_frame": 999},
                          {"occlusion": "invalid"}, {"note": "x" * 1001}):
            with self.assertRaises(ValueError): self.save(**overrides)
        self.assertEqual(self.store.all(), [])

    def test_undo_new_label(self):
        event = self.save()["event_id"]
        result = self.store.undo(event)
        self.assertIsNone(result["annotation"]["label"])
        self.assertEqual(result["annotation"]["revision"], 2)
        self.assertEqual(self.store.export("jsonl"), b"")
        with self.assertRaises(Conflict): self.store.undo(event)

    def test_undo_restores_previous_label_and_note(self):
        self.save()
        event = self.save(label="standing", note="修改", expected_revision=1)["event_id"]
        restored = self.store.undo(event)["annotation"]
        self.assertEqual((restored["label"], restored["note"], restored["occlusion"]), ("sitting", "桌面遮挡", "severe"))

    def test_undo_cannot_overwrite_later_edit(self):
        event = self.save()["event_id"]
        self.save(label="standing", expected_revision=1)
        with self.assertRaises(Conflict): self.store.undo(event)

    def test_clear_retains_revision_and_history(self):
        self.save()
        self.save(label=None, expected_revision=1)
        self.assertEqual(self.store.all()[0]["revision"], 2)
        with self.assertRaises(Conflict): self.save()
        self.save(label="ignore", expected_revision=2)
        self.assertEqual(self.store.all()[0]["label"], "ignore")

    def test_export_includes_geometry_and_only_human_labels(self):
        self.save()
        self.save(person_id=2, label="uncertain")
        rows = [json.loads(r) for r in self.store.export("jsonl").decode().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["training_eligible"])
        self.assertFalse(rows[1]["training_eligible"])
        self.assertEqual(rows[0]["source_frame"], 1)
        self.assertEqual(len(rows[0]["keypoints"]), 17)
        self.assertEqual(rows[0]["bbox_xyxy"], [10, 5, 35, 55])
        self.assertNotIn("posture", rows[0])

    def test_csv_utf8_notes_and_formula_escaping(self):
        self.save(note="=SUM(1,2)")
        raw = self.store.export("csv")
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        row = next(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(row["note"], "'=SUM(1,2)")
        self.assertEqual(len(json.loads(row["keypoints"])), 17)

    def test_pins_persist(self):
        self.store.pin(5)
        self.store.pin(5)
        self.assertEqual(self.store.pins(), [5])

    def test_database_rejects_different_dataset(self):
        identity = self.dataset.identity
        self.dataset.identity = "different"
        try:
            with self.assertRaises(ValueError): AnnotationStore(self.path, self.dataset)
        finally: self.dataset.identity = identity

    def test_http_routes_and_write_protection(self):
        batches = self.root / "http-batches"
        batch = batches / "sample"
        batch.mkdir(parents=True)
        manifest = {"version": 1, "batch": "sample", "dataset_id": self.dataset.identity,
                    "targets": [[4, 2]]}
        (batch / "queue.json").write_text(json.dumps(manifest), encoding="utf-8")
        server = AppServer(("127.0.0.1", 0), self.dataset, self.store, batches)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        def call(path, body=None, headers=None):
            raw = None if body is None else json.dumps(body).encode()
            req = Request(url + path, data=raw, headers=headers or {})
            return urlopen(req, timeout=5)
        try:
            config = json.load(call("/api/config"))
            self.assertEqual(config["hard_queue_api_version"], 1)
            self.assertEqual(json.load(call("/api/hard-queue"))["targets"], [])
            headers = {"Content-Type": "application/json", "X-Annotation-Token": config["csrf_token"]}
            with self.assertRaises(HTTPError) as blocked:
                call("/api/annotation", {}, {"Content-Type": "application/json"})
            self.assertEqual(blocked.exception.code, 403)
            with self.assertRaises(HTTPError) as blocked:
                call("/api/annotation", {}, {**headers, "Origin": "https://evil.invalid"})
            self.assertEqual(blocked.exception.code, 403)
            with self.assertRaises(HTTPError) as blocked:
                call("/api/config", headers={"Host": "evil.invalid"})
            self.assertEqual(blocked.exception.code, 403)
            self.assertIn(b"<!doctype html>", call("/").read())
            self.assertIn(b"X-Annotation-Token", call("/app.js").read())
            self.assertEqual(json.load(call("/api/frame?id=1"))["source_frame"], 1)
            self.assertTrue(call("/api/image?id=1").read().startswith(b"\xff\xd8"))
            result = json.load(call("/api/annotation", {"output_frame": 1, "person_id": 1,
                "label": "standing", "expected_revision": 0}, headers))
            self.assertEqual(result["annotation"]["label"], "standing")
            self.assertEqual(len(json.loads(call("/api/export?format=jsonl").read())["keypoints"]), 17)
            json.load(call("/api/undo", {"event_id": result["event_id"]}, headers))
            self.assertEqual(call("/api/export?format=jsonl").read(), b"")
            json.load(call("/api/jump", {"value": "3", "mode": "frame"}, headers))
            queue = json.load(call("/api/queue?step=1"))
            self.assertIn(3, [f["output_frame"] for f in queue["frames"]])
            before_annotations, before_pins = self.store.all(), self.store.pins()
            with self.assertRaises(HTTPError) as blocked:
                call("/api/hard-queue", {"batch": "sample"}, {"Content-Type": "application/json"})
            self.assertEqual(blocked.exception.code, 403)
            activated = json.load(call("/api/hard-queue", {"batch": "sample"}, headers))
            self.assertEqual(activated["targets"], [[4, 2]])
            queue = json.load(call("/api/queue?step=1"))
            self.assertEqual(queue["hard_queue"]["targets"], [[4, 2]])
            self.assertIn(4, [f["output_frame"] for f in queue["frames"]])
            self.assertEqual(self.store.all(), before_annotations)
            self.assertEqual(self.store.pins(), before_pins)
            # Changes are read from queue.json, never copied into HTML or the database.
            manifest["targets"] = [[6, 1]]
            (batch / "queue.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(json.load(call("/api/hard-queue"))["targets"], [[6, 1]])
            with self.assertRaises(HTTPError) as blocked:
                call("/api/hard-queue", {"batch": "../outside"}, headers)
            self.assertEqual(blocked.exception.code, 400)
            for path in ("/../server.py", "/data/annotations.sqlite3"):
                with self.assertRaises(HTTPError) as blocked: call(path)
                self.assertEqual(blocked.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
