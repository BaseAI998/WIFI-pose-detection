"""Local, CPU-only annotation UI for existing YOLO pose JSONL results.

No inference, training, or writes to source files. Only OpenCV is external.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2

ROOT = Path(__file__).resolve().parent
LABELS = {"sitting", "standing", "uncertain", "ignore"}
OCCLUSIONS = {"unmarked", "clear", "partial", "severe"}
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


class Conflict(Exception):
    """An annotation has been changed since the client loaded it."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Dataset:
    def __init__(self, video, keypoints):
        self.video = Path(video).resolve()
        self.keypoints = Path(keypoints).resolve()
        if not self.video.is_file() or not self.keypoints.is_file():
            raise ValueError("Video or keypoints JSONL not found.")
        self.frames = {}
        digest = hashlib.sha256()
        with self.keypoints.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                frame_id = int(row["output_frame"])
                if frame_id in self.frames:
                    raise ValueError(f"Duplicate output_frame: {frame_id}")
                person_ids = [int(p["person_id"]) for p in row["people"]]
                if len(set(person_ids)) != len(person_ids):
                    raise ValueError(f"Duplicate person_id in frame {frame_id}")
                self.frames[frame_id] = {
                    "output_frame": frame_id, "source_frame": int(row["source_frame"]),
                    "source_time_sec": float(row["source_time_sec"]),
                    "source_time": row.get("source_time", ""),
                    "segment": int(row.get("segment", 1)), "person_ids": person_ids,
                    "offset": offset,
                }
        if not self.frames:
            raise ValueError("The keypoints file contains no frames.")
        self.ordered = sorted(self.frames.values(), key=lambda r: r["output_frame"])
        video_hash = file_hash(self.video)
        self.identity = hashlib.sha256((digest.hexdigest() + video_hash).encode()).hexdigest()
        self.capture = cv2.VideoCapture(str(self.video))
        if not self.capture.isOpened():
            raise ValueError("Cannot decode source video with OpenCV.")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if any(r["source_frame"] < 0 or r["source_frame"] >= count for r in self.ordered):
            self.capture.release()
            raise ValueError("Source frame numbers do not match this video.")
        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def frame(self, frame_id):
        meta = self.frames.get(frame_id)
        if meta is None:
            raise ValueError("Output frame is not present in the keypoints file.")
        with self.keypoints.open("rb") as stream:
            stream.seek(meta["offset"])
            raw = json.loads(stream.readline())
        # Never expose automatic posture predictions: avoid anchoring human labels.
        return {**{k: v for k, v in meta.items() if k not in ("offset", "person_ids")},
                "people": [{"person_id": p["person_id"], "bbox_xyxy": p["bbox_xyxy"],
                            "detection_confidence": p.get("detection_confidence"),
                            "keypoints": p["keypoints"]} for p in raw["people"]]}

    def person(self, frame_id, person_id):
        frame = self.frame(frame_id)
        for person in frame["people"]:
            if person["person_id"] == person_id:
                return frame, person
        raise ValueError("Person does not exist in this frame.")

    def sample(self, seconds, extras=()):
        if not math.isfinite(seconds) or not 0.5 <= seconds <= 60:
            raise ValueError("Sampling interval must be between 0.5 and 60 seconds.")
        selected = set(extras)
        anchors = {}
        for meta in self.ordered:
            segment = meta["segment"]
            anchor = anchors.setdefault(segment, meta["source_time_sec"])
            if meta["person_ids"] and meta["source_time_sec"] >= anchor - 1e-5:
                selected.add(meta["output_frame"])
                jumps = max(1, math.floor((meta["source_time_sec"] - anchor) / seconds) + 1)
                anchors[segment] = anchor + jumps * seconds
        return [{k: v for k, v in self.frames[f].items() if k != "offset"}
                for f in sorted(selected) if f in self.frames and self.frames[f]["person_ids"]]

    def resolve(self, value, mode):
        if mode == "frame":
            frame_id = int(value)
            if frame_id not in self.frames:
                raise ValueError("该输出帧不在关键点文件中。帧号从 0 开始。")
            return frame_id
        if mode != "time":
            raise ValueError("Unknown jump mode.")
        parts = str(value).strip().split(":")
        if not 1 <= len(parts) <= 3:
            raise ValueError("请填写原视频时间，如 06:20。")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid timestamp.")
        closest = min(self.ordered, key=lambda r: abs(r["source_time_sec"] - seconds))
        if abs(closest["source_time_sec"] - seconds) > 0.1:
            raise ValueError("该时间不在已处理的视频区间中。")
        return closest["output_frame"]

    def jpeg(self, frame_id):
        if frame_id not in self.frames:
            raise ValueError("Unknown frame.")
        with self.lock:
            if frame_id in self.cache:
                self.cache.move_to_end(frame_id)
                return self.cache[frame_id]
            source = self.frames[frame_id]["source_frame"]
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, source)
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {source}.")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if not ok:
                raise ValueError("Cannot encode frame.")
            result = encoded.tobytes()
            self.cache[frame_id] = result
            while len(self.cache) > 12:
                self.cache.popitem(last=False)
            return result

    def close(self):
        self.capture.release()


class AnnotationStore:
    def __init__(self, path, dataset):
        self.path = Path(path).resolve()
        if self.path in (dataset.video, dataset.keypoints):
            raise ValueError("Database must not overwrite source data.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS annotations (
                output_frame INTEGER NOT NULL, person_id INTEGER NOT NULL,
                label TEXT, occlusion TEXT NOT NULL, note TEXT NOT NULL,
                revision INTEGER NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (output_frame, person_id));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_frame INTEGER NOT NULL, person_id INTEGER NOT NULL,
                before_json TEXT NOT NULL, after_revision INTEGER NOT NULL,
                created_at TEXT NOT NULL, undone INTEGER NOT NULL DEFAULT 0);
        """)
        identity = self.db.execute("SELECT value FROM meta WHERE key='dataset_id'").fetchone()
        if identity and identity[0] != dataset.identity:
            self.db.close()
            raise ValueError("Database belongs to another dataset. Use a different --database path.")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('dataset_id', ?)", (dataset.identity,))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', '1')")

    def _get(self, frame, person):
        row = self.db.execute("SELECT * FROM annotations WHERE output_frame=? AND person_id=?",
                              (frame, person)).fetchone()
        return dict(row) if row else {"output_frame": frame, "person_id": person,
                                     "label": None, "occlusion": "unmarked", "note": "",
                                     "revision": 0, "updated_at": None}

    def all(self, include_cleared=True):
        with self.lock:
            sql = "SELECT * FROM annotations"
            if not include_cleared:
                sql += " WHERE label IS NOT NULL"
            return [dict(r) for r in self.db.execute(sql + " ORDER BY output_frame, person_id")]

    def save(self, data):
        frame, person = int(data["output_frame"]), int(data["person_id"])
        self.dataset.person(frame, person)
        label = data.get("label")
        occlusion = data.get("occlusion", "unmarked")
        note = data.get("note", "")
        if label is not None and label not in LABELS:
            raise ValueError("Invalid label.")
        if occlusion not in OCCLUSIONS or not isinstance(note, str) or len(note) > 1000:
            raise ValueError("Invalid occlusion or note (maximum 1000 characters).")
        expected = int(data["expected_revision"])
        with self.lock, self.db:
            before = self._get(frame, person)
            if before["revision"] != expected:
                raise Conflict("此标注已被其他页面修改，已刷新，请核对后重试。")
            revision, stamp = expected + 1, utc_now()
            self.db.execute("""INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(output_frame, person_id) DO UPDATE SET
                label=excluded.label, occlusion=excluded.occlusion, note=excluded.note,
                revision=excluded.revision, updated_at=excluded.updated_at""",
                (frame, person, label, occlusion if label else "unmarked", note if label else "",
                 revision, stamp))
            event = self.db.execute("""INSERT INTO events
                (output_frame, person_id, before_json, after_revision, created_at)
                VALUES (?, ?, ?, ?, ?)""", (frame, person, json.dumps(before), revision, stamp)).lastrowid
            return {"annotation": self._get(frame, person), "event_id": event}

    def undo(self, event_id):
        with self.lock, self.db:
            event = self.db.execute("SELECT * FROM events WHERE id=?", (int(event_id),)).fetchone()
            if not event or event["undone"]:
                raise Conflict("这次操作已撤销或不存在。")
            current = self._get(event["output_frame"], event["person_id"])
            if current["revision"] != event["after_revision"]:
                raise Conflict("该样本之后又被修改，不能直接撤销旧操作。请回到样本手动修改。")
            before = json.loads(event["before_json"])
            result = self.save({**before, "expected_revision": current["revision"]})
            self.db.execute("UPDATE events SET undone=1 WHERE id=?", (int(event_id),))
            return result

    def pin(self, frame_id):
        self.dataset.frame(frame_id)
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES (?, '1')", (f"pin:{frame_id}",))

    def pins(self):
        with self.lock:
            return [int(r[0][4:]) for r in self.db.execute("SELECT key FROM meta WHERE key LIKE 'pin:%'")]

    def export_rows(self):
        for annotation in self.all(include_cleared=False):
            frame, person = self.dataset.person(annotation["output_frame"], annotation["person_id"])
            yield {"schema_version": 1, "dataset_id": self.dataset.identity,
                   "video": self.dataset.video.name, "keypoints_file": self.dataset.keypoints.name,
                   "source_frame": frame["source_frame"], "source_time_sec": frame["source_time_sec"],
                   "source_time": frame["source_time"], "segment": frame["segment"],
                   "image_width": self.dataset.width, "image_height": self.dataset.height,
                   **annotation, "training_eligible": annotation["label"] in ("sitting", "standing"),
                   "bbox_xyxy": person["bbox_xyxy"],
                   "detection_confidence": person["detection_confidence"],
                   "keypoints": person["keypoints"]}

    def export(self, kind):
        if kind == "jsonl":
            return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.export_rows()).encode("utf-8")
        if kind != "csv":
            raise ValueError("Unknown export format.")
        fields = ["dataset_id", "video", "keypoints_file", "output_frame", "source_frame",
                  "source_time_sec", "source_time", "segment", "person_id", "label", "occlusion",
                  "note", "training_eligible", "image_width", "image_height", "bbox_xyxy",
                  "detection_confidence", "keypoints", "revision", "updated_at"]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in self.export_rows():
            for key in ("bbox_xyxy", "keypoints"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            for key in ("note", "video", "keypoints_file"):
                if str(row[key]).lstrip().startswith(("=", "+", "-", "@")) or str(row[key]).startswith(("\t", "\r", "\n")):
                    row[key] = "'" + row[key]  # Prevent formula injection in spreadsheet viewers.
            writer.writerow(row)
        return stream.getvalue().encode("utf-8-sig")

    def close(self):
        self.db.close()


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, dataset, store, hard_samples=None):
        super().__init__(address, Handler)
        self.dataset, self.store = dataset, store
        self.token = secrets.token_urlsafe(32)
        self.hard_queue = HardSampleQueue(dataset, hard_samples or ROOT / "hard_samples")


class HardSampleQueue:
    """Read targets only from queue.json; active.json stores just the batch name."""
    def __init__(self, dataset, root):
        self.dataset, self.root = dataset, Path(root).resolve()
        self.lock = threading.RLock()

    def read_batch(self, name, require_match=False):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
            raise ValueError("Invalid hard-sample batch name.")
        path = (self.root / name / "queue.json").resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Batch queue.json is missing or outside the hard-samples directory.")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("version") != 1 or value.get("batch") != name:
            raise ValueError("Invalid queue manifest version or batch name.")
        if value.get("dataset_id") != self.dataset.identity:
            if require_match:
                raise ValueError("Hard-sample batch belongs to another dataset.")
            return {"status": "dataset_mismatch", "batch": name, "targets": [],
                    "dataset_id": self.dataset.identity}
        targets, seen = value.get("targets"), set()
        if not isinstance(targets, list):
            raise ValueError("Hard-sample targets must be a list.")
        for target in targets:
            if (not isinstance(target, list) or len(target) != 2
                    or any(type(v) is not int for v in target)):
                raise ValueError("Every target must be [integer frame, integer person].")
            frame, person = target
            if tuple(target) in seen:
                raise ValueError("Duplicate hard-sample target.")
            if frame not in self.dataset.frames or person not in self.dataset.frames[frame]["person_ids"]:
                raise ValueError("Hard-sample target does not exist in this dataset.")
            seen.add(tuple(target))
        # Whitelist public fields: no predictions/private audit hints for annotators.
        return {"version": 1, "status": "active", "batch": name,
                "dataset_id": self.dataset.identity, "targets": targets}

    def current(self):
        with self.lock:
            pointer = self.root / "active.json"
            if not pointer.is_file():
                return {"status": "none", "batch": None, "targets": [],
                        "dataset_id": self.dataset.identity}
            value = json.loads(pointer.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("version") != 1:
                raise ValueError("Invalid active hard-sample batch pointer.")
            return self.read_batch(value.get("batch"))

    def activate(self, name):
        with self.lock:
            value = self.read_batch(name, require_match=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                                 prefix=".active-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump({"version": 1, "batch": name}, stream)
                    stream.write("\n")
                os.replace(temporary, self.root / "active.json")
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            return value


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Keep the terminal useful; failures remain visible.
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def send_bytes(self, body, mime, status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; "
                         "style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def check_host(self):
        port = self.server.server_port
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self.send_json({"error": "Only localhost access is allowed."}, 403)
            return False
        return True

    def do_GET(self):
        if not self.check_host():
            return
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        value = lambda key, default="": query.get(key, [default])[0]
        dataset, store = self.server.dataset, self.server.store
        try:
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if parsed.path in static:
                name, mime = static[parsed.path]
                self.send_bytes((ROOT / "static" / name).read_bytes(), mime)
            elif parsed.path == "/api/config":
                self.send_json({"dataset_id": dataset.identity, "video": dataset.video.name,
                                "keypoints_file": dataset.keypoints.name, "database": str(store.path),
                                "width": dataset.width, "height": dataset.height, "fps": dataset.fps,
                                "total_frames": len(dataset.frames), "csrf_token": self.server.token,
                                "skeleton": SKELETON, "default_step": 3,
                                "hard_queue_api_version": 1,
                                "hard_samples_directory": str(self.server.hard_queue.root)})
            elif parsed.path == "/api/hard-queue":
                self.send_json(self.server.hard_queue.current())
            elif parsed.path == "/api/queue":
                # Include previously labeled and manually pinned frames at every sampling setting.
                annotations = store.all()
                extras = set(store.pins()) | {a["output_frame"] for a in annotations if a["label"]}
                hard_queue = self.server.hard_queue.current()
                extras.update(f for f, _ in hard_queue["targets"])
                if 2400 in dataset.frames:
                    extras.add(2400)
                self.send_json({"frames": dataset.sample(float(value("step", "3")), extras),
                                "annotations": annotations, "pins": sorted(extras),
                                "hard_queue": hard_queue})
            elif parsed.path == "/api/frame":
                frame_id = int(value("id"))
                self.send_json({**dataset.frame(frame_id),
                                "annotations": [a for a in store.all() if a["output_frame"] == frame_id]})
            elif parsed.path == "/api/image":
                self.send_bytes(dataset.jpeg(int(value("id"))), "image/jpeg")
            elif parsed.path == "/api/export":
                kind = value("format", "jsonl")
                mime = "application/x-ndjson; charset=utf-8" if kind == "jsonl" else "text/csv; charset=utf-8"
                self.send_bytes(store.export(kind), mime, filename=f"posture_labels_{dataset.identity[:8]}.{kind}")
            elif parsed.path == "/api/health":
                self.send_json({"ok": True, "dataset_id": dataset.identity})
            else:
                self.send_json({"error": "Not found"}, 404)
        except Conflict as exc:
            self.send_json({"error": str(exc)}, 409)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.log_error("%s", exc)
            self.send_json({"error": "服务端处理失败，请查看启动终端。原文件未修改。"}, 500)

    def do_POST(self):
        if not self.check_host():
            return
        origin = self.headers.get("Origin")
        port = self.server.server_port
        if (origin and origin not in (f"http://localhost:{port}", f"http://127.0.0.1:{port}")) or not secrets.compare_digest(
                self.headers.get("X-Annotation-Token", ""), self.server.token):
            self.send_json({"error": "请从本机标注页面操作；服务重启后请刷新页面。"}, 403)
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.send_json({"error": "Expected application/json"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/annotation":
                self.send_json(self.server.store.save(data))
            elif self.path == "/api/undo":
                self.send_json(self.server.store.undo(data["event_id"]))
            elif self.path == "/api/hard-queue":
                self.send_json(self.server.hard_queue.activate(data["batch"]))
            elif self.path == "/api/jump":
                frame_id = self.server.dataset.resolve(data["value"], data.get("mode", "frame"))
                if not self.server.dataset.frames[frame_id]["person_ids"]:
                    raise ValueError("这一帧没有检测到人物，请换一个帧号。")
                self.server.store.pin(frame_id)
                self.send_json({"output_frame": frame_id})
            else:
                self.send_json({"error": "Not found"}, 404)
        except Conflict as exc:
            self.send_json({"error": str(exc)}, 409)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.log_error("%s", exc)
            self.send_json({"error": "保存失败，请勿关闭页面。检查终端或磁盘空间后重试。"}, 500)


def main():
    workspace = ROOT.parent
    stem = "VID_20260921_182630"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=workspace / "posture_mlp_v2" / "inputs" / "example" / f"{stem}.mp4")
    parser.add_argument("--keypoints", type=Path, default=workspace / "posture_mlp_v2" / "inputs" / "example" / f"{stem}_keypoints.jsonl")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "annotations.sqlite3")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hard-samples-dir", type=Path, default=ROOT / "hard_samples",
                        help="Batch directories and the active.json selector")
    args = parser.parse_args()
    print("Indexing existing keypoints and checking source video (no YOLO inference)...", flush=True)
    dataset = Dataset(args.video, args.keypoints)
    store = AnnotationStore(args.database, dataset)
    try:
        server = AppServer(("127.0.0.1", args.port), dataset, store, args.hard_samples_dir)
    except OSError as exc:
        store.close()
        dataset.close()
        raise SystemExit(f"Cannot start on port {args.port}: {exc}. Try --port {args.port + 1}.") from exc
    print(f"Ready: http://127.0.0.1:{server.server_port}\nDatabase: {store.path}\nPress Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
        dataset.close()


if __name__ == "__main__":
    main()
