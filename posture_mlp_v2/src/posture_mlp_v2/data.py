"""Source validation; active-mined data is DEVELOPMENT data, not a new test set."""
from __future__ import annotations
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import numpy as np
from .features import as_keypoints
from .paths import find_project_root

# Optional override for integration/tests; never infer data paths from site-packages.
PROJECT_ROOT = None

LABELS = ("sitting", "standing")
ALL_LABELS = (*LABELS, "uncertain", "ignore")
# Three disjoint temporal blocks, separated by six-second embargoes.
BLOCKS = ((0., 17.), (23., 37.), (43., 60.05))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row):
    return int(row["output_frame"]), int(row["person_id"])


def resolve_recorded_path(value, base=None, *, project_root=None):
    """Resolve portable inputs and historical paths without rewriting source hashes."""
    value = str(value)
    path = Path(value)
    if base is not None and not path.is_absolute():
        local = Path(base) / path
        if local.is_file():
            return local.resolve()
    try:
        root = Path(project_root or PROJECT_ROOT).resolve() if (project_root or PROJECT_ROOT) else find_project_root(base)
    except ValueError:
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(value) from None
    normalized = value.replace("\\", "/").casefold()
    manifest_path = root / "runs" / "hard_v2_complete" / "provenance" / "input_example_legacy" / "relocations.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for original, replacement in sorted(manifest.get("relocations", {}).items(),
                                            key=lambda item: len(item[0]), reverse=True):
            original = original.replace("\\", "/").casefold()
            # Bare filenames must not capture moved outputs with the same name.
            if normalized == original or ("/" in original and normalized.endswith("/" + original)):
                target = (root / replacement).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Historical relocation must stay inside the project")
                if target.is_file():
                    return target
    # Known historical aliases prefer frozen project snapshots over live external files.
    if path.is_file():
        return path.resolve()
    # Preserve historical records byte-for-byte after the paired I/O migration.
    for old_prefix, new_prefix in (
        ("input", "inputs"),
        ("posture_mlp_v2/outputs/hard_v2", "outputs/example"),
        ("posture_mlp_v2/archive/input_example_legacy", "runs/hard_v2_complete/provenance/input_example_legacy"),
    ):
        marker = "/" + old_prefix + "/"
        index = normalized.rfind(marker)
        if normalized.startswith(old_prefix + "/"):
            suffix = value.replace("\\", "/")[len(old_prefix) + 1:]
        elif index >= 0:
            suffix = value.replace("\\", "/")[index + len(marker):]
        else:
            continue
        target_root = (root / new_prefix).resolve()
        candidate = (target_root / suffix).resolve()
        if not candidate.is_relative_to(target_root):
            raise ValueError("Historical I/O relocation must stay inside its target directory")
        if candidate.is_file():
            return candidate
    # Saved run artifacts may contain absolute paths from another machine.
    marker = "/posture_mlp_v2/"
    index = normalized.rfind(marker)
    relative = value.replace("\\", "/")
    if index >= 0:
        relative = relative[index + len(marker):]
    elif normalized.startswith("posture_mlp_v2/"):
        relative = relative[len("posture_mlp_v2/"):]
    candidate = (root / relative).resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return candidate
    raise FileNotFoundError(value)


def resolve_sources(metadata):
    metadata = resolve_recorded_path(metadata)
    meta = json.loads(metadata.read_text(encoding="utf-8-sig"))
    kp = resolve_recorded_path(meta["outputs"]["keypoints_jsonl"], metadata.parent)
    video = resolve_recorded_path(meta["input"], metadata.parent)
    ancestor = resolve_recorded_path(meta["cached_keypoints"], metadata.parent) if meta.get("reuses_keypoints") else kp
    return meta, kp, video, ancestor


def resolve_input_directory(directory):
    """Normal input: one video + one frame-per-line JSONL, with no manifest.

    Legacy predictions, if present in the JSONL, are ignored. This function
    derives bookkeeping from actual files; it does not read archived metadata.
    """
    import cv2
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    videos = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    caches = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"]
    if len(videos) != 1 or len(caches) != 1:
        raise ValueError("Input directory requires exactly one video and one keypoint .jsonl file")
    video, kp = videos[0], caches[0]
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open input video")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, count = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if min(width, height, count) <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video dimensions, frame count or FPS")
    frames, people, last_source, last_segment = 0, 0, -1, 0
    segments = {}
    for row in read_jsonl(kp):
        source, segment = row["source_frame"], row["segment"]
        if not isinstance(source, int) or not isinstance(segment, int):
            raise ValueError("source_frame and segment must be integers")
        if row["output_frame"] != frames or not last_source < source < count:
            raise ValueError("Keypoint frames must be ordered and reference valid source frames")
        if segment != last_segment and segment != last_segment + 1:
            raise ValueError("Segments must start at 1 and appear in ascending consecutive order")
        timestamp = row["source_time_sec"]
        if not np.isfinite(timestamp) or abs(timestamp - source / fps) > max(.002, .5 / fps):
            raise ValueError("Keypoint timestamps disagree with the input video's FPS")
        ids = [p["person_id"] for p in row["people"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate person ID inside one frame")
        for person in row["people"]:
            points = as_keypoints(person["keypoints"])
            if not np.isfinite(points).all():
                raise ValueError("Nonfinite keypoints")
            box = np.asarray(person["bbox_xyxy"])
            if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid bounding box")
        extent = segments.setdefault(segment, [source, source])
        extent[1] = source
        last_source, last_segment = source, segment
        frames += 1
        people += len(ids)
    if not frames:
        raise ValueError("Empty keypoint file")

    def boundary(frame):
        seconds = frame / fps
        # Frame-quantized integer-second cuts (e.g. 01:00) retain their intent.
        nearest = round(seconds)
        return float(nearest) if abs(seconds-nearest) <= .5/fps + 1e-6 else seconds

    intervals = [[boundary(first), boundary(last+1)] for first, last in segments.values()]
    meta = {
        "input": str(video), "model": "unspecified_cached_keypoints",
        "video": {"width": width, "height": height, "fps": fps,
                  "source_frame_count": count, "source_duration_sec": count/fps},
        "intervals_sec": intervals, "processed_frames": frames,
        "detected_people_instances": people, "outputs": {"keypoints_jsonl": str(kp)},
        "reuses_keypoints": False, "input_layout": "video_and_keypoints",
        "intervals_inferred": True,
        "interval_inference": "Each segment spans its first source frame through last+1; integer-second boundaries snap within half a frame.",
    }
    return meta, kp, video, kp


def block_id(row, intervals):
    segment = int(row["segment"])
    if not 1 <= segment <= len(intervals):
        raise ValueError("Invalid segment")
    start, end = intervals[segment-1]
    if abs(end-start-60) > .05:
        raise ValueError("This frozen protocol requires 60-second source intervals")
    offset = float(row["source_time_sec"]) - start
    if not -.05 <= offset <= 60.05:
        raise ValueError("Frame lies outside its declared interval")
    offset = max(0., offset)
    for index, (lo, hi) in enumerate(BLOCKS):
        if lo <= offset < hi:
            return index
    return -1


def unique_rows(path):
    rows = list(read_jsonl(path))
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError(f"Duplicate frame/person keys in {path}")
    return rows


def load_data(metadata, old_labels, hard_labels, *, input_dir=None):
    if (metadata is None) == (input_dir is None):
        raise ValueError("Choose input_dir OR legacy metadata, not both")
    meta, kp, video, ancestor = resolve_input_directory(input_dir) if input_dir is not None else resolve_sources(metadata)
    old, hard = unique_rows(old_labels), unique_rows(hard_labels)
    old_map, hard_map = {key(r): r for r in old}, {key(r): r for r in hard}
    if not old_map.keys() <= hard_map.keys():
        raise ValueError("Expected the hard export to contain the previous annotations")
    changes = []
    for k, before in old_map.items():
        after = hard_map[k]
        for field in ("label", "occlusion", "note"):
            if before[field] != after[field]:
                changes.append({"output_frame": k[0], "person_id": k[1], "field": field,
                                "old": before[field], "new": after[field]})
                if after["updated_at"] < before["updated_at"]:
                    raise ValueError("Older hard export cannot override newer labels")
    sources = {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
               for name, path in (("metadata", metadata), ("keypoints", kp), ("video", video),
                                  ("ancestor_keypoints", ancestor), ("old_labels", old_labels),
                                  ("hard_labels", hard_labels)) if path is not None}
    identity = hashlib.sha256((sources["ancestor_keypoints"]["sha256"] + sources["video"]["sha256"]).encode()).hexdigest()
    if meta.get("reuses_keypoints") and sources["ancestor_keypoints"]["sha256"] != meta["cached_keypoints_sha256"]:
        raise ValueError("Ancestor cache hash mismatch")
    for row in hard:
        if row["dataset_id"] != identity:
            raise ValueError("Annotation dataset identity mismatch")
        if row["label"] not in ALL_LABELS or row["occlusion"] not in ("unmarked", "clear", "partial", "severe"):
            raise ValueError("Invalid annotation")
        if bool(row["training_eligible"]) != (row["label"] in LABELS):
            raise ValueError("training_eligible/label mismatch")
        row["origin"] = "old" if key(row) in old_map else "hard_new"
        row["block"] = block_id(row, meta["intervals_sec"])
        if not np.isfinite(as_keypoints(row["keypoints"])).all():
            raise ValueError("Nonfinite input")
    matched, frames, people = set(), 0, 0
    for frame, initial in zip_longest(read_jsonl(kp), read_jsonl(ancestor)):
        if frame is None or initial is None:
            raise ValueError("Cache lineage lengths differ")
        for field in ("output_frame", "source_frame", "source_time_sec", "segment"):
            if frame[field] != initial[field]:
                raise ValueError("Frame lineage mismatch")
        if frame["output_frame"] != frames or len(frame["people"]) != len(initial["people"]):
            raise ValueError("Cache is not complete/sequential")
        ids = [p["person_id"] for p in frame["people"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate person ID within frame")
        for person, before in zip(frame["people"], initial["people"]):
            for field in ("person_id", "bbox_xyxy", "detection_confidence", "keypoints"):
                if person[field] != before[field]:
                    raise ValueError("Detection lineage changed")
            k = (frame["output_frame"], person["person_id"])
            if k not in hard_map:
                continue
            row = hard_map[k]
            for field in ("source_frame", "source_time_sec", "segment"):
                if row[field] != frame[field]:
                    raise ValueError(f"Annotation frame mismatch {k}")
            for field in ("bbox_xyxy", "detection_confidence", "keypoints"):
                if row[field] != person[field]:
                    raise ValueError(f"Annotation keypoints mismatch {k}")
            if (row["image_width"], row["image_height"]) != (meta["video"]["width"], meta["video"]["height"]):
                raise ValueError("Annotation image geometry mismatch")
            matched.add(k)
        frames += 1
        people += len(frame["people"])
    if matched != hard_map.keys():
        raise ValueError("Some labels do not exist in the cache")
    if (frames, people) != (meta["processed_frames"], meta["detected_people_instances"]):
        raise ValueError("Metadata counts mismatch")
    counts = lambda rows: dict(Counter(r["label"] for r in rows))
    audit = {
        "sources": sources, "dataset_id": identity, "old_n": len(old), "hard_export_n": len(hard),
        "input_layout": meta.get("input_layout", "legacy_metadata"), "input_summary": meta,
        "new_n": sum(r["origin"] == "hard_new" for r in hard), "label_counts": counts(hard),
        "new_label_counts": counts([r for r in hard if r["origin"] == "hard_new"]),
        "annotation_changes": changes,
        "manual_occlusion_label_counts": dict(Counter(f"{r['occlusion']}__{r['label']}" for r in hard)),
        "block_label_counts": {str(b): counts([r for r in hard if r["block"] == b]) for b in (-1, 0, 1, 2)},
        "verified_all_cache_detections": people, "verified_frames": frames,
        "evaluation_status": "active_mined_development_only_no_independent_test",
        "identity_split": "same camera and recurring people; P IDs are frame-local, not track identities",
    }
    # Validate disjoint temporal blocks across ALL persons, not merely individual IDs.
    gaps = {}
    for a, b in ((0,1), (0,2), (1,2)):
        ta = {r["source_time_sec"] for r in hard if r["block"] == a}
        tb = {r["source_time_sec"] for r in hard if r["block"] == b}
        if not ta or not tb:
            raise ValueError("Empty development block")
        gap = min(abs(x-y) for x in ta for y in tb)
        if gap < 5.9:
            raise ValueError("Temporal embargo violated")
        gaps[f"{a}__{b}"] = gap
    audit["actual_minimum_gaps_seconds"] = gaps
    return hard, meta, audit
