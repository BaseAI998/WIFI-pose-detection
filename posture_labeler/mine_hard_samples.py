"""Mine cached predictions into a human-review queue; NEVER create human labels.

Mining writes only a new batch directory. --install activates its queue.json via
the running labeler's API after a SQLite online backup. It changes only the
active.json selector, never annotations, database pins, or the webpage source.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def api(base, path, body=None, token=None):
    request = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers={} if body is None else {
                          "Content-Type": "application/json", "X-Annotation-Token": token})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def overlap(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-x) * max(0, bottom-y)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1e-8)


def same_location(a, b):
    # Frame-local P IDs are intentionally NOT used for matching.
    center_a, center_b = (a[:2]+a[2:])/2, (b[:2]+b[2:])/2
    size = max(1, min(np.linalg.norm(a[2:]-a[:2]), np.linalg.norm(b[2:]-b[:2])))
    return overlap(a, b) >= .25 or np.linalg.norm(center_a-center_b)/size < .25


def geometry(points):
    k = np.array([[p["x"], p["y"], p["confidence"]]
                  for p in sorted(points, key=lambda p: p["index"])], dtype=float)
    valid = np.isfinite(k).all(1) & (k[:, 2] >= .35)
    shoulders, hips = np.where(valid[5:7])[0]+5, np.where(valid[11:13])[0]+11
    if not len(shoulders) or not len(hips):
        return None
    s, h = k[shoulders, :2].mean(0), k[hips, :2].mean(0)
    scale = np.linalg.norm(h-s)
    if scale < 3:
        return None
    coords = np.zeros((17, 2))
    coords[valid] = np.clip((k[valid, :2] - (s+h)/2)/scale, -5, 5)
    descriptor = np.r_[coords.ravel(), valid.astype(float), k[:, 2]].astype(np.float32)
    angles, extensions = [], []
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if valid[[hip, knee, ankle]].all():
            a, b = k[hip, :2]-k[knee, :2], k[ankle, :2]-k[knee, :2]
            denom = np.linalg.norm(a)*np.linalg.norm(b)
            if denom > 1e-5:
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a,b)/denom, -1, 1)))))
            extensions.append(float(np.linalg.norm(k[ankle,:2]-k[hip,:2])/scale))
    return {"descriptor": descriptor, "legs_visible": int(valid[13:17].sum()),
            "tilt": float(abs(h[0]-s[0])/scale), "angles": angles, "extensions": extensions}


def select(candidates, count):
    """Round-robin reason quotas, pose diversity and temporal/location exclusion."""
    reasons = ("known_failure_neighborhood", "model_disagreement", "geometry_conflict",
               "low_margin_or_mirror", "occluded_standing_candidate",
               "visible_leg_sitting_candidate", "pose_diversity")
    quotas = dict(zip(reasons, (20, 24, 16, 12, 8, 8, 8)))
    chosen, used = [], set()
    descriptors = np.stack([c["descriptor"] for c in candidates])
    scale = descriptors.std(0)
    scale[scale < .2] = 1
    descriptors = descriptors / scale
    novelty = np.full(len(candidates), 2.0)
    reason_counts, segment_counts = Counter(), Counter()
    for _ in range(count * 2):
        progress = False
        for reason in reasons:
            if len(chosen) >= count:
                return chosen
            if reason_counts[reason] >= quotas[reason]:
                continue
            best, best_score = None, -1
            for i, c in enumerate(candidates):
                if i in used or reason not in c["reasons"] or segment_counts[c["segment"]] >= max(38, count//2):
                    continue
                incompatible = any(
                    (abs(c["source_time_sec"]-old["source_time_sec"]) < 3.0
                     and same_location(c["box"], old["box"]))
                    or (0 < abs(c["source_time_sec"]-old["source_time_sec"]) < 1.0)
                    for old in chosen)
                if incompatible:
                    continue
                # Broad distribution is useful; absolute coordinates are ONLY used to
                # suppress redundant sampling, never as classifier training features.
                score = min(float(novelty[i]), 3) + c["priority"]*.15 - .02*segment_counts[c["segment"]]
                if score > best_score:
                    best, best_score = i, score
            if best is not None:
                selected = {**candidates[best], "selection_bucket": reason}
                chosen.append(selected)
                used.add(best)
                reason_counts[reason] += 1
                segment_counts[selected["segment"]] += 1
                novelty = np.minimum(novelty, np.sqrt(np.mean((descriptors-descriptors[best])**2, axis=1)))
                progress = True
        if not progress:
            # Quotas are preferences, not a reason to add adjacent duplicates.
            if quotas["pose_diversity"] < count:
                quotas["pose_diversity"] = count
            else:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=ROOT.parent / "posture_mlp_v2/runs/hard_v2_complete")
    parser.add_argument("--predictions", type=Path, default=ROOT.parent / "posture_mlp_v2/outputs/example/VID_20260921_182630_keypoints.jsonl")
    args = parser.parse_args()
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.path not in ("", "/"):
        raise ValueError("Only a localhost labeler URL is allowed")
    if not 1 <= args.count <= 300:
        raise ValueError("count must be 1..300")
    output = args.batch_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new batch directory")
    workspace = ROOT.parent
    import sys
    sys.path.insert(0, str(workspace))
    from posture_mlp_v2.data import resolve_recorded_path
    run = args.run_dir.resolve()
    cached = args.predictions.resolve()
    config = api(args.base_url, "/api/config")
    if args.install:
        if config.get("hard_queue_api_version") != 1:
            raise ValueError("Restart the updated labeler before installing a batch.")
        if output.parent != Path(config["hard_samples_directory"]).resolve():
            raise ValueError("--install requires --batch-dir inside the server's hard-samples directory.")
    live = api(args.base_url, "/api/queue?step=3")
    audit = json.loads((run / "data_audit.json").read_text(encoding="utf-8"))
    if config["dataset_id"] != audit["dataset_id"]:
        raise ValueError("Running labeler is not the expected dataset")
    meta_path = cached.parent / "VID_20260921_182630_run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source = audit["sources"]["keypoints"]
    if meta["cached_keypoints_sha256"] != source["sha256"]:
        raise ValueError("Cached prediction ancestry mismatch")
    if file_hash(resolve_recorded_path(source["path"])) != source["sha256"]:
        raise ValueError("Original keypoints changed")
    if file_hash(run / "model/weights.npz") != meta["posture_weights_sha256"]:
        raise ValueError("Model version mismatch")
    all_annotations = live["annotations"]
    labeled = {(r["output_frame"], r["person_id"]): r for r in all_annotations if r["label"]}
    old_queue = {(f["output_frame"], p) for f in live["frames"] for p in f["person_ids"]}
    diagnostics = list(read_rows(run / "labeled_predictions.jsonl"))
    failures = {(r["output_frame"], r["person_id"]) for r in diagnostics
                if r["label"] != "ignore" and (
                    (r["label"] in ("sitting", "standing") and r["prediction"]["label"] != r["label"])
                    or (r["label"] == "uncertain" and r["prediction"]["label"] != "unknown"))}
    labeled_geometry, failed_geometry = [], []
    candidates = []
    for frame in read_rows(cached):
        for person in frame["people"]:
            key = (frame["output_frame"], person["person_id"])
            entry = {"source_time_sec": frame["source_time_sec"], "box": np.array(person["bbox_xyxy"])}
            if key in labeled:
                labeled_geometry.append(entry)
                if key in failures:
                    failed_geometry.append(entry)
            if frame["output_frame"] % 15 or key in labeled or person["detection_confidence"] < .55:
                continue
            g = geometry(person["keypoints"])
            if g is None:
                continue
            pred = person["posture"]
            old = person["previous_posture"]
            reasons = ["pose_diversity"]
            if old["label"] != pred["label"]:
                reasons.append("model_disagreement")
            if pred["confidence"] < .98 or pred.get("mirror_probability_delta", 0) > .15:
                reasons.append("low_margin_or_mirror")
            if g["legs_visible"] < 2 and pred["candidate_label"] == "standing":
                reasons.append("occluded_standing_candidate")
            if g["legs_visible"] >= 3 and pred["candidate_label"] == "sitting":
                reasons.append("visible_leg_sitting_candidate")
            if ((g["angles"] and max(g["angles"]) > 155 and pred["candidate_label"] == "sitting")
                    or (g["tilt"] > .5 and pred["candidate_label"] == "standing")):
                reasons.append("geometry_conflict")
            candidates.append({
                **entry, "output_frame": frame["output_frame"], "person_id": person["person_id"],
                "source_frame": frame["source_frame"], "segment": frame["segment"],
                "source_time": frame["source_time"], "reasons": reasons, **g,
                "old_queue_member": key in old_queue, "priority": len(reasons)-1,
                "baseline_split": next((r["split"] for r in diagnostics if r["output_frame"] == frame["output_frame"]), None),
            })
    filtered = []
    for c in candidates:
        # Each selected target is >=0.9s away from an already-labeled spatially
        # matching person. This is annotation diversity, NOT evaluation isolation.
        if any(abs(c["source_time_sec"]-a["source_time_sec"]) < .9
               and same_location(c["box"], a["box"]) for a in labeled_geometry):
            continue
        if any(abs(c["source_time_sec"]-a["source_time_sec"]) < 5
               and same_location(c["box"], a["box"]) for a in failed_geometry):
            c["reasons"].append("known_failure_neighborhood")
            c["priority"] += 2
        filtered.append(c)
    if not filtered:
        raise ValueError("No sufficiently separated new candidates")
    chosen = select(filtered, args.count)
    chosen.sort(key=lambda c: (c["output_frame"], c["person_id"]))
    output.mkdir(parents=True, exist_ok=True)
    public = {"version": 1, "dataset_id": config["dataset_id"], "batch": output.name,
              "purpose": "development_annotation_not_independent_test",
              "targets": [[c["output_frame"], c["person_id"]] for c in chosen]}
    write_json(output / "queue.json", public)
    private = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
               for c in chosen]
    write_json(output / "selection_audit.json", {
        "candidate_count": len(filtered), "requested": args.count, "selected": len(chosen),
        "reason_counts": dict(Counter(c["selection_bucket"] for c in chosen)),
        "segment_counts": dict(Counter(c["segment"] for c in chosen)),
        "live_label_count_before": len(labeled), "source_predictions": str(cached),
        "source_predictions_sha256": file_hash(cached), "dataset_id": config["dataset_id"],
        "new_target_count_vs_original_queue": sum(not c["old_queue_member"] for c in chosen),
        "selection_policy": "No inferred label is written. Automated candidate tags are NOT human truth.",
        "sampling_only_spatial_matching": True,
        "minimum_temporal_distance_from_spatially_matching_labeled_person_seconds": .9,
        "minimum_temporal_distance_between_spatially_matching_selected_targets_seconds": 3,
        "targets": private,
    })
    if args.install:
        database = Path(config["database"]).resolve()
        # Online SQLite backup works safely while the original service remains live.
        source = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        backup_path = output / "annotations_before_queue.sqlite3"
        with sqlite3.connect(backup_path) as backup:
            source.backup(backup)
        source.close()
        activated = api(args.base_url, "/api/hard-queue", {"batch": output.name}, config["csrf_token"])
        if activated["targets"] != public["targets"]:
            raise RuntimeError("Activated target list does not match queue.json")
        installed = sorted({c["output_frame"] for c in chosen})
        after = api(args.base_url, "/api/queue?step=3")
        if after["annotations"] != all_annotations:
            raise RuntimeError("Annotations changed concurrently; queue was added but inspect current labels")
        available = {(f["output_frame"], p) for f in after["frames"] for p in f["person_ids"]}
        if not {tuple(t) for t in public["targets"]}.issubset(available):
            raise RuntimeError("Some targets are missing from the updated queue")
        write_json(output / "installation.json", {
            "status": "installed", "active_batch": output.name, "frames_available": installed,
            "annotation_database_mutated": False, "target_count": len(chosen),
            "human_annotations_unchanged": True, "human_label_count": len(labeled),
            "database_backup": str(backup_path), "all_queue_person_instances": len(available),
            "new_queue_person_instances": len(available-old_queue),
            "new_queue_frames": len({f for f, _ in available}-{f for f, _ in old_queue}),
        })
    print(json.dumps({"selected_targets": len(chosen), "frames": len({c["output_frame"] for c in chosen}),
                      "buckets": dict(Counter(c["selection_bucket"] for c in chosen)),
                      "segments": dict(Counter(c["segment"] for c in chosen)),
                      "batch_dir": str(output), "installed": args.install}, indent=2))


if __name__ == "__main__":
    main()
