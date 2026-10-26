"""Apply a frozen MLP to cached poses and redraw source frames. NO YOLO calls."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import time
import cv2
import numpy as np

from ..data import read_jsonl, resolve_sources, resolve_input_directory, sha256, write_json
from ..features import as_keypoints
from ..model import PostureClassifier
from ..paths import find_project_root
from ._config import parse_config

LINKS = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16))
COLORS = {"standing": (70, 210, 70), "sitting": (0, 165, 255), "unknown": (170, 170, 170)}
QA_FRAMES = (0, 900, 1619, 1709, 2400, 3327, 4000, 5125, 5305, 5393)


def resolve_output_directory(input_dir, output_dir=None, *, project_root=None):
    """Pair inputs/<dataset> with outputs/<dataset>, independent of working dir."""
    source = Path(input_dir).resolve() if input_dir is not None else None
    if output_dir is None:
        if source is None:
            raise ValueError("Legacy --metadata mode requires an explicit --output-dir")
        if not source.name:
            raise ValueError("Input must be a named dataset directory")
        root = Path(project_root).resolve() if project_root is not None else (
            source.parent.parent if source.parent.name == "inputs" else find_project_root(source))
        output = (root / "outputs" / source.name).resolve()
    else:
        output = Path(output_dir).resolve()
    if source is not None and (output == source or output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError("Input and output directories must be separate, not nested")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output already contains results; preserve or move it before re-running: {output}")
    return output


def label_rectangle(x1, y1, tw, th, baseline, image_size, occupied):
    """Greedy label lanes; never change any detector box to accommodate text."""
    width, height = image_size
    tx = min(max(0, x1), max(0, width - tw - 10))
    label_height = th + baseline + 7
    preferred = max(34 + label_height, min(height - 3, y1))
    offsets = [0] + [v for i in range(1, 12) for v in (-i * label_height, i * label_height)]
    for offset in offsets:
        ty = preferred + offset
        box = (tx, ty - label_height, tx + tw + 8, ty + 2)
        if box[1] < 34 or box[3] >= height:
            continue
        if not any(box[0] < b[2] + 2 and box[2] + 2 > b[0]
                   and box[1] < b[3] + 2 and box[3] + 2 > b[1] for b in occupied):
            occupied.append(box)
            return tx, ty
    occupied.append((tx, preferred - label_height, tx + tw + 8, preferred + 2))
    return tx, preferred


def draw_person(frame, person, result, keypoint_threshold, occupied=None, draw_label=True):
    k = as_keypoints(person["keypoints"])
    visible = np.isfinite(k).all(1) & (k[:, 2] >= keypoint_threshold)
    color = COLORS[result["label"]]
    x1, y1, x2, y2 = map(lambda v: int(round(v)), person["bbox_xyxy"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    for a, b in LINKS:
        if visible[a] and visible[b]:
            cv2.line(frame, tuple(np.rint(k[a, :2]).astype(int)),
                     tuple(np.rint(k[b, :2]).astype(int)), color, 2, cv2.LINE_AA)
    for x, y, _ in k[visible]:
        cv2.circle(frame, (round(x), round(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if not draw_label:
        return
    text = f"P{person['person_id']} {result['label'].upper()}"
    if result["label"] != "unknown":
        text += f" p={result['confidence']:.2f}"
    font, size = cv2.FONT_HERSHEY_SIMPLEX, .5
    (tw, th), baseline = cv2.getTextSize(text, font, size, 1)
    tx, ty = label_rectangle(x1, y1, tw, th, baseline,
                            (frame.shape[1], frame.shape[0]), [] if occupied is None else occupied)
    if abs(ty - y1) > th + baseline + 7:
        cv2.line(frame, (tx, ty), (x1, y1), color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (tx, ty - th - baseline - 5), (tx + tw + 8, ty + 2), color, -1)
    cv2.putText(frame, text, (tx + 4, ty - baseline), font, size, (15, 15, 15), 1, cv2.LINE_AA)


def make_review(source_video, predictions_file, cache_path, output):
    """Only diagnostic images; labels are displayed, never used to change predictions."""
    rows = list(read_jsonl(predictions_file))
    selected = [r for r in rows if r["output_frame"] in (1619, 1709, 5125, 5215, 5305) and (
        (r["label"] in ("sitting", "standing") and r["label"] != r["prediction"]["label"])
        or (r["label"] == "uncertain" and r["prediction"]["label"] != "unknown"))]
    selected += [r for r in rows if r["output_frame"] == 2400 and r["person_id"] in (1, 2, 3, 4)]
    requested = {(r["output_frame"], r["person_id"]): r for r in selected}
    detections = {}
    for frame in read_jsonl(cache_path):
        for person in frame["people"]:
            key = (frame["output_frame"], person["person_id"])
            if key in requested:
                detections[key] = person
    cap = cv2.VideoCapture(str(source_video))
    tiles = []
    try:
        for r in selected:
            key = (r["output_frame"], r["person_id"])
            person = detections[key]
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["source_frame"])
            ok, original = cap.read()
            if not ok:
                raise RuntimeError(f"Review decode failed: {key}")
            drawn = original.copy()
            draw_person(drawn, person, r["prediction"], .35, draw_label=False)
            box = person["bbox_xyxy"]
            w, h = box[2] - box[0], box[3] - box[1]
            x1, x2 = max(0, int(box[0] - .25*w)), min(original.shape[1], int(box[2] + .25*w))
            y1, y2 = max(0, int(box[1] - .15*h)), min(original.shape[0], int(box[3] + .15*h))
            # Side by side clean crop and overlay, with aspect ratio preserved.
            pair = []
            for image in (original, drawn):
                crop = image[y1:y2, x1:x2]
                scale = min(310 / crop.shape[1], 300 / crop.shape[0])
                crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
                canvas = np.full((300, 310, 3), 30, np.uint8)
                ox, oy = (310-crop.shape[1])//2, (300-crop.shape[0])//2
                canvas[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = crop
                pair.append(canvas)
            tile = np.full((365, 620, 3), 30, np.uint8)
            tile[65:] = np.hstack(pair)
            title = f"Frame {key[0]} P{key[1]} | {r['split'].upper()} | GT={r['label']}"
            cv2.putText(tile, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
            subtitle = f"HARD-V2={r['prediction']['label']} | p={r['prediction']['confidence']:.6f}"
            cv2.putText(tile, subtitle, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
            cv2.imwrite(str(output / f"review_f{key[0]:05d}_p{key[1]}.jpg"), tile)
    finally:
        cap.release()
    if tiles:
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(output / "errors_and_frame2400.jpg"), sheet)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--input-dir", type=Path, help="Directory containing one video and one keypoint JSONL")
    source.add_argument("--metadata", type=Path, help="Legacy manifest (historical reproduction only)")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--project-root", type=Path, help="Data project root; never inferred from installed source")
    p.add_argument("--output-dir", type=Path,
                   help="Optional override; defaults to this module's outputs/<input-folder-name>")
    p.add_argument("--skip-video", action=argparse.BooleanOptionalAction, default=False, help="Write predictions only; no decode or images")
    p.add_argument("--labeled-predictions", type=Path, help="Optional saved diagnostics for review crops")
    args = parse_config(p, required=("model_dir",), source_required=True)
    output = resolve_output_directory(args.input_dir, args.output_dir, project_root=args.project_root)
    meta, kp, video, ancestor = resolve_input_directory(args.input_dir) if args.input_dir else resolve_sources(args.metadata)
    classifier = PostureClassifier(args.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    qa = output / "qa"
    if not args.skip_video:
        qa.mkdir()
    kp_hash = sha256(kp)
    model_hash = sha256(args.model_dir / "weights.npz")
    video_path = output / f"{video.stem}_hard_v2_pose_posture.mp4"
    json_path = output / f"{video.stem}_keypoints.jsonl"
    capture = writer = None
    frames = people_count = 0
    counts, rejects, last_source, started = Counter(), Counter(), -2, time.perf_counter()
    width, height, fps = meta["video"]["width"], meta["video"]["height"], meta["video"]["fps"]
    if not args.skip_video:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError("Cannot open original video")
        if (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (width, height):
            capture.release()
            raise ValueError("Video geometry does not match the cache")
        if abs(capture.get(cv2.CAP_PROP_FPS) - fps) > .01:
            capture.release()
            raise ValueError("Video fps does not match the cache")
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Cannot open output video")
    try:
        with json_path.open("w", encoding="utf-8") as stream:
            iterator = iter(read_jsonl(kp))
            while batch := list(islice(iterator, 64)):
                keypoints = [person["keypoints"] for frame in batch for person in frame["people"]]
                predictions = iter(classifier.predict_many(keypoints))
                for frame in batch:
                    if frame["output_frame"] != frames or frame["source_frame"] <= last_source:
                        raise ValueError("Nonsequential cached frames")
                    image = None
                    if capture is not None:
                        if frame["source_frame"] != last_source + 1:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, frame["source_frame"])
                        ok, image = capture.read()
                        if not ok:
                            raise RuntimeError(f"Decode failed at source frame {frame['source_frame']}")
                    last_source = frame["source_frame"]
                    occupied = []
                    for person in frame["people"]:
                        prediction = next(predictions)
                        if "posture" in person:
                            person["previous_posture"] = person["posture"]
                        person["posture"] = prediction
                        counts[prediction["label"]] += 1
                        rejects.update(prediction["rejection_reasons"])
                        people_count += 1
                        if image is not None:
                            draw_person(image, person, prediction, classifier.config["keypoint_threshold"], occupied)
                    if image is not None:
                        cv2.rectangle(image, (0, 0), (width, 32), (20, 20, 20), -1)
                        text = (f"EXPERIMENTAL HARD-V2 | source {frame['source_time_sec']:.3f}s | "
                                f"frame {frames} | p is conditional, not guaranteed reliability")
                        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
                        writer.write(image)
                        if frames in QA_FRAMES:
                            cv2.imwrite(str(qa / f"frame_{frames:05d}.jpg"), image)
                    stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
                    frames += 1
                if frames % 512 == 0:
                    print(f"Relabeled/rendered {frames} frames, {people_count} persons, "
                          f"{time.perf_counter()-started:.1f}s", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
    if frames != meta["processed_frames"] or people_count != meta["detected_people_instances"]:
        raise ValueError("Incomplete output")
    if sha256(kp) != kp_hash:
        raise ValueError("Source cache changed during rendering")
    if args.labeled_predictions and not args.skip_video:
        make_review(video, args.labeled_predictions, kp, qa)
    result = {
        "input": str(video), "model": meta["model"], "posture_model": str(args.model_dir.resolve()),
        "dataset_name": args.input_dir.resolve().name if args.input_dir else None,
        "input_directory": str(args.input_dir.resolve()) if args.input_dir else None,
        "output_directory": str(output),
        "posture_weights_sha256": model_hash, "posture_config_sha256": sha256(args.model_dir / "model.json"),
        "algorithm_version": "hard-2.0", "experimental": True,
        "reuses_keypoints": True, "cached_keypoints": str(kp), "cached_keypoints_sha256": kp_hash,
        "scene_calibration": None, "absolute_spatial_features": False,
        "video": meta["video"], "intervals_sec": meta["intervals_sec"],
        "processed_frames": frames, "detected_people_instances": people_count,
        "posture_counts": dict(counts), "rejection_reasons": dict(rejects),
        "elapsed_seconds_without_yolo": time.perf_counter() - started,
        "posture_config": {"temperature": classifier.temperature, "reject_threshold": classifier.threshold, "rejection_policy": classifier.policy,
                           "keypoint_threshold": classifier.config["keypoint_threshold"]},
        "outputs": {"keypoints_jsonl": str(json_path),
                    "annotated_video": None if args.skip_video else str(video_path),
                    "qa": None if args.skip_video else str(qa)},
        "limitations": [
            "Experimental single-camera baseline; calibration support is small.",
            "No labels are used to override inference, including known ignored duplicate detections.",
            "Cached detector duplicates/false positives remain; this is posture-only inference.",
            "Frame-local person IDs are preserved, NOT persistent tracking IDs.",
        ],
    }
    write_json(output / f"{video.stem}_run_metadata.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
