"""Verify completed outputs against every cached detection and frozen inference."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import cv2
import numpy as np

from ..data import read_jsonl, sha256, write_json, resolve_recorded_path
from ..model import PostureClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    args = parser.parse_args()
    resolve = lambda path: resolve_recorded_path(path, project_root=args.run_dir.resolve().parents[1])
    audit = json.loads((args.run_dir / "data_audit.json").read_text(encoding="utf-8"))
    metadata = json.loads(args.output_metadata.read_text(encoding="utf-8"))
    for field, original in audit["sources"].items():
        if sha256(resolve(original["path"])) != original["sha256"]:
            raise AssertionError(f"Source changed: {field}")
    classifier = PostureClassifier(args.run_dir / "model")
    info = json.loads((args.run_dir / "model" / "training_info.json").read_text(encoding="utf-8"))
    manifest = list(read_jsonl(args.run_dir / "split_manifest.jsonl"))
    train_keys = {(r["output_frame"],r["person_id"]) for r in manifest if r["final_role"] == "train"}
    calibration_keys = {(r["output_frame"],r["person_id"]) for r in manifest if r["final_role"] == "calibration"}
    if train_keys & calibration_keys:
        raise AssertionError("Training/calibration sample leakage")
    if not set(map(tuple,info["reference_keys"])) <= train_keys:
        raise AssertionError("Rejection references include calibration/embargo examples")
    if not set(map(tuple,info["training_keys"])) <= train_keys:
        raise AssertionError("Binary fitting includes calibration/embargo examples")
    for fold in range(3):
        for family in ("baseline_mlp","regularized_mlp","linear"):
            f = json.loads((args.run_dir/"folds"/f"fold_{fold}_{family}"/"training_info.json").read_text(encoding="utf-8"))
            allowed = {(r["output_frame"],r["person_id"]) for r in manifest if r["block"] == fold}
            if not set(map(tuple,f["reference_keys"])) <= allowed or not set(map(tuple,f["training_keys"])) <= allowed:
                raise AssertionError("Development fold reference leakage")
    if sha256(args.run_dir / "model" / "weights.npz") != metadata["posture_weights_sha256"]:
        raise AssertionError("Model weights changed")
    if sha256(args.run_dir / "model" / "model.json") != metadata["posture_config_sha256"]:
        raise AssertionError("Model configuration changed")
    expected_labels = {(r["output_frame"], r["person_id"]): r
                       for r in read_jsonl(args.run_dir / "labeled_predictions.jsonl")}
    matched = set()
    counts, reasons = Counter(), Counter()
    frames = people = reproducibility_checked = 0
    for source, output in zip_longest(read_jsonl(resolve(audit["sources"]["keypoints"]["path"])),
                                     read_jsonl(resolve(metadata["outputs"]["keypoints_jsonl"]))):
        if source is None or output is None:
            raise AssertionError("Frame count mismatch")
        for field in ("output_frame", "source_frame", "source_time_sec", "source_time", "segment"):
            if source[field] != output[field]:
                raise AssertionError(f"Frame identity mismatch: {field}")
        if len(source["people"]) != len(output["people"]):
            raise AssertionError("Person count mismatch")
        for old, new in zip(source["people"], output["people"]):
            for field in ("person_id", "bbox_xyxy", "detection_confidence", "keypoints"):
                if old[field] != new[field]:
                    raise AssertionError(f"Original detection changed: {field}")
            if "posture" in old and old["posture"] != new.get("previous_posture"):
                raise AssertionError("Previous posture provenance lost")
            p = new["posture"]
            values = list(p["probabilities"].values())
            if not np.isfinite(values).all() or not np.isclose(sum(values), 1):
                raise AssertionError("Invalid probabilities")
            if p["label"] not in ("sitting", "standing", "unknown"):
                raise AssertionError("Invalid label")
            if (p["label"] == "unknown") != bool(p["rejection_reasons"]):
                raise AssertionError("Unknown/rejection inconsistency")
            if p["label"] != "unknown" and p["confidence"] < classifier.threshold:
                raise AssertionError("Threshold bypassed")
            counts[p["label"]] += 1
            reasons.update(p["rejection_reasons"])
            key = (output["output_frame"], new["person_id"])
            if key in expected_labels:
                expected = expected_labels[key]["prediction"]
                if expected["label"] != p["label"] or expected["rejection_reasons"] != p["rejection_reasons"]:
                    raise AssertionError(f"Labeled inference mismatch: {key}")
                np.testing.assert_allclose(list(expected["probabilities"].values()), values, atol=1e-5)
                matched.add(key)
            if frames % 211 == 0:
                result = classifier.predict(new["keypoints"])
                if result["label"] != p["label"]:
                    raise AssertionError("Batch/single inference mismatch")
                np.testing.assert_allclose(list(result["probabilities"].values()), values, atol=1e-5)
                reproducibility_checked += 1
            people += 1
        frames += 1
    if len(matched) != len(expected_labels):
        raise AssertionError("Not all annotated examples appear in output")
    if (frames, people) != (audit["verified_frames"], audit["verified_all_cache_detections"]):
        raise AssertionError("Incomplete results")
    if dict(counts) != metadata["posture_counts"] or dict(reasons) != metadata["rejection_reasons"]:
        raise AssertionError("Output summary mismatch")
    decoded = 0
    video = metadata["outputs"]["annotated_video"]
    if video:
        capture = cv2.VideoCapture(str(resolve(video)))
        if not capture.isOpened():
            raise AssertionError("Output video cannot be opened")
        try:
            if abs(capture.get(cv2.CAP_PROP_FPS) - metadata["video"]["fps"]) > .01:
                raise AssertionError("Output video FPS mismatch")
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape[:2] != (metadata["video"]["height"], metadata["video"]["width"]):
                    raise AssertionError("Output video dimensions mismatch")
                decoded += 1
        finally:
            capture.release()
        if decoded != frames:
            raise AssertionError("Output video is truncated")
    result = {"status": "passed", "all_source_hashes_unchanged": True,
              "all_fold_and_final_training_reference_leakage_checks": "passed",
              "all_detections_and_keypoints_unchanged": True, "frames": frames,
              "person_instances": people, "all_labeled_instances_matched": len(matched),
              "single_vs_batch_predictions_checked": reproducibility_checked,
              "video_frames_fully_decoded": decoded, "posture_counts": dict(counts)}
    write_json(args.output_metadata.parent / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
