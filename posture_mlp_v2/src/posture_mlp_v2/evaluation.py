"""Selective classification metrics: abstention counts as a recall miss."""
from __future__ import annotations
import math
import numpy as np
from .data import LABELS


def ratio(a, b):
    return float(a / b) if b else None


def wilson(k, n):
    if not n:
        return None
    z = 1.96
    p, d = k / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, mid - half), min(1, mid + half)]


def classification_metrics(truth, prediction):
    if len(truth) != len(prediction):
        raise ValueError("Truth/prediction length mismatch")
    known = [(a, b) for a, b in zip(truth, prediction) if a in LABELS]
    matrix = {a: {b: 0 for b in (*LABELS, "unknown")} for a in LABELS}
    for a, b in known:
        matrix[a][b] += 1
    accepted = sum(b != "unknown" for a, b in known)
    errors = sum(a != b and b != "unknown" for a, b in known)
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[a][label] for a in LABELS)
        precision, recall = ratio(tp, predicted), ratio(tp, support)
        # An absent class is not a perfect class.
        f1 = ratio(2 * tp, support + predicted)
        per_class[label] = {"support": support, "predicted": predicted, "precision": precision,
                            "recall": recall, "f1": f1,
                            "unknown": matrix[label]["unknown"]}
    uncertain = [b for a, b in zip(truth, prediction) if a == "uncertain"]
    ignored = [b for a, b in zip(truth, prediction) if a == "ignore"]
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    return {
        "known_n": len(known), "accepted": accepted, "errors_among_accepted": errors,
        "coverage": ratio(accepted, len(known)), "unknown_rate": ratio(len(known) - accepted, len(known)),
        "selective_error": ratio(errors, accepted), "selective_error_wilson95": wilson(errors, accepted),
        "accuracy_counting_unknown_as_miss": ratio(sum(a == b for a, b in known), len(known)),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "confusion": matrix, "per_class": per_class,
        "uncertain_n": len(uncertain), "uncertain_rejected": uncertain.count("unknown"),
        "uncertain_rejection_rate": ratio(uncertain.count("unknown"), len(uncertain)),
        "ignore_n": len(ignored), "ignore_rejected": ignored.count("unknown"),
        "ignore_policy": "Excluded from binary metrics, model fitting, calibration and threshold selection.",
    }


def evaluate(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Rows/prediction length mismatch")
    def subset(indices):
        return classification_metrics([rows[i]["label"] for i in indices],
                                      [predictions[i]["label"] for i in indices])
    group_values = ("legs_clear_proxy", "legs_partial_proxy", "legs_hidden_proxy")
    return {
        "overall": subset(list(range(len(rows)))),
        "visibility_proxy_groups": {
            g: subset([i for i, p in enumerate(predictions) if p["visibility_group"] == g])
            for g in group_values},
        "manual_occlusion_groups": {
            g: subset([i for i, r in enumerate(rows) if r["occlusion"] == g])
            for g in ("unmarked", "clear", "partial", "severe")},
    }


def calibration_metrics(y, probability):
    y, p = np.asarray(y), np.asarray(probability)
    if not len(y):
        return {"n": 0}
    confidence = p.max(1)
    correct = p.argmax(1) == y
    ece, bins = 0.0, []
    for lo in np.linspace(0.5, 0.9, 5):
        hi = lo + 0.1
        mask = (confidence >= lo - 1e-8) & (confidence < hi if hi < 0.999 else confidence <= 1)
        if mask.any():
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean() * abs(acc - conf)
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                         "accuracy": acc, "confidence": conf})
    return {"n": len(y), "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-9)).mean()),
            "brier": float(np.mean((p[:, 1] - y) ** 2)), "ece_5_bins": float(ece), "bins": bins}
