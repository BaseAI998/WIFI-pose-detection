"""NumPy-only deployment: binary classifier + training-support abstention.

Human uncertain examples are rejection references, NEVER a third softmax class.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .features import FEATURE_VERSION, MaskedScaler, extract, mirror
from .data import LABELS


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def descriptor_indices(names):
    return np.array([i for i, n in enumerate(names)
                     if (n.endswith((".x", ".y", ".confidence", ".visible"))
                         or n.startswith("angle_"))], dtype=int)


def paired_distances(a, b, reference_a, reference_b):
    """One distance per HUMAN example, not two votes for its mirrored views."""
    def distance(x, y):
        square = (x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T
        return np.sqrt(np.maximum(square / x.shape[1], 0))
    return np.minimum.reduce([distance(a, reference_a), distance(a, reference_b),
                              distance(b, reference_a), distance(b, reference_b)])


def support_evidence(a, b, ref_a, ref_b, ref_labels, candidates, radii):
    """Reflection invariant references; time/position never used at inference."""
    distance = paired_distances(a, b, ref_a, ref_b)
    result = []
    for d, candidate in zip(distance, candidates):
        known = ref_labels < 2
        same = ref_labels == candidate
        uncertain = ref_labels == 2
        same_distance = float(d[same].min()) if same.any() else 1e6
        uncertain_distance = float(d[uncertain].min()) if uncertain.any() else 1e6
        nearest_known = int(ref_labels[np.flatnonzero(known)[np.argmin(d[known])]]) if known.any() else -1
        result.append({
            "support_distance": same_distance,
            "support_radius": float(radii[candidate]),
            "uncertain_distance": uncertain_distance,
            "uncertain_ratio": uncertain_distance / max(same_distance, 1e-6),
            "nearest_known_disagrees": bool(nearest_known != candidate),
        })
    return result


def build_evidence(features, logits, reflected_logits, temperature, support=None):
    a, b = softmax(logits / temperature), softmax(reflected_logits / temperature)
    p = softmax((logits + reflected_logits) / (2*temperature))
    result = []
    for index, (f, q, first, second) in enumerate(zip(features, p, a, b)):
        reasons = list(f.quality_reasons)
        if first.argmax() != second.argmax():
            reasons.append("mirror_label_disagreement")
        item = {
            "candidate_label": LABELS[int(q.argmax())], "confidence": float(q.max()),
            "probabilities": dict(zip(LABELS, map(float, q))),
            "quality_reasons": reasons, "visibility_group": f.visibility_group,
            "mirror_probability_delta": float(abs(first[1] - second[1])),
        }
        if support is not None:
            item.update(support[index])
        result.append(item)
    return result


def decide(evidence, policy):
    if isinstance(policy, (float, int)):
        policy = {"threshold": float(policy), "support_gate": False}
    reasons = list(evidence["quality_reasons"])
    if evidence["confidence"] < policy["threshold"]:
        reasons.append("low_conservative_confidence")
    if policy.get("support_gate", False):
        if evidence["support_distance"] > evidence["support_radius"] * policy["radius_multiplier"]:
            reasons.append("outside_training_support")
        if evidence["uncertain_ratio"] <= policy["uncertain_ratio"]:
            reasons.append("near_human_uncertain_pose")
        if evidence["nearest_known_disagrees"]:
            reasons.append("local_class_evidence_conflict")
    return {**evidence, "label": "unknown" if reasons else evidence["candidate_label"],
            "rejection_reasons": reasons, "accepted": not reasons,
            "algorithm_version": "hard-2.0", "evidence_source": "binary_model_with_support_rejection"}


class PostureClassifier:
    def __init__(self, directory):
        directory = Path(directory)
        self.config = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if self.config["feature_version"] != FEATURE_VERSION or self.config["classes"] != list(LABELS):
            raise ValueError("Feature/class schema mismatch")
        self.temperature = float(self.config["temperature"])
        if not np.isfinite(self.temperature) or self.temperature < 1:
            raise ValueError("v2 calibration must NOT amplify confidence (T >= 1)")
        self.policy = self.config["rejection_policy"]
        self.threshold = self.policy["threshold"]
        with np.load(directory / "weights.npz", allow_pickle=False) as saved:
            self.arrays = {k: saved[k].copy() for k in saved.files}
        self.scaler = MaskedScaler(self.arrays["mean"], self.arrays["scale"], self.arrays["masks"])

    def forward(self, x):
        for i in range(self.config["linear_layers"]):
            x = x @ self.arrays[f"w{i}"] + self.arrays[f"b{i}"]
            if i + 1 < self.config["linear_layers"]:
                x = np.maximum(x, 0)
        return x

    def evidence(self, points):
        if len(points) == 0:
            return []
        features = [extract(p, self.config["keypoint_threshold"]) for p in points]
        flipped = [extract(mirror(p), self.config["keypoint_threshold"]) for p in points]
        if features[0].names != self.config["feature_names"]:
            raise ValueError("Feature order mismatch")
        a, b = self.scaler.transform(features), self.scaler.transform(flipped)
        logits, reflected = self.forward(a), self.forward(b)
        candidates = ((logits+reflected)/2).argmax(1)
        indices = self.arrays["descriptor_indices"]
        support = support_evidence(a[:, indices], b[:, indices], self.arrays["reference_a"],
                                   self.arrays["reference_b"], self.arrays["reference_labels"],
                                   candidates, self.arrays["support_radii"])
        return build_evidence(features, logits, reflected, self.temperature, support)

    def predict_many(self, points):
        return [decide(e, self.policy) for e in self.evidence(points)]

    def predict(self, points):
        return self.predict_many([points])[0]
