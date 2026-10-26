"""Translation/scale invariant features, with explicit validity for every geometry.

No boxes, pixel locations, timestamps, person IDs, seat priors or old predictions
are model inputs. Image vertical is preserved (rotation is NOT normalized out).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FEATURE_VERSION = "coco17_masked_geometry_v1"
MIRROR = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle")


def as_keypoints(points) -> np.ndarray:
    if isinstance(points, list) and points and isinstance(points[0], dict):
        if len(points) != 17 or sorted(int(p["index"]) for p in points) != list(range(17)):
            raise ValueError("Exactly one keypoint per COCO index 0..16 is required")
        points = [[p["x"], p["y"], p["confidence"]]
                  for p in sorted(points, key=lambda p: int(p["index"]))]
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (17, 3):
        raise ValueError(f"Expected (17, 3) COCO keypoints, got {result.shape}")
    return result


def mirror(points) -> np.ndarray:
    result = as_keypoints(points)[MIRROR].copy()
    result[:, 0] *= -1
    return result


@dataclass
class Features:
    values: np.ndarray
    observed: np.ndarray
    names: list[str]
    mask_columns: np.ndarray
    quality_reasons: list[str]
    visibility_group: str


def extract(points, threshold: float = 0.35) -> Features:
    if not 0 < threshold <= 1:
        raise ValueError("keypoint confidence threshold must be in (0, 1]")
    k = as_keypoints(points)
    finite = np.isfinite(k).all(1)
    conf = np.where(finite, np.clip(k[:, 2], 0, 1), 0)
    visible = finite & (conf >= threshold)
    xy = np.where(finite[:, None], k[:, :2], 0)
    shoulders, hips = np.flatnonzero(visible[5:7]) + 5, np.flatnonzero(visible[11:13]) + 11
    same_side_torso = (visible[5] and visible[11]) or (visible[6] and visible[12])
    reasons = []
    center, scale = np.zeros(2), 1.0
    normalized = bool(len(shoulders) and len(hips) and same_side_torso)
    if normalized:
        shoulder, hip = xy[shoulders].mean(0), xy[hips].mean(0)
        center = (shoulder + hip) * 0.5
        scale = float(np.linalg.norm(hip - shoulder))
        normalized = scale > 1e-6
    if not normalized:
        reasons.append("missing_or_degenerate_torso")
        scale = 1.0
    if visible.sum() < 4:
        reasons.append("too_few_visible_keypoints")
    coords = np.zeros((17, 2))
    if normalized:
        coords[visible] = (xy[visible] - center) / scale
        if np.max(np.abs(coords[visible]), initial=0) > 10:
            reasons.append("implausible_normalized_geometry")
        coords = np.clip(coords, -10, 10)

    values, observed, names, masks = [], [], [], []

    def scalar(name, value, valid=True, mask=False):
        names.append(name)
        values.append(float(value) if valid else 0.0)
        observed.append(bool(valid))
        masks.append(mask)

    def geometry(name, value, valid):
        scalar(name, value, valid)
        scalar(name + ".available", valid, mask=True)

    for j, name in enumerate(JOINTS):
        valid = bool(visible[j] and normalized)
        geometry(name + ".x", coords[j, 0], valid)
        geometry(name + ".y", coords[j, 1], valid)
        scalar(name + ".confidence", conf[j])
        scalar(name + ".visible", visible[j], mask=True)

    pairs = ((5, 6), (11, 12), (5, 11), (6, 12), (5, 7), (7, 9),
             (6, 8), (8, 10), (11, 13), (12, 14), (13, 15), (14, 16),
             (11, 15), (12, 16), (13, 14), (15, 16))
    for a, b in pairs:
        valid = bool(normalized and visible[a] and visible[b])
        vec = coords[b] - coords[a]
        length = float(np.linalg.norm(vec))
        name = f"bone_{a}_{b}"
        geometry(name + ".dx", vec[0], valid)
        geometry(name + ".dy", vec[1], valid)
        geometry(name + ".length", length, valid)
        geometry(name + ".verticality", abs(vec[1]) / max(length, 1e-8), valid and length > 1e-8)

    angles = ((5, 11, 13), (6, 12, 14), (11, 13, 15), (12, 14, 16),
              (5, 7, 9), (6, 8, 10))
    for a, b, c in angles:
        v, w = coords[a] - coords[b], coords[c] - coords[b]
        product = float(np.linalg.norm(v) * np.linalg.norm(w))
        valid = bool(normalized and visible[[a, b, c]].all() and product > 1e-8)
        cosine = float(np.clip(np.dot(v, w) / max(product, 1e-8), -1, 1))
        geometry(f"angle_{a}_{b}_{c}.cos", cosine, valid)

    for name, indices in (("upper", list(range(5, 13))), ("lower", list(range(11, 17))),
                          ("distal_legs", list(range(13, 17))), ("all", list(range(17)))):
        scalar(name + ".visibility", visible[indices].mean())
        scalar(name + ".mean_confidence", conf[indices].mean())
    scalar("torso.available", normalized, mask=True)
    leg_visible = int(visible[13:17].sum())
    group = "legs_clear_proxy" if leg_visible >= 3 else (
        "legs_partial_proxy" if leg_visible else "legs_hidden_proxy")
    return Features(np.asarray(values, dtype=np.float32), np.asarray(observed, dtype=bool),
                    names, np.asarray(masks, dtype=bool), reasons, group)


class MaskedScaler:
    """Fit on training observations only; missing geometry stays exactly zero."""
    def __init__(self, mean, scale, masks):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.masks = np.asarray(masks, dtype=bool)

    @classmethod
    def fit(cls, features: list[Features]):
        x = np.stack([f.values for f in features])
        observed = np.stack([f.observed for f in features])
        n = observed.sum(0).clip(1)
        mean = (x * observed).sum(0) / n
        variance = (((x - mean) ** 2) * observed).sum(0) / n
        scale = np.sqrt(variance)
        # Near-constant dimensions must not amplify unseen variation enormously.
        scale[scale < 0.05] = 1.0
        masks = features[0].mask_columns
        mean[masks], scale[masks] = 0, 1
        return cls(mean, scale, masks)

    def transform(self, features: list[Features]) -> np.ndarray:
        x = np.stack([f.values for f in features])
        observed = np.stack([f.observed for f in features])
        return np.where(observed, np.clip((x - self.mean) / self.scale, -8, 8), 0).astype(np.float32)
