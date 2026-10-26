"""Conservative calibration and low-risk operating points; never inspect evaluation labels."""
from __future__ import annotations
import numpy as np
from .data import LABELS
from .model import softmax, decide
from .evaluation import classification_metrics, calibration_metrics

PROBABILITY_THRESHOLDS = (.65, .70, .75, .80, .85, .90, .925, .95, .975, .99, .995, .999)
RADIUS_MULTIPLIERS = (.75, 1.0, 1.5, 2.0)
UNCERTAIN_RATIOS = (.8, 1.0, 1.25)


def conservative_temperature(logits, y):
    logits, y = np.asarray(logits), np.asarray(y, dtype=int)
    if set(y) != {0,1}:
        raise ValueError("Both classes needed for conservative calibration")
    grid = np.exp(np.linspace(0, np.log(8.), 121))
    nll = []
    for t in grid:
        prob = softmax(logits/t)
        nll.append(float(-np.log(prob[np.arange(len(y)),y].clip(1e-12)).mean()))
    index = int(np.argmin(nll))
    return float(grid[index]), {"method": "temperature_NLL_T_greater_equal_1",
                               "n": len(y), "bounds": [1.,8.], "upper_boundary": index == len(grid)-1,
                               "before": calibration_metrics(y,softmax(logits)),
                               "after": calibration_metrics(y,softmax(logits/grid[index]))}


def feasible(metrics):
    return (metrics["accepted"] >= 10
            and metrics["selective_error"] is not None and metrics["selective_error"] <= .03
            and (metrics["uncertain_n"] == 0 or metrics["uncertain_rejection_rate"] >= .90)
            and all(metrics["per_class"][c]["predicted"] >= 2 for c in LABELS))


def select_policy(rows, evidence, support_gate=True):
    truth = [r["label"] for r in rows]
    grid = []
    radii = RADIUS_MULTIPLIERS if support_gate else (1.0,)
    ratios = UNCERTAIN_RATIOS if support_gate else (1.0,)
    for radius in radii:
        for ratio in ratios:
            for threshold in PROBABILITY_THRESHOLDS:
                policy = {"threshold": threshold, "support_gate": support_gate,
                          "radius_multiplier": radius, "uncertain_ratio": ratio}
                predictions = [decide(e,policy)["label"] for e in evidence]
                metrics = classification_metrics(truth,predictions)
                grid.append({"policy":policy, "metrics":metrics, "feasible":feasible(metrics)})
    good = [r for r in grid if r["feasible"]]
    if good:
        chosen = max(good, key=lambda r: (
            r["metrics"]["accepted"], -r["metrics"]["selective_error"],
            r["metrics"]["uncertain_rejection_rate"] or 0,
            r["policy"]["threshold"], -r["policy"]["radius_multiplier"], r["policy"]["uncertain_ratio"]))
    else:
        policy = {"threshold": 1.000001, "support_gate":support_gate,
                  "radius_multiplier":1.0, "uncertain_ratio":1.0}
        chosen = {"policy":policy, "metrics":classification_metrics(truth,["unknown"]*len(rows)), "feasible":False}
    return chosen["policy"], {
        "chosen":chosen, "grid":grid, "support_gate":support_gate,
        "criterion":"maximize coverage subject to <=3% empirical known error, >=90% uncertain rejection, >=10 accepted, >=2 predictions per class",
        "warning":"Calibration-selected empirical risk, NOT a statistical guarantee; fail closed if no supported policy.",
    }
