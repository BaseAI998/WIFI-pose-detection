"""Predeclared lightweight model families; all hyperparameters frozen before evaluation."""
from __future__ import annotations
import numpy as np
from .features import MaskedScaler, extract, mirror
from .model import descriptor_indices, paired_distances, support_evidence, build_evidence
from .data import LABELS

FAMILIES = {
    "baseline_mlp": dict(hidden=[32,16], dropout=.15, lr=.002, decay=.03, epochs=250, smoothing=0.),
    "regularized_mlp": dict(hidden=[16,8], dropout=.35, lr=.002, decay=.20, epochs=180, smoothing=.08),
    "linear": dict(hidden=[], dropout=0., lr=.005, decay=.5, epochs=250, smoothing=.05),
}


def forward(arrays, x, layers):
    for i in range(layers):
        x = x @ arrays[f"w{i}"] + arrays[f"b{i}"]
        if i + 1 < layers:
            x = np.maximum(x, 0)
    return x


def training_radii(reference_a, reference_b, labels, times):
    d = paired_distances(reference_a, reference_b, reference_a, reference_b)
    # Do not treat the same frame or immediate temporal neighbors as independent support.
    separated = np.abs(times[:, None] - times[None, :]) >= 3.0
    radii = []
    for cls in (0, 1):
        nearest = []
        for i in np.flatnonzero(labels == cls):
            mask = separated[i] & (labels == cls)
            if mask.any():
                nearest.append(float(d[i, mask].min()))
        if len(nearest) < 2:
            raise ValueError("Insufficient separated same-class support for radius estimation")
        radii.append(max(.15, float(np.quantile(nearest, .95))))
    return np.array(radii, dtype=np.float32)


def fit_model(rows, family, seed=20261019):
    import torch
    from torch import nn
    cfg = FAMILIES[family]
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    features = [extract(r["keypoints"]) for r in rows]
    binary_ids = [i for i, r in enumerate(rows) if r["label"] in LABELS and not features[i].quality_reasons]
    reference_ids = [i for i,r in enumerate(rows) if r["label"] != "ignore" and not features[i].quality_reasons]
    augmented = []
    y = []
    for i in binary_ids:
        augmented.extend([features[i], extract(mirror(rows[i]["keypoints"]))])
        y.extend([LABELS.index(rows[i]["label"])]*2)
    if set(y) != {0,1}:
        raise ValueError("Both classes are required in the training block")
    scaler = MaskedScaler.fit(augmented)  # Only binary TRAIN observations.
    x = scaler.transform(augmented)
    y = np.array(y)
    dims = [x.shape[1], *cfg["hidden"], 2]
    modules, linear = [], []
    for i in range(len(dims)-1):
        layer = nn.Linear(dims[i], dims[i+1])
        linear.append(layer)
        modules.append(layer)
        if i < len(dims)-2:
            modules.extend([nn.ReLU(), nn.Dropout(cfg["dropout"])])
    network = nn.Sequential(*modules)
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg["lr"], weight_decay=cfg["decay"])
    weights = len(y) / (2*np.bincount(y, minlength=2))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32),
                                   label_smoothing=cfg["smoothing"])
    xx, yy = torch.tensor(x), torch.tensor(y, dtype=torch.long)
    history = []
    for epoch in range(cfg["epochs"]):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(network(xx), yy)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 25 == 0 or epoch == cfg["epochs"]-1:
            history.append({"epoch": epoch+1, "loss": float(loss.detach())})
    network.eval()
    arrays = {}
    for i, layer in enumerate(linear):
        arrays[f"w{i}"] = layer.weight.detach().numpy().T.copy()
        arrays[f"b{i}"] = layer.bias.detach().numpy().copy()
    with torch.no_grad():
        expected = network(xx).numpy()
    actual = forward(arrays, x, len(linear))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    a = scaler.transform([features[i] for i in reference_ids])
    b = scaler.transform([extract(mirror(rows[i]["keypoints"])) for i in reference_ids])
    indices = descriptor_indices(features[0].names)
    labels = np.array([LABELS.index(rows[i]["label"]) if rows[i]["label"] in LABELS else 2 for i in reference_ids])
    times = np.array([rows[i]["source_time_sec"] for i in reference_ids])
    radii = training_radii(a[:,indices], b[:,indices], labels, times)
    arrays.update(mean=scaler.mean, scale=scaler.scale, masks=scaler.masks, descriptor_indices=indices,
                  reference_a=a[:,indices], reference_b=b[:,indices], reference_labels=labels, support_radii=radii)
    info = {
        "family": family, "hyperparameters": cfg, "seed": seed, "feature_names": features[0].names,
        "linear_layers": len(linear), "parameter_count": sum(p.numel() for p in network.parameters()),
        "binary_training_n": len(binary_ids), "reference_n": len(reference_ids),
        "reference_label_counts": {str(k): int((labels==k).sum()) for k in (0,1,2)},
        "reference_keys": [[rows[i]["output_frame"],rows[i]["person_id"]] for i in reference_ids],
        "training_keys": [[rows[i]["output_frame"],rows[i]["person_id"]] for i in binary_ids],
        "support_radii": radii.tolist(), "train_history": history,
        "numpy_export_max_error": float(abs(expected-actual).max()), "torch_version": torch.__version__,
    }
    return arrays, info


def raw_evidence(arrays, info, rows):
    f = [extract(r["keypoints"]) for r in rows]
    flipped = [extract(mirror(r["keypoints"])) for r in rows]
    scaler = MaskedScaler(arrays["mean"], arrays["scale"], arrays["masks"])
    a, b = scaler.transform(f), scaler.transform(flipped)
    l, r = forward(arrays,a,info["linear_layers"]), forward(arrays,b,info["linear_layers"])
    indices = arrays["descriptor_indices"]
    support = support_evidence(a[:,indices], b[:,indices], arrays["reference_a"], arrays["reference_b"],
                               arrays["reference_labels"], ((l+r)/2).argmax(1), arrays["support_radii"])
    return f, l, r, support


def evidence_at_temperature(raw, temperature):
    f, left, right, support = raw
    return build_evidence(f, left, right, temperature, support)
