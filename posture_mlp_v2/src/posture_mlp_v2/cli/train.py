"""Frozen temporal development comparison; there is NO new independent test data."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from ..data import BLOCKS, LABELS, load_data, write_json, write_jsonl, key
from ..features import FEATURE_VERSION, extract, MaskedScaler
from ..model import PostureClassifier, decide, paired_distances
from ..training import FAMILIES, fit_model, raw_evidence, evidence_at_temperature
from ..calibration import conservative_temperature, select_policy, feasible
from ..evaluation import evaluate, classification_metrics, calibration_metrics
from ._config import parse_config


def detailed_metrics(rows, predictions):
    report = evaluate(rows,predictions)
    report["by_origin"] = {}
    for origin in ("old","hard_new"):
        indices = [i for i,r in enumerate(rows) if r["origin"] == origin]
        report["by_origin"][origin] = evaluate([rows[i] for i in indices], [predictions[i] for i in indices])
    accepted = [(r,p) for r,p in zip(rows,predictions) if r["label"] != "ignore" and p["label"] != "unknown"]
    unsafe = sum(r["label"] == "uncertain" or p["label"] != r["label"] for r,p in accepted)
    report["operational_risk_including_uncertain"] = {
        "accepted":len(accepted), "unsafe_acceptances":unsafe,
        "rate":unsafe/len(accepted) if accepted else None,
        "meaning":"Accepted uncertain examples count as unsafe, not necessarily factually wrong standing/sitting.",
    }
    return report


def export_model(directory, arrays, info, temperature, policy, audit):
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory/"weights.npz",**arrays)
    config = {
        "algorithm_version":"hard-2.0","feature_version":FEATURE_VERSION,
        "feature_names":info["feature_names"],"classes":list(LABELS),"keypoint_threshold":.35,
        "linear_layers":info["linear_layers"],"family":info["family"],
        "temperature":temperature,"rejection_policy":policy,
        "source_dataset_id":audit["dataset_id"],"hard_labels_sha256":audit["sources"]["hard_labels"]["sha256"],
        "absolute_spatial_features":False,"scene_calibration":None,
        "experimental":True,"independent_test_available":False,
        "reference_semantics":"Training-only labelled known/uncertain neighbors; no timestamps/IDs as inference features",
    }
    write_json(directory/"model.json",config)
    write_json(directory/"training_info.json",info)


def annotation_consistency(rows, arrays, info):
    """Suggest human review of pose-near contradictory labels; NEVER relabel."""
    relevant = [r for r in rows if r["label"] in (*LABELS,"uncertain") and not extract(r["keypoints"]).quality_reasons]
    raw = raw_evidence(arrays,info,relevant)
    scaler = MaskedScaler(arrays["mean"],arrays["scale"],arrays["masks"])
    from ..features import mirror
    a = scaler.transform(raw[0])[:,arrays["descriptor_indices"]]
    b = scaler.transform([extract(mirror(r["keypoints"])) for r in relevant])[:,arrays["descriptor_indices"]]
    d = paired_distances(a,b,a,b)
    candidates = []
    for i,r in enumerate(relevant):
        if r["label"] != "uncertain":
            continue
        possible = [j for j,s in enumerate(relevant) if s["label"] == "standing" and abs(s["source_time_sec"]-r["source_time_sec"]) >= 1.0]
        if possible:
            j = min(possible,key=lambda j:d[i,j])
            if d[i,j] < .75:
                fields = ("output_frame","source_frame","source_time_sec","person_id","label","occlusion","origin","bbox_xyxy")
                candidates.append({"distance":float(d[i,j]),
                                   "uncertain":{k:r[k] for k in fields},
                                   "standing":{k:relevant[j][k] for k in fields}})
    candidates.sort(key=lambda r:r["distance"])
    # Do not flood the reviewer with one person's adjacent frames.
    diverse = []
    for c in candidates:
        if all(abs(c["uncertain"]["source_time_sec"]-r["uncertain"]["source_time_sec"]) >= 4 for r in diverse):
            diverse.append(c)
        if len(diverse) == 10:
            break
    return {"policy":"Flag only. Geometric similarity does not prove label inconsistency.",
            "standing_standard":"Confidently standing, including bent-over standing, is standing.",
            "uncertain_standard":"Support/contact cannot be determined, transition, or ambiguous half-squat.",
            "review_pairs":diverse,"labels_automatically_changed":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir",type=Path,help="Directory containing one video and one keypoint JSONL")
    source.add_argument("--metadata",type=Path,help="Legacy manifest (historical reproduction only)")
    parser.add_argument("--old-labels",type=Path)
    parser.add_argument("--hard-labels",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--project-root",type=Path,help="Base directory for config paths")
    parser.add_argument("--seed",type=int,default=20261019)
    args = parse_config(parser, required=("old_labels", "hard_labels", "output_dir"), source_required=True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a NEW experiment directory; frozen experiments are never overwritten")
    print("Validating all labels, source hashes and unchanged cached geometry...",flush=True)
    rows,meta,audit = load_data(args.metadata,args.old_labels,args.hard_labels,input_dir=args.input_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"data_audit.json",audit)
    write_json(out/"input_summary.json",meta)
    protocol = {
        "status":"development comparison, NOT independent final test",
        "seed":args.seed,"families":FAMILIES,"calibration_methods":["none","conservative_T"],
        "temporal_blocks_relative_seconds":BLOCKS,"embargo_seconds":6,
        "fold_roles":[{"train":i,"calibration":(i+1)%3,"evaluation":(i+2)%3} for i in range(3)],
        "final_fit":{"train_blocks":[0,2],"calibration_block":1,"test":None},
        "old_test_status":"Now development. Active-mined hard labels also development.",
        "selection":"feasible development risk/uncertain rejection first, then maximize coverage; no final-test claim",
        "target_known_error":.03,"target_uncertain_rejection":.9,
        "absolute_spatial_features":False,"scene_calibration":False,
        "probability_only_ablation":True,
        "support":"Training-only nearest-class consistency, class radius, uncertain-neighborhood rejection",
        "no_image_or_occlusion_branch":"Insufficient evidence to justify extra models before this comparison",
    }
    write_json(out/"protocol.json",protocol)
    write_jsonl(out/"split_manifest.jsonl",[
        {**{k:r[k] for k in ("output_frame","source_frame","source_time_sec","person_id","label","occlusion","origin","block")},
         "final_role":"embargo" if r["block"] == -1 else "calibration" if r["block"] == 1 else "train"}
        for r in rows])
    print(json.dumps(audit["block_label_counts"],indent=2),flush=True)
    buckets = {i:[r for r in rows if r["block"] == i] for i in range(3)}
    comparison = {f"{f}__{c}":{"rows":[],"predictions":[],"probability_only":[],"folds":[]}
                  for f in FAMILIES for c in ("none","conservative_T")}
    diagnostics = []
    for fold in range(3):
        training,calibration,validation = buckets[fold],buckets[(fold+1)%3],buckets[(fold+2)%3]
        for family in FAMILIES:
            print(f"Fold {fold}: {family}, train={len(training)} cal={len(calibration)} eval={len(validation)}",flush=True)
            arrays,info = fit_model(training,family,args.seed+fold)
            cal_raw = raw_evidence(arrays,info,calibration)
            val_raw = raw_evidence(arrays,info,validation)
            proper = [i for i,r in enumerate(calibration) if r["label"] in LABELS and not cal_raw[0][i].quality_reasons]
            t,temperature_report = conservative_temperature(
                ((cal_raw[1]+cal_raw[2])/2)[proper], [LABELS.index(calibration[i]["label"]) for i in proper])
            fold_dir = out/"folds"/f"fold_{fold}_{family}"
            fold_dir.mkdir(parents=True)
            write_json(fold_dir/"training_info.json",info)
            for method in ("none","conservative_T"):
                temperature = 1. if method == "none" else t
                cal_ev = evidence_at_temperature(cal_raw,temperature)
                policy,selection = select_policy(calibration,cal_ev,True)
                probability_policy,probability_selection = select_policy(calibration,cal_ev,False)
                ev = evidence_at_temperature(val_raw,temperature)
                predicted = [decide(e,policy) for e in ev]
                probability_predicted = [decide(e,probability_policy) for e in ev]
                name = f"{family}__{method}"
                comp = comparison[name]
                comp["rows"].extend(validation)
                comp["predictions"].extend(predicted)
                comp["probability_only"].extend(probability_predicted)
                record = {
                    "fold":fold,"temperature":temperature,"policy":policy,
                    "calibration_selection":selection,"temperature_fit":temperature_report,
                    "probability_only_calibration_selection":probability_selection,
                    "evaluation":detailed_metrics(validation,predicted),
                    "probability_only_evaluation":detailed_metrics(validation,probability_predicted),
                }
                write_json(fold_dir/f"{method}.json",record)
                comp["folds"].append({k:record[k] for k in ("fold","temperature","policy","evaluation","probability_only_evaluation")})
                for r,p in zip(validation,predicted):
                    diagnostics.append({k:r[k] for k in ("output_frame","source_frame","source_time_sec","person_id",
                                                        "label","occlusion","origin","block")})
                    diagnostics[-1].update(candidate=name,fold=fold,prediction=p)
    summaries = {}
    for name,comp in comparison.items():
        if len({key(r) for r in comp["rows"]}) != len(comp["rows"]):
            raise AssertionError("A development sample was evaluated multiple times for one candidate")
        summaries[name] = {"support_gated":detailed_metrics(comp["rows"],comp["predictions"]),
                           "probability_only":detailed_metrics(comp["rows"],comp["probability_only"]),
                           "folds":comp["folds"]}
    # Explicitly a DEVELOPMENT choice. No unseen test exists in the supplied files.
    def rank(name):
        m = summaries[name]["support_gated"]["overall"]
        valid = feasible(m)
        risk = 1. if m["selective_error"] is None else m["selective_error"]
        u = m["uncertain_rejection_rate"] if m["uncertain_n"] else 1.
        violations = max(0,risk-.03) + max(0,.9-u)
        return (valid, -violations, m["coverage"] or 0, -risk, name.endswith("conservative_T"),
                -len(FAMILIES[name.split("__")[0]]["hidden"]))
    chosen = max(summaries,key=rank)
    write_json(out/"development_comparison.json",{"selected":chosen,"candidates":summaries,
                                                 "not_independent_test":True})
    write_jsonl(out/"crossfit_predictions.jsonl",diagnostics)
    print(f"Development choice frozen: {chosen}",flush=True)
    family,method = chosen.split("__")
    final_train = [r for r in rows if r["block"] in (0,2)]
    final_cal = buckets[1]
    arrays,info = fit_model(final_train,family,args.seed)
    cal_raw = raw_evidence(arrays,info,final_cal)
    proper = [i for i,r in enumerate(final_cal) if r["label"] in LABELS and not cal_raw[0][i].quality_reasons]
    t,temperature_report = conservative_temperature(((cal_raw[1]+cal_raw[2])/2)[proper],
                                                   [LABELS.index(final_cal[i]["label"]) for i in proper])
    temperature = 1. if method == "none" else t
    policy,selection = select_policy(final_cal,evidence_at_temperature(cal_raw,temperature))
    export_model(out/"model",arrays,info,temperature,policy,audit)
    write_json(out/"final_calibration.json",{"temperature_fit":temperature_report,"actual_temperature":temperature,
                                            "policy_selection":selection,"not_independent_final_test":True})
    classifier = PostureClassifier(out/"model")
    predictions = classifier.predict_many([r["keypoints"] for r in rows])
    saved = []
    for r,p in zip(rows,predictions):
        saved.append({k:r[k] for k in ("output_frame","source_frame","source_time_sec","person_id","label","occlusion","origin","block")})
        saved[-1].update(prediction=p,split="embargo" if r["block"] == -1 else "calibration" if r["block"] == 1 else "train")
    write_jsonl(out/"labeled_predictions.jsonl",saved)
    # These final-model numbers are explicitly in-sample/calibration, never OOF or test.
    write_json(out/"final_fit_diagnostics.json",{"not_independent":True,**detailed_metrics(rows,predictions)})
    write_json(out/"annotation_consistency.json",annotation_consistency(rows,arrays,info))
    times = []
    sample = [r["keypoints"] for r in rows]
    classifier.predict_many(sample)
    for _ in range(3):
        started = time.perf_counter()
        classifier.predict_many(sample)
        times.append(time.perf_counter()-started)
    write_json(out/"benchmark.json",{"median_ms_per_person":1000*float(np.median(times))/len(sample),
                                    "includes":"masked features, reflected view, NumPy model, support rejection",
                                    "excludes":"YOLO, video decode/render, I/O","batch_size":len(sample)})
    write_json(out/"completion.json",{"status":"complete","independent_test_available":False,
                                      "selected":chosen,"final_policy_feasible":selection["chosen"]["feasible"]})
    print(json.dumps({"selected":chosen,"crossfit_development":summaries[chosen]["support_gated"]["overall"],
                      "final_policy":policy,"final_calibration":selection["chosen"]["metrics"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
