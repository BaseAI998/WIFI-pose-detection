import tempfile
import json
from pathlib import Path
import unittest
import numpy as np
from posture_mlp_v2.features import extract, mirror, MaskedScaler
from posture_mlp_v2.data import block_id
from posture_mlp_v2.model import softmax, paired_distances, support_evidence, decide, PostureClassifier
from posture_mlp_v2.calibration import conservative_temperature, select_policy
from posture_mlp_v2.training import fit_model, raw_evidence, evidence_at_temperature
from posture_mlp_v2.cli.train import export_model


def pose(seated=False):
    xy = np.array([[0,-2],[-.1,-2.1],[.1,-2.1],[-.3,-2],[.3,-2],[-.5,-1],[.5,-1],
                   [-.7,-.4],[.7,-.4],[-.6,0],[.6,0],[-.3,0],[.3,0],[-.3,1],[.3,1],[-.3,2],[.3,2]])
    if seated:
        xy[13:15] += [.8,-.7]
        xy[15:17] += [.8,-.7]
    return np.c_[xy*80+[400,200],np.ones(17)*.95]


class V2Tests(unittest.TestCase):
    def test_translation_scale_invariance(self):
        k = pose()
        q = k.copy()
        q[:,:2] = q[:,:2]*4+[500,-100]
        np.testing.assert_allclose(extract(k).values,extract(q).values,atol=1e-6)

    def test_missing_geometry_never_observed(self):
        k = pose()
        k[13:,2] = .1
        a = extract(k)
        k[13:,:2] = 10000
        np.testing.assert_array_equal(a.values,extract(k).values)
        scaler = MaskedScaler.fit([extract(pose()),a])
        self.assertTrue((scaler.transform([a])[0][~a.observed] == 0).all())

    def test_calibration_never_amplifies_confidence(self):
        logits = np.array([[3,-3],[-3,3],[2,-2],[-2,2]])
        t,info = conservative_temperature(logits,[0,1,0,1])
        self.assertGreaterEqual(t,1)
        self.assertTrue((softmax(logits/t).max(1) <= softmax(logits).max(1)+1e-12).all())

    def test_wrong_confident_data_softens_temperature(self):
        t,_ = conservative_temperature(np.array([[5,-5],[5,-5],[-5,5],[-5,5]]),[0,1,1,0])
        self.assertGreater(t,1)

    def test_temporal_embargo_all_persons(self):
        intervals = [[60,120]]
        for offset,b in ((0,0),(16.9,0),(17,-1),(22.99,-1),(23,1),(36.99,1),(37,-1),(42.99,-1),(43,2)):
            self.assertEqual(block_id({"segment":1,"source_time_sec":60+offset},intervals),b)

    def test_pair_distance_is_reflection_invariant(self):
        rng = np.random.default_rng(1)
        a,b,c,d = [rng.normal(size=(3,10)).astype("float32") for _ in range(4)]
        np.testing.assert_allclose(paired_distances(a,b,c,d),paired_distances(b,a,c,d))
        np.testing.assert_allclose(paired_distances(a,b,c,d),paired_distances(a,b,d,c))

    def test_uncertain_reference_rejects_even_probability_one(self):
        ev = {"candidate_label":"standing","confidence":1.0,"quality_reasons":[],
              "support_distance":.3,"support_radius":1.,"uncertain_ratio":.4,
              "nearest_known_disagrees":False}
        policy = {"threshold":.9,"support_gate":True,"radius_multiplier":1,"uncertain_ratio":1}
        p = decide(ev,policy)
        self.assertEqual(p["label"],"unknown")
        self.assertIn("near_human_uncertain_pose",p["rejection_reasons"])

    def test_far_away_reference_rejects(self):
        ev = {"candidate_label":"sitting","confidence":.99,"quality_reasons":[],
              "support_distance":2.,"support_radius":.4,"uncertain_ratio":10,
              "nearest_known_disagrees":False}
        p = decide(ev,{"threshold":.8,"support_gate":True,"radius_multiplier":1,"uncertain_ratio":1})
        self.assertIn("outside_training_support",p["rejection_reasons"])

    def test_class_evidence_conflict_rejects(self):
        ev = {"candidate_label":"standing","confidence":.999,"quality_reasons":[],
              "support_distance":.1,"support_radius":1,"uncertain_ratio":10,
              "nearest_known_disagrees":True}
        self.assertEqual(decide(ev,{"threshold":.8,"support_gate":True,"radius_multiplier":1,"uncertain_ratio":1})["label"],"unknown")

    def test_no_supported_threshold_fails_closed(self):
        rows = [{"label":"standing"}]*2
        ev = [{"candidate_label":"standing","confidence":1.,"quality_reasons":[]}]
        policy,report = select_policy(rows,ev*2,False)
        self.assertGreater(policy["threshold"],1)
        self.assertFalse(report["chosen"]["feasible"])

    def test_ignored_labels_do_not_change_selection(self):
        rows = [{"label":"sitting"}]*6+[{"label":"standing"}]*6+[{"label":"uncertain"}]
        ev = [{"candidate_label":r["label"] if r["label"] != "uncertain" else "sitting",
               "confidence":.9 if r["label"] != "uncertain" else .55,"quality_reasons":[]} for r in rows]
        a,_ = select_policy(rows,ev,False)
        b,_ = select_policy(rows+[{"label":"ignore"}],ev+[{"candidate_label":"standing","confidence":1.,"quality_reasons":[]}],False)
        self.assertEqual(a,b)
        self.assertGreaterEqual(a["threshold"],.65)

    def test_insufficient_joints_reject_regardless_of_score(self):
        ev = {"candidate_label":"standing","confidence":1.,"quality_reasons":["missing_or_degenerate_torso"]}
        self.assertEqual(decide(ev,.7)["label"],"unknown")

    def test_export_roundtrip_and_reference_not_three_class_softmax(self):
        rows = []
        for i in range(10):
            label = "sitting" if i%2 == 0 else "standing"
            rows.append({"label":label,"keypoints":pose(label=="sitting"),
                         "output_frame":i,"person_id":1,"source_time_sec":i*4.})
        rows.append({"label":"uncertain","keypoints":pose(),"output_frame":100,"person_id":1,"source_time_sec":45.})
        arrays,info = fit_model(rows,"linear")
        self.assertEqual(arrays["b0"].shape,(2,))
        self.assertEqual(info["binary_training_n"],10)
        self.assertIn(2,arrays["reference_labels"])
        raw = raw_evidence(arrays,info,rows)
        evidence = evidence_at_temperature(raw,1.)
        policy = {"threshold":.8,"support_gate":True,"radius_multiplier":1.5,"uncertain_ratio":1.}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"model"
            export_model(directory,arrays,info,1.,policy,{"dataset_id":"test","sources":{"hard_labels":{"sha256":"test"}}})
            classifier = PostureClassifier(directory)
            result = classifier.predict_many([r["keypoints"] for r in rows])
            json.dumps(result,allow_nan=False)
            for e,r in zip(evidence,result):
                self.assertEqual(decide(e,policy)["label"],r["label"])
                np.testing.assert_allclose(list(e["probabilities"].values()),list(r["probabilities"].values()),atol=1e-6)
            self.assertEqual(classifier.predict_many([]),[])


if __name__ == "__main__":
    unittest.main()
