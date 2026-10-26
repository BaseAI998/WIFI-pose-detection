"""Portable input paths and historical hash-preserving relocation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import sys
import cv2
import numpy as np

from posture_mlp_v2.data import resolve_recorded_path, resolve_sources, resolve_input_directory, load_data, sha256
from posture_mlp_v2.cli.render import resolve_output_directory


class PairedDirectoryTests(unittest.TestCase):
    def test_default_output_uses_dataset_name(self):
        with tempfile.TemporaryDirectory() as temp:
            module = Path(temp)/"posture_mlp_v2"
            source = module/"inputs"/"meeting_02"
            with patch("posture_mlp_v2.cli.render.find_project_root",return_value=module):
                self.assertEqual(resolve_output_directory(source),module/"outputs"/"meeting_02")
            self.assertFalse((module/"outputs").exists())

    def test_existing_result_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/"outputs"/"example"
            output.mkdir(parents=True)
            result = output/"keep.txt"
            result.write_text("unchanged")
            with patch("posture_mlp_v2.cli.render.find_project_root",return_value=root):
                with self.assertRaises(FileExistsError):
                    resolve_output_directory(root/"inputs"/"example")
            self.assertEqual(result.read_text(),"unchanged")

    def test_input_output_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"inputs"/"example"
            for output in (source,source/"nested",source.parent):
                with self.assertRaises(ValueError):
                    resolve_output_directory(source,output)

    def test_legacy_mode_requires_explicit_output(self):
        with self.assertRaises(ValueError):
            resolve_output_directory(None)

    def test_old_input_path_resolves_to_new_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            target=root/"posture_mlp_v2"/"inputs"/"example"/"video.mp4"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"original")
            with patch("posture_mlp_v2.data.PROJECT_ROOT",root/"posture_mlp_v2"):
                self.assertEqual(resolve_recorded_path("Z:/project/input/example/video.mp4"),target)

    def test_output_does_not_resolve_to_same_named_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            module=root/"posture_mlp_v2"
            input_file=module/"inputs"/"example"/"poses.jsonl"
            output_file=module/"outputs"/"example"/"poses.jsonl"
            for path,content in ((input_file,b"input"),(output_file,b"output")):
                path.parent.mkdir(parents=True)
                path.write_bytes(content)
            archive=module/"runs"/"hard_v2_complete"/"provenance"/"input_example_legacy"
            archive.mkdir(parents=True)
            (archive/"relocations.json").write_text(json.dumps({
                "relocations":{"poses.jsonl":"inputs/example/poses.jsonl"}}))
            with patch("posture_mlp_v2.data.PROJECT_ROOT",root/"posture_mlp_v2"):
                for folder in ("hard_v2","example"):
                    self.assertEqual(resolve_recorded_path(
                        f"Z:/project/posture_mlp_v2/outputs/{folder}/poses.jsonl"),output_file)
                self.assertEqual(resolve_recorded_path("poses.jsonl"),input_file)
                with self.assertRaises(FileNotFoundError):
                    resolve_recorded_path("Z:/unrelated/poses.jsonl")

    def test_old_path_cannot_escape_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("posture_mlp_v2.data.PROJECT_ROOT",Path(temp)):
                with self.assertRaises(ValueError):
                    resolve_recorded_path("Z:/project/input/../../outside.txt")


class InputLayoutTests(unittest.TestCase):
    def test_relative_manifest_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("video.mp4", "keypoints.jsonl", "original.jsonl"):
                (root/name).write_bytes(b"test")
            manifest = root/"input.json"
            manifest.write_text(json.dumps({
                "input":"video.mp4", "outputs":{"keypoints_jsonl":"keypoints.jsonl"},
                "reuses_keypoints":True, "cached_keypoints":"original.jsonl",
            }),encoding="utf-8")
            _, keypoints, video, ancestor = resolve_sources(manifest)
            self.assertEqual(keypoints,root/"keypoints.jsonl")
            self.assertEqual(video,root/"video.mp4")
            self.assertEqual(ancestor,root/"original.jsonl")

    def test_historical_absolute_path_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            example = root/"posture_mlp_v2"/"inputs"/"example"
            example.mkdir(parents=True)
            (example/"kept.jsonl").write_bytes(b"original bytes")
            archive = root/"posture_mlp_v2"/"runs"/"hard_v2_complete"/"provenance"/"input_example_legacy"
            archive.mkdir(parents=True)
            (archive/"relocations.json").write_text(json.dumps({
                "relocations":{"old/keypoints.jsonl":"inputs/example/kept.jsonl"}
            }),encoding="utf-8")
            with patch("posture_mlp_v2.data.PROJECT_ROOT",root/"posture_mlp_v2"):
                resolved=resolve_recorded_path("Z:/old-machine/project/old/keypoints.jsonl")
                self.assertEqual(resolved,example/"kept.jsonl")

    def test_relocation_cannot_escape_example(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            example=root/"posture_mlp_v2"/"inputs"/"example"
            example.mkdir(parents=True)
            archive = root/"posture_mlp_v2"/"runs"/"hard_v2_complete"/"provenance"/"input_example_legacy"
            archive.mkdir(parents=True)
            (archive/"relocations.json").write_text(json.dumps({
                "relocations":{"old/keypoints.jsonl":"../outside.jsonl"}
            }),encoding="utf-8")
            with patch("posture_mlp_v2.data.PROJECT_ROOT",root/"posture_mlp_v2"):
                with self.assertRaises(ValueError):
                    resolve_recorded_path("Z:/old/keypoints.jsonl")

    def test_missing_source_is_not_silently_replaced(self):
        with self.assertRaises(FileNotFoundError):
            resolve_recorded_path("Z:/not-a-real-dataset/unmatched_file.jsonl")


class TwoFileInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"input"
        self.root.mkdir()
        self.video = self.root/"sample.avi"
        self.kp = self.root/"sample_keypoints.jsonl"
        writer = cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*"MJPG"),10.,(64,64))
        if not writer.isOpened():
            self.skipTest("MJPG unavailable")
        for _ in range(30):
            writer.write(np.zeros((64,64,3),np.uint8))
        writer.release()
        self.rows = [{"output_frame":i,"source_frame":source,"source_time_sec":source/10,
                      "source_time":f"{source/10:.3f}","segment":1 if i<2 else 2,"people":[]}
                     for i,source in enumerate((10,11,20,21))]
        self.save()

    def save(self):
        self.kp.write_text("\n".join(json.dumps(r) for r in self.rows)+"\n",encoding="utf-8")

    def test_exactly_two_files_no_archive_dependency(self):
        with patch("posture_mlp_v2.data.resolve_recorded_path",side_effect=AssertionError("Legacy path lookup forbidden")):
            meta,kp,video,ancestor = resolve_input_directory(self.root)
        self.assertEqual(kp,self.kp)
        self.assertEqual(video,self.video)
        self.assertEqual(ancestor,kp)
        self.assertEqual(meta["processed_frames"],4)
        self.assertEqual(meta["detected_people_instances"],0)
        np.testing.assert_allclose(meta["intervals_sec"],[[1.,1.2],[2.,2.2]])
        self.assertEqual(len(list(self.root.iterdir())),2)

    def test_multiple_caches_fail_instead_of_guessing(self):
        (self.root/"other.jsonl").write_text("",encoding="utf-8")
        with self.assertRaises(ValueError):
            resolve_input_directory(self.root)

    def test_labels_subdirectory_does_not_count_as_keypoint_cache(self):
        labels = self.root/"labels"
        labels.mkdir()
        (labels/"baseline.jsonl").write_text("{}\n")
        (labels/"hard_v1.jsonl").write_text("{}\n")
        meta,kp,*_ = resolve_input_directory(self.root)
        self.assertEqual(kp,self.kp)
        self.assertEqual(meta["processed_frames"],4)

    def test_multiple_videos_fail_instead_of_guessing(self):
        (self.root/"other.mp4").write_bytes(b"")
        with self.assertRaises(ValueError):
            resolve_input_directory(self.root)

    def test_empty_cache_fails(self):
        self.kp.write_text("",encoding="utf-8")
        with self.assertRaises(ValueError):
            resolve_input_directory(self.root)

    def test_timestamp_mismatch_fails(self):
        self.rows[0]["source_time_sec"]=12.
        self.save()
        with self.assertRaises(ValueError):
            resolve_input_directory(self.root)

    def test_out_of_range_frame_fails(self):
        self.rows[-1]["source_frame"]=100
        self.save()
        with self.assertRaises(ValueError):
            resolve_input_directory(self.root)

    def test_repeated_segment_fails(self):
        self.rows[-1]["segment"]=1
        self.save()
        with self.assertRaises(ValueError):
            resolve_input_directory(self.root)

    def test_new_cache_without_posture_works(self):
        person = {"person_id":1,"bbox_xyxy":[1,1,50,60],"detection_confidence":.9,
                  "keypoints":[{"index":i,"x":20.,"y":20.+i,"confidence":.1} for i in range(17)]}
        self.rows[0]["people"]=[person]
        self.save()
        meta,*_=resolve_input_directory(self.root)
        self.assertEqual(meta["detected_people_instances"],1)
        self.assertNotIn("posture",person)

    def test_render_cli_two_files_smoke(self):
        project = Path(__file__).resolve().parents[2]
        model = project/"posture_mlp_v2"/"runs"/"hard_v2_complete"/"model"
        if not model.is_dir():
            self.skipTest("Bundled model unavailable")
        self.test_new_cache_without_posture_works()
        before = sha256(self.kp)
        output=Path(self.temp.name)/"output"
        result=subprocess.run([sys.executable,"-B","-m","posture_mlp_v2.cli.render",
                               "--input-dir",str(self.root),"--model-dir",str(model),
                               "--output-dir",str(output)],
                              cwd=project,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        meta=json.loads((output/"sample_run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["processed_frames"],4)
        self.assertEqual(meta["detected_people_instances"],1)
        cap=cv2.VideoCapture(meta["outputs"]["annotated_video"])
        decoded=0
        while cap.read()[0]:
            decoded+=1
        cap.release()
        self.assertEqual(decoded,4)
        self.assertEqual(sha256(self.kp),before)

    def test_training_input_modes_are_exclusive(self):
        with self.assertRaises(ValueError):
            load_data(None,"missing","missing")
        with self.assertRaises(ValueError):
            load_data("manifest","missing","missing",input_dir=self.root)

    def test_render_without_output_arg_creates_matching_folder(self):
        from posture_mlp_v2.cli.render import main
        from contextlib import redirect_stdout
        import io
        project=Path(__file__).resolve().parents[2]
        model=project/"posture_mlp_v2"/"runs"/"hard_v2_complete"/"model"
        if not model.is_dir():
            self.skipTest("Bundled model unavailable")
        self.test_new_cache_without_posture_works()
        module=Path(self.temp.name)/"module"
        argv=["render","--input-dir",str(self.root),"--model-dir",str(model),"--skip-video"]
        with patch("posture_mlp_v2.cli.render.find_project_root",return_value=module),patch.object(sys,"argv",argv),redirect_stdout(io.StringIO()):
            main()
        output=module/"outputs"/self.root.name
        metadata=json.loads((output/"sample_run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["dataset_name"],self.root.name)
        self.assertEqual(Path(metadata["input_directory"]),self.root)
        self.assertEqual(Path(metadata["output_directory"]),output)
        self.assertEqual(metadata["processed_frames"],4)
        self.assertFalse((output/"qa").exists())

    def test_render_config_from_unrelated_working_directory(self):
        project=Path(__file__).resolve().parents[1]
        model=project/"runs"/"hard_v2_complete"/"model"
        self.test_new_cache_without_posture_works()
        config=Path(self.temp.name)/"render.yaml"
        import yaml
        config.write_text(yaml.safe_dump({"project_root":str(Path(self.temp.name)),
                                          "input_dir":"input", "model_dir":str(model),
                                          "skip_video":True}),encoding="utf-8")
        unrelated=Path(self.temp.name)/"unrelated"
        unrelated.mkdir()
        result=subprocess.run([sys.executable,"-B","-m","posture_mlp_v2.cli.render",
                               "--config",str(config)],cwd=unrelated,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        output=Path(self.temp.name)/"outputs"/"input"
        metadata=json.loads((output/"sample_run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["processed_frames"],4)
        self.assertFalse((output/"qa").exists())


if __name__=="__main__":
    unittest.main()
