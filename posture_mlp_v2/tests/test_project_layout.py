"""Installed-package, config precedence and relocated provenance regression tests."""
import argparse
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from posture_mlp_v2.cli._config import parse_config
from posture_mlp_v2.cli.render import resolve_output_directory
from posture_mlp_v2.data import resolve_recorded_path


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "configs" / "test.yaml"
        self.config.parent.mkdir()

    def parse(self, text, *extra):
        self.config.write_text(text, encoding="utf-8")
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--input-dir", type=Path)
        group.add_argument("--metadata", type=Path)
        parser.add_argument("--project-root", type=Path)
        parser.add_argument("--model-dir", type=Path)
        parser.add_argument("--seed", type=int, default=20261019)
        parser.add_argument("--skip-video", action=argparse.BooleanOptionalAction, default=False)
        return parse_config(parser, required=("model_dir",), source_required=True,
                            argv=["--config", str(self.config), *extra])

    def test_config_relative_paths(self):
        args = self.parse("project_root: ..\ninput_dir: inputs/demo\nmodel_dir: runs/model\nseed: 123\n")
        self.assertEqual(args.input_dir, self.root / "inputs" / "demo")
        self.assertEqual(args.model_dir, self.root / "runs" / "model")
        self.assertEqual(args.seed, 123)

    def test_cli_precedence_and_boolean_override(self):
        args = self.parse("input_dir: demo\nmodel_dir: model\nseed: 123\nskip_video: true\n",
                          "--seed", "456", "--model-dir", "cli-model", "--no-skip-video")
        self.assertEqual(args.seed, 456)
        self.assertEqual(args.model_dir, Path("cli-model"))
        self.assertFalse(args.skip_video)

    def test_cli_project_root_rebases_config_paths(self):
        other = self.root / "other-project"
        args = self.parse("project_root: ..\ninput_dir: inputs/demo\nmodel_dir: model\n",
                          "--project-root", str(other))
        self.assertEqual(args.input_dir, other / "inputs" / "demo")

    def test_cli_can_switch_source_mode(self):
        args = self.parse("input_dir: demo\nmodel_dir: model\n", "--metadata", "old.json")
        self.assertIsNone(args.input_dir)
        self.assertEqual(args.metadata, Path("old.json"))
        args = self.parse("metadata: old.json\nmodel_dir: model\n", "--input-dir=demo")
        self.assertIsNone(args.metadata)

    def test_invalid_config_is_rejected(self):
        for text in (
            "- not-a-mapping\n", "unknown: 1\n", "model_dir: [bad]\n",
            "skip_video: 'false'\n", "seed: true\n", "seed: 1.5\n",
            "input_dir: demo\n", "model_dir: model\n",
            "input_dir: demo\nmetadata: old\nmodel_dir: model\n", "input_dir: [\n",
        ):
            with self.subTest(text=text), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(text)


class InstalledLayoutTests(unittest.TestCase):
    def test_source_package_can_be_imported_outside_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([
                sys.executable, "-B", "-c",
                "from posture_mlp_v2.model import PostureClassifier; import sys; "
                "assert 'torch' not in sys.modules; assert 'cv2' not in sys.modules; "
                "assert 'yaml' not in sys.modules; print(PostureClassifier.__module__)",
            ], cwd=temp, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("posture_mlp_v2.model", result.stdout)

    def test_cli_entrypoints_import(self):
        for command in ("train", "render", "verify", "review", "report"):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, "-B", "-m", f"posture_mlp_v2.cli.{command}", "--help"],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_standard_input_pairs_without_source_location(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "posture_mlp_v2.cli.render.find_project_root", side_effect=AssertionError("No lookup needed")
        ):
            root = Path(temp)
            self.assertEqual(resolve_output_directory(root / "inputs" / "demo"), root / "outputs" / "demo")

    def test_explicit_root_for_external_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(resolve_output_directory(root / "external" / "demo", project_root=root / "project"),
                             root / "project" / "outputs" / "demo")

    def test_old_archive_path_maps_to_run_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "runs" / "hard_v2_complete" / "provenance" / "input_example_legacy" / "source.json"
            target.parent.mkdir(parents=True)
            target.write_text("{}")
            self.assertEqual(resolve_recorded_path(
                "Z:/old/posture_mlp_v2/archive/input_example_legacy/source.json", project_root=root), target)

    def test_project_is_portable_without_labeler_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "inputs" / "example" / "labels" / "baseline.jsonl"
            labels.parent.mkdir(parents=True)
            labels.write_text("original labels")
            provenance = root / "runs" / "hard_v2_complete" / "provenance" / "input_example_legacy"
            provenance.mkdir(parents=True)
            (provenance / "relocations.json").write_text(json.dumps({"relocations": {
                "posture_labeler/exports/baseline.jsonl": "inputs/example/labels/baseline.jsonl"}}))
            self.assertEqual(resolve_recorded_path(
                "Z:/old/posture_labeler/exports/baseline.jsonl", project_root=root), labels)
            external = root / "live" / "posture_labeler" / "exports" / "baseline.jsonl"
            external.parent.mkdir(parents=True)
            external.write_text("subsequently edited labels")
            self.assertEqual(resolve_recorded_path(external, project_root=root), labels)

    def test_training_config_reaches_validation_without_starting_training(self):
        from posture_mlp_v2.cli.train import main
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp, patch.object(sys, "argv", [
            "train", "--config", str(project / "configs" / "train.yaml"),
            "--output-dir", str(Path(temp) / "new-run"),
        ]), patch("posture_mlp_v2.cli.train.load_data", side_effect=RuntimeError("validation sentinel")) as load:
            with self.assertRaisesRegex(RuntimeError, "validation sentinel"):
                main()
            self.assertEqual(load.call_args.args[1], project / "inputs" / "example" / "labels" / "baseline.jsonl")
            self.assertEqual(load.call_args.kwargs["input_dir"], project / "inputs" / "example")
            self.assertFalse((Path(temp) / "new-run").exists())


if __name__ == "__main__":
    unittest.main()
