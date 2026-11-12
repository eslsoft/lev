import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lev.dataset import prepare
from lev.io import read, write
from lev.workflow import check_calibration, config, execute, model_identity, plan

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_model_must_be_pinned(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            model_identity("jaredpalmer/kev-4b")
        self.assertIn(
            "hub", model_identity(config(ROOT / "configs/lev-4b.toml")["init_from"])
        )

    def test_calibration_is_bound_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cal.json"
            write(path, {"checkpoint": {"hub": "a"}, "temperature": 1.2})
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                check_calibration(path, {"hub": "b"})

    def test_calibration_follows_checkpoint_content_after_download(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cal.json"
            write(
                path,
                {
                    "checkpoint": {
                        "path": "/remote/checkpoint",
                        "files": {"head.pt": "original"},
                    },
                    "temperature": 2.0,
                },
            )
            self.assertEqual(
                check_calibration(
                    path,
                    {"path": "/local/checkpoint", "files": {"head.pt": "original"}},
                )["temperature"],
                2.0,
            )
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                check_calibration(
                    path, {"path": "/local/checkpoint", "files": {"head.pt": "changed"}}
                )

    def test_process_failure_is_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "failed"
            job = {
                "out": str(out),
                "operation": "benchmark",
                "env": {},
                "command": [
                    sys.executable,
                    "-c",
                    'print("cause preserved"); raise SystemExit(7)',
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "failed"):
                execute(job)
            self.assertEqual(read(out / "status.json")["returncode"], 7)
            self.assertEqual(read(out / "status.json")["status"], "failed")
            self.assertIn("cause preserved", (out / "run.log").read_text())

    def test_cli_data_flow_requires_no_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lev",
                    "prepare",
                    "--data",
                    str(ROOT / "data/examples/format-only.jsonl"),
                    "--rubric",
                    str(ROOT / "rubrics/en-v1.json"),
                    "--examples",
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            self.assertTrue(read(out / "manifest.json")["example_only"])

    def test_training_refuses_example_bundle_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            prepare(
                ROOT / "data/examples/format-only.jsonl",
                ROOT / "rubrics/en-v1.json",
                out,
                examples=True,
            )
            args = argparse.Namespace(
                command="train",
                config=ROOT / "configs/lev-4b.toml",
                kev=temp,
                out=Path(temp) / "run",
                dataset=out,
                device="cuda",
            )
            with (
                patch("lev.workflow.checkout", return_value={"path": temp}),
                patch("lev.workflow.contract"),
            ):
                with self.assertRaisesRegex(ValueError, "example-only"):
                    plan(args)
