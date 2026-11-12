"""Execute the pinned upstream's real JSON reader/exporter without importing a GPU stack.

This proves serialization compatibility, not training or model execution.
Run with LEV_KEV_PATH pointing at the existing checkout.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lev.dataset import prepare
from lev.workflow import config, plan
from lev.upstream import checkout, contract

ROOT = Path(__file__).resolve().parents[1]
KEV = os.environ.get("LEV_KEV_PATH")


@unittest.skipUnless(KEV, "set LEV_KEV_PATH for local Kev source contract checks")
class KevContractTests(unittest.TestCase):
    def test_real_loader_and_request_projection(self):
        cfg = config(ROOT / "configs/lev-4b.toml")
        checkout(KEV, cfg["kev_commit"])
        contract(KEV)
        tree = ast.parse((Path(KEV) / "kev/data.py").read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("load_records", "api_request")
        ]
        self.assertEqual(len(nodes), 2)
        scope = {"Path": Path, "json": json, "hashlib": hashlib}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "kev/data.py", "exec"),
            scope,
        )
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            prepare(
                ROOT / "data/examples/format-only.jsonl",
                ROOT / "rubrics/en-v1.json",
                out,
                examples=True,
            )
            rows = scope["load_records"](out / "train.jsonl")
            self.assertEqual(len(rows), 2)
            request = scope["api_request"](rows[0])
            self.assertEqual(set(request), {"state", "questions"})
            self.assertEqual(
                set(request["questions"]["number"]), {"type", "instructions"}
            )
            self.assertNotIn("label", json.dumps(request))

    def test_benchmark_plan_uses_explicit_training_context(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            prepare(
                ROOT / "data/examples/format-only.jsonl",
                ROOT / "rubrics/en-v1.json",
                out,
                examples=True,
            )
            cfg = config(ROOT / "configs/lev-4b.toml")
            args = argparse.Namespace(
                command="benchmark",
                kev=KEV,
                config=ROOT / "configs/lev-4b.toml",
                dataset=out,
                out=Path(temp) / "read",
                checkpoint=cfg["init_from"],
                split="development",
                device="cpu",
                calibration=None,
            )
            job = plan(args)
            self.assertIn("lev.bridge", job["command"])
            self.assertEqual(
                job["command"][job["command"].index("--max-state") + 1], "7552"
            )
            self.assertFalse((Path(temp) / "read").exists())

    def test_real_training_parser_accepts_lev_recipe(self):
        # Run only the actual argument parser, never main()/model loading.
        tree = ast.parse((Path(KEV) / "kev/train.py").read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "parse_args"
        )

        class FullFT:
            @staticmethod
            def snapshot_fractions(value):
                return []

            @staticmethod
            def too_many_snapshots(*args):
                return None

        scope = {
            "argparse": argparse,
            "Path": Path,
            "os": os,
            "math": math,
            "DEVICES": ("cpu", "mps", "cuda"),
            "DEVICE_HELP": "",
            "MAX_STATE": 384,
            "MAX_TRAIN_STATE": 65536,
            "full_ft": FullFT,
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "kev/train.py", "exec"),
            scope,
        )
        cfg = config(ROOT / "configs/lev-4b.toml")
        with tempfile.TemporaryDirectory() as temp:
            argv = [
                "train",
                "--base",
                cfg["base"],
                "--base_revision",
                cfg["base_revision"],
                "--init_from",
                cfg["init_from"],
                "--data",
                str(Path(temp) / "train.jsonl"),
                "--device",
                "cuda",
                "--dtype",
                "bf16",
                "--out",
                str(Path(temp) / "new-run"),
            ]
            for key, value in cfg["training"].items():
                argv += ["--" + key, str(value)]
            with patch.object(sys, "argv", argv):
                parsed = scope["parse_args"]()
            self.assertEqual(parsed.max_state, 7552)
            self.assertEqual(parsed.init_from, cfg["init_from"])
