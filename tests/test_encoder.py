import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lev.dataset import bundle, prepare
from lev.encoder import config, examples, execute, fingerprint, plan, state_text
from lev.encoder_train import order
from lev.io import jsonl, read, write
from lev.report import summarize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/encoder-modernbert-base.toml"


def record(label: object = True, qtype: str = "noul"):
    return {
        "state": {"passage": "Michael was one of the greatest drivers."},
        "questions": {
            "q0": {
                "type": qtype,
                "instructions": "Judge the passage.",
                "label": label,
                "src": "grammar",
            }
        },
        "_meta": {
            "id": "r1",
            "group_id": "g1",
            "source": "lev-en",
            "variant": "clean",
            "split": "train",
            "case_id": "c1",
            "scope": "partial",
        },
    }


def args(**overrides):
    base = {
        "command": "encode",
        "config": CONFIG,
        "dataset": None,
        "out": None,
        "device": "cuda",
        "checkpoint": None,
        "split": "development",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def frozen(temp, name="demo"):
    """A frozen example bundle: four partitions, no model needed."""
    out = Path(temp) / name
    prepare(
        ROOT / "data/examples/format-only.jsonl",
        ROOT / "rubrics/en-v1.json",
        out,
        examples=True,
    )
    return out


class EncoderConfigTests(unittest.TestCase):
    def test_shipped_configs_pin_the_backbone(self):
        for name in ("encoder-modernbert-base.toml", "encoder-modernbert-large.toml"):
            cfg = config(ROOT / "configs" / name)
            self.assertRegex(cfg["backbone_revision"], r"^[a-f0-9]{40}$")
            self.assertLessEqual(cfg["max_len"], 8192)

    def test_unpinned_backbone_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cfg.toml"
            path.write_text(
                CONFIG.read_text().replace(
                    "8949b909ec900327062f0ebf497f51aef5e6f0c8", "main"
                )
            )
            with self.assertRaisesRegex(ValueError, "full commit SHA"):
                config(path)

    def test_unknown_keys_and_bad_values_are_refused(self):
        original = CONFIG.read_text()
        cases = (
            (original + "\nstray = 1\n", "unknown fields"),
            (original.replace('dtype = "fp32"', 'dtype = "fp8"'), "dtype must be"),
            (original.replace("max_len = 3072", "max_len = 99"), "max_len"),
            (original.replace("warmup = 0.06", "warmup = 1.5"), "warmup"),
            (original.replace("epochs = 3", "epochs = 0"), "invalid training integer"),
            (
                original.replace("max_tokens = 8192", "max_tokens = 0"),
                "invalid training integer",
            ),
        )
        for text, message in cases:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "cfg.toml"
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    config(path)


class EncoderExampleTests(unittest.TestCase):
    def test_state_text_is_deterministic(self):
        self.assertEqual(state_text({"b": "2", "a": "1"}), "a: 1\nb: 2")
        self.assertEqual(state_text("plain"), "plain")
        with self.assertRaisesRegex(ValueError, "string or an object"):
            state_text(["x"])

    def test_pairs_carry_state_and_question_only(self):
        item = examples([record()])[0]
        self.assertEqual(
            item["text"],
            "passage: Michael was one of the greatest drivers.\n\nJudge the passage.",
        )
        self.assertEqual(item["label"], 1)
        self.assertNotIn("label", item["text"])
        self.assertEqual(
            (item["group"], item["source"], item["variant"]), ("g1", "lev-en", "clean")
        )

    def test_false_labels_project_to_zero(self):
        self.assertEqual(examples([record(label=False)])[0]["label"], 0)

    def test_unresolved_and_non_boolean_labels_are_refused(self):
        for bad in (None, 1, "true"):
            with self.assertRaisesRegex(ValueError, "JSON boolean"):
                examples([record(label=bad)])

    def test_only_noul_questions_are_supported(self):
        with self.assertRaisesRegex(ValueError, "only noul"):
            examples([record(qtype="choice")])


class EncoderPlanTests(unittest.TestCase):
    def test_training_refuses_an_example_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "example-only"):
                plan(args(dataset=frozen(temp), out=Path(temp) / "run"))

    def test_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = frozen(temp)
            taken = Path(temp) / "taken"
            taken.mkdir()
            with self.assertRaisesRegex(ValueError, "new output directory"):
                plan(args(dataset=dataset, out=taken))

    def test_every_training_value_reaches_the_command(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = frozen(temp)
            # A frozen example bundle may not train; the command shape is what is under test.
            with patch(
                "lev.encoder.bundle",
                return_value=([], {"example_only": False, "research_only": False}),
            ):
                job = plan(args(dataset=dataset, out=Path(temp) / "run"))
            cfg = config(CONFIG)
            self.assertEqual(job["engine"], "encoder")
            self.assertEqual(job["operation"], "train")
            self.assertEqual(job["split"], "train")
            self.assertEqual(job["command"][1:3], ["-m", "lev.encoder_train"])
            self.assertEqual(job["command"][3], "train")
            self.assertEqual(
                job["command"][job["command"].index("--max-len") + 1],
                str(cfg["max_len"]),
            )
            self.assertEqual(
                job["command"][job["command"].index("--revision") + 1],
                cfg["backbone_revision"],
            )
            for key, value in cfg["training"].items():
                flag = "--" + key.replace("_", "-")
                self.assertIn(flag, job["command"])
                self.assertEqual(
                    job["command"][job["command"].index(flag) + 1], str(value)
                )

    def test_benchmark_binds_checkpoint_content_not_its_path(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint"
            checkpoint.mkdir()
            write(checkpoint / "lev-encoder.json", {"engine": "encoder"})
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            job = plan(
                args(
                    command="encode-benchmark",
                    dataset=frozen(temp),
                    out=Path(temp) / "run",
                    checkpoint=checkpoint,
                )
            )
            self.assertEqual(job["operation"], "benchmark")
            self.assertEqual(job["split"], "development")
            self.assertEqual(
                set(job["checkpoint"]["files"]),
                {"lev-encoder.json", "model.safetensors"},
            )
            self.assertIn("predict", job["command"])

    def test_a_directory_without_metadata_is_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "not an encoder checkpoint"):
                fingerprint(temp)

    def test_process_failure_is_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "failed"
            job = {
                "out": str(out),
                "operation": "benchmark",
                "config": {"training": {"seed": 42}},
                "command": [
                    sys.executable,
                    "-c",
                    'print("cause preserved"); raise SystemExit(7)',
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "failed"):
                execute(job)
            status = read(out / "status.json")
            self.assertEqual(status["returncode"], 7)
            self.assertEqual(status["status"], "failed")
            self.assertIn("cause preserved", (out / "run.log").read_text())


class EncoderReportContractTests(unittest.TestCase):
    """The point of the encoder path: `lev report` scores it with unchanged code."""

    def test_encoder_rows_satisfy_the_kev_prediction_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = frozen(temp)
            records, _ = bundle(dataset)
            rows = [
                {
                    "id": item["id"],
                    "question": item["question"],
                    "keys": ["false", "true"],
                    "type": "noul",
                    "label": item["label"],
                    "p": [0.25, 0.75],
                }
                for item in examples(jsonl(dataset / "test.jsonl"))
            ]
            split = [r for r in records if r["split"] == "test"]
            result = summarize(split, rows, 0.5)
            self.assertEqual(result["decisions"]["requested"], len(rows))
            self.assertEqual(result["decisions"]["missing"], 0)
            # p_true 0.75 at threshold 0.5 accepts every question
            self.assertEqual(result["decisions"]["completed"], len(rows))

    def test_a_missing_row_stays_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = frozen(temp)
            records, _ = bundle(dataset)
            items = examples(jsonl(dataset / "test.jsonl"))
            rows = [
                {
                    "id": item["id"],
                    "question": item["question"],
                    "keys": ["false", "true"],
                    "type": "noul",
                    "label": item["label"],
                    "p": [0.25, 0.75],
                }
                for item in items[1:]
            ]
            split = [r for r in records if r["split"] == "test"]
            result = summarize(split, rows, 0.5)
            self.assertEqual(result["decisions"]["requested"], len(items))
            self.assertEqual(result["decisions"]["missing"], 1)
            self.assertEqual(
                result["decisions"]["accuracy_all_requested"]["denominator"], len(items)
            )

    def test_a_label_that_drifted_from_the_frozen_data_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = frozen(temp)
            records, _ = bundle(dataset)
            items = examples(jsonl(dataset / "test.jsonl"))
            rows = [
                {
                    "id": item["id"],
                    "question": item["question"],
                    "keys": ["false", "true"],
                    "type": "noul",
                    "label": 1 - item["label"],
                    "p": [0.25, 0.75],
                }
                for item in items
            ]
            split = [r for r in records if r["split"] == "test"]
            with self.assertRaisesRegex(ValueError, "differs from frozen data"):
                summarize(split, rows, 0.5)


class EncoderCliTests(unittest.TestCase):
    def test_dry_run_reports_the_plan_without_touching_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = frozen(temp)
            checkpoint = Path(temp) / "checkpoint"
            checkpoint.mkdir()
            write(
                checkpoint / "lev-encoder.json",
                {"engine": "encoder", "max_len": 3072},
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lev",
                    "encode-benchmark",
                    "--dataset",
                    str(dataset),
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(Path(temp) / "run"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            shown = json.loads(result.stdout)
            self.assertTrue(shown["dry_run"])
            self.assertEqual(shown["engine"], "encoder")
            self.assertFalse((Path(temp) / "run").exists())

    def test_a_missing_checkpoint_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lev",
                    "encode-benchmark",
                    "--dataset",
                    str(frozen(temp)),
                    "--out",
                    str(Path(temp) / "run"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("checkpoint", result.stderr)


class EncoderBatchOrderTests(unittest.TestCase):
    """Padding to the pack's worst case is what pushed fp32 past 16 GiB."""

    def test_a_long_record_never_drags_short_ones_into_its_batch(self):
        lengths = [20] * 8 + [2400]
        groups = order(lengths, batch=4, budget=8192, seed=42)
        longest = max(groups, key=lambda g: max(lengths[i] for i in g))
        self.assertEqual([lengths[i] for i in longest], [2400])
        for group in groups:
            self.assertLessEqual(max(lengths[i] for i in group) * len(group), 8192)

    def test_every_record_lands_in_exactly_one_batch(self):
        lengths = [17, 2400, 33, 45, 200, 1, 900]
        groups = order(lengths, batch=3, budget=2048, seed=7)
        self.assertEqual(
            sorted(i for g in groups for i in g), list(range(len(lengths)))
        )

    def test_the_budget_caps_batches_independently_of_batch(self):
        self.assertEqual(len(order([1000] * 8, batch=8, budget=3000, seed=1)), 3)


if __name__ == "__main__":
    unittest.main()
