"""Exercise real shard files and aggregation boundaries; no model execution claimed."""

from pathlib import Path
import tempfile
import unittest

from lev.io import digest, encoded, read, write
from lev.vocabulary_jobs import merge, shard, verify_shards


def lines(path, values):
    path.write_text("".join(encoded(v) + "\n" for v in values))


def fixture(root):
    dataset = root / "dataset"
    dataset.mkdir()
    words = []
    records = []
    native = []
    for rank, word in enumerate(["alpha", "beta", "gamma"], 1):
        ids = [f"{word}-{n}" for n in range(2 if rank == 1 else 1)]
        words.append(
            dict(
                word=word,
                rank=rank,
                record_ids=ids,
                judgment_count=len(ids),
                exposure="unseen",
                coverage={},
                gaps=[],
            )
        )
        for rid in ids:
            records.append(
                dict(
                    id=rid,
                    state=dict(headword=word),
                    questions=dict(q=dict(label=True, kind="test")),
                )
            )
            native.append(
                dict(
                    _meta=dict(id=rid),
                    state=dict(headword=word),
                    questions=dict(
                        q=dict(type="noul", label=True, instructions="Test fixture")
                    ),
                )
            )
    lines(dataset / "words.jsonl", words)
    lines(dataset / "records.jsonl", records)
    lines(dataset / "test.jsonl", native)
    for name in ["rubric.json", "coverage.json", "source-manifest.json"]:
        write(dataset / name, {})
    write(
        dataset / "manifest.json",
        dict(
            format="lev-vocabulary-test-v1",
            test_only=True,
            unique_headwords=3,
            files={p.name: digest(p) for p in dataset.iterdir()},
        ),
    )
    return dataset


def run_fixture(root, shards, spec):
    run = root / f"run-{spec['index']}"
    run.mkdir()
    (run / "native").mkdir()
    data = read_lines(shards / spec["file"])
    lines(run / "test.jsonl", data)
    job = dict(
        shard=spec["index"],
        shards_sha256=digest(shards / "manifest.json"),
        checkpoint="/fixed",
        checkpoint_files={"head.pt": "fixed"},
        temperature=2.0,
        calibration_sha256="cal",
        selection_sha256="selected",
        max_state=7552,
        threshold=0.5,
        kev_commit="fixed",
    )
    write(run / "job.json", job)
    write(run / "modal-status.json", dict(status="succeeded"))
    write(
        run / "native/rows.json",
        [
            dict(
                id=r["_meta"]["id"],
                question="q",
                type="noul",
                keys=["false", "true"],
                label=1,
                p=[0.1, 0.9],
            )
            for r in data
        ],
    )
    write(
        run / "native/lev-context.json",
        dict(
            data_sha256=spec["sha256"],
            checkpoint="/fixed",
            inference_temperature=2.0,
            context=dict(max_state=7552, truncate=False),
            environment=dict(dtype="fp32"),
        ),
    )
    return run


def read_lines(path):
    from lev.io import jsonl

    return jsonl(path)


class VocabularyJobTests(unittest.TestCase):
    def test_every_word_and_request_is_assigned_once_and_scored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = fixture(root)
            shards = root / "shards"
            plan = shard(dataset, shards, 2)
            verify_shards(shards)
            self.assertEqual(plan["requests"], 4)
            self.assertEqual(plan["judgments"], 4)
            self.assertEqual(sum(s["words"] for s in plan["shards"]), 3)
            locations = {
                r["_meta"]["id"]: s["index"]
                for s in plan["shards"]
                for r in read_lines(shards / s["file"])
            }
            self.assertEqual(locations["alpha-0"], locations["alpha-1"])
            runs = [run_fixture(root, shards, s) for s in plan["shards"]]
            result = merge(dataset, shards, runs, root / "report.json")
            self.assertEqual(
                result["overall"]["words_all_scheduled_items_correct"]["numerator"], 3
            )

    def test_missing_duplicate_or_mixed_precision_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = fixture(root)
            shards = root / "shards"
            plan = shard(dataset, shards, 2)
            runs = [run_fixture(root, shards, s) for s in plan["shards"]]
            for values in [runs[:1], runs + [runs[0]]]:
                with self.assertRaises(ValueError):
                    merge(dataset, shards, values, root / "bad.json")
            path = runs[1] / "native/lev-context.json"
            value = read(path)
            value["environment"]["dtype"] = "bf16"
            path.write_text(encoded(value))
            with self.assertRaises(ValueError):
                merge(dataset, shards, runs, root / "bad.json")

    def test_a100_form_factors_preserved_but_other_environment_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = fixture(root)
            shards = root / "shards"
            plan = shard(dataset, shards, 2)
            runs = [run_fixture(root, shards, s) for s in plan["shards"]]
            names = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
            for run, name in zip(runs, names):
                path = run / "native/lev-context.json"
                value = read(path)
                value["environment"]["gpu"] = name
                path.write_text(encoded(value))
            result = merge(dataset, shards, runs, root / "report.json")
            self.assertEqual(len(result["observed_inference_environments"]), 2)
            self.assertEqual([s["inference_environment"]["gpu"] for s in result["sources"]], names)
            path = runs[1] / "native/lev-context.json"
            original = read(path)
            for key, changed in [("gpu", "NVIDIA H100"), ("dtype", "bf16"),
                                 ("packages", {"torch": "different"})]:
                value = {**original, "environment": {**original["environment"], key: changed}}
                path.write_text(encoded(value))
                with self.assertRaises(ValueError):
                    merge(dataset, shards, runs, root / "bad.json")

    def test_missing_prediction_does_not_shrink_word_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = fixture(root)
            shards = root / "shards"
            plan = shard(dataset, shards, 2)
            runs = [run_fixture(root, shards, s) for s in plan["shards"]]
            path = runs[0] / "native/rows.json"
            rows = read(path)
            path.write_text(encoded(rows[1:]))
            result = merge(dataset, shards, runs, root / "partial.json")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(
                result["overall"]["words_all_scheduled_items_correct"]["denominator"], 3
            )
            self.assertEqual(
                result["overall"]["words_all_scheduled_items_correct"]["numerator"], 2
            )


if __name__ == "__main__":
    unittest.main()
