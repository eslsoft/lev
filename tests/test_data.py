import json
import tempfile
import unittest
from pathlib import Path

from lev.dataset import bundle, prepare, to_kev, validate
from lev.io import jsonl, parse, read

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.rows = jsonl(ROOT / "data/examples/format-only.jsonl")
        self.rubric = read(ROOT / "rubrics/en-v1.json")

    def test_label_and_adjudication_do_not_enter_model_context(self):
        r = self.rows[0]
        r["questions"]["number"]["review"]["rationale"] = "SECRET_GOLD_REASON"
        converted = to_kev(r, self.rubric)
        self.assertNotIn("SECRET_GOLD_REASON", json.dumps(converted))
        self.assertEqual(converted["questions"]["number"]["type"], "noul")
        self.assertIs(converted["questions"]["number"]["label"], True)
        self.assertNotIn("review", converted["questions"]["number"])

    def test_family_leakage_is_rejected(self):
        self.rows[2]["family_id"] = self.rows[0]["family_id"]
        with self.assertRaisesRegex(ValueError, "crosses split"):
            validate(self.rows, self.rubric)

    def test_renamed_identical_state_leakage_is_rejected(self):
        self.rows[2]["state"] = self.rows[0]["state"]
        with self.assertRaisesRegex(ValueError, "identical state"):
            validate(self.rows, self.rubric)

    def test_duplicate_records_and_non_boolean_labels(self):
        with self.assertRaisesRegex(ValueError, "duplicate record"):
            validate([*self.rows, self.rows[0]], self.rubric)
        self.rows[0]["questions"]["number"]["label"] = 1
        with self.assertRaisesRegex(ValueError, "bool or null"):
            validate(self.rows, self.rubric)

    def test_jsonl_preserves_unicode_line_separators_inside_strings(self):
        rows = [{"text": "a\u2028b\u2029c\u0085d"}, {"text": "second"}]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "unicode.jsonl"
            path.write_text(
                "\r\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\r\n"
            )
            self.assertEqual(jsonl(path), rows)

    def test_strict_json_rejects_duplicates_and_nan(self):
        for value in ('{"x":1,"x":2}', '{"p":NaN}'):
            with self.assertRaises(ValueError):
                parse(value)

    def test_adjudicated_labels_require_evidence(self):
        self.rows[0]["questions"]["number"]["review"]["status"] = "adjudicated"
        with self.assertRaisesRegex(ValueError, "label and evidence"):
            validate(self.rows, self.rubric)

    def test_export_is_reproducible_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            for out in (a, b):
                prepare(
                    ROOT / "data/examples/format-only.jsonl",
                    ROOT / "rubrics/en-v1.json",
                    out,
                    examples=True,
                )
            self.assertEqual(read(a / "manifest.json"), read(b / "manifest.json"))
            records, manifest = bundle(a)
            self.assertEqual(len(records), 8)
            self.assertTrue(manifest["example_only"])
            (a / "test.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "modified"):
                bundle(a)

    def test_changing_example_flag_does_not_make_data_trainable(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            prepare(
                ROOT / "data/examples/format-only.jsonl",
                ROOT / "rubrics/en-v1.json",
                out,
                examples=True,
            )
            manifest = read(out / "manifest.json")
            manifest["example_only"] = False
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "rubric is draft"):
                bundle(out)

    def test_examples_and_unresolved_records_cannot_be_production_data(self):
        with tempfile.TemporaryDirectory() as temp:
            data, rubric = Path(temp) / "rows.jsonl", Path(temp) / "rubric.json"
            self.rubric["status"] = "approved"
            rubric.write_text(json.dumps(self.rubric))
            data.write_text("\n".join(json.dumps(r) for r in self.rows))
            with self.assertRaisesRegex(ValueError, "example records"):
                prepare(data, rubric, Path(temp) / "out")
            for r in self.rows:
                r["origin"] = "generated"
            data.write_text("\n".join(json.dumps(r) for r in self.rows))
            with self.assertRaisesRegex(ValueError, "unresolved"):
                prepare(data, rubric, Path(temp) / "out")
            self.rows[0]["questions"]["number"].update(label=None)
            data.write_text("\n".join(json.dumps(r) for r in self.rows))
            with self.assertRaisesRegex(ValueError, "unresolved"):
                prepare(data, rubric, Path(temp) / "demo", examples=True)


if __name__ == "__main__":
    unittest.main()
