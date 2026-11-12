import copy
import json
from pathlib import Path
import tempfile
import unittest

from lev.dataset import bundle, prepare, to_kev, validate
from lev.io import jsonl, read

ROOT = Path(__file__).resolve().parents[1]


class ResearchDataTests(unittest.TestCase):
    def setUp(self):
        self.rubric = read(ROOT / "rubrics/en-general-research-v1.json")
        self.rows = jsonl(ROOT / "data/examples/format-only.jsonl")
        for row in self.rows:
            row["rubric_version"] = self.rubric["version"]
            # Synthetic fixture only, never saved as a training artifact.
            row["origin"] = "generated"
            for question in row["questions"].values():
                question["kind"] = "previously_unregistered_task"
                question["review"] = dict(
                    status="source_labeled",
                    reviewer="test-fixture",
                    rationale="fixture mapping",
                    evidence=[dict(reference="fixture", locator="line:1")],
                )

    def test_free_question_category_and_private_provenance(self):
        validate(self.rows, self.rubric)
        self.rows[0]["questions"]["number"]["review"]["rationale"] = "SECRET_REVIEW"
        payload = json.dumps(to_kev(self.rows[0], self.rubric))
        self.assertNotIn("SECRET_REVIEW", payload)
        self.assertIn("previously_unregistered_task", payload)

    def test_research_mode_cannot_be_promoted_by_changing_manifest_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, rubric = root / "records.jsonl", root / "rubric.json"
            data.write_text("\n".join(json.dumps(r) for r in self.rows))
            rubric.write_text(json.dumps(self.rubric))
            with self.assertRaisesRegex(ValueError, "production export"):
                prepare(data, rubric, root / "production")
            out = root / "research"
            prepare(data, rubric, out, research=True)
            self.assertTrue(bundle(out)[1]["research_only"])
            manifest = read(out / "manifest.json")
            manifest["research_only"] = False
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "production export"):
                bundle(out)

    def test_research_rejects_examples_draft_and_missing_evidence(self):
        from lev.dataset import exportable

        for field, value in (("status", "draft"), ("evidence", [])):
            rows = copy.deepcopy(self.rows)
            next(iter(rows[0]["questions"].values()))["review"][field] = value
            with self.assertRaises(ValueError):
                validate(rows, self.rubric)
                exportable(rows, self.rubric, examples=False, research=True)
        with self.assertRaises(ValueError):
            exportable(self.rows, self.rubric, examples=True, research=True)
