"""Uncorroborated or tampered morphology cannot silently become family evidence."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_corroborated_families import verified_relations  # noqa: E402
from lev.io import digest, encoded, write  # noqa: E402


class CorroboratedFamiliesTests(unittest.TestCase):
    def test_only_supported_statuses_are_adopted_and_hash_changes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            statuses = [
                "corroborated_neutral",
                "corroborated_qualified",
                "invalid_metadata_form",
                "not_corroborated_not_a_false_label",
            ]
            file = root / "comparisons.jsonl"
            file.write_text("".join(encoded(dict(status=s)) + "\n" for s in statuses))
            write(root / "summary.json", dict(status="audit_complete"))
            write(
                root / "manifest.json",
                dict(
                    source_files={},
                    output_files={p.name: digest(p) for p in root.iterdir()},
                ),
            )
            self.assertEqual(
                [r["status"] for r in verified_relations(root)], statuses[:2]
            )
            file.write_text(encoded(dict(status="corroborated_neutral")) + "\n")
            with self.assertRaisesRegex(ValueError, "output changed"):
                verified_relations(root)


if __name__ == "__main__":
    unittest.main()
