"""Do not confuse participles, qualifiers, missing evidence, and false labels."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_morphology_dictionary import classify  # noqa: E402


class MorphologyDictionaryTests(unittest.TestCase):
    def test_past_participle_does_not_attest_finite_past(self):
        entries = [
            dict(
                pos="verb",
                forms=[
                    dict(form="gone", tags=["past", "participle"]),
                    dict(form="went", tags=["past"]),
                ],
            )
        ]
        self.assertEqual(
            classify("go", "gone", "V;PST", entries)[0],
            "not_corroborated_not_a_false_label",
        )
        self.assertEqual(
            classify("go", "gone", "V;V.PTCP;PST", entries)[0], "corroborated_neutral"
        )
        self.assertEqual(
            classify("go", "went", "V;PST", entries)[0], "corroborated_neutral"
        )

    def test_nonstandard_variant_keeps_qualification(self):
        entries = [
            dict(
                pos="adj",
                forms=[dict(form="gooder", tags=["comparative", "nonstandard"])],
            )
        ]
        status, evidence = classify("good", "gooder", "ADJ;CMPR", entries)
        self.assertEqual(status, "corroborated_qualified")
        self.assertEqual(evidence[0]["qualifications"], ["nonstandard"])

    def test_metadata_and_wrong_pos_are_not_gold(self):
        entries = [
            dict(
                pos="verb",
                forms=[
                    dict(form="banks", tags=["present", "third-person", "singular"])
                ],
            )
        ]
        self.assertEqual(
            classify("bank", "banks", "N;PL", entries)[0],
            "not_corroborated_not_a_false_label",
        )
        self.assertEqual(
            classify("efflux", "countable", "N;PL", [])[0], "invalid_metadata_form"
        )
        self.assertEqual(
            classify("bank", "bank", "N;SG", entries)[0], "outside_inflection_scope"
        )


if __name__ == "__main__":
    unittest.main()
