"""Exercise source edits, unresolved labels, and cross-split contamination rules."""

from copy import deepcopy
from pathlib import Path
import json
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_context_corpus import author_groups, corrected, select, spelling_edges  # noqa: E402


class SimpleFamilies:
    @staticmethod
    def word(value):
        return "lex:" + value.casefold()

    @staticmethod
    def find(value):
        return value


def essay(identity, author, split, text, fixed):
    return dict(
        source="wi",
        locator=f"source:{identity}",
        native_split=split,
        essay=dict(id=identity, userid=author, text=text),
        corrected=fixed,
        issue=None,
    )


def sense(word, split, line):
    return dict(
        source="wic",
        locator=f"{split}:{line}",
        target=word,
        native_split=split,
        pos="N",
        contexts=["A target .", "The target ."],
        indices=[1, 1],
        label=False,
    )


class ContextDataTests(unittest.TestCase):
    def test_explicit_linked_spelling_redirect_is_reserved_but_synonym_is_not(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            entries = [
                dict(
                    word="harbour",
                    senses=[
                        dict(
                            glosses=["Standard spelling of harbor."],
                            links=[["harbor", "harbor#English"]],
                        )
                    ],
                ),
                dict(
                    word="missus",
                    senses=[
                        dict(
                            glosses=["Wife or girlfriend."],
                            links=[["wife", "wife#English"]],
                        )
                    ],
                ),
            ]
            (path / "entries.jsonl").write_text(
                "\n".join(json.dumps(r) for r in entries)
            )
            edges = spelling_edges(path)
            self.assertEqual(
                [(e["word"], e["alias"]) for e in edges], [("harbour", "harbor")]
            )

    def test_simultaneous_offsets_support_insertion_deletion_and_replacement(self):
        self.assertEqual(
            corrected("He go home.", [[3, 5, "goes"], [10, 10, " now"]]),
            "He goes home now.",
        )
        self.assertEqual(
            corrected("She is very happy.", [[7, 12, ""]]), "She is happy."
        )
        self.assertEqual(corrected("fine", []), "fine")

    def test_unknown_overlapping_out_of_range_edits_never_become_gold(self):
        for edits in (
            [[0, 1, None]],
            [[0, 3, "x"], [2, 4, "y"]],
            [[-1, 0, "x"]],
            [[0, 9, "x"]],
        ):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                corrected("abcd", edits)

    def test_author_and_corrected_duplicate_connections_are_transitive(self):
        rows = [
            essay("1", "a", "train", "bad one", "good one"),
            essay("2", "b", "dev", "good one", "better one"),
            essay("3", "b", "train", "bad three", "good three"),
        ]
        author_groups(rows)
        self.assertEqual(len({r["group"] for r in rows}), 1)
        records, events = select([], deepcopy(rows), SimpleFamilies(), set(), set(), [])
        self.assertTrue(records)
        self.assertNotIn("train", {r["split"] for r in records})
        self.assertEqual(
            sum(
                e["reason"] == "native_holdout_author_or_duplicate_document"
                for e in events
            ),
            2,
        )

    def test_native_lexical_holdout_is_never_used_in_training(self):
        rows = [
            sense("term", "train", 1),
            sense("term", "test", 1),
            sense("reserved", "train", 2),
        ]
        records, events = select(
            rows, [], SimpleFamilies(), {"lex:reserved"}, set(), []
        )
        self.assertEqual([r["split"] for r in records], ["test"])
        self.assertEqual(
            [e["reason"] for e in events],
            [
                "native_holdout_family",
                None,
                "reserved_vocabulary_or_historical_holdout_family",
            ],
        )

    def test_all_original_and_corrected_variants_stay_together_and_partial(self):
        records, _ = select(
            [],
            [essay("1", "a", "train", "He go.", "He goes.")],
            SimpleFamilies(),
            set(),
            set(),
            [],
        )
        self.assertEqual(
            {r["questions"]["q0"]["label"] for r in records}, {True, False}
        )
        self.assertEqual(len({r["family_id"] for r in records}), 1)
        self.assertEqual({r["scope"] for r in records}, {"partial"})
        for row in records:
            self.assertNotIn("corrected", row["state"])
            self.assertNotIn("edits", row["state"])

    def test_unresolved_essay_is_retained_in_exclusion_audit(self):
        row = essay("1", "a", "train", "unclear", "unclear")
        row["issue"] = "unresolved_edit"
        records, events = select([], [row], SimpleFamilies(), set(), set(), [])
        self.assertEqual(records, [])
        self.assertEqual(events[0]["reason"], "unresolved_edit")


if __name__ == "__main__":
    unittest.main()
