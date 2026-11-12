"""Context supplement exclusions retain audit rows and never turn labels into predictions."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_vocabulary_context import select  # noqa: E402


class KnownFamilies:
    @staticmethod
    def word(word):
        return word

    @staticmethod
    def find(word):
        return {"banks": "bank"}.get(word, word)


class ContextSelectionTests(unittest.TestCase):
    def row(self, word, locator, label=True):
        return dict(
            target=word,
            locator=locator,
            native_split="test",
            label=label,
            contexts=["A context", "Another context"],
            indices=[0, 1],
        )

    def test_family_aliases_and_outside_words_are_recorded_not_silently_dropped(self):
        rows = [self.row("banks", "a"), self.row("cat", "b"), self.row("other", "c")]
        selected, events = select(rows, {"banks", "cat"}, KnownFamilies(), {"bank"})
        self.assertEqual([r[0]["target"] for r in selected], ["cat"])
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["reason"], "prior_family_exposure")
        self.assertEqual(events[2]["reason"], "outside_35000")

    def test_duplicates_not_counted_as_new_judgments(self):
        rows = [self.row("cat", "a"), self.row("cat", "b")]
        selected, events = select(rows, {"cat"}, KnownFamilies(), set())
        self.assertEqual(len(selected), 1)
        self.assertEqual(events[1]["reason"], "duplicate_context_pair")
        rows[1]["label"] = False
        with self.assertRaisesRegex(ValueError, "contradictory"):
            select(rows, {"cat"}, KnownFamilies(), set())


if __name__ == "__main__":
    unittest.main()
