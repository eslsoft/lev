"""Keep source minimal pairs intact and invert source labels only with question polarity."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_grammar_replay import QUERIES, key, pair_records, select_pairs, state_texts  # noqa: E402


class GrammarReplayTests(unittest.TestCase):
    def row(
        self, identifier, good="The dogs bark.", bad="The dogs barks.", term="agreement"
    ):
        return dict(
            pairID=identifier,
            sentence_good=good,
            sentence_bad=bad,
            linguistics_term=term,
        )

    def test_one_previously_seen_sentence_excludes_entire_pair(self):
        selected, events = select_pairs(
            [self.row(1)], "file", {"construction:agreement"}, {"the dogs bark."}, 100
        )
        self.assertEqual(selected, [])
        self.assertEqual(events[0]["reason"], "previously_used_sentence")

    def test_holdout_and_identical_pair_stay_in_audit(self):
        rows = [self.row(1, term="heldout"), self.row(2, good="Same.", bad="same.")]
        selected, events = select_pairs(
            rows, "file", {"construction:agreement"}, set(), 100
        )
        self.assertEqual(selected, [])
        self.assertEqual(
            {e["reason"] for e in events},
            {"nontraining_construction", "identical_good_bad"},
        )

    def test_question_choice_shared_by_pair_and_all_polarities_correct(self):
        covered = set()
        for i in range(100):
            locator = f"file:{i}"
            records = pair_records(locator, self.row(i))
            claim, invert = QUERIES[int(key(locator)[:8], 16) % len(QUERIES)]
            covered.add(claim)
            self.assertEqual(
                [r["questions"]["q0"]["claim"] for r in records], [claim, claim]
            )
            self.assertEqual(
                [r["questions"]["q0"]["label"] for r in records], [not invert, invert]
            )
        self.assertEqual(len(covered), len(QUERIES))

    def test_nested_contexts_are_reserved(self):
        self.assertEqual(
            state_texts({"contexts": ["  The dogs bark. ", "Other."], "index": 2}),
            {"the dogs bark.", "other."},
        )


if __name__ == "__main__":
    unittest.main()
