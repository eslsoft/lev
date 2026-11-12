"""Prove per-word completeness, source-label boundaries and test-only export safety."""

from copy import deepcopy
import gzip
from pathlib import Path
import sys
import tempfile
import unittest

from lev.dataset import bundle, to_kev
from lev.io import digest, encoded, write
from lev.vocabulary import load_words, score_words

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_vocabulary_benchmark import audit  # noqa: E402
from build_vocabulary_benchmark import (  # noqa: E402
    VERSION,
    distractor,
    negative_pronunciation,
    questions_for,
    records_for,
    wordnet,
)
from vocabulary_wiktionary import independent, load_supplement, supplement_questions  # noqa: E402


class WordScoringTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            dict(
                id="r1",
                state={"headword": "bank"},
                questions={"a": dict(label=True, kind="sense_inventory")},
            ),
            dict(
                id="r2",
                state={"headword": "bank"},
                questions={"b": dict(label=False, kind="sense_inventory")},
            ),
            dict(
                id="r3",
                state={"headword": "cat"},
                questions={"c": dict(label=True, kind="sense_inventory")},
            ),
        ]
        self.words = [
            dict(
                word="bank",
                rank=1,
                exposure="prior_train",
                record_ids=["r1", "r2"],
                judgment_count=2,
                coverage={},
                gaps=["collocations"],
            ),
            dict(
                word="cat",
                rank=2,
                exposure="no_detected_prior_target_family",
                record_ids=["r3"],
                judgment_count=1,
                coverage={},
                gaps=[],
            ),
        ]
        self.rows = [
            dict(
                id="r1",
                question="a",
                keys=["false", "true"],
                type="noul",
                label=1,
                p=[0.1, 0.9],
            ),
            dict(
                id="r2",
                question="b",
                keys=["false", "true"],
                type="noul",
                label=0,
                p=[0.9, 0.1],
            ),
            dict(
                id="r3",
                question="c",
                keys=["false", "true"],
                type="noul",
                label=1,
                p=[0.1, 0.9],
            ),
        ]

    def test_missing_request_prevents_word_passing_without_shrinking_denominator(self):
        result = score_words(self.words, self.records, [self.rows[0], self.rows[2]])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(
            result["overall"]["words_all_scheduled_items_correct"],
            {"numerator": 1, "denominator": 2, "rate": 0.5},
        )
        self.assertEqual(result["overall"]["word_macro_accuracy"], 0.75)
        self.assertEqual(result["overall"]["missing_judgments"], 1)
        self.assertFalse(result["words"][0]["all_scheduled_items_correct"])

    def test_one_error_fails_word_even_when_other_senses_pass(self):
        self.rows[1]["p"] = [0.1, 0.9]
        result = score_words(self.words, self.records, self.rows)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["overall"]["words_fully_executed"], 2)
        self.assertEqual(
            result["overall"]["words_all_scheduled_items_correct"]["numerator"], 1
        )
        self.assertFalse(result["words"][0]["complete_entry_reviewed"])

    def test_no_predictions_is_not_a_model_accuracy_result(self):
        result = score_words(self.words, self.records, [])
        self.assertEqual(result["status"], "not_run")
        self.assertEqual(result["overall"]["words_fully_executed"], 0)
        self.assertEqual(result["overall"]["missing_judgments"], 3)

    def test_false_acceptance_rejection_and_missing_are_distinct(self):
        self.rows[0]["p"] = [0.8, 0.2]
        self.rows[1]["p"] = [0.5, 0.5]
        result = score_words(self.words, self.records, self.rows[:2])
        totals = result["overall"]
        self.assertEqual(totals["false_acceptances"], 1)
        self.assertEqual(totals["false_rejections"], 1)
        self.assertEqual(totals["missing_judgments"], 1)
        self.assertEqual(totals["positive_labels"], 2)
        self.assertEqual(totals["negative_labels"], 1)
        self.assertEqual(result["words"][1]["false_rejections"], 0)

    def test_duplicate_unknown_or_relabeled_predictions_rejected(self):
        for rows in [
            self.rows + [self.rows[0]],
            [dict(self.rows[0], id="unknown")],
            [dict(self.rows[0], label=0)],
            [dict(self.rows[0], p=[float("nan"), 0.9])],
        ]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                score_words(self.words, self.records, rows)

    def test_word_registry_cannot_hide_request_or_duplicate_word(self):
        for words in [self.words[:1], self.words + [self.words[0]]]:
            with self.assertRaises(ValueError):
                score_words(words, self.records, self.rows)


class InventoryTests(unittest.TestCase):
    def test_pure_inflections_do_not_count_as_new_headwords(self):
        self.assertFalse(
            independent(
                {"pos": "verb"},
                {"glosses": ["past of go"], "form_of": [{"word": "go"}]},
            )
        )
        self.assertFalse(independent({"pos": "name"}, {"glosses": ["a personal name"]}))
        self.assertTrue(
            independent({"pos": "pron"}, {"glosses": ["the person being addressed"]})
        )

    def test_function_word_senses_keep_parent_glosses_and_all_alternatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "downloads.jsonl").write_text(
                encoded({"status": "downloaded"}) + "\n"
            )
            (root / "you.jsonl").write_text(
                encoded(
                    dict(
                        word="you",
                        lang_code="en",
                        pos="pron",
                        senses=[
                            {"glosses": ["second person", "as subject"]},
                            {"glosses": ["the addressee as object"]},
                        ],
                    )
                )
                + "\n"
            )
            (root / "i.jsonl").write_text(
                encoded(
                    dict(
                        word="I",
                        lang_code="en",
                        pos="pron",
                        senses=[
                            {"glosses": ["the speaker as subject"]},
                        ],
                    )
                )
                + "\n"
            )
            words, senses = load_supplement(root)
            self.assertTrue(words["you"]["eligible"])
            self.assertIn("i", words)
            questions = supplement_questions("you", words["you"], senses)
            positive = [
                q["claim"]
                for q in questions.values()
                if q["kind"] == "wiktionary_sense_inventory" and q["label"]
            ]
            negative = [
                q["claim"]
                for q in questions.values()
                if q["kind"] == "wiktionary_sense_inventory" and not q["label"]
            ]
            self.assertTrue(any("second person; as subject" in q for q in positive))
            self.assertTrue(all("the speaker as subject" in q for q in negative))
            self.assertEqual(len(positive), 2)

    def test_pronunciation_negative_excludes_all_valid_variants(self):
        variants = {"R IY1 D", "R EH1 D"}
        negative = negative_pronunciation("read", "R IY1 D", variants)
        self.assertNotIn(negative, variants)
        self.assertEqual(
            sum(
                a != b for a, b in zip("R IY1 D".split(), negative.split(), strict=True)
            ),
            1,
        )

    def test_other_sense_and_duplicate_gloss_cannot_be_negative(self):
        synsets = {
            "bank1": dict(
                definition="financial institution", pos="noun", domain="noun.group"
            ),
            "bank2": dict(definition="sloping land", pos="noun", domain="noun.group"),
            "duplicate": dict(
                definition="Financial institution", pos="noun", domain="noun.group"
            ),
            "other": dict(
                definition="organized musicians", pos="noun", domain="noun.group"
            ),
        }
        pools = {("noun", "noun.group"): list(synsets)}
        self.assertEqual(
            distractor("bank", "bank1", {"senses": ["bank1", "bank2"]}, synsets, pools),
            "other",
        )
        del synsets["other"]
        pools[("noun", "noun.group")].remove("other")
        with self.assertRaises(ValueError):
            distractor("bank", "bank1", {"senses": ["bank1", "bank2"]}, synsets, pools)

    def test_actual_bundle_audits_labels_and_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, dataset = root / "raw", root / "dataset"
            raw.mkdir()
            dataset.mkdir()
            xml = """<LexicalResource><Lexicon version="2025">
            <LexicalEntry id="cat"><Lemma writtenForm="cat" partOfSpeech="n"/><Sense synset="cat-s"/></LexicalEntry>
            <LexicalEntry id="dog"><Lemma writtenForm="dog" partOfSpeech="n"/><Sense synset="dog-s"/></LexicalEntry>
            <Synset id="cat-s" partOfSpeech="n" lexfile="noun.animal"><Definition>small feline mammal</Definition></Synset>
            <Synset id="dog-s" partOfSpeech="n" lexfile="noun.animal"><Definition>domesticated canine mammal</Definition></Synset>
            </Lexicon></LexicalResource>"""
            with gzip.open(raw / "wordnet.xml.gz", "wt") as stream:
                stream.write(xml)
            (raw / "cmudict.dict").write_text("cat K AE1 T\n")
            words, synsets = wordnet(raw / "wordnet.xml.gz")
            sources = {
                name: digest(raw / name) for name in ["wordnet.xml.gz", "cmudict.dict"]
            }
            questions = questions_for(
                "cat",
                words["cat"],
                synsets,
                {("noun", "noun.animal"): ["cat-s", "dog-s"]},
                {"K AE1 T": 1},
                sources,
            )
            records = records_for("cat", "lex:cat", questions)
            rubric = dict(
                format_version=2,
                version=VERSION,
                language="en",
                status="research",
                rules={"default": "Judge the proposition."},
            )
            word = dict(
                word="cat",
                rank=1,
                record_ids=[r["id"] for r in records],
                judgment_count=len(questions),
            )
            (dataset / "words.jsonl").write_text(encoded(word) + "\n")
            write(dataset / "rubric.json", rubric)
            write(dataset / "coverage.json", {})
            write(
                dataset / "source-manifest.json",
                {"files": {str(raw / k): v for k, v in sources.items()}},
            )

            def freeze():
                (dataset / "records.jsonl").write_text(
                    "".join(encoded(r) + "\n" for r in records)
                )
                (dataset / "test.jsonl").write_text(
                    "".join(encoded(to_kev(r, rubric)) + "\n" for r in records)
                )
                manifest = dict(
                    format="lev-vocabulary-test-v1",
                    test_only=True,
                    unique_headwords=1,
                    files={
                        p.name: digest(p)
                        for p in dataset.iterdir()
                        if p.name != "manifest.json"
                    },
                )
                (dataset / "manifest.json").write_text(encoded(manifest))

            freeze()
            self.assertEqual(audit(dataset, raw)["status"], "passed")
            with self.assertRaises(ValueError):
                bundle(dataset)
            original = deepcopy(records)
            next(iter(records[0]["questions"].values()))["label"] ^= True
            freeze()
            with self.assertRaisesRegex(ValueError, "source-label mismatch"):
                audit(dataset, raw)
            records = original
            freeze()
            (dataset / "test.jsonl").write_text("modified")
            with self.assertRaisesRegex(ValueError, "modified"):
                load_words(dataset)


if __name__ == "__main__":
    unittest.main()
