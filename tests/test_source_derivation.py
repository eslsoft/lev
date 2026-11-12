"""Check deterministic label derivation against tiny explicit source facts."""

import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_research_corpus import Corpus  # noqa: E402


class SourceDerivationTests(unittest.TestCase):
    def test_shared_irregular_form_joins_every_lemma_not_just_first_owner(self):
        xml = """<LexicalResource><Lexicon>
          <LexicalEntry id="good"><Lemma writtenForm="good"/><Form writtenForm="better"/><Form writtenForm="best"/></LexicalEntry>
          <LexicalEntry id="well"><Lemma writtenForm="well"/><Form writtenForm="better"/><Form writtenForm="best"/></LexicalEntry>
          </Lexicon></LexicalResource>"""
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            with gzip.open(raw / "wordnet.xml.gz", "wt") as stream:
                stream.write(xml)
            corpus = Corpus(raw)
            corpus.wordnet()
            roots = {
                corpus.find(corpus.lexical(word))
                for word in ("good", "well", "better", "best")
            }
            self.assertEqual(len(roots), 1)

    def test_stress_counts_nuclei_and_rejects_ambiguous_primary_stress(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            (raw / "cmudict.dict").write_text(
                "correct K AH0 R EH1 K T\n"
                "ambiguous AE1 M B IH1 G Y UW0 AH0 S\n"
                "cat K AE1 T\n"
            )
            corpus = Corpus(raw)
            corpus.cmu()
            self.assertEqual(len(corpus.rows), 1)
            questions = {q["claim"]: q for q in corpus.rows[0]["questions"].values()}
            self.assertTrue(
                questions["Is syllable 2 the one carrying primary stress?"]["label"]
            )
            self.assertFalse(
                questions["Is syllable 1 the one carrying primary stress?"]["label"]
            )
            self.assertTrue(
                all(
                    q["review"]["status"] == "source_derived"
                    for q in questions.values()
                )
            )
            self.assertEqual(sum(e["status"] == "excluded" for e in corpus.audit), 2)

    def test_hypernym_direction_and_endpoint_grouping(self):
        xml = """<LexicalResource><Lexicon>
          <LexicalEntry id="dog-n"><Lemma writtenForm="dog" partOfSpeech="n"/><Sense id="dog-s" synset="child"/></LexicalEntry>
          <LexicalEntry id="animal-n"><Lemma writtenForm="animal" partOfSpeech="n"/><Sense id="animal-s" synset="parent"/></LexicalEntry>
          <Synset id="child" members="dog-n"><Definition>domestic canine</Definition><SynsetRelation relType="hypernym" target="parent"/></Synset>
          <Synset id="parent" members="animal-n"><Definition>living animal</Definition></Synset>
          </Lexicon></LexicalResource>"""
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            with gzip.open(raw / "wordnet.xml.gz", "wt") as stream:
                stream.write(xml)
            corpus = Corpus(raw)
            corpus.wordnet()
            self.assertEqual(len(corpus.rows), 1)
            row = corpus.rows[0]
            questions = {q["claim"]: q["label"] for q in row["questions"].values()}
            a_is_child = row["state"]["sense_A"] == "domestic canine"
            self.assertEqual(
                questions["Does sense A denote a subtype of sense B?"], a_is_child
            )
            self.assertEqual(
                questions[
                    "Is sense B more specific than sense A in this is-a relation?"
                ],
                not a_is_child,
            )
            self.assertEqual(corpus.find("lex:dog"), corpus.find("lex:animal"))

    def test_lexical_aliases_share_family(self):
        corpus = Corpus(Path("unused"))
        corpus.aliases["children"] = "child"
        self.assertEqual(corpus.lexical("children"), corpus.lexical("CHILD"))
        corpus.union([corpus.lexical("child"), "document:one"])
        self.assertEqual(
            corpus.find(corpus.lexical("children")), corpus.find("document:one")
        )
