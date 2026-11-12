"""Evidence-backed research labels, never represented as independent adjudication."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from lev.dataset import prepare
from lev.io import digest, encoded, write
from fetch_research_sources import fetch

VERSION = "en-general-research-v1"
WIKT_WORDS = "apple book river bank light run walk write read speak think know teach learn build break carry choose draw drive eat fall fly give grow hold keep leave meet pay put ride rise sell send sing sit sleep stand swim take tell throw understand wear win wind work beautiful quickly happy child tooth foot mouse goose good bad old new hard soft sound word sentence language meaning grammar knowledge evidence".split()
POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def partition(family):
    n = int(key("lev-research-42:" + family)[:8], 16) % 100
    return (
        "train"
        if n < 70
        else "development"
        if n < 80
        else "calibration"
        if n < 90
        else "test"
    )


class Corpus:
    def __init__(self, raw):
        self.raw, self.rows, self.audit = raw, [], []
        self.parents = {}
        self.aliases = {}

    def find(self, item):
        self.parents.setdefault(item, item)
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def union(self, items):
        roots = sorted({self.find(x) for x in items})
        for root in roots[1:]:
            self.parents[root] = roots[0]

    def lexical(self, word):
        word = word.casefold()
        return "lex:" + self.aliases.get(word, word)

    def event(self, source, locator, status, reason):
        self.audit.append(
            dict(source=source, locator=locator, status=status, reason=reason)
        )

    def add(
        self,
        source,
        locator,
        anchors,
        state,
        questions,
        rationale,
        status="source_derived",
    ):
        self.union(anchors)
        rid = key(source + ":" + locator)[:20]
        qmap = {}
        # Rotate question order independently of truth; stable neutral IDs.
        questions = sorted(questions, key=lambda q: key(rid + q[0]))
        for i, (claim, label, kind) in enumerate(questions):
            qmap[f"q{i}"] = dict(
                kind=kind,
                claim=claim,
                label=label,
                review=dict(
                    status=status,
                    reviewer="source:" + source,
                    rationale=rationale,
                    evidence=[
                        dict(reference=source, locator=locator),
                        dict(
                            reference=str(self.raw / source),
                            locator="sha256:" + digest(self.raw / source),
                        ),
                    ],
                ),
            )
        self.rows.append(
            dict(
                id=rid,
                family_id=anchors[0],
                case_id=rid,
                split="train",
                language="en",
                scope="partial",
                origin="generated",
                rubric_version=VERSION,
                state=state,
                questions=qmap,
            )
        )
        self.event(
            source,
            locator,
            "selected",
            "deterministic eligible sampling; source provenance retained",
        )

    def wordnet(self, limit=180):
        lex = ET.parse(gzip.open(self.raw / "wordnet.xml.gz")).getroot().find("Lexicon")
        synsets = {s.attrib["id"]: s for s in lex.findall("Synset")}
        entries = {}
        candidates = []
        for e in lex.findall("LexicalEntry"):
            lemma = e.find("Lemma").attrib["writtenForm"].casefold()
            entries[e.attrib["id"]] = lemma
            self.aliases.setdefault(lemma, lemma)
            for form in e.findall("Form"):
                self.aliases.setdefault(form.attrib["writtenForm"].casefold(), lemma)
        # A form can belong to several lemmas (good/well -> better/best).
        # An alias alone remembers only the first owner; join every explicit edge.
        for e in lex.findall("LexicalEntry"):
            self.union(
                [self.lexical(entries[e.attrib["id"]])]
                + [self.lexical(f.attrib["writtenForm"]) for f in e.findall("Form")]
            )
        # Published derivational relations join lexical families before any split assignment.
        sense_lemmas = {
            s.attrib["id"]: entries[e.attrib["id"]]
            for e in lex.findall("LexicalEntry")
            for s in e.findall("Sense")
        }
        for e in lex.findall("LexicalEntry"):
            for sense in e.findall("Sense"):
                for relation in sense.findall("SenseRelation"):
                    if (
                        relation.attrib["relType"] == "derivation"
                        and relation.attrib["target"] in sense_lemmas
                    ):
                        self.union(
                            [
                                self.lexical(entries[e.attrib["id"]]),
                                self.lexical(sense_lemmas[relation.attrib["target"]]),
                            ]
                        )
        for sid, syn in synsets.items():
            rels = [
                r.attrib["target"]
                for r in syn.findall("SynsetRelation")
                if r.attrib["relType"] == "hypernym"
            ]
            members = [
                entries[x]
                for x in syn.attrib.get("members", "").split()
                if x in entries
            ]
            if (
                len(rels) != 1
                or not members
                or not all(re.fullmatch("[a-z]{3,16}", w) for w in members)
            ):
                self.event(
                    "wordnet.xml.gz",
                    sid,
                    "excluded",
                    "pilot requires one explicit hypernym and simple lexical members",
                )
                continue
            parent = synsets[rels[0]]
            parents = [
                entries[x]
                for x in parent.attrib.get("members", "").split()
                if x in entries
            ]
            if (
                not parents
                or not syn.findtext("Definition")
                or not parent.findtext("Definition")
            ):
                self.event(
                    "wordnet.xml.gz",
                    sid,
                    "excluded",
                    "missing parent members or definitions",
                )
                continue
            candidates.append((sid, syn, parent, members, parents))
        chosen = sorted(candidates, key=lambda x: key(x[0]))[:limit]
        selected = {x[0] for x in chosen}
        for sid, *_ in candidates:
            if sid not in selected:
                self.event(
                    "wordnet.xml.gz", sid, "excluded", "outside fixed pilot hash sample"
                )
        for sid, syn, parent, members, parents in chosen:
            flip = int(key(sid)[0], 16) % 2
            a, b = (parent, syn) if flip else (syn, parent)
            state = {
                "sense_A": a.findtext("Definition"),
                "sense_B": b.findtext("Definition"),
                "convention": "WordNet sense-level is-a hierarchy; subtype is more specific, not merely associated.",
            }
            self.add(
                "wordnet.xml.gz",
                sid + "->" + parent.attrib["id"],
                [self.lexical(w) for w in members + parents],
                state,
                [
                    (
                        "Does sense A denote a subtype of sense B?",
                        not bool(flip),
                        "semantic_relation",
                    ),
                    (
                        "Is sense B more specific than sense A in this is-a relation?",
                        bool(flip),
                        "semantic_relation",
                    ),
                ],
                "Source explicitly asserts the directed hypernym edge. Reverse subtype is false under the strict sense-level taxonomy. Source-derived, not independently adjudicated by Lev.",
            )

    def cmu(self, limit=140):
        candidates = []
        for number, line in enumerate(
            (self.raw / "cmudict.dict").read_text().splitlines(), 1
        ):
            fields = line.split()
            if not fields:
                continue
            word, phones = fields[0], fields[1:]
            vowels = [p for p in phones if re.fullmatch(r"[A-Z]+[012]", p)]
            if (
                not re.fullmatch("[a-z]{3,16}", word)
                or len(vowels) < 2
                or sum(p.endswith("1") for p in vowels) != 1
            ):
                self.event(
                    "cmudict.dict",
                    str(number),
                    "excluded",
                    "requires simple headword, multisyllabic pronunciation and one primary stress",
                )
                continue
            candidates.append((number, word, phones, vowels))
        chosen = sorted(candidates, key=lambda x: key(x[1]))[:limit]
        selected = {x[0] for x in chosen}
        for number, *_ in candidates:
            if number not in selected:
                self.event(
                    "cmudict.dict",
                    str(number),
                    "excluded",
                    "outside fixed pilot hash sample",
                )
        for number, word, phones, vowels in chosen:
            index = next(i + 1 for i, p in enumerate(vowels) if p.endswith("1"))
            wrong = index % len(vowels) + 1
            self.add(
                "cmudict.dict",
                str(number),
                [self.lexical(word)],
                {
                    "word": word,
                    "pronunciation": " ".join(phones),
                    "notation": "CMU ARPABET: each vowel is one syllabic nucleus; 0 unstressed, 1 primary stress, 2 secondary stress. Judge this supplied pronunciation only.",
                },
                [
                    (
                        f"Is syllable {index} the one carrying primary stress?",
                        True,
                        "pronunciation",
                    ),
                    (
                        f"Is syllable {wrong} the one carrying primary stress?",
                        False,
                        "pronunciation",
                    ),
                ],
                "Count annotated vowel nuclei in the supplied source pronunciation. A unique stress-1 vowel establishes the positive index and contradicts the other index. No claim about unlisted pronunciations.",
            )

    def ud(self, limit=180, unique_groups=False):
        candidates = []
        for block in (self.raw / "ud.conllu").read_text().split("\n\n"):
            lines = block.splitlines()
            meta = dict(
                line[2:].split(" = ", 1)
                for line in lines
                if line.startswith("# ") and " = " in line
            )
            if "sent_id" not in meta:
                continue
            tokens = [
                line.split("\t")
                for line in lines
                if line and not line.startswith("#") and line.split("\t")[0].isdigit()
            ]
            eligible = [
                t
                for t in tokens
                if t[3] in {"NOUN", "VERB", "ADJ", "ADV"}
                and re.fullmatch("[a-z]{4,16}", t[2])
                and t[7] in {"nsubj", "obj", "amod", "advmod"}
            ]
            if not eligible or not 5 <= len(tokens) <= 30:
                self.event(
                    "ud.conllu",
                    meta["sent_id"],
                    "excluded",
                    "no unambiguous pilot target or length outside 5-30 tokens",
                )
                continue
            t = min(eligible, key=lambda t: key(meta["sent_id"] + t[0]))
            candidates.append((meta, tokens, t))
        ordered = sorted(candidates, key=lambda x: key(x[0]["sent_id"]))
        if unique_groups:
            chosen, seen = [], set()
            for candidate in ordered:
                meta, _, target = candidate
                groups = {
                    self.find(self.lexical(target[2])),
                    self.find("document:" + meta["sent_id"].rsplit("-", 1)[0]),
                }
                if groups & seen:
                    continue
                chosen.append(candidate)
                seen.update(groups)
                if len(chosen) == limit:
                    break
        else:
            chosen = ordered[:limit]
        selected = {x[0]["sent_id"] for x in chosen}
        for meta, *_ in candidates:
            if meta["sent_id"] not in selected:
                self.event(
                    "ud.conllu",
                    meta["sent_id"],
                    "excluded",
                    "outside fixed pilot hash sample",
                )
        roles = {
            "nsubj": "nominal subject",
            "obj": "direct object",
            "amod": "adjectival modifier",
            "advmod": "adverbial modifier",
        }
        for meta, tokens, t in chosen:
            wrong = {
                "nsubj": "obj",
                "obj": "nsubj",
                "amod": "advmod",
                "advmod": "amod",
            }[t[7]]
            doc = meta["sent_id"].rsplit("-", 1)[0]
            self.add(
                "ud.conllu",
                meta["sent_id"] + ":token=" + t[0],
                [self.lexical(t[2]), "document:" + doc],
                {
                    "tokens": [{"index": int(x[0]), "text": x[1]} for x in tokens],
                    "target_index": int(t[0]),
                    "convention": "Universal Dependencies basic analysis. Judge the target token's dependency role, not a possible role in another sentence.",
                },
                [
                    (
                        f"Does the target token function as a {roles[t[7]]}?",
                        True,
                        "syntax",
                    ),
                    (
                        f"Does the target token function as a {roles[wrong]}?",
                        False,
                        "syntax",
                    ),
                ],
                "The EWT source assigns the explicit basic dependency relation in column 8. The alternative is a different exclusive basic relation under this annotation. Lev has not independently re-adjudicated the sentence.",
            )

    def blimp(self):
        for file in sorted((self.raw / "blimp").glob("*.jsonl")):
            allrows = [json.loads(line) for line in file.read_text().splitlines()]
            chosen = sorted(allrows, key=lambda x: key(file.name + str(x["pairID"])))[
                :12
            ]
            selected = {str(x["pairID"]) for x in chosen}
            for row in allrows:
                if str(row["pairID"]) not in selected:
                    self.event(
                        str(file.relative_to(self.raw)),
                        str(row["pairID"]),
                        "excluded",
                        "outside fixed pilot hash sample",
                    )
            for row in chosen:
                for field, label in (("sentence_good", True), ("sentence_bad", False)):
                    self.add(
                        str(file.relative_to(self.raw)),
                        str(row["pairID"]) + ":" + field,
                        ["construction:" + row["linguistics_term"]],
                        {
                            "sentence": row[field],
                            "variety": "English under the BLiMP grammaticality task; evaluate syntax, not whether the event is likely.",
                        },
                        [
                            (
                                "Is this sentence grammatically acceptable under the stated task?",
                                label,
                                "grammaticality",
                            )
                        ],
                        "Original BLiMP sentence_good/sentence_bad label from linguist-written generation templates. Human validation was aggregate, not an independent adjudication of this individual Lev sample.",
                        "source_labeled",
                    )

    def wiktionary(self):
        rawdir = self.raw / "wiktionary"
        rawdir.mkdir(exist_ok=True)
        events = self.raw / "wiktionary-downloads.jsonl"
        for word in WIKT_WORDS:
            path = rawdir / (word + ".jsonl")
            url = f"https://kaikki.org/dictionary/English/meaning/{word[0]}/{word[:2]}/{word}.jsonl"
            if not path.exists():
                try:
                    sha = fetch(url, path)
                    event = dict(word=word, url=url, sha256=sha, status="downloaded")
                except Exception as error:
                    event = dict(word=word, url=url, status="failed", error=str(error))
                with events.open("a") as out:
                    out.write(json.dumps(event) + "\n")
                if event["status"] != "downloaded":
                    raise RuntimeError(f"Wiktionary download failed; see {events}")
            download_log = [
                json.loads(line) for line in events.read_text().splitlines()
            ]
            matching = [event for event in download_log if event["word"] == word]
            if (
                not matching
                or matching[-1]["status"] != "downloaded"
                or matching[-1]["sha256"] != digest(path)
            ):
                raise ValueError(
                    f"Wiktionary snapshot missing provenance or modified: {word}"
                )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for n, entry in enumerate(rows):
                for form in entry.get("forms", []):
                    if re.fullmatch("[a-z]+", form.get("form", "")):
                        self.union([self.lexical(word), self.lexical(form["form"])])
                locator = f"entry={n}"
                if entry.get("pos") not in {
                    "noun",
                    "verb",
                    "adj",
                    "adv",
                } or not entry.get("senses"):
                    self.event(
                        str(path.relative_to(self.raw)),
                        locator,
                        "excluded",
                        "unsupported pilot part of speech or no senses",
                    )
                    continue
                sense = entry["senses"][0]
                glosses = sense.get("glosses", [])
                if not glosses:
                    self.event(
                        str(path.relative_to(self.raw)),
                        locator,
                        "excluded",
                        "no explicit gloss",
                    )
                    continue
                pos = {"adj": "adjective", "adv": "adverb"}.get(
                    entry["pos"], entry["pos"]
                )
                wrong = {
                    "noun": "verb",
                    "verb": "noun",
                    "adjective": "adverb",
                    "adverb": "adjective",
                }[pos]
                self.add(
                    str(path.relative_to(self.raw)),
                    locator + ":sense=0",
                    [self.lexical(word)],
                    {
                        "headword": word,
                        "sense_description": " / ".join(glosses),
                        "usage_tags": sense.get("tags", []),
                        "scope": "The particular Wiktionary lexical sense described here, not every possible use of the spelling.",
                    },
                    [
                        (
                            f"Is {pos} the part of speech of this lexical sense?",
                            True,
                            "lexical_analysis",
                        ),
                        (
                            f"Is {wrong} the part of speech of this lexical sense?",
                            False,
                            "lexical_analysis",
                        ),
                    ],
                    f"Kaikki extraction of English Wiktionary at {url}; the original entry explicitly supplies its pos and first sense. Alternative POS refers only to this sense. Extraction/source errors remain possible.",
                )
                for si in range(1, len(entry["senses"])):
                    self.event(
                        str(path.relative_to(self.raw)),
                        f"entry={n}:sense={si}",
                        "excluded",
                        "pilot selects first sense per lexical entry",
                    )

    def finish(self, out):
        # Identical input content is grouped even if independent sources reused it.
        states = defaultdict(list)
        for row in self.rows:
            states[" ".join(encoded(row["state"]).casefold().split())].append(
                row["family_id"]
            )
        for anchors in states.values():
            self.union(anchors)
        for row in self.rows:
            row["family_id"] = self.find(row["family_id"])
            row["split"] = partition(row["family_id"])
            # Predeclared held-out constructions, selected before any model scoring.
            construction_splits = {
                "construction:anaphor_agreement": "development",
                "construction:binding": "calibration",
                "construction:quantifiers": "test",
            }
            if row["family_id"].startswith("construction:"):
                row["split"] = construction_splits.get(row["family_id"], "train")
        out.mkdir(parents=True, exist_ok=False)
        with (out / "records.jsonl").open("x") as stream:
            for row in sorted(self.rows, key=lambda r: r["id"]):
                stream.write(encoded(row) + "\n")
        with (out / "selection-audit.jsonl").open("x") as stream:
            for event in self.audit:
                stream.write(encoded(event) + "\n")
        write(
            out / "source-manifest.json",
            {
                "files": {
                    str(p.relative_to(self.raw)): digest(p)
                    for p in self.raw.rglob("*")
                    if p.is_file()
                },
                "policy": "Research source labels/derivations; no independent Lev adjudication. WiC excluded because its license is noncommercial. BLiMP grouped by construction, lexical tasks by target families/derivations and relation endpoints, UD also by document. Incidental vocabulary in sentences may recur; not a universal vocabulary-disjoint benchmark.",
            },
        )
        counts = Counter(
            (r["split"], q["kind"]) for r in self.rows for q in r["questions"].values()
        )
        write(
            out / "summary.json",
            {
                "records": len(self.rows),
                "questions": sum(counts.values()),
                "splits": {
                    s: dict((k, n) for (split, k), n in counts.items() if split == s)
                    for s in ("train", "development", "calibration", "test")
                },
                "audit_counts": dict(Counter(e["status"] for e in self.audit)),
                "labels": dict(
                    Counter(
                        str(q["label"])
                        for r in self.rows
                        for q in r["questions"].values()
                    )
                ),
            },
        )
        return out / "records.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bundle", required=True)
    a = p.parse_args()
    corpus = Corpus(Path(a.raw))
    downloads = [
        json.loads(line)
        for line in (corpus.raw / "download-events.jsonl").read_text().splitlines()
    ]
    for entry in downloads:
        if (
            entry["status"] != "downloaded"
            or digest(corpus.raw / entry["file"]) != entry["sha256"]
        ):
            raise ValueError("Source failed download or changed: " + entry["file"])
    for name in ("wordnet", "cmu", "ud", "blimp", "wiktionary"):
        getattr(corpus, name)()
        print(name, len(corpus.rows), flush=True)
    records = corpus.finish(Path(a.out))
    print(
        prepare(records, "rubrics/en-general-research-v1.json", a.bundle, research=True)
    )


if __name__ == "__main__":
    main()
