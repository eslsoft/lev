"""Expand train coverage with OEWN sense-level hypernymy.

The 35k inventory benchmark tests OEWN POS/sense/pronunciation membership for
its own 35,000 headwords, so no training item here asks any of those questions.
Hypernymy is a different task and its labels come from the directed is-a edge,
not from surface negation.

Every new item is train-only. Held-out splits of the base bundle are copied
verbatim, and any family that already touches a held-out split is excluded.

Families come from the canonical `lexical_graph`, not a private union graph, so
family ids stay comparable with every other Lev pack.
"""

from collections import Counter
from pathlib import Path
import gzip
import re
import xml.etree.ElementTree as ET

from audit_family_isolation import Families
from build_context_corpus import RAW as CONTEXT_RAW, lexical_graph, read_wic
from build_generalization_corpus import phrasing
from build_research_corpus import key
from lev.dataset import bundle, prepare
from lev.io import digest, encoded, write

BASE = Path("artifacts/en-joint-002")
OUT = Path("data/research/en-joint-006")
ARTIFACT = Path("artifacts/en-joint-006")
RUBRIC = Path("rubrics/en-general-research-v1.json")
RAW = Path("data/raw/en-research-20261021-a")
WORDNET = RAW / "wordnet.xml.gz"
TARGET = Path("artifacts/en-us-vocab-35000-v3/words.txt")
VERSION = "en-general-research-v1"
CONVENTION = (
    "WordNet sense-level is-a hierarchy; subtype is more specific, "
    "not merely associated."
)


def eligible(lex):
    """Same eligibility rule as the frozen pilot, applied to every synset."""
    entries = {}
    for entry in lex.findall("LexicalEntry"):
        lemma = entry.find("Lemma")
        if lemma is not None:
            entries[entry.attrib["id"]] = lemma.attrib["writtenForm"].casefold()
    synsets = {s.attrib["id"]: s for s in lex.findall("Synset")}

    def members_of(syn):
        return [
            entries[x] for x in syn.attrib.get("members", "").split() if x in entries
        ]

    kept, rejected = [], Counter()
    for sid, syn in synsets.items():
        parents = [
            r.attrib["target"]
            for r in syn.findall("SynsetRelation")
            if r.attrib["relType"] == "hypernym"
        ]
        members = members_of(syn)
        if len(parents) != 1 or not members:
            rejected["not_single_hypernym"] += 1
            continue
        if not all(re.fullmatch("[a-z]{3,16}", w) for w in members):
            rejected["irregular_members"] += 1
            continue
        parent = synsets.get(parents[0])
        if parent is None or not members_of(parent):
            rejected["missing_parent_members"] += 1
            continue
        if not syn.findtext("Definition") or not parent.findtext("Definition"):
            rejected["missing_definition"] += 1
            continue
        kept.append((sid, syn, parent, members, members_of(parent)))
    return kept, rejected


def main():
    if OUT.exists() or ARTIFACT.exists():
        raise ValueError("new output directories required")
    base, base_manifest = bundle(BASE)
    with gzip.open(WORDNET) as stream:
        lex = ET.parse(stream).getroot().find("Lexicon")
    if lex is None:
        raise ValueError("wordnet.xml.gz has no Lexicon element")
    candidates, rejected = eligible(lex)
    families: Families
    families, _, _, prior = lexical_graph(read_wic(CONTEXT_RAW))

    # Pass 1: link document families to their words exactly as the project audit
    # does, then refuse any synset whose members touch a holdout family. No union
    # is created for a synset, so no new train/holdout bridge can be introduced.
    families.attach(base + prior)
    holdout_fams = {
        families.find(r["family_id"]) for r in base + prior if r["split"] != "train"
    }
    refused = set()
    for sid, _syn, _parent, members, parents in candidates:
        roots = {families.find(families.word(w)) for w in members + parents}
        if roots & holdout_fams:
            refused.add(sid)

    blocked = holdout_fams
    target = {
        line.strip().lower()
        for line in TARGET.read_text().splitlines()
        if line.strip()
    }

    audit = []
    rows, dropped = [], Counter()
    words, target_words = set(), set()
    for sid, syn, parent, members, parents in candidates:
        if sid in refused:
            dropped["would_merge_holdout"] += 1
            audit.append(
                dict(
                    source="wordnet.xml.gz",
                    locator=sid,
                    status="excluded",
                    reason="would_merge_holdout",
                )
            )
            continue
        family = families.find(families.word(members[0]))
        if family in blocked:
            dropped["holdout_family"] += 1
            audit.append(
                dict(
                    source="wordnet.xml.gz",
                    locator=sid,
                    status="excluded",
                    reason="holdout_family",
                )
            )
            continue
        flip = int(key(sid)[0], 16) % 2
        a, b = (parent, syn) if flip else (syn, parent)
        state = dict(
            sense_A=a.findtext("Definition").strip(),
            sense_B=b.findtext("Definition").strip(),
            convention=CONVENTION,
        )
        rid = key("wordnet.xml.gz:" + sid + "->" + parent.attrib["id"])[:20]
        claims = [
            ("Does sense A denote a subtype of sense B?", not bool(flip)),
            (
                "Is sense B more specific than sense A in this is-a relation?",
                bool(flip),
            ),
        ]
        questions = {}
        for index, (claim, label) in enumerate(claims):
            text, truth = phrasing(claim, label, "train", rid + str(index))
            questions[f"q{index}"] = dict(
                kind="semantic_relation",
                claim=text,
                label=truth,
                review=dict(
                    status="source_derived",
                    reviewer="source:wordnet.xml.gz",
                    rationale=(
                        "Source explicitly asserts the directed hypernym edge. "
                        "Reverse subtype is false under the strict sense-level "
                        "taxonomy. Source-derived, not independently adjudicated by "
                        "Lev. The wrapper asks for the original answer or its logical "
                        "negation, with label transformed accordingly."
                    ),
                    evidence=[
                        dict(
                            reference="wordnet.xml.gz",
                            locator=sid + "->" + parent.attrib["id"],
                        ),
                        dict(
                            reference=str(WORDNET),
                            locator="sha256:" + digest(WORDNET),
                        ),
                    ],
                ),
            )
        rows.append(
            dict(
                id=rid,
                case_id=rid,
                family_id=family,
                split="train",
                language="en",
                scope="partial",
                origin="generated",
                rubric_version=VERSION,
                state=state,
                questions=questions,
            )
        )
        audit.append(
            dict(
                source="wordnet.xml.gz",
                locator=sid,
                status="selected",
                reason=None,
            )
        )
        words.update(members + parents)
        target_words.update(w for w in members + parents if w in target)

    base_ids = {row["id"] for row in base}
    seen = set()
    unique = []
    for row in rows:
        if row["id"] in seen or row["id"] in base_ids:
            dropped["duplicate_record"] += 1
            continue
        seen.add(row["id"])
        unique.append(row)

    OUT.mkdir(parents=True)
    combined = base + unique
    with (OUT / "records.jsonl").open("x") as stream:
        for row in combined:
            stream.write(encoded(row) + "\n")
    with (OUT / "selection-audit.jsonl").open("x") as stream:
        for event in audit:
            stream.write(encoded(event) + "\n")
        for reason, count in sorted(rejected.items()):
            stream.write(
                encoded(
                    dict(
                        source="wordnet.xml.gz",
                        locator="lexicon",
                        status="excluded",
                        reason=reason,
                        count=count,
                    )
                )
                + "\n"
            )

    prepare(OUT / "records.jsonl", RUBRIC, ARTIFACT, research=True)
    exported, manifest = bundle(ARTIFACT)

    for split in ("development", "calibration", "test"):
        before = digest(BASE / f"{split}.jsonl")
        after = digest(ARTIFACT / f"{split}.jsonl")
        if before != after:
            raise ValueError(f"{split} changed: {before} != {after}")

    split_of = {}
    for row in exported:
        split_of.setdefault(families.find(row["family_id"]), set()).add(row["split"])
    crossing = [f for f, s in split_of.items() if len(s) > 1]
    if crossing:
        raise ValueError(f"{len(crossing)} families cross splits")

    train_families = {
        families.find(r["family_id"]) for r in exported if r["split"] == "train"
    }
    covered = {
        w for w in target if families.find(families.word(w)) in train_families
    }
    summary = dict(
        version=ARTIFACT.name,
        base=BASE.name,
        base_manifest_sha256=base_manifest.get("source_sha256"),
        new_records=len(unique),
        new_questions=sum(len(r["questions"]) for r in unique),
        new_synset_words=len(words),
        new_synset_target_words=len(target_words),
        eligible_synsets=len(candidates),
        rejected=dict(rejected),
        dropped=dict(dropped),
        requests=len(exported),
        requests_by_split=dict(Counter(r["split"] for r in exported)),
        judgments_by_split=dict(
            Counter(r["split"] for r in exported for _ in r["questions"])
        ),
        by_kind=dict(
            Counter(q["kind"] for r in exported for q in r["questions"].values())
        ),
        families=len({families.find(r["family_id"]) for r in exported}),
        train_families=len(train_families),
        target_words_covered=len(covered),
        target_words_covered_ratio=round(len(covered) / len(target), 4),
        families_crossing_splits=0,
        benchmark_policy=(
            "The frozen 35k inventory benchmark tests OEWN POS, sense and CMUdict "
            "pronunciation membership for its own headwords. No such question is "
            "generated here; hypernymy only."
        ),
        heldout_byte_identical=True,
        status="passed",
    )
    write(OUT / "summary.json", summary)
    write(
        OUT / "source-manifest.json",
        dict(
            files={
                str(p): digest(p)
                for p in (WORDNET, BASE / "manifest.json", Path(__file__))
            },
            output_manifest_sha256=digest(ARTIFACT / "manifest.json"),
        ),
    )
    print(encoded(summary))


if __name__ == "__main__":
    main()
