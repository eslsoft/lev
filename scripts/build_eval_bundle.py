"""Build a multi-task evaluation pack that is deliberately not a training pack.

The training pack cannot carry a large development set: the base train passages
cover such a wide vocabulary that almost any new held-out synset shares a
lexical family with some train passage, and the project audit rejects that.

This pack therefore ships the en-joint-010 train split unchanged for reference
and loadability, replaces nothing in the existing holdout, and appends fresh
development/calibration/test items for kinds the 147-item development set never
measured. Overlap with train families is expected and is measured here rather
than assumed away.
"""

from pathlib import Path
from collections import Counter
import argparse
import json
import re
import zipfile

from audit_family_isolation import Families
from build_context_corpus import RAW as CONTEXT_RAW, key, lexical_graph, read_wic
from build_grammar_replay import QUERIES
from lev.dataset import bundle, prepare
from lev.io import digest, encoded, write

BASE = Path("artifacts/en-joint-010")
RUBRIC = Path("rubrics/en-general-research-v1.json")
RESEARCH = Path("data/raw/en-research-20261021-a")
CMU = RESEARCH / "cmudict.dict"
BLIMP = Path("data/raw/en-general-003/blimp.zip")
UD = RESEARCH / "ud.conllu"
WIKTIONARY = RESEARCH / "wiktionary"
BLIMP_SHA = "172a1a714d72e72a89e8384dd531483f3d0f8a9aee3d33dccaf7992d8cecbb73"
VERSION = "en-eval-001"
NOTATION = (
    "CMU ARPABET: each vowel is one syllabic nucleus; 0 unstressed, "
    "1 primary stress, 2 secondary stress. Judge this supplied pronunciation only."
)


def eval_split(family):
    """Send fresh families to development mostly, so each kind stays measurable.

    The split is chosen per family, never per record: records sharing a family
    must land in the same split or the project validator rejects the bundle.
    """
    bucket = int(key("lev-eval:" + family)[:8], 16) % 10
    if bucket < 6:
        return "development"
    if bucket < 8:
        return "calibration"
    return "test"


def pronunciation(raw, limit, families, known, excluded):
    """Fresh CMUdict headwords judged on their supplied stress pattern."""
    candidates = []
    for number, line in enumerate(raw.read_text().splitlines(), 1):
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
            continue
        candidates.append((number, word, phones, vowels))
    chosen = sorted(candidates, key=lambda x: key("eval-cmu:" + x[1]))[:limit]
    rows = []
    for number, word, phones, vowels in chosen:
        family = families.find(families.word(word))
        if family in known:
            excluded["cmudict_family_already_assigned"] += 1
            continue
        index = next(i + 1 for i, p in enumerate(vowels) if p.endswith("1"))
        wrong = index % len(vowels) + 1
        rid = key(VERSION + ":cmudict:" + str(number))[:20]
        rows.append(
            dict(
                id=rid,
                case_id=rid,
                family_id=family,
                split=eval_split(family),
                language="en",
                scope="partial",
                origin="generated",
                rubric_version="en-general-research-v1",
                state=dict(
                    word=word,
                    pronunciation=" ".join(phones),
                    notation=NOTATION,
                ),
                questions={
                    "q0": dict(
                        kind="pronunciation",
                        claim=f"Is syllable {index} the one carrying primary stress?",
                        label=True,
                        review=dict(
                            status="source_derived",
                            reviewer="source:cmudict.dict",
                            rationale=(
                                "The source pronunciation marks exactly one stress-1 "
                                "vowel, which fixes the true syllable index and "
                                "contradicts the alternative. Source-derived, not "
                                "independently adjudicated by Lev."
                            ),
                            evidence=[
                                dict(reference=str(raw), locator=str(number)),
                                dict(
                                    reference=str(raw),
                                    locator="sha256:" + digest(raw),
                                ),
                            ],
                        ),
                    ),
                    "q1": dict(
                        kind="pronunciation",
                        claim=f"Is syllable {wrong} the one carrying primary stress?",
                        label=False,
                        review=dict(
                            status="source_derived",
                            reviewer="source:cmudict.dict",
                            rationale=(
                                "Same source pronunciation; the unique stress-1 vowel "
                                "is a different syllable, so this index is false."
                            ),
                            evidence=[
                                dict(reference=str(raw), locator=str(number)),
                                dict(
                                    reference=str(raw),
                                    locator="sha256:" + digest(raw),
                                ),
                            ],
                        ),
                    ),
                },
            )
        )
    return rows, len(candidates)


def grammaticality(limit, families, known, excluded):
    """Fresh BLiMP minimal pairs.

    These reuse the construction families that the training pack already holds.
    That is deliberate: the pack measures whether the encoder memorised those
    constructions, not whether it generalises to new ones. The summary says so.
    """
    if digest(BLIMP) != BLIMP_SHA:
        raise ValueError("BLiMP source changed")
    rows = []
    with zipfile.ZipFile(BLIMP) as archive:
        names = sorted(
            n for n in archive.namelist() if n.endswith(".jsonl") and "__MACOSX" not in n
        )
        for name in names:
            raw = archive.read(name)
            try:
                items = [json.loads(line) for line in raw.splitlines()]
            except json.JSONDecodeError as error:
                raise ValueError(f"BLiMP member {name} is not valid JSONL") from error
            for item in items:
                locator = name + ":" + str(item["pairID"])
                if int(key("eval-blimp:" + locator)[:8], 16) % 100 >= limit:
                    continue
                claim, invert = QUERIES[int(key(locator)[:8], 16) % len(QUERIES)]
                family = families.find("construction:" + item["linguistics_term"])
                if family in known:
                    excluded["blimp_family_already_assigned"] += 1
                    continue
                for field, truth in (("sentence_good", True), ("sentence_bad", False)):
                    identity = locator + ":" + field
                    rid = key(VERSION + ":blimp:" + identity)[:24]
                    rows.append(
                        dict(
                            id=rid,
                            case_id=rid,
                            family_id=family,
                            split=eval_split(family),
                            language="en",
                            scope="partial",
                            origin="curated",
                            rubric_version="en-general-research-v1",
                            state=dict(
                                sentence=item[field],
                                variety=(
                                    "English under the BLiMP grammaticality task; "
                                    "evaluate syntax and the stipulated reading, not "
                                    "event plausibility."
                                ),
                            ),
                            questions={
                                "q0": dict(
                                    kind="grammaticality",
                                    claim=claim,
                                    label=not truth if invert else truth,
                                    review=dict(
                                        status="source_labeled",
                                        reviewer="source:BLiMP",
                                        rationale=(
                                            "Published minimal-pair good/bad label"
                                            + (
                                                " logically inverted for the negative question."
                                                if invert
                                                else " for the affirmative question."
                                            )
                                            + " Template-authored source; not independent "
                                            "Lev adjudication."
                                        ),
                                        evidence=[
                                            dict(
                                                reference=str(BLIMP),
                                                locator=identity,
                                            )
                                        ],
                                    ),
                                )
                            },
                        )
                    )
    return rows


def syntax(raw, limit, families, known, excluded):
    """UD EWT dependency-role items for sentences the base never assigns."""
    roles = {
        "nsubj": "nominal subject",
        "obj": "direct object",
        "amod": "adjectival modifier",
        "advmod": "adverbial modifier",
    }
    wrong_of = {"nsubj": "obj", "obj": "nsubj", "amod": "advmod", "advmod": "amod"}
    candidates = []
    for block in raw.read_text().split("\n\n"):
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
            and t[7] in roles
        ]
        if not eligible or not 5 <= len(tokens) <= 30:
            continue
        target = min(eligible, key=lambda t: key(meta["sent_id"] + t[0]))
        candidates.append((meta, tokens, target))
    chosen = sorted(candidates, key=lambda x: key("eval-ud:" + x[0]["sent_id"]))[:limit]
    rows = []
    for meta, tokens, t in chosen:
        doc = meta["sent_id"].rsplit("-", 1)[0]
        lex = families.find(families.word(t[2]))
        if lex in known or families.find("document:" + doc) in known:
            excluded["ud_family_already_assigned"] += 1
            continue
        rid = key(VERSION + ":ud:" + meta["sent_id"] + ":token=" + t[0])[:24]
        wrong = wrong_of[t[7]]
        rows.append(
            dict(
                id=rid,
                case_id=rid,
                family_id=lex,
                split=eval_split(lex),
                language="en",
                scope="partial",
                origin="generated",
                rubric_version="en-general-research-v1",
                state=dict(
                    tokens=[{"index": int(x[0]), "text": x[1]} for x in tokens],
                    target_index=int(t[0]),
                    convention=(
                        "Universal Dependencies basic analysis. Judge the target "
                        "token's dependency role, not a possible role in another "
                        "sentence."
                    ),
                ),
                questions={
                    "q0": dict(
                        kind="syntax",
                        claim=f"Does the target token function as a {roles[t[7]]}?",
                        label=True,
                        review=dict(
                            status="source_derived",
                            reviewer="source:ud.conllu",
                            rationale=(
                                "The EWT source assigns the explicit basic dependency "
                                "relation in column 8. Source-derived, not independently "
                                "adjudicated by Lev."
                            ),
                            evidence=[
                                dict(reference=str(raw), locator=meta["sent_id"]),
                                dict(reference=str(raw), locator="sha256:" + digest(raw)),
                            ],
                        ),
                    ),
                    "q1": dict(
                        kind="syntax",
                        claim=f"Does the target token function as a {roles[wrong]}?",
                        label=False,
                        review=dict(
                            status="source_derived",
                            reviewer="source:ud.conllu",
                            rationale=(
                                "The alternative is a different exclusive basic "
                                "relation under this annotation."
                            ),
                            evidence=[
                                dict(reference=str(raw), locator=meta["sent_id"]),
                                dict(reference=str(raw), locator="sha256:" + digest(raw)),
                            ],
                        ),
                    ),
                },
            )
        )
    return rows, len(candidates)


def lexical_analysis(raw_dir, families, known, excluded):
    """Wiktionary part-of-speech items for headwords the base never assigns."""
    names = {"adj": "adjective", "adv": "adverb"}
    wrong_of = {
        "noun": "verb",
        "verb": "noun",
        "adjective": "adverb",
        "adverb": "adjective",
    }
    rows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        word = path.stem
        lex = families.find(families.word(word))
        if lex in known:
            excluded["wiktionary_family_already_assigned"] += 1
            continue
        try:
            entries = [json.loads(line) for line in path.read_text().splitlines()]
        except json.JSONDecodeError as error:
            raise ValueError(f"Wiktionary {path} is not valid JSONL") from error
        for number, entry in enumerate(entries):
            pos = entry.get("pos")
            if (
                not isinstance(pos, str)
                or pos not in {"noun", "verb", "adj", "adv"}
                or not entry.get("senses")
            ):
                continue
            glosses = entry["senses"][0].get("glosses", [])
            if not glosses:
                continue
            pos = names.get(pos, pos)
            rid = key(VERSION + ":wiktionary:" + word + ":" + str(number))[:24]
            rows.append(
                dict(
                    id=rid,
                    case_id=rid,
                    family_id=lex,
                    split=eval_split(lex),
                    language="en",
                    scope="partial",
                    origin="generated",
                    rubric_version="en-general-research-v1",
                    state=dict(
                        headword=word,
                        sense_description=" / ".join(glosses),
                        usage_tags=entry["senses"][0].get("tags", []),
                        scope=(
                            "The particular Wiktionary lexical sense described here, "
                            "not every possible use of the spelling."
                        ),
                    ),
                    questions={
                        "q0": dict(
                            kind="lexical_analysis",
                            claim=f"Is {pos} the part of speech of this lexical sense?",
                            label=True,
                            review=dict(
                                status="source_derived",
                                reviewer="source:wiktionary",
                                rationale=(
                                    "The original entry explicitly supplies its pos and "
                                    "first sense. Source-derived, not independently "
                                    "adjudicated by Lev."
                                ),
                                evidence=[
                                    dict(reference=str(path), locator=f"entry={number}"),
                                    dict(
                                        reference=str(path),
                                        locator="sha256:" + digest(path),
                                    ),
                                ],
                            ),
                        ),
                        "q1": dict(
                            kind="lexical_analysis",
                            claim=(
                                f"Is {wrong_of[pos]} the part of speech of this "
                                "lexical sense?"
                            ),
                            label=False,
                            review=dict(
                                status="source_derived",
                                reviewer="source:wiktionary",
                                rationale=(
                                    "The alternative part of speech refers only to this "
                                    "sense and contradicts the source pos."
                                ),
                                evidence=[
                                    dict(reference=str(path), locator=f"entry={number}"),
                                    dict(
                                        reference=str(path),
                                        locator="sha256:" + digest(path),
                                    ),
                                ],
                            ),
                        ),
                    },
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--pronunciation", type=int, default=600)
    parser.add_argument("--grammaticality-percent", type=int, default=8)
    parser.add_argument("--syntax", type=int, default=1200)
    args = parser.parse_args()
    OUT = Path("data/research") / args.version
    ARTIFACT = Path("artifacts") / args.version
    if OUT.exists() or ARTIFACT.exists():
        raise ValueError("new output directories required")

    base, base_manifest = bundle(BASE)
    families: Families
    families, _, _, prior = lexical_graph(read_wic(CONTEXT_RAW))
    families.attach(base + prior)

    known = {}
    for row in base:
        known.setdefault(families.find(row["family_id"]), row["split"])
    excluded = Counter()
    cmu_rows, cmu_candidates = pronunciation(
        CMU, args.pronunciation, families, known, excluded
    )
    blimp_rows = grammaticality(
        args.grammaticality_percent, families, known, excluded
    )
    syntax_rows, ud_candidates = syntax(UD, args.syntax, families, known, excluded)
    lex_rows = lexical_analysis(WIKTIONARY, families, known, excluded)
    fresh = cmu_rows + blimp_rows + syntax_rows + lex_rows

    base_ids = {r["id"] for r in base}
    seen = set()
    unique = []
    for row in fresh:
        if row["id"] in seen or row["id"] in base_ids:
            continue
        seen.add(row["id"])
        unique.append(row)

    combined = base + unique
    OUT.mkdir(parents=True)
    with (OUT / "records.jsonl").open("x") as stream:
        for row in combined:
            stream.write(encoded(row) + "\n")
    prepare(OUT / "records.jsonl", RUBRIC, ARTIFACT, research=True)
    exported, manifest = bundle(ARTIFACT)

    # The training split and every existing held-out record must survive verbatim.
    exported_by_id = {r["id"]: r for r in exported}
    for row in base:
        if exported_by_id.get(row["id"]) != row:
            raise ValueError(f"base record changed: {row['id']}")

    # Overlap must be measured after attach() links documents to their words:
    # that link is what actually merges a fresh held-out family into a train one.
    merged: Families
    merged, _, _, merged_prior = lexical_graph(read_wic(CONTEXT_RAW))
    merged.attach(exported + merged_prior)
    root_split = {}
    for row in exported:
        root_split.setdefault(merged.find(row["family_id"]), set()).add(row["split"])
    crossing = sorted(r for r, s in root_split.items() if len(s) > 1)
    train_roots = {r for r, s in root_split.items() if "train" in s}
    overlap = Counter()
    fresh_ids = {row["id"] for row in unique}
    for row in exported:
        if row["id"] not in fresh_ids:
            continue
        if merged.find(row["family_id"]) in train_roots:
            overlap[next(iter(row["questions"].values()))["kind"]] += 1

    summary = dict(
        version=ARTIFACT.name,
        base=BASE.name,
        base_manifest_sha256=base_manifest.get("source_sha256"),
        purpose=(
            "Measure per-kind accuracy. Not a training pack: fresh items may share "
            "lexical families with train, because the base passages cover too wide "
            "a vocabulary for any new held-out synset to stay isolated."
        ),
        fresh_records=len(unique),
        fresh_questions=sum(len(r["questions"]) for r in unique),
        pronunciation_candidates=cmu_candidates,
        syntax_candidates=ud_candidates,
        judgments_by_split=dict(
            Counter(r["split"] for r in exported for _ in r["questions"])
        ),
        by_split_kind={
            split: dict(
                Counter(
                    q["kind"]
                    for r in exported
                    if r["split"] == split
                    for q in r["questions"].values()
                )
            )
            for split in ("train", "development", "calibration", "test")
        },
        fresh_by_kind=dict(
            Counter(q["kind"] for r in unique for q in r["questions"].values())
        ),
        excluded_fresh=dict(excluded),
        fresh_sharing_train_family=dict(overlap),
        root_crossing_splits=len(crossing),
        root_crossing_examples=crossing[:8],
        root_crossing_note=(
            "These roots merge a train document with a held-out word through shared "
            "passage vocabulary. prepare() validates literal family_id only, which "
            "stays clean; this stricter root view is what audit_family_isolation.py "
            "reports, so the count is stated rather than hidden."
        ),
        grammaticality_caveat=(
            "BLiMP construction families already held by the training pack were "
            "skipped, so these items come only from constructions the pack never saw. "
            "A high score here is therefore not memorisation of a trained construction."
        ),
        not_generated=[],
        ud_note=(
            "UD items use the EWT basic dependency relation as the label. A record "
            "whose lexical family or source document is already assigned to the "
            "training pack is skipped, so these sentences are new to the model."
        ),
        adjudication="source_derived or source_labeled only; not independently adjudicated",
        status="prepared_not_trained",
    )
    write(OUT / "summary.json", summary)
    write(
        OUT / "source-manifest.json",
        dict(
            files={
                str(p): digest(p)
                for p in (CMU, BLIMP, BASE / "manifest.json", RUBRIC, Path(__file__))
            },
            output_manifest_sha256=digest(ARTIFACT / "manifest.json"),
        ),
    )
    print(encoded(summary))


if __name__ == "__main__":
    main()
