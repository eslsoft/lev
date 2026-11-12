"""Freeze contextual research data without reusing reserved lexical targets.

WiC targets use lexical closure; W&I asks passage-level correction acceptability,
so its independence unit is the author/document, not incidental passage tokens.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

from audit_family_isolation import Families
from vocabulary_wiktionary import load_supplement
from lev.dataset import prepare
from lev.io import digest, encoded, jsonl, write

VERSION = "en-context-002"
RAW = Path("data/raw/en-context-001")
LEXICAL_RAW = Path("data/raw/en-research-20261021-a")
SUPPLEMENT = Path("data/raw/en-us-vocab-35000-v1/wiktionary-supplement")
BENCHMARK = Path("artifacts/en-us-vocab-35000-v3")
PRIOR = [Path(f"data/research/en-general-{v}/records.jsonl") for v in ("002", "005")]
RUBRIC = Path("rubrics/en-general-research-v1.json")


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def read_wic(raw):
    rows = []
    with zipfile.ZipFile(raw / "wic.zip") as archive:
        for split in ("train", "dev", "test"):
            data = archive.read(f"{split}/{split}.data.txt").decode().splitlines()
            gold = archive.read(f"{split}/{split}.gold.txt").decode().splitlines()
            if len(data) != len(gold):
                raise ValueError("WiC data/gold length mismatch")
            for line, (text, label) in enumerate(zip(data, gold), 1):
                word, pos, indices, a, b = text.split("\t")
                offsets = [int(i) for i in indices.split("-")]
                if label not in {"T", "F"} or pos not in {"N", "V"}:
                    raise ValueError("unknown WiC label/POS")
                if len(offsets) != 2 or any(
                    not 0 <= i < len(s.split()) for i, s in zip(offsets, (a, b))
                ):
                    raise ValueError("WiC target offset out of bounds")
                rows.append(
                    dict(
                        source="wic",
                        locator=f"{split}/{split}.data.txt:line={line}",
                        native_split=split,
                        target=word,
                        pos=pos,
                        indices=offsets,
                        contexts=[a, b],
                        label=label == "T",
                    )
                )
    return rows


def corrected(text, edits):
    """Apply simultaneous source character edits, rejecting ambiguous spans."""
    previous = 0
    pieces = []
    for start, end, replacement in edits:
        if replacement is None:
            raise ValueError("unresolved_edit")
        if type(start) is not int or type(end) is not int:
            raise ValueError("invalid_edit_offsets")
        if not (previous <= start <= end <= len(text)):
            raise ValueError("overlapping_or_invalid_edits")
        if not isinstance(replacement, str):
            raise ValueError("invalid_edit_text")
        pieces.extend((text[previous:start], replacement))
        previous = end
    pieces.append(text[previous:])
    return "".join(pieces)


def read_wi(raw):
    rows = []
    with tarfile.open(raw / "wi-locness.tar.gz") as archive:
        for name in sorted(archive.getnames()):
            if not name.startswith("wi+locness/json/") or not name.endswith(".json"):
                continue
            for line, text in enumerate(
                archive.extractfile(name).read().decode().splitlines(), 1
            ):
                essay = json.loads(text)
                row = dict(
                    source="wi",
                    locator=f"{name}:line={line}",
                    native_split="train" if ".train." in name else "dev",
                    level=Path(name).name.split(".")[0],
                    essay=essay,
                )
                row["issue"] = None
                if row["level"] == "N":
                    row["issue"] = "locness_not_included_restricted_redistribution"
                elif not essay.get("userid"):
                    row["issue"] = "missing_author_identity"
                elif len(essay["edits"]) != 1:
                    row["issue"] = "multiple_annotators_require_adjudication"
                if len(essay["edits"]) == 1:
                    try:
                        row["corrected"] = corrected(
                            essay["text"], essay["edits"][0][1]
                        )
                        if row["issue"] is not None:
                            pass
                        elif max(len(essay["text"]), len(row["corrected"])) > 16000:
                            row["issue"] = "passage_over_16000_characters"
                        elif not normalized(essay["text"]) or not normalized(
                            row["corrected"]
                        ):
                            row["issue"] = "empty_passage"
                        elif row["corrected"] != essay["text"] and normalized(
                            row["corrected"]
                        ) == normalized(essay["text"]):
                            row["issue"] = "only_case_or_whitespace_edits"
                    except ValueError as error:
                        row["issue"] = row["issue"] or str(error)
                rows.append(row)
    return rows


def spelling_edges(directory):
    """Some Kaikki spelling redirects lack alt_of; accept only explicit linked glosses."""
    edges = []
    for file in sorted(directory.glob("*.jsonl")):
        if file.name == "downloads.jsonl":
            continue
        for ei, entry in enumerate(jsonl(file)):
            for si, sense in enumerate(entry.get("senses", [])):
                for gloss in sense.get("glosses", []):
                    match = re.fullmatch(
                        r"(?:[A-Za-z, -]+ )?(?:standard|alternative|variant|US) spelling of ([a-zA-Z]+)\.",
                        gloss,
                        re.IGNORECASE,
                    )
                    if match and any(
                        link[0].casefold() == match[1].casefold()
                        for link in sense.get("links", [])
                    ):
                        edges.append(
                            dict(
                                word=entry["word"].casefold(),
                                alias=match[1].casefold(),
                                source=str(file),
                                entry=ei,
                                sense=si,
                                gloss=gloss,
                            )
                        )
    return edges


def lexical_graph(wic, *, include_aliases=True):
    families = Families(LEXICAL_RAW)
    supplement, _ = load_supplement(SUPPLEMENT)
    for word, info in supplement.items():
        families.union(
            [families.word(word), *[families.word(f) for f in info["forms"]]]
        )
    if include_aliases:
        aliases, _ = load_supplement(RAW / "lexical-aliases")
        for word, info in aliases.items():
            families.union(
                [families.word(word), *[families.word(f) for f in info["forms"]]]
            )
    for directory in (
        (SUPPLEMENT, RAW / "lexical-aliases") if include_aliases else (SUPPLEMENT,)
    ):
        for edge in spelling_edges(directory):
            families.union([families.word(edge["word"]), families.word(edge["alias"])])
    prior = [r for path in PRIOR for r in jsonl(path)]
    families.attach(prior)
    # Source-designated target tokens add observed inflections before reservation.
    # Identical contexts connect all their targets; no sentence is shared across splits.
    for row in wic:
        families.union(
            [
                families.word(row["target"]),
                *[
                    families.word(s.split()[i])
                    for s, i in zip(row["contexts"], row["indices"])
                ],
                *["wic-context:" + key(normalized(s)) for s in row["contexts"]],
            ]
        )
    reserved = {
        families.find(families.word(w["word"]))
        for w in jsonl(BENCHMARK / "words.jsonl")
    }
    reserved |= {families.find(r["family_id"]) for r in prior if r["split"] != "train"}
    prior_train = {
        families.find(r["family_id"]) for r in prior if r["split"] == "train"
    }
    return families, reserved, prior_train, prior


def author_groups(rows):
    """Close shared authors, original texts and corrected variants before splitting."""
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(items):
        roots = sorted({find(i) for i in items})
        for root in roots[1:]:
            parents[root] = roots[0]

    for row in rows:
        essay = row["essay"]
        row["group"] = "essay:" + str(essay["id"])
        anchors = [row["group"], "text:" + key(normalized(essay["text"]))]
        if essay.get("userid"):
            anchors.append("author:" + str(essay["userid"]))
        if "corrected" in row:
            anchors.append("text:" + key(normalized(row["corrected"])))
        union(anchors)
    for row in rows:
        row["group"] = find(row["group"])
    return rows


def holdout_split(group):
    return ("development", "calibration", "test")[int(key(group)[:8], 16) % 3]


def record(
    identity, family, split, state, kind, claim, label, source, locator, rationale
):
    rid = key(VERSION + ":" + identity)[:24]
    return dict(
        id=rid,
        family_id=family,
        case_id=rid,
        split=split,
        language="en",
        scope="partial",
        origin="curated",
        rubric_version="en-general-research-v1",
        state=state,
        questions={
            "q0": dict(
                kind=kind,
                claim=claim,
                label=label,
                review=dict(
                    status="source_labeled"
                    if kind == "contextual_sense"
                    else "source_derived",
                    reviewer="source:WiC"
                    if kind == "contextual_sense"
                    else "source:Cambridge-W&I",
                    rationale=rationale,
                    evidence=[dict(reference=str(RAW / source), locator=locator)],
                ),
            )
        },
    )


def select(wic, wi, families, reserved, prior_train, prior):
    records, events = [], []
    native = defaultdict(set)
    for row in wic:
        row["group"] = families.find(families.word(row["target"]))
        native[row["group"]].add(row["native_split"])
    seen = set()
    for row in wic:
        group, reason = row["group"], None
        splits = native[group]
        if group in reserved:
            reason = "reserved_vocabulary_or_historical_holdout_family"
        elif row["native_split"] == "train" and splits - {"train"}:
            reason = "native_holdout_family"
        elif row["native_split"] == "dev" and "test" in splits:
            reason = "native_test_family"
        elif row["native_split"] != "train" and group in prior_train:
            reason = "prior_training_family"
        identity = (
            row["target"].casefold(),
            tuple(row["contexts"]),
            tuple(row["indices"]),
        )
        if reason is None and identity in seen:
            reason = "duplicate_context_pair"
        split = (
            "train"
            if row["native_split"] == "train"
            else (
                "test"
                if row["native_split"] == "test"
                else ("development", "calibration")[int(key(group)[:8], 16) % 2]
            )
        )
        event = dict(
            source="wic",
            locator=row["locator"],
            target=row["target"],
            family_id=group,
            native_split=row["native_split"],
            status="excluded" if reason else "included",
            reason=reason,
            record_ids=[],
        )
        if reason is None:
            seen.add(identity)
            rec = record(
                row["locator"],
                group,
                split,
                dict(
                    target_word=row["target"],
                    part_of_speech=row["pos"],
                    contexts=row["contexts"],
                    target_token_indices_zero_based=row["indices"],
                    convention="English; use the marked occurrences and the WiC sense distinction.",
                ),
                "contextual_sense",
                "Is the marked word used with the same meaning in both contexts?",
                row["label"],
                "wic.zip",
                row["locator"],
                "Published WiC T/F sense-equivalence label, aligned by line to the native gold file; not independently adjudicated by Lev.",
            )
            records.append(rec)
            event["record_ids"].append(rec["id"])
        events.append(event)
    author_groups(wi)
    holdout_groups = {r["group"] for r in wi if r["native_split"] != "train"}
    # Prior sentence text is conservatively reserved even when previously train.
    previous_texts = set()
    for row in prior:
        state = row["state"]
        if isinstance(state, dict):
            for field in ("text", "sentence", "context", "passage"):
                if isinstance(state.get(field), str):
                    previous_texts.add(normalized(state[field]))
    conflict_groups = {
        r["group"]
        for r in wi
        if normalized(r["essay"]["text"]) in previous_texts
        or normalized(r.get("corrected", "")) in previous_texts
    }
    seen_texts = {}
    for row in wi:
        reason = row["issue"]
        if (
            reason is None
            and row["native_split"] == "train"
            and row["group"] in holdout_groups
        ):
            reason = "native_holdout_author_or_duplicate_document"
        if reason is None and row["group"] in conflict_groups:
            reason = "historical_passage_overlap"
        event = dict(
            source="wi",
            locator=row["locator"],
            native_split=row["native_split"],
            essay_id=row["essay"]["id"],
            author_id=row["essay"].get("userid"),
            family_id="wi:" + row["group"],
            status="excluded" if reason else "included",
            reason=reason,
            record_ids=[],
            duplicate_variants=[],
        )
        if reason is None:
            split = (
                "train"
                if row["native_split"] == "train"
                else holdout_split(row["group"])
            )
            original, fixed = row["essay"]["text"], row["corrected"]
            variants = [("original", original, original == fixed)]
            if original != fixed:
                variants.append(("corrected", fixed, True))
            for variant, text, label in variants:
                signature = normalized(text)
                if signature in seen_texts:
                    previous = seen_texts[signature]
                    if previous["label"] != label:
                        raise ValueError(
                            "contradictory W&I passage judgments; preserve and adjudicate before export"
                        )
                    event["duplicate_variants"].append(
                        dict(variant=variant, record_id=previous["id"])
                    )
                    continue
                rec = record(
                    row["locator"] + ":" + variant,
                    event["family_id"],
                    split,
                    dict(
                        passage=text,
                        convention="Written English under the Cambridge Write & Improve annotator's correction standard, including grammar, word choice, spelling and punctuation. Judge the passage, not the truth of its subject matter. Regional English variants are not inherently errors.",
                    ),
                    "passage_correction_acceptability",
                    "Does this passage satisfy the source annotation's correction requirements without further changes?",
                    label,
                    "wi-locness.tar.gz",
                    row["locator"] + ":" + variant,
                    "One human annotator's explicit character edits. Original is false only when edits change it; corrected is true relative to that annotation after every edit is applied. No claim of universally flawless prose or independent Lev adjudication. Unresolved corrections excluded.",
                )
                records.append(rec)
                event["record_ids"].append(rec["id"])
                seen_texts[signature] = dict(id=rec["id"], label=label)
            if not event["record_ids"]:
                event.update(status="excluded", reason="duplicate_passage_variants")
        events.append(event)
    return records, events


def source_files():
    files = [
        *RAW.rglob("*"),
        LEXICAL_RAW / "wordnet.xml.gz",
        LEXICAL_RAW / "ud.conllu",
        *(LEXICAL_RAW / "wiktionary").glob("*.jsonl"),
        *SUPPLEMENT.glob("*.jsonl"),
        *PRIOR,
        BENCHMARK / "words.jsonl",
        BENCHMARK / "manifest.json",
        RUBRIC,
        Path(__file__),
        Path("scripts/audit_family_isolation.py"),
        Path("scripts/vocabulary_wiktionary.py"),
        Path("scripts/fetch_context_aliases.py"),
        Path("scripts/audit_context_corpus.py"),
        Path("tests/test_context.py"),
        Path("tests/test_data.py"),
        *Path("src/lev").glob("*.py"),
    ]
    return {str(p): digest(p) for p in sorted(set(files)) if p.is_file()}


def build(out, artifact):
    if out.exists() or artifact.exists():
        raise ValueError("new output directories required")
    for event in jsonl(RAW / "downloads.jsonl"):
        if (
            event["status"] != "downloaded"
            or digest(RAW / event["file"]) != event["sha256"]
        ):
            raise ValueError("raw source download failed or changed")
    for event in jsonl(RAW / "lexical-aliases/downloads.jsonl"):
        if (
            event["status"] != "downloaded"
            or digest(RAW / "lexical-aliases" / (event["word"] + ".jsonl"))
            != event["sha256"]
        ):
            raise ValueError("alias evidence missing or changed")
    wic, wi = read_wic(RAW), read_wi(RAW)
    families, reserved, prior_train, prior = lexical_graph(wic)
    records, events = select(wic, wi, families, reserved, prior_train, prior)
    out.mkdir(parents=True)
    for name, rows in (("records.jsonl", records), ("selection.jsonl", events)):
        with (out / name).open("x") as stream:
            for row in rows:
                stream.write(encoded(row) + "\n")
    stats = {}
    for split in ("train", "development", "calibration", "test"):
        subset = [r for r in records if r["split"] == split]
        words = sorted(
            {
                r["state"]["target_word"].casefold()
                for r in subset
                if "target_word" in r["state"]
            }
        )
        stats[split] = dict(
            requests=len(subset),
            judgments=sum(len(r["questions"]) for r in subset),
            explicit_target_words=len(words),
            target_words=words,
            groups=len({r["family_id"] for r in subset}),
            kinds=dict(
                Counter(q["kind"] for r in subset for q in r["questions"].values())
            ),
            labels=dict(
                Counter(
                    str(q["label"]).lower()
                    for r in subset
                    for q in r["questions"].values()
                )
            ),
        )
    summary = dict(
        version=VERSION,
        splits=stats,
        selection=dict(
            Counter(f"{e['source']}:{e['reason'] or 'included'}" for e in events)
        ),
        candidate_units=len(events),
        included_units=sum(e["status"] == "included" for e in events),
        lexical_reservation="All 35000 benchmark targets and historical holdout families, after observed WiC inflection/context closure.",
        passage_reservation="Author/document plus normalized original/corrected duplicates; native dev never trained. Incidental passage vocabulary is NOT lexical target coverage.",
        scope="All partial. Research only; no independently adjudicated production labels; no model execution.",
        gaps=[
            "35k contextual lexical coverage remains mostly excluded by frozen-test reservation",
            "No separately adjudicated collocation or morphology challenge set",
            "W&I annotations do not guarantee universal acceptability",
            "Source licenses restrict use to non-commercial research; no publication or deployment authorized by this export",
        ],
    )
    write(out / "summary.json", summary)
    write(
        out / "source-manifest.json",
        dict(
            files=source_files(),
            records_sha256=digest(out / "records.jsonl"),
            selection_sha256=digest(out / "selection.jsonl"),
        ),
    )
    prepare(out / "records.jsonl", RUBRIC, artifact, research=True)
    print(encoded(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("data/research/en-context-002")
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/en-context-002")
    )
    args = parser.parse_args()
    build(args.out, args.artifact)
