"""Expand only established training constructions, preserving every held-out byte."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import zipfile

from build_context_corpus import PRIOR, key, normalized
from lev.dataset import bundle, prepare
from lev.io import digest, encoded, jsonl, write

VERSION = "en-joint-002"
SOURCE = Path("data/raw/en-general-003/blimp.zip")
SOURCE_SHA = "172a1a714d72e72a89e8384dd531483f3d0f8a9aee3d33dccaf7992d8cecbb73"
PARENT = Path("artifacts/en-joint-001")
RUBRIC = Path("rubrics/en-general-research-v1.json")
QUERIES = (
    ("Is the sentence grammatically acceptable under the stated English task?", False),
    (
        "Does the sentence satisfy the grammatical requirements of the stated task?",
        False,
    ),
    ("Is the sentence grammatically unacceptable under the stated English task?", True),
    (
        "Does the sentence violate the grammatical requirements of the stated task?",
        True,
    ),
)


def state_texts(value):
    if isinstance(value, str):
        return {normalized(value)}
    if isinstance(value, dict):
        return set().union(*(state_texts(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(state_texts(v) for v in value))
    return set()


def select_pairs(rows, name, allowed, used, limit):
    events, selected = [], []
    for item in sorted(
        rows, key=lambda r: key(VERSION + ":" + name + ":" + str(r["pairID"]))
    ):
        term = item["linguistics_term"]
        good, bad = normalized(item["sentence_good"]), normalized(item["sentence_bad"])
        reason = (
            "nontraining_construction"
            if "construction:" + term not in allowed
            else "identical_good_bad"
            if good == bad
            else "previously_used_sentence"
            if good in used or bad in used
            else "fixed_pair_quota"
            if len(selected) >= limit
            else None
        )
        locator = name + ":" + str(item["pairID"])
        events.append(
            dict(
                locator=locator,
                construction=term,
                status="excluded" if reason else "included",
                reason=reason,
            )
        )
        if reason is None:
            selected.append((locator, item))
            used.update((good, bad))
    return selected, events


def pair_records(locator, item):
    claim, invert = QUERIES[int(key(locator)[:8], 16) % len(QUERIES)]
    result = []
    for field, truth in (("sentence_good", True), ("sentence_bad", False)):
        identity = locator + ":" + field
        rid = key(VERSION + ":" + identity)[:24]
        result.append(
            dict(
                id=rid,
                case_id=rid,
                family_id="construction:" + item["linguistics_term"],
                split="train",
                language="en",
                scope="partial",
                origin="curated",
                rubric_version="en-general-research-v1",
                state=dict(
                    sentence=item[field],
                    variety="English under the BLiMP grammaticality task; evaluate syntax and the stipulated reading, not event plausibility.",
                ),
                questions={
                    "q0": dict(
                        kind="grammaticality",
                        claim=claim,
                        label=not truth if invert else truth,
                        review=dict(
                            status="source_labeled",
                            reviewer="source:BLiMP",
                            rationale="Published minimal-pair good/bad label"
                            + (
                                " logically inverted for the negative question."
                                if invert
                                else " for the affirmative question."
                            )
                            + " Template-authored source; not independent Lev adjudication.",
                            evidence=[dict(reference=str(SOURCE), locator=identity)],
                        ),
                    )
                },
            )
        )
    return result


def build(out, provenance):
    if out.exists() or provenance.exists():
        raise ValueError("new output directories required")
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError("BLiMP source changed")
    records, _ = bundle(PARENT)
    prior = [r for p in PRIOR for r in jsonl(p)]
    allowed = {
        r["family_id"]
        for r in records
        if r["split"] == "train" and r["family_id"].startswith("construction:")
    }
    blocked = {r["family_id"] for r in records + prior if r["split"] != "train"}
    if allowed & blocked:
        raise ValueError("training construction touches historical holdout")
    used = set().union(*(state_texts(r["state"]) for r in records + prior))
    expanded, events, additions = deepcopy(records), [], []
    with zipfile.ZipFile(SOURCE) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".jsonl") or name.startswith("__MACOSX"):
                continue
            rows = [json.loads(line) for line in archive.read(name).splitlines()]
            selected, audit = select_pairs(rows, name, allowed, used, 100)
            events.extend(audit)
            for locator, item in selected:
                additions.extend(pair_records(locator, item))
    expanded.extend(additions)
    provenance.mkdir(parents=True)
    for name, rows in (("records.jsonl", expanded), ("selection-audit.jsonl", events)):
        with (provenance / name).open("x") as stream:
            for row in rows:
                stream.write(encoded(row) + "\n")
    prepare(provenance / "records.jsonl", RUBRIC, out, research=True)
    for split in ("development", "calibration", "test"):
        if digest(out / (split + ".jsonl")) != digest(PARENT / (split + ".jsonl")):
            raise ValueError("held-out export changed: " + split)
    summary = dict(
        status="prepared_not_trained",
        source_pairs=len(events),
        added_requests=len(additions),
        selected_pairs=sum(e["status"] == "included" for e in events),
        decisions_by_split=dict(
            Counter(r["split"] for r in expanded for _ in r["questions"])
        ),
        requests_by_split=dict(Counter(r["split"] for r in expanded)),
        selection=dict(Counter(e["reason"] or "included" for e in events)),
        new_supervised_word_coverage_claimed=0,
        heldout_exports_byte_identical=True,
    )
    write(provenance / "summary.json", summary)
    write(
        provenance / "source-manifest.json",
        dict(
            files={
                str(p): digest(p)
                for p in (
                    SOURCE,
                    Path("data/raw/en-research-20261021-a/blimp-README"),
                    PARENT / "manifest.json",
                    PARENT / "records.jsonl",
                    RUBRIC,
                    *PRIOR,
                    Path(__file__),
                    Path("docs/experiments/en-joint-002.md"),
                )
            },
            artifact_manifest_sha256=digest(out / "manifest.json"),
        ),
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    print(encoded(build(args.out, args.provenance)))
