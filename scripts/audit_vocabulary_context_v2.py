"""Re-read WiC source lines and verify frozen contextual labels and export content."""

import argparse
from pathlib import Path
import zipfile

from audit_corroborated_families import verified_relations
from build_context_corpus import RAW, lexical_graph, read_wic
from lev.dataset import bundle, to_kev, validate
from lev.io import digest, encoded, jsonl, read, write
from lev.vocabulary import load_words, verify_bundle


def audit(dataset):
    verify_bundle(dataset)
    words = load_words(dataset)
    sources = read(dataset / "source-manifest.json")
    for path, expected in sources["files"].items():
        if digest(path) != expected:
            raise ValueError("source changed: " + path)
    source = {}
    with zipfile.ZipFile("data/raw/en-context-001/wic.zip") as archive:
        for split in ("train", "dev", "test"):
            data = archive.read(f"{split}/{split}.data.txt").decode().splitlines()
            labels = archive.read(f"{split}/{split}.gold.txt").decode().splitlines()
            for index, (line, gold) in enumerate(zip(data, labels, strict=True), 1):
                source[f"{split}/{split}.data.txt:line={index}"] = (line, gold)
    events = jsonl(dataset / "selection-audit.jsonl")
    if len(events) != len(source) or {e["locator"] for e in events} != set(source):
        raise ValueError("selection audit omits or duplicates source rows")
    included = {e["locator"] for e in events if e["status"] == "included"}
    records = jsonl(dataset / "records.jsonl")
    families, _, _, prior = lexical_graph(read_wic(RAW))
    for name in ("001", "002"):
        audit_path = Path(f"runs/en-morphology-001-dictionary-audit-{name}")
        if str(audit_path / "manifest.json") not in sources["files"]:
            raise ValueError("missing frozen corroboration dependency")
        for relation in verified_relations(audit_path):
            families.union(
                [families.word(relation["target"]), families.word(relation["form"])]
            )
    joint, _ = bundle("artifacts/en-joint-001")
    blocked = {families.find(r["family_id"]) for r in prior + joint}
    rubric = read(dataset / "rubric.json")
    validate(records, rubric)
    found = set()
    for r in records:
        q = r["questions"]["q0"]
        locator = q["review"]["evidence"][0]["locator"]
        if locator in found or locator not in included:
            raise ValueError("record does not match included source population")
        found.add(locator)
        text, gold = source[locator]
        word, pos, offsets, first, second = text.split("\t")
        state = r["state"]
        family = families.find(families.word(word))
        if family in blocked or r["family_id"] != family:
            raise ValueError(
                "context target touches prior Lev family or has wrong family id"
            )
        if (
            q["label"] is not (gold == "T")
            or state["headword"] != word.casefold()
            or state["target_word"] != word
            or state["part_of_speech"] != pos
            or state["contexts"] != [first, second]
            or state["target_token_indices_zero_based"]
            != [int(x) for x in offsets.split("-")]
            or r["split"] != "test"
            or r["scope"] != "partial"
        ):
            raise ValueError("context or label differs from source")
    if found != included:
        raise ValueError("included source missing from export")
    if jsonl(dataset / "test.jsonl") != [to_kev(r, rubric) for r in records]:
        raise ValueError("native input differs from source records")
    expected = {r["id"]: r["state"]["headword"] for r in records}
    assigned = [(rid, w["word"]) for w in words for rid in w["record_ids"]]
    if len(assigned) != len(expected) or dict(assigned) != expected:
        raise ValueError("word/request registry mismatch")
    return dict(
        status="passed",
        source_rows=len(source),
        judgments=len(records),
        target_words=len(words),
        source_files_checked=len(sources["files"]),
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
        model_executed=False,
        known_family_isolation="passed against all historical and joint records",
        auditor_sha256=digest(__file__),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    write(args.out, result)
    print(encoded(result))
