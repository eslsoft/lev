"""Check every exported judgment against original labels/edits and isolation units.

Character edits are reconstructed right-to-left independently of the builder's
left-to-right splice implementation. This is data validation, not model execution.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tarfile
import zipfile

from audit_family_isolation import Families
from build_context_corpus import (
    BENCHMARK,
    LEXICAL_RAW,
    PRIOR,
    RAW,
    SUPPLEMENT,
    spelling_edges,
)
from lev.dataset import bundle
from lev.io import digest, jsonl, read, write
from vocabulary_wiktionary import load_supplement


def norm(text):
    return " ".join(text.casefold().split())


def apply_reference(text, edits):
    boundary = len(text)
    result = text
    for start, end, value in reversed(edits):
        if value is None or not 0 <= start <= end <= boundary:
            raise ValueError("unresolved/overlapping reference edits")
        result = result[:start] + value + result[end:]
        boundary = start
    return result


def audit(artifact, provenance):
    manifest = read(provenance / "source-manifest.json")
    for file, expected in manifest["files"].items():
        if digest(file) != expected:
            raise ValueError("source changed: " + file)
    for name in ("records", "selection"):
        if digest(provenance / (name + ".jsonl")) != manifest[name + "_sha256"]:
            raise ValueError("provenance changed: " + name)
    records, frozen = bundle(artifact)
    if not frozen["research_only"] or frozen["example_only"]:
        raise ValueError("not a research bundle")
    if sorted(jsonl(provenance / "records.jsonl"), key=lambda r: r["id"]) != records:
        raise ValueError("frozen records differ from provenance")
    sources = {}
    with zipfile.ZipFile(RAW / "wic.zip") as archive:
        for split in ("train", "dev", "test"):
            texts = archive.read(f"{split}/{split}.data.txt").decode().splitlines()
            labels = archive.read(f"{split}/{split}.gold.txt").decode().splitlines()
            assert len(texts) == len(labels)
            for line, (text, label) in enumerate(zip(texts, labels), 1):
                word, pos, offsets, a, b = text.split("\t")
                sources[("wic", f"{split}/{split}.data.txt:line={line}")] = dict(
                    word=word,
                    pos=pos,
                    indices=list(map(int, offsets.split("-"))),
                    contexts=[a, b],
                    label=label == "T",
                    native=split,
                )
    with tarfile.open(RAW / "wi-locness.tar.gz") as archive:
        for name in archive.getnames():
            if not name.startswith("wi+locness/json/") or not name.endswith(".json"):
                continue
            for line, text in enumerate(
                archive.extractfile(name).read().decode().splitlines(), 1
            ):
                essay = json.loads(text)
                fixed = None
                if len(essay["edits"]) == 1:
                    try:
                        fixed = apply_reference(essay["text"], essay["edits"][0][1])
                    except ValueError:
                        pass
                sources[("wi", f"{name}:line={line}")] = dict(
                    essay=essay,
                    fixed=fixed,
                    native="train" if ".train." in name else "dev",
                    level=Path(name).name.split(".")[0],
                )
    events = jsonl(provenance / "selection.jsonl")
    event_map = {(e["source"], e["locator"]): e for e in events}
    if len(event_map) != len(events) or set(event_map) != set(sources):
        raise ValueError("missing or duplicated candidate accounting")
    by_id = {r["id"]: r for r in records}
    mapped = [rid for e in events for rid in e["record_ids"]]
    if len(mapped) != len(set(mapped)) or set(mapped) != set(by_id):
        raise ValueError("candidate-to-record alignment is incomplete")

    families = Families(LEXICAL_RAW)
    for directory in (SUPPLEMENT, RAW / "lexical-aliases"):
        items, _ = load_supplement(directory)
        for word, info in items.items():
            families.union(
                [families.word(word), *[families.word(f) for f in info["forms"]]]
            )
    for directory in (SUPPLEMENT, RAW / "lexical-aliases"):
        for edge in spelling_edges(directory):
            families.union([families.word(edge["word"]), families.word(edge["alias"])])
    prior = [r for file in PRIOR for r in jsonl(file)]
    families.attach(prior)
    # Context strings themselves are union nodes here (builder uses their hashes).
    for (kind, _), source in sources.items():
        if kind == "wic":
            families.union(
                [
                    families.word(source["word"]),
                    *[
                        families.word(s.split()[i])
                        for s, i in zip(source["contexts"], source["indices"])
                    ],
                    *["audit-context:" + norm(s) for s in source["contexts"]],
                ]
            )
        else:
            essay = source["essay"]
            anchors = ["wi-doc:" + str(essay["id"]), "wi-text:" + norm(essay["text"])]
            if essay.get("userid"):
                anchors.append("wi-user:" + str(essay["userid"]))
            if source["fixed"] is not None:
                anchors.append("wi-text:" + norm(source["fixed"]))
            families.union(anchors)
    reserved = {
        families.find(families.word(w["word"]))
        for w in jsonl(BENCHMARK / "words.jsonl")
    }
    reserved |= {families.find(r["family_id"]) for r in prior if r["split"] != "train"}
    trained = {families.find(r["family_id"]) for r in prior if r["split"] == "train"}
    native_groups = defaultdict(set)
    for (kind, _), source in sources.items():
        anchor = (
            families.word(source["word"])
            if kind == "wic"
            else "wi-doc:" + str(source["essay"]["id"])
        )
        native_groups[families.find(anchor)].add(source["native"])
    current_groups = defaultdict(set)
    labels_checked = 0
    for key, event in event_map.items():
        source = sources[key]
        assert event["native_split"] == source["native"]
        if event["status"] == "excluded":
            assert event["reason"] and not event["record_ids"]
            continue
        assert (
            event["status"] == "included"
            and event["reason"] is None
            and event["record_ids"]
        )
        for rid in event["record_ids"]:
            row = by_id[rid]
            assert row["scope"] == "partial" and set(row["questions"]) == {"q0"}
            question, state = row["questions"]["q0"], row["state"]
            assert row["family_id"] == event["family_id"]
            evidence = question["review"]["evidence"]
            assert len(evidence) == 1
            if key[0] == "wic":
                assert set(state) == {
                    "target_word",
                    "part_of_speech",
                    "contexts",
                    "target_token_indices_zero_based",
                    "convention",
                }
                assert state["target_word"] == source["word"]
                assert state["part_of_speech"] == source["pos"]
                assert state["contexts"] == source["contexts"]
                assert state["target_token_indices_zero_based"] == source["indices"]
                assert question["label"] is source["label"]
                assert (
                    question["claim"]
                    == "Is the marked word used with the same meaning in both contexts?"
                )
                assert question["kind"] == "contextual_sense"
                assert question["review"]["status"] == "source_labeled"
                assert evidence[0] == dict(
                    reference=str(RAW / "wic.zip"), locator=key[1]
                )
                group = families.find(families.word(source["word"]))
                assert group not in reserved
                if row["split"] != "train":
                    assert group not in trained
                if source["native"] == "test":
                    assert row["split"] == "test"
                if source["native"] == "dev":
                    assert (
                        row["split"] in {"development", "calibration"}
                        and "test" not in native_groups[group]
                    )
            else:
                assert source["level"] != "N" and source["essay"].get("userid")
                assert source["fixed"] is not None
                assert set(state) == {"passage", "convention"}
                locator, variant = evidence[0]["locator"].rsplit(":", 1)
                assert locator == key[1] and variant in {"original", "corrected"}
                assert evidence[0]["reference"] == str(RAW / "wi-locness.tar.gz")
                expected = (
                    source["essay"]["text"]
                    if variant == "original"
                    else source["fixed"]
                )
                assert state["passage"] == expected
                assert question["label"] is (expected == source["fixed"])
                assert (
                    question["claim"]
                    == "Does this passage satisfy the source annotation's correction requirements without further changes?"
                )
                assert question["kind"] == "passage_correction_acceptability"
                assert question["review"]["status"] == "source_derived"
                group = families.find("wi-doc:" + str(source["essay"]["id"]))
            current_groups[group].add(row["split"])
            if row["split"] == "train":
                assert source["native"] == "train" and native_groups[group] == {"train"}
            else:
                assert source["native"] != "train"
            labels_checked += 1
    if any(len(splits) != 1 for splits in current_groups.values()):
        raise ValueError("current split leakage")
    summary = read(provenance / "summary.json")
    for split, values in summary["splits"].items():
        subset = [r for r in records if r["split"] == split]
        assert values["requests"] == values["judgments"] == len(subset)
        assert values["target_words"] == sorted(
            {
                r["state"]["target_word"].casefold()
                for r in subset
                if "target_word" in r["state"]
            }
        )
        assert values["explicit_target_words"] == len(values["target_words"])
    return dict(
        status="passed",
        judgments_checked=labels_checked,
        candidates_accounted=len(events),
        lexical_and_author_components=len(current_groups),
        splits=dict(Counter(r["split"] for r in records)),
        source_files_checked=len(manifest["files"]),
        frozen_native_export="bundle() validated every native record against provenance",
        model_execution=False,
        benchmark_manifest_sha256=digest(BENCHMARK / "manifest.json"),
        limitations="Source truth and known family/author/document isolation only; no independent human adjudication, unseen-construction guarantee or model accuracy.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, default=Path("artifacts/en-context-002"))
    p.add_argument(
        "--provenance", type=Path, default=Path("data/research/en-context-002")
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.artifact, a.provenance)
    write(a.out, result)
    print(json.dumps(result))
