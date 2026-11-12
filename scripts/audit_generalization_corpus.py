"""Independently check source-label polarity and frozen data invariants."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

from lev.dataset import bundle
from lev.io import digest, jsonl, read, write

NEGATIVE_PREFIXES = (
    "Would the correct answer be no:",
    "Is a negative answer warranted",
    "Would answering yes",
    "Should this question receive a no",
    "Is the affirmative response",
)


def audit(dataset, previous, extra, out, raw):
    rows, manifest = bundle(dataset)
    prior = {r["id"]: r for r in jsonl(previous)}
    checked = Counter()
    for name, expected in read(previous.parent / "source-manifest.json")[
        "files"
    ].items():
        assert digest(raw / name) == expected, name
    for event in jsonl(extra / "download-events.jsonl"):
        assert event["status"] == "downloaded"
        assert digest(extra / event["file"]) == event["sha256"]
    cmu = (raw / "cmudict.dict").read_text().splitlines()
    lex = ET.parse(gzip.open(raw / "wordnet.xml.gz")).getroot().find("Lexicon")
    synsets = {s.attrib["id"]: s for s in lex.findall("Synset")}
    ud = {}
    for block in (raw / "ud.conllu").read_text().split("\n\n"):
        lines = block.splitlines()
        sid = next(
            (
                line.removeprefix("# sent_id = ")
                for line in lines
                if line.startswith("# sent_id = ")
            ),
            None,
        )
        if sid:
            ud[sid] = [
                line.split("\t")
                for line in lines
                if line and not line.startswith("#") and line.split("\t")[0].isdigit()
            ]
    legacy_grammar = {
        str(p.relative_to(raw)): {str(row["pairID"]): row for row in jsonl(p)}
        for p in (raw / "blimp").glob("*.jsonl")
    }
    wikt = {
        str(p.relative_to(raw)): jsonl(p) for p in (raw / "wiktionary").glob("*.jsonl")
    }
    with zipfile.ZipFile(extra / "blimp.zip") as z:
        grammar = {
            name: {
                str(row["pairID"]): row
                for row in map(json.loads, z.read(name).splitlines())
            }
            for name in z.namelist()
            if name.endswith(".jsonl") and not name.startswith("__MACOSX")
        }
    dialogues = json.loads((extra / "implicatures.json").read_text())["examples"]
    family_splits = {}
    for r in rows:
        assert family_splits.setdefault(r["family_id"], r["split"]) == r["split"]
        if r["id"] in prior:
            assert r["split"] == prior[r["id"]]["split"] == "train"
            assert (
                r["state"] == prior[r["id"]]["state"]
                and r["questions"] == prior[r["id"]]["questions"]
            )
            checked["unchanged_replay_requests"] += 1
        for q in r["questions"].values():
            invert = q["claim"].startswith(NEGATIVE_PREFIXES)
            ev = q["review"]["evidence"][0]
            if ev["reference"].endswith("/blimp.zip"):
                name, pair, field = ev["locator"].rsplit(":", 2)
                original = field == "sentence_good"
                assert grammar[name][pair][field] == r["state"]["sentence"]
                assert q["label"] == (original != invert)
                checked["blimp_decisions"] += 1
            elif ev["reference"].startswith("blimp/"):
                pair, field = ev["locator"].rsplit(":", 1)
                item = legacy_grammar[ev["reference"]][pair]
                assert r["state"]["sentence"] == item[field]
                assert q["label"] == ((field == "sentence_good") != invert)
                checked["blimp_decisions"] += 1
            elif ev["reference"].endswith("/implicatures.json"):
                i = int(re.match(r"examples\[(\d+)\]", ev["locator"]).group(1))
                item = dialogues[i]
                assert item["input"] == r["state"]["dialogue"]
                assert q["label"] == (bool(item["target_scores"]["yes"]) != invert)
                assert r["split"] == "test"
                checked["evaluation_only_implicatures"] += 1
            elif q["kind"] == "pronunciation":
                source = cmu[int(ev["locator"]) - 1].split()
                assert source[0] == r["state"]["word"]
                assert source[1:] == r["state"]["pronunciation"].split()
                vowels = [
                    p
                    for p in r["state"]["pronunciation"].split()
                    if re.fullmatch(r"[A-Z]+[012]", p)
                ]
                if "syllabic nuclei" in q["claim"]:
                    n = int(re.search(r"exactly (\d+)", q["claim"]).group(1))
                    original = n == len(vowels)
                else:
                    n = int(re.search(r"syllable (\d+)", q["claim"]).group(1))
                    original = vowels[n - 1].endswith("1")
                assert q["label"] == (original != invert)
                checked["pronunciation_decisions"] += 1
            elif ev["reference"] == "wordnet.xml.gz":
                child, parent = (synsets[sid] for sid in ev["locator"].split("->"))
                assert any(
                    rel.attrib["relType"] == "hypernym"
                    and rel.attrib["target"] == parent.attrib["id"]
                    for rel in child.findall("SynsetRelation")
                )
                a, b = child.findtext("Definition"), parent.findtext("Definition")
                assert a != b, "Indistinguishable sense definitions"
                state_pair = (r["state"]["sense_A"], r["state"]["sense_B"])
                assert state_pair in ((a, b), (b, a))
                forward = state_pair == (a, b)
                if q["claim"].endswith("Does sense A denote a subtype of sense B?"):
                    original = forward
                else:
                    assert q["claim"].endswith(
                        "Is sense B more specific than sense A in this is-a relation?"
                    )
                    original = not forward
                assert q["label"] == (original != invert)
                checked["semantic_relation_decisions"] += 1
            elif ev["reference"] == "ud.conllu":
                sid, index = ev["locator"].split(":token=")
                tokens = ud[sid]
                assert r["state"]["target_index"] == int(index)
                assert r["state"]["tokens"] == [
                    {"index": int(t[0]), "text": t[1]} for t in tokens
                ]
                target = next(t for t in tokens if t[0] == index)
                roles = {
                    "nsubj": "nominal subject",
                    "obj": "direct object",
                    "amod": "adjectival modifier",
                    "advmod": "adverbial modifier",
                }
                claimed = re.search(r"function as a (.+)\?", q["claim"]).group(1)
                assert claimed in roles.values()
                assert q["label"] == ((claimed == roles[target[7]]) != invert)
                checked["syntax_decisions"] += 1
            elif ev["reference"].startswith("wiktionary/"):
                index = int(
                    re.fullmatch(r"entry=(\d+):sense=0", ev["locator"]).group(1)
                )
                entry = wikt[ev["reference"]][index]
                assert r["state"]["sense_description"] == " / ".join(
                    entry["senses"][0]["glosses"]
                )
                assert r["state"]["headword"] == Path(ev["reference"]).stem
                pos = {"adj": "adjective", "adv": "adverb"}.get(
                    entry["pos"], entry["pos"]
                )
                claimed = re.search(
                    r"Is (noun|verb|adjective|adverb) the part of speech", q["claim"]
                ).group(1)
                assert q["label"] == ((claimed == pos) != invert)
                checked["lexical_analysis_decisions"] += 1
            else:
                raise ValueError("Unverified source: " + ev["reference"])
            checked["all_source_decisions"] += 1
    write(
        out,
        dict(
            dataset_sha256=digest(dataset / "manifest.json"),
            status="passed",
            checked=dict(checked),
            families=len(family_splits),
            research_only=manifest["research_only"],
            limitation="Every label is checked against its published source and declared transformation. This is not independent linguistic adjudication; full family closure is audited separately.",
        ),
    )
    print(json.dumps(dict(checked), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument(
        "--previous", default="data/research/en-general-002/records.jsonl", type=Path
    )
    p.add_argument("--extra", default="data/raw/en-general-003", type=Path)
    p.add_argument("--raw", default="data/raw/en-research-20261021-a", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    audit(a.dataset, a.previous, a.extra, a.out, a.raw)
