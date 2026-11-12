"""Check every frozen vocabulary label against the raw inventories and real export."""

import argparse
from collections import Counter, defaultdict
import gzip
from itertools import zip_longest
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from lev.dataset import to_kev
from lev.io import digest, encoded, jsonl, parse, read, write
from lev.vocabulary import load_words


def audit(dataset, raw):
    words = load_words(dataset)
    word_index = {w["word"]: w for w in words}
    rubric = read(dataset / "rubric.json")
    source_manifest = read(dataset / "source-manifest.json")
    for path, expected in source_manifest["files"].items():
        if digest(path) != expected:
            raise ValueError(f"raw source changed: {path}")
    lexicon = ET.parse(gzip.open(raw / "wordnet.xml.gz")).getroot().find("Lexicon")
    definitions = {
        s.get("id"): (s.findtext("Definition") or "").strip()
        for s in lexicon.findall("Synset")
    }
    senses, positions = defaultdict(set), defaultdict(set)
    pos_names = {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "s": "adjective",
        "r": "adverb",
    }
    for entry in lexicon.findall("LexicalEntry"):
        lemma = entry.find("Lemma")
        word = lemma.get("writtenForm")
        if word in word_index:
            senses[word].update(
                s.get("synset")
                for s in entry.findall("Sense")
                if definitions.get(s.get("synset"))
            )
            positions[word].add(pos_names[lemma.get("partOfSpeech")])
    pronunciations = defaultdict(set)
    wikt_glosses, wikt_pos, wikt_entries = defaultdict(set), defaultdict(set), {}
    wikt_pos_names = {
        "noun": "noun",
        "verb": "verb",
        "adj": "adjective",
        "adv": "adverb",
        "pron": "pronoun",
        "det": "determiner",
        "article": "article",
        "prep": "preposition",
        "conj": "conjunction",
        "intj": "interjection",
        "num": "numeral",
        "particle": "particle",
    }
    for name in source_manifest["files"]:
        path = Path(name)
        if (
            path.parent.name != "wiktionary-supplement"
            or path.suffix != ".jsonl"
            or path.name == "downloads.jsonl"
        ):
            continue
        entries = jsonl(path)
        wikt_entries[name] = entries
        for entry in entries:
            word = entry["word"].casefold()
            if entry.get("pos") in wikt_pos_names:
                wikt_pos[word].add(entry["pos"])
                wikt_glosses[word].update(
                    "; ".join(s.get("glosses", [])).strip()
                    for s in entry.get("senses", [])
                    if s.get("glosses")
                )
    covered_wikt, covered_wikt_pos = defaultdict(set), defaultdict(set)
    for line in (raw / "cmudict.dict").read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) > 1 and not fields[0].startswith(";;;"):
            word = re.sub(r"\(\d+\)$", "", fields[0]).casefold()
            pronunciations[word].add(" ".join(fields[1:]))
    counts, kinds, labels = Counter(), Counter(), Counter()
    record_ids, covered_senses, covered_pronunciations, covered_pos = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    ids = set()
    with (
        (dataset / "records.jsonl").open() as records_file,
        (dataset / "test.jsonl").open() as exported_file,
    ):
        for raw_record, raw_export in zip_longest(records_file, exported_file):
            if raw_record is None or raw_export is None:
                raise ValueError("different record/export lengths")
            record, exported = parse(raw_record), parse(raw_export)
            if exported != to_kev(record, rubric):
                raise ValueError("actual Kev export differs from provenance records")
            word = record["state"]["headword"]
            if word not in word_index or record["id"] in ids:
                raise ValueError("unknown target or duplicate record")
            ids.add(record["id"])
            record_ids[word].add(record["id"])
            if (
                record["split"] != "test"
                or record["scope"] != "partial"
                or len(record["questions"]) > 8
            ):
                raise ValueError("invalid test scope or request size")
            for q in record["questions"].values():
                first = q["review"]["evidence"][0]
                locator = dict(
                    piece.split("=", 1) for piece in first["locator"].split(";")
                )
                if locator["word"] != word or type(q["label"]) is not bool:
                    raise ValueError("invalid target/label")
                kind = q["kind"]
                if kind == "sense_inventory":
                    sid = locator["synset"]
                    definition = definitions[sid]
                    if '"' + definition + '"' not in q["claim"]:
                        raise ValueError("candidate definition does not match source")
                    truth = sid in senses[word] or definition.casefold() in {
                        definitions[s].casefold() for s in senses[word]
                    }
                    if truth:
                        covered_senses[word].update(
                            s
                            for s in senses[word]
                            if definitions[s].casefold() == definition.casefold()
                        )
                elif kind == "pos_inventory":
                    pos = locator["pos"]
                    if q["claim"] not in {
                        f"Does OEWN 2025 record a {pos} entry for this headword?",
                        f"Does OEWN 2025 record an {pos} entry for this headword?",
                    }:
                        raise ValueError("POS source locator and question disagree")
                    truth = pos in positions[word]
                    covered_pos[word].add(pos)
                elif kind == "pronunciation_inventory":
                    candidate = locator["candidate"]
                    if '"' + candidate + '"' not in q["claim"]:
                        raise ValueError("pronunciation locator and question disagree")
                    truth = candidate in pronunciations[word]
                    if truth:
                        covered_pronunciations[word].add(candidate)
                elif kind == "wiktionary_sense_inventory":
                    source_entry = wikt_entries[first["reference"]][
                        int(locator["entry"])
                    ]
                    sense = source_entry["senses"][int(locator["sense"])]
                    definition = "; ".join(sense["glosses"]).strip()
                    if '"' + definition + '"' not in q["claim"]:
                        raise ValueError(
                            "Wiktionary question and raw definition disagree"
                        )
                    truth = definition.casefold() in {
                        g.casefold() for g in wikt_glosses[word]
                    }
                    if truth:
                        covered_wikt[word].add(definition.casefold())
                elif kind == "wiktionary_pos_inventory":
                    code = locator["pos"]
                    if q["claim"] not in {
                        f"Does the specified English Wiktionary snapshot record a {wikt_pos_names[code]} entry for this headword?",
                        f"Does the specified English Wiktionary snapshot record an {wikt_pos_names[code]} entry for this headword?",
                    }:
                        raise ValueError("Wiktionary POS question and source disagree")
                    truth = code in wikt_pos[word]
                    covered_wikt_pos[word].add(code)
                else:
                    raise ValueError("unsupported source derivation")
                if q["label"] != truth:
                    raise ValueError(f"source-label mismatch: {word}/{kind}")
                counts[word] += 1
                kinds[kind] += 1
                labels[str(truth).lower()] += 1
    for word, card in word_index.items():
        if counts[word] != card["judgment_count"] or record_ids[word] != set(
            card["record_ids"]
        ):
            raise ValueError(f"word coverage mismatch: {word}")
        if covered_senses[word] != senses[word] or covered_pos[word] != (
            set(pos_names.values()) if senses[word] else set()
        ):
            raise ValueError(f"missing inventory sense/POS coverage: {word}")
        if card.get("coverage", {}).get("Wiktionary_inventory"):
            if covered_wikt[word] != {
                g.casefold() for g in wikt_glosses[word]
            } or covered_wikt_pos[word] != set(wikt_pos_names):
                raise ValueError(f"missing Wiktionary inventory coverage: {word}")
        if covered_pronunciations[word] != pronunciations[word]:
            raise ValueError(f"missing recorded pronunciation: {word}")
    return dict(
        status="passed",
        words=len(words),
        records=len(ids),
        judgments=sum(counts.values()),
        by_kind=dict(kinds),
        labels=dict(labels),
        dataset_manifest_sha256=digest(dataset / "manifest.json"),
        checks=[
            "all raw-source hashes",
            "all source-derived labels",
            "all actual Kev exports",
            "all word/request memberships",
            "all recorded source senses and pronunciations",
            "all four inventory POS checks",
            "all supplemented Wiktionary definition and POS inventories",
        ],
        limitation="Source-inventory correctness is not independent adjudication of unrestricted English semantics. No model inference performed.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/en-research-20261021-a")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset, args.raw)
    write(args.out, result)
    print(encoded(result))
