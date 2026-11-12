"""Corroborate candidate inflections with structured dictionary forms; no gold export."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from lev.io import digest, encoded, jsonl, read, write

FEATURES = {
    "N;PL": ("noun", {"plural"}),
    "V;PRS;3;SG": ("verb", {"present", "third-person", "singular"}),
    "V;PST": ("verb", {"past"}),
    "V;V.PTCP;PST": ("verb", {"past", "participle"}),
    "V;V.PTCP;PRS": ("verb", {"present", "participle"}),
    "ADJ;CMPR": ("adj", {"comparative"}),
    "ADJ;SPRL": ("adj", {"superlative"}),
}
GRAMMATICAL = {tag for _, tags in FEATURES.values() for tag in tags}
MARKERS = {"countable", "uncountable", "comparable", "comparative", "superlative"}


def matches(entries, form, features):
    if features not in FEATURES:
        return []
    pos, required = FEATURES[features]
    found = []
    for entry_index, entry in enumerate(entries):
        if entry.get("pos") != pos:
            continue
        for form_index, item in enumerate(entry.get("forms", [])):
            tags = set(item.get("tags", []))
            # A past participle does not independently attest a finite past tense.
            if features == "V;PST" and "participle" in tags:
                continue
            if item.get("form") == form and required <= tags:
                found.append(
                    dict(
                        entry_index=entry_index,
                        form_index=form_index,
                        tags=sorted(tags),
                        qualifications=sorted(tags - GRAMMATICAL),
                    )
                )
    return found


def classify(lemma, form, features, entries):
    if lemma != form and form in MARKERS:
        return "invalid_metadata_form", []
    if features not in FEATURES:
        return "outside_inflection_scope", []
    support = matches(entries, form, features)
    if not support:
        return "not_corroborated_not_a_false_label", []
    return (
        "corroborated_qualified"
        if all(e["qualifications"] for e in support)
        else "corroborated_neutral"
    ), support


def audit(directory, out):
    if out.exists():
        raise ValueError("new output directory required")
    result = read(directory / "result.json")
    selection = read(directory / "selection.json")
    events = jsonl(directory / "downloads.jsonl")
    expected = set(selection["words"])
    if len(events) != len(expected) or {e["word"] for e in events} != expected:
        raise ValueError("download population incomplete or duplicated")
    if result["requested"] != len(expected):
        raise ValueError("download summary differs from selection")
    source = Path("data/raw/en-morphology-001/eng")
    if digest(source) != selection["source_sha256"]:
        raise ValueError("candidate source changed")
    dictionaries, files = {}, {}
    for event in events:
        if event["status"] == "downloaded":
            path = directory / (event["word"] + ".jsonl")
            if digest(path) != event["sha256"]:
                raise ValueError("dictionary changed: " + event["word"])
            entries = jsonl(path)
            if any(
                e.get("word") != event["word"] or e.get("lang_code") != "en"
                for e in entries
            ):
                raise ValueError("dictionary headword/language mismatch")
            dictionaries[event["word"]] = entries
            files[str(path)] = event["sha256"]
        elif event["status"] not in ("not_found", "failed"):
            raise ValueError("unexpected source status")
    out.mkdir(parents=True)
    counts, word_status = Counter(), defaultdict(Counter)
    distinct = defaultdict(set)
    with (out / "comparisons.jsonl").open("x") as stream:
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            lemma, form, features = line.split("\t")
            target = lemma.casefold()
            if target not in expected:
                continue
            if target not in dictionaries:
                status, support = "dictionary_unavailable", []
            else:
                status, support = classify(lemma, form, features, dictionaries[target])
            counts[status] += 1
            word_status[target][status] += 1
            distinct[status].add((target, form, features))
            stream.write(
                encoded(
                    dict(
                        source_line=line_number,
                        target=target,
                        source_lemma=lemma,
                        form=form,
                        features=features,
                        status=status,
                        dictionary_support=support,
                    )
                )
                + "\n"
            )
    write(out / "words.json", {w: dict(word_status[w]) for w in selection["words"]})
    summary = dict(
        status="audit_complete",
        requested_words=len(expected),
        dictionaries_available=len(dictionaries),
        source_rows=dict(counts),
        distinct_relations={k: len(v) for k, v in distinct.items()},
        words_with_corroborated_inflections=sum(
            any(k.startswith("corroborated_") for k in v) for v in word_status.values()
        ),
        words_with_neutral_corroboration=sum(
            v["corroborated_neutral"] > 0 for v in word_status.values()
        ),
        remaining_candidate_words=selection["remaining_candidates"],
        gold_created=False,
        model_execution=False,
        interpretation="Structured parser corroboration, potentially sharing underlying Wiktionary evidence; not independent expert adjudication. Missing support is not a false linguistic label. Qualified variants remain qualified. Raw-source rows outside this requested batch are not assessed.",
    )
    write(out / "summary.json", summary)
    files.update(
        {
            str(p): digest(p)
            for p in (
                source,
                directory / "result.json",
                directory / "selection.json",
                directory / "downloads.jsonl",
                Path(__file__),
            )
        }
    )
    write(
        out / "manifest.json",
        dict(
            source_files=files,
            output_files={p.name: digest(p) for p in out.iterdir() if p.is_file()},
        ),
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(encoded(audit(args.dictionary, args.out)))
