"""Add genuine Wiktionary headwords, including function words missing from WordNet."""

from collections import defaultdict
import hashlib
import re

from lev.io import digest, jsonl

NAMES = {
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


def gloss(sense):
    return "; ".join(sense.get("glosses", [])).strip()


def independent(entry, sense):
    return (
        entry.get("pos") in NAMES
        and bool(gloss(sense))
        and not sense.get("form_of")
        and not sense.get("alt_of")
        and not set(sense.get("tags", [])) & {"form-of", "alt-of"}
    )


def load_supplement(directory):
    if directory is None:
        return {}, []
    events = jsonl(directory / "downloads.jsonl")
    if any(e["status"] == "failed" for e in events):
        raise ValueError("unresolved supplement download failures")
    output = defaultdict(
        lambda: dict(senses=[], pos=set(), eligible=False, forms=set(), sources=set())
    )
    for file in sorted(directory.glob("*.jsonl")):
        if file.name == "downloads.jsonl":
            continue
        for entry_index, entry in enumerate(jsonl(file)):
            word = entry["word"].casefold()
            if not re.fullmatch("[a-z]+", word) or entry["lang_code"] != "en":
                raise ValueError("supplement contains unexpected language/word")
            info = output[word]
            info["sources"].add(str(file))
            if entry.get("pos") in NAMES:
                info["pos"].add(entry["pos"])
            info["forms"].update(
                f["form"].casefold()
                for f in entry.get("forms", [])
                if re.fullmatch("[a-zA-Z]+", f.get("form", ""))
            )
            for sense_index, sense in enumerate(entry.get("senses", [])):
                info["eligible"] |= independent(entry, sense)
                for relation in ("form_of", "alt_of"):
                    info["forms"].update(
                        r["word"].casefold()
                        for r in sense.get(relation, [])
                        if re.fullmatch("[a-zA-Z]+", r.get("word", ""))
                    )
                if entry.get("pos") in NAMES and gloss(sense):
                    info["senses"].append(
                        dict(
                            word=word,
                            definition=gloss(sense),
                            pos=entry["pos"],
                            source=str(file),
                            entry=entry_index,
                            sense=sense_index,
                        )
                    )
    all_senses = [s for info in output.values() for s in info["senses"]]
    return dict(output), all_senses


def supplement_questions(word, info, all_senses):
    questions = {}
    own = {s["definition"].casefold() for s in info["senses"]}
    pools = defaultdict(list)
    for sense in all_senses:
        pools[sense["pos"]].append(sense)
    hashes = {s["source"]: digest(s["source"]) for s in info["senses"]}

    def add(kind, claim, truth, source, locator):
        qid = hashlib.sha256(f"{word}:{kind}:{claim}".encode()).hexdigest()[:20]
        item = dict(
            kind=kind,
            claim=claim,
            label=truth,
            review=dict(
                status="source_derived",
                reviewer="source:Wiktionary/Kaikki",
                rationale="Source-scoped dictionary inventory membership. Alternative senses and all downloaded entries checked; not a universal claim that unrecorded usage is invalid.",
                evidence=[
                    dict(reference=source, locator=locator),
                    dict(
                        reference=source,
                        locator="sha256:" + hashes.setdefault(source, digest(source)),
                    ),
                ],
            ),
        )
        if qid in questions:
            if questions[qid]["label"] != truth:
                raise ValueError("contradictory Wiktionary labels")
            for evidence in item["review"]["evidence"]:
                if evidence not in questions[qid]["review"]["evidence"]:
                    questions[qid]["review"]["evidence"].append(evidence)
        else:
            questions[qid] = item

    for original in info["senses"]:
        pool = pools[original["pos"]]
        start = int(
            hashlib.sha256((word + original["definition"]).encode()).hexdigest()[:8], 16
        ) % len(pool)
        candidate = next(
            (
                pool[(start + i) % len(pool)]
                for i in range(len(pool))
                if pool[(start + i) % len(pool)]["definition"].casefold() not in own
                and pool[(start + i) % len(pool)]["word"] != word
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"no same-POS Wiktionary contrast for {word}")
        for sense, truth in [(original, True), (candidate, False)]:
            add(
                "wiktionary_sense_inventory",
                'In the specified English Wiktionary snapshot, does this headword have a sense with the definition: "'
                + sense["definition"]
                + '"?',
                truth,
                sense["source"],
                f"word={word};entry={sense['entry']};sense={sense['sense']}",
            )
    for code, name in sorted(NAMES.items()):
        add(
            "wiktionary_pos_inventory",
            f"Does the specified English Wiktionary snapshot record {'an' if name[0] in 'aeiou' else 'a'} {name} entry for this headword?",
            code in info["pos"],
            sorted(info["sources"])[0],
            f"word={word};pos={code}",
        )
    return questions
