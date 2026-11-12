"""Freeze a test-only, US-prevalence-ranked vocabulary benchmark; no model labels."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

from audit_family_isolation import Families
from lev.dataset import exportable, to_kev, validate
from lev.io import digest, encoded, jsonl, read, write
from vocabulary_wiktionary import load_supplement, supplement_questions

PREVALENCE_SHA = "3b2cbee33f7a4eab50e71a7a09461fe28c32bb16c53435327b94cf7b951b76af"
POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
XML = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
VERSION = "en-us-vocab-35000-v1"


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prevalence(path):
    if digest(path) != PREVALENCE_SHA:
        raise ValueError(
            "unexpected prevalence workbook; preserve a new source version"
        )
    with zipfile.ZipFile(path) as archive:
        strings = [
            "".join(e.itertext())
            for e in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]

        def sheet(number):
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
            rows = root.findall("m:sheetData/m:row", XML)
            output = []
            for row in rows:
                values = {}
                for cell in row:
                    value = cell.findtext("m:v", None, XML)
                    if value is not None:
                        values[re.sub(r"\d", "", cell.get("r"))] = (
                            strings[int(value)]
                            if cell.get("t") == "s"
                            else float(value)
                        )
                output.append(values)
            return output

        main, us = sheet(1), sheet(2)
    if main[0] != {
        "A": "Word",
        "B": "Pknown",
        "C": "Nobs",
        "D": "Prevalence",
        "E": "FreqZipfUS",
    }:
        raise ValueError("unexpected prevalence schema")
    if us[0] != {"A": "Word", "B": "Prevalence_UK", "C": "Prevalence_US"}:
        raise ValueError("unexpected US prevalence schema")
    lookup = {r["A"]: r for r in main[1:]}
    if len(lookup) != len(main) - 1 or len(us) != len(main):
        raise ValueError("duplicate or mismatched prevalence rows")
    result = []
    for number, row in enumerate(us[1:], 2):
        general = lookup[row["A"]]
        if not all(math.isfinite(x) for x in (row["C"], general["E"])):
            raise ValueError("non-finite ranking value")
        result.append(
            dict(
                word="i" if row["A"] == "I" else row["A"],
                source_word=row["A"],
                prevalence_us=row["C"],
                frequency_zipf_us=general["E"],
                source_row=number,
            )
        )
    return sorted(
        result, key=lambda r: (-r["prevalence_us"], -r["frequency_zipf_us"], r["word"])
    )


def wordnet(path):
    lex = ET.parse(gzip.open(path)).getroot().find("Lexicon")
    if lex.get("version") != "2025":
        raise ValueError("expected OEWN 2025")
    synsets = {}
    for syn in lex.findall("Synset"):
        definition = syn.findtext("Definition")
        if definition and definition.strip():
            synsets[syn.get("id")] = dict(
                definition=definition.strip(),
                pos=POS[syn.get("partOfSpeech")],
                domain=syn.get("lexfile"),
                members=syn.get("members", "").split(),
            )
    words = defaultdict(lambda: {"senses": [], "entries": [], "pos": set()})
    for entry in lex.findall("LexicalEntry"):
        lemma = entry.find("Lemma")
        word = lemma.get("writtenForm")
        if not re.fullmatch("[a-z]+", word):
            continue
        senses = [
            s.get("synset")
            for s in entry.findall("Sense")
            if s.get("synset") in synsets
        ]
        if not senses:
            continue
        words[word]["senses"].extend(senses)
        words[word]["entries"].append(entry.get("id"))
        words[word]["pos"].add(POS[lemma.get("partOfSpeech")])
    for info in words.values():
        info["senses"] = sorted(set(info["senses"]))
        info["entries"].sort()
    return dict(words), synsets


def cmu(path):
    pronunciations = defaultdict(dict)
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith(";;;"):
            continue
        spelling, *phones = line.split()
        word = re.sub(r"\(\d+\)$", "", spelling).casefold()
        if re.fullmatch("[a-z]+", word) and phones:
            pronunciations[word][" ".join(phones)] = line_number
    return dict(pronunciations)


def negative_pronunciation(word, original, variants):
    phones = original.split()
    # Preserve vowel/consonant class and stress, avoiding invalid ARPABET tokens.
    vowels = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
    consonants = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
    positions = sorted(
        range(len(phones)), key=lambda i: stable(f"{word}:{original}:{i}")
    )
    for index in positions:
        phone = phones[index]
        vowel = phone[-1:] in {"0", "1", "2"}
        options = [p + phone[-1] for p in vowels] if vowel else consonants
        for replacement in sorted(options, key=lambda p: stable(f"{word}:{index}:{p}")):
            candidate = " ".join(phones[:index] + [replacement] + phones[index + 1 :])
            if candidate not in variants:
                return candidate
    raise ValueError(f"no defensible pronunciation contrast for {word}")


def distractor(word, sid, info, synsets, pools):
    own = set(info["senses"])
    own_text = {synsets[s]["definition"].casefold() for s in own}
    source = synsets[sid]
    pool = pools[(source["pos"], source["domain"])]
    start = int(stable(f"distractor:{word}:{sid}")[:12], 16) % len(pool)
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        definition = synsets[candidate]["definition"]
        if candidate not in own and definition.casefold() not in own_text:
            if not re.search(r"\b" + re.escape(word) + r"\b", definition, re.I):
                return candidate
    raise ValueError(f"no same-domain contrast for {word}/{sid}; do not silently drop")


def questions_for(word, info, synsets, pools, variants, source_hashes):
    questions = {}

    def add(kind, proposition, label, source, locator, rationale):
        qid = stable(f"{word}:{kind}:{proposition}")[:20]
        question = dict(
            kind=kind,
            claim=proposition,
            label=label,
            review=dict(
                status="source_derived",
                reviewer="source:" + source,
                rationale=rationale,
                evidence=[
                    dict(reference=source, locator=locator),
                    dict(reference=source, locator="sha256:" + source_hashes[source]),
                ],
            ),
        )
        if qid in questions and questions[qid] != question:
            # Identical glosses may be attached to several synsets. Keep all source locators.
            old = questions[qid]
            if old["label"] != label or old["claim"] != proposition:
                raise ValueError("question hash/label collision")
            if question["review"]["evidence"][0] not in old["review"]["evidence"]:
                old["review"]["evidence"].append(question["review"]["evidence"][0])
        else:
            questions[qid] = question

    definition_templates = [
        'Does the OEWN 2025 inventory assign the headword a sense defined as: "{}"?',
        'Is this a definition of a recorded OEWN 2025 sense of the headword: "{}"?',
        'For this headword, is the following meaning recorded in OEWN 2025: "{}"?',
    ]
    for sid in info["senses"]:
        other = distractor(word, sid, info, synsets, pools)
        for candidate, label in [(sid, True), (other, False)]:
            definition = synsets[candidate]["definition"]
            template = definition_templates[
                int(stable(word + definition)[:8], 16) % len(definition_templates)
            ]
            add(
                "sense_inventory",
                template.format(definition),
                label,
                "wordnet.xml.gz",
                f"word={word};synset={candidate}",
                "Membership of the candidate sense definition in this headword's complete OEWN 2025 inventory. "
                "Negative checks every recorded sense and identical gloss, not universal invalidity in English.",
            )
    for pos in sorted(set(POS.values())):
        add(
            "pos_inventory",
            f"Does OEWN 2025 record {'an' if pos[0] in 'aeiou' else 'a'} {pos} entry for this headword?",
            pos in info["pos"],
            "wordnet.xml.gz",
            f"word={word};pos={pos}",
            "Closed-source entry membership, not a claim that unrecorded English usage is impossible.",
        )
    for pronunciation, line in sorted(variants.items()):
        contrast = negative_pronunciation(word, pronunciation, variants)
        for candidate, label in [(pronunciation, True), (contrast, False)]:
            add(
                "pronunciation_inventory",
                f'Is the ARPABET transcription "{candidate}" recorded for this headword in the specified CMUdict snapshot?',
                label,
                "cmudict.dict",
                f"word={word};line={line};candidate={candidate}",
                "Exact recorded pronunciation membership across all listed variants; a nonlisted variant is not declared "
                "impossible for every US speaker. Negatives replace one legal phoneme and exclude all listed variants.",
            )
    return dict(sorted(questions.items()))


def records_for(word, family, questions):
    items = list(questions.items())
    records = []
    case = stable("vocab-word:" + word)[:20]
    for offset in range(0, len(items), 8):
        records.append(
            dict(
                id=stable(f"{VERSION}:{word}:{offset}")[:20],
                case_id=case,
                family_id=family,
                split="test",
                scope="partial",
                origin="generated",
                language="en",
                rubric_version=VERSION,
                state=dict(
                    headword=word,
                    convention=(
                        "Word-level lexical inventory review. OEWN means Open English WordNet 2025. "
                        "Wiktionary questions refer to the pinned English Kaikki extraction in the source manifest. "
                        "Pronunciation questions use the pinned CMUdict snapshot, a US English pronunciation resource. "
                        "Judge the explicitly named source inventory, not whether an unlisted usage could ever occur."
                    ),
                ),
                questions=dict(items[offset : offset + 8]),
            )
        )
    return records


def build(args):
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    ranking = prevalence(args.prevalence / "English_Word_Prevalences.xlsx")
    words, synsets = wordnet(args.raw / "wordnet.xml.gz")
    supplement, supplement_senses = load_supplement(args.wiktionary)
    eligible_words = set(words) | {
        w for w, info in supplement.items() if info["eligible"]
    }
    pronunciations = cmu(args.raw / "cmudict.dict")
    pools = defaultdict(list)
    for sid, info in sorted(synsets.items()):
        pools[(info["pos"], info["domain"])].append(sid)
    eligible = [r for r in ranking if r["word"] in eligible_words]
    if len(eligible) < args.count:
        raise ValueError("insufficient evidence-backed headwords; do not pad")
    selected = eligible[: args.count]
    chosen = {r["word"] for r in selected}
    with (root / "selection-audit.jsonl").open("x") as stream:
        for rank, row in enumerate(ranking, 1):
            reason = (
                "selected"
                if row["word"] in chosen
                else "outside_fixed_prevalence_rank"
                if row["word"] in eligible_words
                else "no_supported_independent_headword"
            )
            stream.write(
                encoded(dict(**row, population_rank=rank, selection=reason)) + "\n"
            )
    families = Families(args.raw)
    for word, info in supplement.items():
        families.union(
            [families.word(word), *[families.word(f) for f in info["forms"]]]
        )
    prior = [r for path in args.prior for r in jsonl(path)]
    families.attach(prior)
    exposure = defaultdict(set)
    for record in prior:
        exposure[families.find(record["family_id"])].add(record["split"])
    sources = {
        name: digest(args.raw / name) for name in ["wordnet.xml.gz", "cmudict.dict"]
    }
    rubric = read("rubrics/en-general-research-v1.json")
    rubric["version"] = VERSION
    write(root / "rubric.json", rubric)
    summary = Counter()
    kinds, labels = Counter(), Counter()
    counts_per_word, wordlist = [], []
    records_path = root / "records.jsonl"
    with (
        records_path.open("x") as records_file,
        (root / "test.jsonl").open("x") as test_file,
    ):
        for rank, selected_word in enumerate(selected, 1):
            word = selected_word["word"]
            info = words.get(word, dict(senses=[], pos=set()))
            family = families.find(families.word(word))
            contact = sorted(exposure.get(family, set()))
            group = (
                "prior_train"
                if "train" in contact
                else "prior_evaluation_only"
                if contact
                else "no_detected_prior_target_family"
            )
            questions = questions_for(
                word, info, synsets, pools, pronunciations.get(word, {}), sources
            )
            if word not in words:
                questions = {
                    k: q for k, q in questions.items() if q["kind"] != "pos_inventory"
                }
            if word in supplement and supplement[word]["senses"]:
                questions.update(
                    supplement_questions(word, supplement[word], supplement_senses)
                )
            questions = dict(sorted(questions.items()))
            records = records_for(word, family, questions)
            validate(records, rubric)
            exportable(records, rubric, examples=False, research=True)
            for record in records:
                records_file.write(encoded(record) + "\n")
                test_file.write(encoded(to_kev(record, rubric)) + "\n")
            count = len(questions)
            wordlist.append(
                dict(
                    **selected_word,
                    rank=rank,
                    family_id=family,
                    historical_splits=contact,
                    exposure=group,
                    record_ids=[r["id"] for r in records],
                    judgment_count=count,
                    inventory_senses=len(info["senses"]),
                    wiktionary_inventory_senses=len(
                        supplement.get(word, {}).get("senses", [])
                    ),
                    parts_of_speech=sorted(info["pos"]),
                    pronunciation_variants=len(pronunciations.get(word, {})),
                    coverage={
                        "OEWN_sense_inventory": word in words,
                        "OEWN_pos_inventory": word in words,
                        "Wiktionary_inventory": word in supplement
                        and bool(supplement[word]["senses"]),
                        "CMU_pronunciation_inventory": word in pronunciations,
                    },
                    gaps=[
                        "contextual_sense_disambiguation",
                        "collocations",
                        "register",
                        "morphology",
                        "etymology",
                    ]
                    + (
                        []
                        if word in pronunciations
                        else ["source_pronunciation_unavailable"]
                    ),
                    complete_entry_reviewed=False,
                )
            )
            counts_per_word.append(count)
            summary[group] += 1
            summary["records"] += len(records)
            summary["judgments"] += count
            summary["words_with_pronunciation"] += word in pronunciations
            summary["inventory_senses"] += len(info["senses"])
            summary["wiktionary_inventory_senses"] += len(
                supplement.get(word, {}).get("senses", [])
            )
            summary["words_with_wiktionary"] += word in supplement and bool(
                supplement[word]["senses"]
            )
            kinds.update(q["kind"] for q in questions.values())
            labels.update(str(q["label"]).lower() for q in questions.values())
    with (root / "words.jsonl").open("x") as stream:
        for word in wordlist:
            stream.write(encoded(word) + "\n")
    with (root / "words.txt").open("x") as stream:
        stream.write("\n".join(r["word"] for r in wordlist) + "\n")
    shutil.copy2(args.prevalence / "license.txt", root / "prevalence-LICENSE.txt")
    for name in ["wordnet-LICENSE", "cmudict-LICENSE"]:
        shutil.copy2(args.raw / name, root / name)
    shutil.copy2(args.protocol, root / "PROTOCOL.md")
    if args.wiktionary:
        shutil.copy2(
            args.wiktionary / "ATTRIBUTION.md", root / "Wiktionary-ATTRIBUTION.md"
        )
    write(
        root / "coverage.json",
        dict(
            unique_headwords=len(wordlist),
            eligible_headwords=len(eligible),
            prevalence_rows=len(ranking),
            summary=dict(summary),
            by_kind=dict(kinds),
            labels=dict(labels),
            min_judgments_per_word=min(counts_per_word),
            max_judgments_per_word=max(counts_per_word),
            min_selected_prevalence_us=min(r["prevalence_us"] for r in wordlist),
            model_evaluation_status="not_run",
            full_entry_coverage=0,
        ),
    )
    write(
        root / "source-manifest.json",
        dict(
            files={str(args.raw / name): sha for name, sha in sources.items()}
            | {str(p): digest(p) for p in args.prevalence.rglob("*") if p.is_file()}
            | {str(p): digest(p) for p in (args.raw / "wiktionary").glob("*.jsonl")}
            | {str(args.raw / "ud.conllu"): digest(args.raw / "ud.conllu")}
            | {str(p): digest(p) for p in args.prior},
            builder_sha256=digest(__file__),
            supplement_builder_sha256=digest("scripts/vocabulary_wiktionary.py"),
            family_auditor_sha256=digest("scripts/audit_family_isolation.py"),
            ranking="US prevalence descending, US SUBTLEX Zipf descending, headword ascending; supported dictionary headwords, not appended inflections",
            sources=[
                "https://osf.io/g4xrt/",
                "https://github.com/globalwordnet/english-wordnet",
                "https://github.com/cmusphinx/cmudict",
                "https://kaikki.org/dictionary/English/",
            ],
            attribution="Brysbaert, Mandera, McCormick & Keuleers (2019); Open English WordNet team and Princeton WordNet; Carnegie Mellon CMUdict contributors.",
            license="Research only; prevalence-derived selection is CC BY-NC-SA 4.0. Preserve separate WordNet and CMUdict terms. No commercial training authorization.",
        ),
    )
    write(
        root / "manifest.json",
        dict(
            format="lev-vocabulary-test-v1",
            version=VERSION,
            test_only=True,
            research_only=True,
            unique_headwords=len(wordlist),
            files={p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
        ),
    )
    print(encoded(read(root / "coverage.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/en-research-20261021-a")
    )
    parser.add_argument(
        "--prevalence", type=Path, default=Path("data/raw/en-us-vocab-35000-v1")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=35000)
    parser.add_argument("--wiktionary", type=Path)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/experiments/en-us-vocab-35000-v1.md"),
    )
    parser.add_argument(
        "--prior",
        type=Path,
        action="append",
        default=[
            Path("data/research/en-general-002/records.jsonl"),
            Path("data/research/en-general-005/records.jsonl"),
        ],
    )
    args = parser.parse_args()
    VERSION = args.version
    build(args)
