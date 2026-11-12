"""Add a source-labeled, family-disjoint context subset without changing the 35k suite."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil

from build_context_corpus import (
    BENCHMARK,
    LEXICAL_RAW,
    PRIOR,
    RAW,
    RUBRIC,
    SUPPLEMENT,
    key,
    lexical_graph,
    normalized,
    read_wic,
)
from audit_corroborated_families import verified_relations

from lev.dataset import bundle, exportable, to_kev, validate
from lev.io import digest, encoded, read, write
from lev.vocabulary import load_words

VERSION = "en-us-vocab-context-002"
PROTOCOL = Path("docs/experiments/en-us-vocab-context-002.md")
JOINT = Path("artifacts/en-joint-001")
AUDITS = [
    Path("runs/en-morphology-001-dictionary-audit-001"),
    Path("runs/en-morphology-001-dictionary-audit-002"),
]


def select(rows, population, families, blocked):
    selected, events, seen = [], [], {}
    for row in rows:
        word = row["target"].casefold()
        family = families.find(families.word(word))
        identity = (
            word,
            tuple(normalized(s) for s in row["contexts"]),
            tuple(row["indices"]),
        )
        reason = (
            "outside_35000"
            if word not in population
            else "prior_family_exposure"
            if family in blocked
            else None
        )
        if reason is None and identity in seen:
            if seen[identity] != row["label"]:
                raise ValueError("contradictory duplicate source labels")
            reason = "duplicate_context_pair"
        if reason is None:
            seen[identity] = row["label"]
            selected.append((row, family))
        events.append(
            dict(
                locator=row["locator"],
                target=word,
                family_id=family,
                native_split=row["native_split"],
                status="excluded" if reason else "included",
                reason=reason,
            )
        )
    return selected, events


def build(out):
    if out.exists():
        raise ValueError("new output directory required")
    population = load_words(BENCHMARK)
    by_word = {w["word"]: w for w in population}
    source = read_wic(RAW)
    families, _, _, prior = lexical_graph(source)
    for audit in AUDITS:
        for row in verified_relations(audit):
            if not row["dictionary_support"]:
                raise ValueError("corroboration lacks dictionary evidence")
            families.union([families.word(row["target"]), families.word(row["form"])])
    joint, _ = bundle(JOINT)
    blocked = {families.find(r["family_id"]) for r in prior}
    blocked |= {
        families.find(r["family_id"])
        for r in joint
        if not r["family_id"].startswith("wi:")
    }
    selected, events = select(source, by_word, families, blocked)
    rubric = read(RUBRIC)
    grouped = defaultdict(list)
    for row, family in selected:
        word = row["target"].casefold()
        rid = key(VERSION + ":" + row["locator"])[:24]
        grouped[word].append(
            dict(
                id=rid,
                family_id=family,
                case_id=rid,
                split="test",
                language="en",
                scope="partial",
                origin="curated",
                rubric_version=rubric["version"],
                state=dict(
                    headword=word,
                    target_word=row["target"],
                    part_of_speech=row["pos"],
                    contexts=row["contexts"],
                    target_token_indices_zero_based=row["indices"],
                    convention="English; use the marked occurrences and the WiC sense distinction.",
                ),
                questions={
                    "q0": dict(
                        kind="contextual_sense",
                        claim="Is the marked word used with the same meaning in both contexts?",
                        label=row["label"],
                        review=dict(
                            status="source_labeled",
                            reviewer="source:WiC",
                            rationale="Published WiC T/F sense-equivalence label; not independently adjudicated by Lev.",
                            evidence=[
                                dict(
                                    reference=str(RAW / "wic.zip"),
                                    locator=row["locator"],
                                )
                            ],
                        ),
                    )
                },
            )
        )
    records, words = [], []
    for word in sorted(grouped, key=lambda w: by_word[w]["rank"]):
        rows = grouped[word]
        records.extend(rows)
        words.append(
            dict(
                word=word,
                rank=len(words) + 1,
                population_rank=by_word[word]["rank"],
                exposure=by_word[word]["exposure"],
                record_ids=[r["id"] for r in rows],
                judgment_count=len(rows),
                coverage={"contextual_sense": len(rows)},
                gaps=["all_other_word_knowledge", "independent_expert_adjudication"],
                known_lev_family_exposure="none_detected",
                pretraining_exposure="unknown",
            )
        )
    validate(records, rubric)
    exportable(records, rubric, examples=False, research=True)
    out.mkdir(parents=True)
    for name, rows in (
        ("records.jsonl", records),
        ("words.jsonl", words),
        ("test.jsonl", [to_kev(r, rubric) for r in records]),
        ("selection-audit.jsonl", events),
    ):
        with (out / name).open("x") as stream:
            for row in rows:
                stream.write(encoded(row) + "\n")
    write(out / "rubric.json", rubric)
    coverage = dict(
        unique_headwords=len(words),
        judgments=len(records),
        population_words=len(population),
        population_words_without_context=len(population) - len(words),
        source_rows=len(source),
        selection=dict(Counter(e["reason"] or "included" for e in events)),
        native_splits=dict(Counter(r["native_split"] for r, _ in selected)),
        labels=dict(Counter(str(r["questions"]["q0"]["label"]) for r in records)),
        full_entry_coverage=0,
        model_evaluation_status="not_run",
    )
    write(out / "coverage.json", coverage)
    deps = [
        RAW / "wic.zip",
        RUBRIC,
        PROTOCOL,
        Path(__file__),
        Path("scripts/build_context_corpus.py"),
        Path("scripts/audit_family_isolation.py"),
        Path("scripts/vocabulary_wiktionary.py"),
        LEXICAL_RAW / "wordnet.xml.gz",
        LEXICAL_RAW / "ud.conllu",
        JOINT / "manifest.json",
        JOINT / "records.jsonl",
        BENCHMARK / "manifest.json",
        BENCHMARK / "words.jsonl",
        *PRIOR,
    ]
    deps.append(Path("scripts/audit_corroborated_families.py"))
    for audit in AUDITS:
        deps.extend(
            [
                audit / "manifest.json",
                audit / "comparisons.jsonl",
                audit / "summary.json",
            ]
        )
    for directory in (SUPPLEMENT, RAW / "lexical-aliases", LEXICAL_RAW / "wiktionary"):
        deps.extend(sorted(directory.glob("*.jsonl")))
    write(
        out / "source-manifest.json",
        dict(
            files={str(p): digest(p) for p in deps},
            source="https://pilehvar.github.io/wic/",
            license="CC BY-NC 4.0; target selection additionally CC BY-NC-SA 4.0; research only",
            attribution="Pilehvar and Camacho-Collados (2019), WiC; target-word selection attribution retained in parent benchmark.",
            parent_manifest_sha256=digest(BENCHMARK / "manifest.json"),
        ),
    )
    shutil.copyfile(PROTOCOL, out / "PROTOCOL.md")
    shutil.copyfile(RAW / "ATTRIBUTION.md", out / "ATTRIBUTION.md")
    write(
        out / "manifest.json",
        dict(
            format="lev-vocabulary-test-v1",
            version=VERSION,
            test_only=True,
            research_only=True,
            unique_headwords=len(words),
            parent_manifest_sha256=digest(BENCHMARK / "manifest.json"),
            files={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        ),
    )
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(encoded(build(parser.parse_args().out)))
