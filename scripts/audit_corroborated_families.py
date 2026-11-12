"""Extend known-family audits only with hash-verified dictionary corroborations."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from build_context_corpus import RAW, lexical_graph, read_wic
from lev.dataset import bundle
from lev.io import digest, encoded, jsonl, read, write
from lev.vocabulary import load_words, verify_bundle


def verified_relations(audit):
    manifest = read(audit / "manifest.json")
    for path, expected in manifest["source_files"].items():
        if digest(path) != expected:
            raise ValueError("corroboration source changed: " + path)
    for name, expected in manifest["output_files"].items():
        if Path(name).name != name or digest(audit / name) != expected:
            raise ValueError("corroboration output changed: " + name)
    if read(audit / "summary.json")["status"] != "audit_complete":
        raise ValueError("dictionary corroboration audit is incomplete")
    return [
        row
        for row in jsonl(audit / "comparisons.jsonl")
        if row["status"] in {"corroborated_neutral", "corroborated_qualified"}
    ]


def audit_families(audits, training, vocabulary, context, out):
    if out.exists():
        raise ValueError("new output directory required")
    verify_bundle(context)
    for path, expected in read(context / "source-manifest.json")["files"].items():
        if digest(path) != expected:
            raise ValueError("base family source changed: " + path)
    families, _, _, prior = lexical_graph(read_wic(RAW))
    edges = defaultdict(list)
    for audit in audits:
        for row in verified_relations(audit):
            if not row["dictionary_support"]:
                raise ValueError("corroborated edge has no dictionary evidence")
            target, form = row["target"], row["form"]
            families.union([families.word(target), families.word(form)])
            edges[(target, form)].append(
                dict(
                    audit=str(audit),
                    source_line=row["source_line"],
                    dictionary_support=row["dictionary_support"],
                )
            )
    records, _ = bundle(training)
    groups = defaultdict(list)
    for row in records:
        groups[families.find(row["family_id"])].append(row)
    conflicts = [
        dict(
            root=root,
            records=[
                dict(id=r["id"], family_id=r["family_id"], split=r["split"])
                for r in rows
            ],
        )
        for root, rows in groups.items()
        if len({r["split"] for r in rows}) > 1
    ]
    blocked = {families.find(r["family_id"]) for r in records + prior}
    context_records = jsonl(context / "records.jsonl")
    context_conflicts = [
        dict(
            id=r["id"],
            word=r["state"]["headword"],
            family=families.find(r["family_id"]),
        )
        for r in context_records
        if families.find(r["family_id"]) in blocked
    ]
    train_roots = {
        families.find(r["family_id"])
        for r in records
        if r["split"] == "train"
        and not r["family_id"].startswith(("wi:", "construction:"))
    }
    words = []
    for word in load_words(vocabulary):
        root = families.find(families.word(word["word"]))
        words.append(
            dict(
                word=word["word"],
                rank=word["rank"],
                frozen_historical_exposure=word["exposure"],
                candidate_supervised_target_family_seen=root in train_roots,
                current_known_family=root,
                pretraining_exposure="unknown",
                incidental_context_tokens_counted=False,
            )
        )
    out.mkdir(parents=True)
    for name, rows in (
        ("words.jsonl", words),
        (
            "edges.jsonl",
            [
                dict(target=k[0], form=k[1], evidence=v)
                for k, v in sorted(edges.items())
            ],
        ),
    ):
        with (out / name).open("x") as stream:
            for row in rows:
                stream.write(encoded(row) + "\n")
    report = dict(
        status="conflicts_require_revision" if conflicts else "passed",
        training_family_isolation_passed=not conflicts,
        training_conflicts=conflicts,
        context_conflicts=context_conflicts,
        context_requires_revision=bool(context_conflicts),
        corroborated_distinct_edges=len(edges),
        words=len(words),
        candidate_supervised_target_family_counts=dict(
            Counter(
                "seen"
                if r["candidate_supervised_target_family_seen"]
                else "not_detected"
                for r in words
            )
        ),
        training_manifest_sha256=digest(training / "manifest.json"),
        vocabulary_manifest_sha256=digest(vocabulary / "manifest.json"),
        context_manifest_sha256=digest(context / "manifest.json"),
        words_sha256=digest(out / "words.jsonl"),
        edges_sha256=digest(out / "edges.jsonl"),
        corroboration_manifests={str(p): digest(p / "manifest.json") for p in audits},
        script_sha256=digest(__file__),
        model_scores_read=False,
        interpretation="Known supervised target-family exposure, expanded by dictionary-supported inflections only. Historical registry unchanged. Pretraining exposure unknown; no word mastery claim. A passed training-family audit does not admit a context suite with listed conflicts.",
    )
    write(out / "manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--training", type=Path, default=Path("artifacts/en-joint-001"))
    parser.add_argument(
        "--vocabulary", type=Path, default=Path("artifacts/en-us-vocab-35000-v3")
    )
    parser.add_argument(
        "--context", type=Path, default=Path("artifacts/en-us-vocab-context-001")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_families(
        args.audit, args.training, args.vocabulary, args.context, args.out
    )
    print(
        encoded(
            {
                k: v
                for k, v in result.items()
                if k not in {"training_conflicts", "context_conflicts"}
            }
        )
    )
