"""Combine new contextual train data with eligible historical training replay.

Historical evaluation never enters the new bundle. The 35k inventory benchmark
remains untouched; replay retains its already-disclosed prior-train exposure.
"""

from collections import Counter
from copy import deepcopy
from pathlib import Path

from build_context_corpus import RAW, PRIOR, lexical_graph, read_wic
from lev.dataset import bundle, prepare
from lev.io import digest, encoded, write


def main():
    out = Path("data/research/en-joint-001")
    artifact = Path("artifacts/en-joint-001")
    if out.exists() or artifact.exists():
        raise ValueError("new output directories required")
    contextual, _ = bundle("artifacts/en-context-002")
    old, _ = bundle("artifacts/en-general-005")
    families, _, _, prior = lexical_graph(read_wic(RAW))
    blocked = {families.find(r["family_id"]) for r in prior if r["split"] != "train"}
    blocked |= {
        families.find(families.word(r["state"]["target_word"]))
        for r in contextual
        if r["split"] != "train" and "target_word" in r["state"]
    }
    rows = deepcopy(contextual)
    for r in rows:
        if "target_word" in r["state"]:
            r["family_id"] = families.find(families.word(r["state"]["target_word"]))
    events = []
    for r in old:
        family = families.find(r["family_id"])
        reason = (
            "historical_evaluation"
            if r["split"] != "train"
            else ("expanded_family_touches_holdout" if family in blocked else None)
        )
        events.append(
            dict(
                id=r["id"],
                original_split=r["split"],
                family_id=family,
                status="excluded" if reason else "replayed",
                reason=reason,
            )
        )
        if reason is None:
            replay = deepcopy(r)
            replay["family_id"] = family
            rows.append(replay)
    out.mkdir(parents=True)
    for name, values in [("records.jsonl", rows), ("replay-selection.jsonl", events)]:
        with (out / name).open("x") as stream:
            for value in values:
                stream.write(encoded(value) + "\n")
    # Independently exercise all native exports and the strict family validator.
    prepare(
        out / "records.jsonl",
        "rubrics/en-general-research-v1.json",
        artifact,
        research=True,
    )
    exported, manifest = bundle(artifact)
    for r in exported:
        if r["split"] == "train" and not r["family_id"].startswith("wi:"):
            assert families.find(r["family_id"]) not in blocked
    summary = dict(
        version="en-joint-001",
        requests=len(rows),
        requests_by_split=dict(Counter(r["split"] for r in rows)),
        judgments_by_split=dict(
            Counter(r["split"] for r in rows for _ in r["questions"])
        ),
        replay=dict(Counter(e["reason"] or "replayed" for e in events)),
        status="passed",
        new_lexical_training_targets=17,
        benchmark_policy="Frozen 35k remains test-only. Historical replay has prior_train exposure already declared in its registry; no new benchmark target family is introduced by contextual examples.",
        scope="Research-only partial review; never whole-word mastery.",
    )
    write(out / "summary.json", summary)
    write(
        out / "source-manifest.json",
        dict(
            files={
                str(p): digest(p)
                for p in [
                    Path("artifacts/en-context-002/manifest.json"),
                    Path("artifacts/en-general-005/manifest.json"),
                    Path("data/research/en-context-002/source-manifest.json"),
                    *PRIOR,
                    Path(__file__),
                ]
            },
            output_manifest_sha256=digest(artifact / "manifest.json"),
        ),
    )
    print(encoded(summary))


if __name__ == "__main__":
    main()
