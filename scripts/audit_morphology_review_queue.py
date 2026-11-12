"""Reserve prior holdout families before reviewing new morphology contexts."""

from collections import Counter, defaultdict
from pathlib import Path

from audit_corroborated_families import verified_relations
from build_context_corpus import RAW, lexical_graph, read_wic, source_files
from lev.dataset import bundle
from lev.io import digest, encoded, jsonl, read, write
from lev.vocabulary import verify_bundle


def main():
    queue = Path("runs/en-morphology-001-review-queue-001")
    training = Path("artifacts/en-joint-002")
    context = Path("artifacts/en-us-vocab-context-002")
    out = Path("runs/en-morphology-001-review-queue-family-audit-001")
    if out.exists():
        raise ValueError("new output directory required")
    if digest(queue / "queue.jsonl") != read(queue / "manifest.json")["queue_sha256"]:
        raise ValueError("review queue changed")
    verify_bundle(context)
    records, _ = bundle(training)
    families, _, _, prior = lexical_graph(read_wic(RAW))
    audits = [Path(f"runs/en-morphology-001-dictionary-audit-{n:03d}") for n in (1, 2)]
    for audit in audits:
        for row in verified_relations(audit):
            if not row["dictionary_support"]:
                raise ValueError("corroborated relation lacks evidence")
            families.union([families.word(row["target"]), families.word(row["form"])])
    exposure = defaultdict(list)
    for source, rows in (("historical", prior), ("joint002", records),
                         ("context002", jsonl(context / "records.jsonl"))):
        for row in rows:
            exposure[families.find(row["family_id"])].append(
                dict(source=source, id=row["id"], split=row["split"]))
    results = []
    for row in jsonl(queue / "queue.jsonl"):
        candidate = row["candidate"]
        word = candidate["target"]
        root = families.find(families.word(word))
        seen = exposure[root]
        blocked = any(r["split"] != "train" for r in seen)
        results.append(dict(word=word, family=root, prior_exposure=seen,
                            status="excluded_prior_holdout_family" if blocked else "family_eligible_pending_label_review",
                            label=None, training_exported=False))
    if len({r["word"] for r in results}) != len(results):
        raise ValueError("duplicate queue headword")
    out.mkdir()
    with (out / "words.jsonl").open("x") as stream:
        for row in results:
            stream.write(encoded(row) + "\n")
    report = dict(words=len(results), status_counts=dict(Counter(r["status"] for r in results)),
                  words_sha256=digest(out / "words.jsonl"), queue_sha256=digest(queue / "queue.jsonl"),
                  training_manifest_sha256=digest(training / "manifest.json"),
                  context_manifest_sha256=digest(context / "manifest.json"),
                  corroboration_manifests={str(p): digest(p / "manifest.json") for p in audits},
                  family_source_files=source_files(), script_sha256=digest(__file__),
                  model_scores_read=False, labels_assigned=0, training_exported=False,
                  note="Family eligibility alone does not admit a draft label. Historical vocabulary inventory exposure remains separately disclosed; no unseen-word or pretraining claim.")
    write(out / "manifest.json", report)
    print(encoded({k: report[k] for k in ("words", "status_counts", "labels_assigned")}))


if __name__ == "__main__":
    main()
