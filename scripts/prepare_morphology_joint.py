"""Freeze the next research bundle without submitting a training experiment."""

from pathlib import Path

from lev.dataset import bundle, prepare
from lev.io import digest, encoded, jsonl, read, write


def main():
    parent = Path("artifacts/en-joint-002")
    contribution = Path("data/annotations/en-morphology-relations-001")
    provenance = Path("data/annotations/en-joint-003")
    out = Path("artifacts/en-joint-003")
    if out.exists() or provenance.exists():
        raise ValueError("new output directories required")
    records, _ = bundle(parent)
    manifest = read(contribution / "manifest.json")
    if digest(contribution / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("contribution changed")
    if digest(parent / "manifest.json") != manifest["training_manifest_sha256"]:
        raise ValueError("parent changed")
    for name, expected in manifest["source_files"].items():
        if digest(name) != expected:
            raise ValueError("contribution source changed: " + name)
    additions = jsonl(contribution / "records.jsonl")
    if any(r["split"] != "train" or r["scope"] != "partial" for r in additions):
        raise ValueError("only partial training contributions permitted")
    if {r["id"] for r in records} & {r["id"] for r in additions}:
        raise ValueError("duplicate contribution ID")
    provenance.mkdir()
    with (provenance / "records.jsonl").open("x") as stream:
        for row in records + additions:
            stream.write(encoded(row) + "\n")
    prepare(provenance / "records.jsonl", Path("rubrics/en-general-research-v1.json"), out, research=True)
    for split in ("development", "calibration", "test"):
        if digest(out / (split + ".jsonl")) != digest(parent / (split + ".jsonl")):
            raise ValueError("heldout export changed: " + split)
    frozen, _ = bundle(out)
    by_id = {r["id"]: r for r in frozen}
    if any(by_id[r["id"]] != r for r in records + additions):
        raise ValueError("record changed during export")
    summary = dict(status="prepared_not_trained", added_requests=len(additions),
                   added_judgments=sum(len(r["questions"]) for r in additions),
                   added_target_words=len({r["state"]["headword"] for r in additions}),
                   total_training_requests=sum(r["split"] == "train" for r in frozen),
                   heldout_exports_byte_identical=True, all_parent_records_preserved=True,
                   artifact_manifest_sha256=digest(out / "manifest.json"),
                   parent_manifest_sha256=digest(parent / "manifest.json"),
                   contribution_manifest_sha256=digest(contribution / "manifest.json"),
                   exposure_manifest_sha256=digest("runs/en-morphology-relations-001-prospective-exposure/manifest.json"),
                   script_sha256=digest(__file__), training_submitted=False,
                   limitations="Research labels only. Inflection relations do not establish sentence understanding or complete word mastery. Previously inspected heldout sets are historical regression evidence. Tokenization, source-semantic audit and a preregistered experiment remain required before training.")
    write(provenance / "summary.json", summary)
    print(encoded(summary))


if __name__ == "__main__":
    main()
