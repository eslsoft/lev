"""Apply registered gates to verified development runs; never read test scores."""

import argparse
from pathlib import Path

from lev.dataset import bundle
from lev.io import digest, jsonl, read, write
from lev.report import summarize


def score(run, dataset_hash, config_hash, checkpoint):
    for name in ("modal-status.json", "result/status.json"):
        status = read(run / name)
        if status != {"status": "succeeded", "returncode": 0}:
            raise ValueError("run did not succeed: " + str(run))
    records, _ = bundle(run / "dataset")
    job = read(run / "result/job.json")
    if job["split"] != "development" or job["calibration_sha256"] is not None:
        raise ValueError("expected uncalibrated development run")
    if digest(run / "dataset/manifest.json") != dataset_hash or job["dataset_sha256"] != dataset_hash:
        raise ValueError("dataset identity mismatch")
    if job["config_sha256"] != config_hash:
        raise ValueError("config identity mismatch")
    for name in ("adapter_model.safetensors", "head.pt"):
        if job["checkpoint"]["files"][name] != digest(checkpoint / name):
            raise ValueError("checkpoint identity mismatch")
    records = [r for r in records if r["split"] == "development"]
    rows = read(run / "result/native/rows.json")
    return records, rows, summarize(records, rows, .5)


def subset(records, rows):
    ids = {r["id"] for r in records}
    return summarize(records, [r for r in rows if r["id"] in ids], .5)["decisions"]


def check(root):
    plan = read(root / "gate-verification-plan.json")
    if digest(root / "registration.json") != plan["registration_sha256"]:
        raise ValueError("registration changed")
    registration = read(root / "registration.json")
    checkpoint = Path(plan["training_run"]) / "result/checkpoint"
    if read(root / "training-verification.json")["status"] != "passed":
        raise ValueError("training verification did not pass")
    records, rows, development = score(Path(plan["development_run"]), registration["dataset_sha256"], registration["config_sha256"], checkpoint)
    _, _, retention = score(Path(plan["retention_run"]), plan["retention_dataset_sha256"], registration["config_sha256"], checkpoint)
    correct = lambda m: m["accuracy_all_requested"]["numerator"]
    measured = dict(retention_total_correct=correct(retention["decisions"]),
                    retention_grammar_correct=correct(retention["by_kind"]["grammaticality"]),
                    grammar_correct=correct(retention["by_kind"]["grammaticality"]),
                    missing=development["decisions"]["missing"] + retention["decisions"]["missing"])
    for kind, name in (("contextual_sense", "context_correct"), ("passage_correction_acceptability", "passage_correct")):
        measured[name] = correct(development["by_kind"][kind])
    if plan["version"] == "v06":
        contribution = Path("data/annotations/en-gum-morphology-development-001/records.jsonl")
        if digest(contribution) != plan["gum_records_sha256"]:
            raise ValueError("GUM contribution changed")
        gum = jsonl(contribution)
        lookup = {r["id"]: r for r in records}
        if any(lookup.get(r["id"]) != r for r in gum):
            raise ValueError("GUM records differ from scored bundle")
        gum_ids = {r["id"] for r in gum}
        measured["gum_correct"] = correct(subset(gum, rows))
        measured["parent_development_correct"] = correct(subset([r for r in records if r["id"] not in gum_ids], rows))
        for term, feature in (("number", "number"), ("tense", "tense"), ("verb form", "verbform")):
            selected = [r for r in gum if all(term in q["claim"] for q in r["questions"].values())]
            if not selected:
                raise ValueError("missing GUM feature population: " + feature)
            measured["gum_" + feature + "_correct"] = correct(subset(selected, rows))
    decisions = {}
    for name, threshold in registration["gate"].items():
        metric, direction = name.rsplit("_", 1)
        if direction not in {"min", "max"}:
            raise ValueError("unknown gate direction")
        value = measured[metric]
        decisions[name] = dict(value=value, threshold=threshold,
                               passed=value >= threshold if direction == "min" else value <= threshold)
    out = root / "development-gate-result.json"
    if out.exists():
        raise ValueError("new gate result required")
    write(out, dict(status="passed" if all(d["passed"] for d in decisions.values()) else "rejected",
                    gates=decisions, measured=measured, development=development, retention=retention,
                    plan_sha256=digest(root / "gate-verification-plan.json"), script_sha256=digest(__file__),
                    final_goal_complete=False, test_execution=False))
    print(read(out)["status"], decisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, required=True)
    check(parser.parse_args().cycle)
