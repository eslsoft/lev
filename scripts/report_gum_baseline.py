"""Verify and report the registered v04 contextual morphology baseline."""

from collections import defaultdict
from pathlib import Path

from build_context_corpus import key
from lev.dataset import bundle
from lev.io import digest, jsonl, read, write
from lev.report import align, summarize


def grouped(records, predictions):
    ids = {r["id"] for r in records}
    return summarize(records, [p for p in predictions if p["id"] in ids], 0.5)


def main():
    cycle = Path("runs/lev-joint005-morphology-development-cycle")
    registration = read(cycle / "registration.json")
    run = Path("runs/modal-" + registration["baseline_run_id"])
    out = cycle / "v04-baseline-report.json"
    if out.exists():
        raise ValueError("new report path required")
    for name in ("modal-status.json", "result/status.json"):
        status = read(run / name)
        if status["status"] != "succeeded" or status["returncode"] != 0:
            raise ValueError("baseline execution did not succeed")
    records, _ = bundle(run / "dataset")
    job = read(run / "result/job.json")
    expected = registration["dataset_manifest_sha256"]
    if digest(run / "dataset/manifest.json") != expected or job["dataset_sha256"] != expected:
        raise ValueError("baseline dataset identity mismatch")
    if job["config_sha256"] != registration["config_sha256"]:
        raise ValueError("baseline config identity mismatch")
    if job["split"] != "development" or job["calibration_sha256"] is not None:
        raise ValueError("expected uncalibrated development execution")
    for name, expected in registration["baseline_checkpoint"].items():
        if job["checkpoint"]["files"][name] != expected:
            raise ValueError("checkpoint mismatch: " + name)
    development = [r for r in records if r["split"] == "development"]
    predictions = read(run / "result/native/rows.json")
    align(development, predictions)
    contribution = Path("data/annotations/en-gum-morphology-development-001")
    manifest = read(contribution / "manifest.json")
    if digest(contribution / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("GUM contribution changed")
    gum = jsonl(contribution / "records.jsonl")
    if len(gum) != registration["gum_requests"] or sum(len(r["questions"]) for r in gum) != registration["gum_judgments"]:
        raise ValueError("GUM population differs from registration")
    actual = {r["id"]: r for r in development}
    if any(actual.get(r["id"]) != r for r in gum):
        raise ValueError("GUM contribution differs from scored records")
    candidate_root = Path("runs/en-gum-morphology-development-candidates-001")
    candidates_manifest = read(candidate_root / "manifest.json")
    if digest(candidate_root / "manifest.json") != manifest["candidate_manifest_sha256"]:
        raise ValueError("GUM candidate manifest changed")
    if digest(candidate_root / "candidates.jsonl") != candidates_manifest["candidates_sha256"]:
        raise ValueError("GUM candidate metadata changed")
    metadata = {}
    for row in jsonl(candidate_root / "candidates.jsonl"):
        locator = row["sentence_id"] + ":token=" + str(row["target_index"])
        metadata[locator] = row
    words, features = defaultdict(list), defaultdict(list)
    for row in gum:
        locator = next(iter(row["questions"].values()))["review"]["evidence"][0]["locator"]
        candidate = metadata[locator]
        feature, _ = min(candidate["features"], key=lambda p: key(candidate["sentence_id"] + str(candidate["target_index"]) + p[0]))
        words[candidate["word"]].append(row)
        features[feature].append(row)
    per_word = {word: grouped(group, predictions)["decisions"] for word, group in sorted(words.items())}
    gum_ids = {r["id"] for r in gum}
    report = dict(
        status="verified_execution_report", threshold=0.5,
        overall=grouped(development, predictions), gum=grouped(gum, predictions),
        parent=grouped([r for r in development if r["id"] not in gum_ids], predictions),
        by_feature={name: grouped(group, predictions)["decisions"] for name, group in sorted(features.items())},
        words=per_word, word_count=len(per_word),
        word_macro_accuracy=sum(v["accuracy_all_requested"]["rate"] for v in per_word.values()) / len(per_word),
        words_all_scheduled_correct=sum(v["accuracy_all_requested"]["numerator"] == v["requested"] for v in per_word.values()),
        registration_sha256=digest(cycle / "registration.json"),
        job_sha256=digest(run / "result/job.json"), predictions_sha256=digest(run / "result/native/rows.json"),
        script_sha256=digest(__file__), goal_complete=False,
        limitations="Partial source-derived development questions. All-scheduled-correct is not complete word mastery. Words share one document/lexical component, not independent samples. No final-test claim.",
    )
    write(out, report)
    print(report["gum"]["decisions"])


if __name__ == "__main__":
    main()
