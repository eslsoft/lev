"""Report candidate-specific supervised exposure without rewriting frozen history."""

import argparse
from pathlib import Path

from .io import digest, fingerprint, jsonl, read, write
from .report import rate


def summarize_exposure(report_path, exposure, checkpoint):
    report = read(report_path)
    manifest = read(exposure / "manifest.json")
    if manifest["status"] != "passed":
        raise ValueError("exposure audit did not pass")
    if digest(exposure / "words.jsonl") != manifest["words_sha256"]:
        raise ValueError("exposure rows changed")
    if report["dataset_manifest_sha256"] != manifest["vocabulary_manifest_sha256"]:
        raise ValueError("exposure audit belongs to another vocabulary benchmark")
    if fingerprint(checkpoint) != report["inference_identity"]["checkpoint_files"]:
        raise ValueError("report belongs to another checkpoint")
    model = read(checkpoint / "lev-model.json")
    if model["dataset_sha256"] != manifest["training_manifest_sha256"]:
        raise ValueError("exposure audit belongs to another training dataset")
    rows = jsonl(exposure / "words.jsonl")
    by_word = {row["word"]: row for row in rows}
    scored = report["words"]
    if (
        len(by_word) != len(rows)
        or len(rows) != manifest["words"]
        or len({row["word"] for row in scored}) != len(scored)
        or set(by_word) != {row["word"] for row in scored}
    ):
        raise ValueError("exposure and scored word populations differ")
    groups = {"seen": [], "not_detected": []}
    for row in scored:
        exposure_row = by_word[row["word"]]
        seen = exposure_row["candidate_supervised_target_family_seen"]
        if type(seen) is not bool:
            raise ValueError("exposure must be Boolean")
        if (
            row["rank"] != exposure_row["rank"]
            or row["exposure"] != exposure_row["frozen_historical_exposure"]
        ):
            raise ValueError("frozen word metadata differs")
        groups["seen" if seen else "not_detected"].append(row)
    summaries = {}
    for key, group in groups.items():
        expected = sum(row["expected"] for row in group)
        completed = sum(row["completed"] for row in group)
        summaries[key] = dict(
            words=len(group),
            expected_judgments=expected,
            completed_judgments=completed,
            missing_judgments=expected - completed,
            judgment_accuracy=rate(sum(row["correct"] for row in group), expected),
            words_all_scheduled_items_correct=rate(
                sum(row["all_scheduled_items_correct"] for row in group), len(group)
            ),
            false_acceptances=sum(row["false_acceptances"] for row in group),
            false_rejections=sum(row["false_rejections"] for row in group),
        )
    return dict(
        status=report["status"],
        by_candidate_supervised_target_exposure=summaries,
        word_report_sha256=digest(report_path),
        exposure_manifest_sha256=digest(exposure / "manifest.json"),
        inference_identity=report["inference_identity"],
        interpretation="Known supervised target-family exposure only. Not detected does not mean unseen in pretraining. Passing scheduled partial items does not prove complete word mastery. Missing predictions stay in denominators.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("report", "exposure", "checkpoint", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    write(args.out, summarize_exposure(args.report, args.exposure, args.checkpoint))


if __name__ == "__main__":
    main()
