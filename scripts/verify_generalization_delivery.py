"""Verify the corrected evaluation against frozen weights, data and raw execution."""

from pathlib import Path

from lev.dataset import bundle
from lev.io import digest, fingerprint, read, write
from lev.report import summarize
from lev.workflow import check_calibration, model_identity


def main():
    training = Path("runs/modal-lev-general004-v02-train-01")
    final = Path("runs/modal-lev-general005-v02-final-01")
    dataset = Path("artifacts/en-general-005")
    records, manifest = bundle(dataset)
    old, _ = bundle("artifacts/en-general-004")
    old_by_id = {r["id"]: r for r in old}
    new_by_id = {r["id"]: r for r in records}
    correction = read("data/research/en-general-005/correction.json")
    assert set(old_by_id) - set(new_by_id) == set(correction["excluded_ids"])
    assert set(new_by_id) - set(old_by_id) == {correction["replacement_id"]}
    assert all(old_by_id[k] == new_by_id[k] for k in set(old_by_id) & set(new_by_id))
    unchanged = {}
    for split in ("train", "development", "calibration"):
        name = split + ".jsonl"
        assert digest(dataset / name) == digest(Path("artifacts/en-general-004") / name)
        unchanged[name] = digest(dataset / name)
    assert read("data/research/en-general-005/family-audit.json")["status"] == "passed"
    assert read("runs/en-general005-full-source-audit.json")["checked"][
        "all_source_decisions"
    ] == sum(len(r["questions"]) for r in records)
    assert (
        read("runs/en-general004-v02-training-verification.json")["status"] == "passed"
    )
    assert read(final / "modal-status.json")["status"] == "succeeded"
    assert read(final / "experiment.json")["status"] == "succeeded"
    selection = read("runs/en-general004-v02-selection.json")
    assert (
        selection["candidate_selected"]
        and selection["status"] == "selected-before-test"
    )
    assert selection["comparison_sha256"] == digest(
        "runs/en-general004-candidate-comparison.json"
    )
    assert read("runs/en-general004-candidate-comparison.json")["identities"]["lev02"][
        "files"
    ] == fingerprint(training / "result/checkpoint")
    identity = model_identity(str(training / "result/checkpoint"))
    cal = check_calibration(final / "calibration-fit/calibration.json", identity)
    assert cal["dataset_sha256"] == digest(dataset / "manifest.json")
    assert cal["rows_sha256"] == digest(final / "calibration-read/native/rows.json")
    assert cal["questions"] == sum(
        len(r["questions"]) for r in records if r["split"] == "calibration"
    )
    reports = {}
    for stage, split in [
        ("calibration-read", "calibration"),
        ("test", "test"),
        ("comparison-0-test", "test"),
        ("comparison-1-test", "test"),
    ]:
        directory = final / stage
        assert read(directory / "status.json") == {
            "returncode": 0,
            "status": "succeeded",
        }
        job = read(directory / "job.json")
        assert (
            job["dataset_sha256"] == digest(dataset / "manifest.json")
            and job["split"] == split
        )
        assert job["created_at"] > selection["decided_at"]
        assert all(
            digest(Path("runs/en-general005-source/src/lev") / name) == sha
            for name, sha in job["lev_source"].items()
        )
        rows = read(directory / "native/rows.json")
        expected = [r for r in records if r["split"] == split]
        report = summarize(expected, rows, 0.5)
        report.update(research_only=manifest["research_only"], split=split)
        assert report == read(directory / "lev-report.json")
        assert report["decisions"]["missing"] == 0
        context = read(directory / "native/lev-context.json")
        assert context["data_sha256"] == digest(dataset / (split + ".jsonl"))
        assert (
            context["coverage"]["rejected_records"]
            == context["coverage"]["truncated_records"]
            == 0
        )
        if stage in ("calibration-read", "test"):
            assert job["checkpoint"]["files"] == identity["files"]
        elif stage == "comparison-0-test":
            assert (
                job["checkpoint"]["hub"]
                == "jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
            )
        else:
            assert job["checkpoint"]["files"] == fingerprint(
                "runs/modal-lev-general-20261021-01/result/checkpoint"
            )
        if stage == "test":
            assert job["calibration_sha256"] == digest(
                final / "calibration-fit/calibration.json"
            )
            assert context["inference_temperature"] == cal["temperature"]
        reports[stage] = report["decisions"]
    assert read(final / "calibration-fit/status.json") == {
        "returncode": 0,
        "status": "succeeded",
    }
    write(
        "runs/en-general005-delivery-verification.json",
        dict(
            status="passed",
            training_dataset_sha256=digest("artifacts/en-general-004/manifest.json"),
            evaluation_dataset_sha256=digest(dataset / "manifest.json"),
            unchanged_exports=unchanged,
            checkpoint=identity,
            calibration_sha256=digest(final / "calibration-fit/calibration.json"),
            temperature=cal["temperature"],
            selection=selection,
            excluded_request_ids=correction["excluded_ids"],
            replacement_id=correction["replacement_id"],
            reports=reports,
            note="Weights and selection frozen before tests. 005 corrects one source-discovered family overlap; 930 test decisions are shared with 004, so this is not an independent replication.",
        ),
    )
    print(
        "Verified frozen model selection, identical train/development/calibration data, all source labels, family isolation, relocated calibration binding, raw reports and every evaluation source hash."
    )


if __name__ == "__main__":
    main()
