"""Score the admitted context supplement after the v04 vocabulary jobs finish."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import time

from lev.io import digest, read, write
from lev.vocabulary_jobs import verify_shards

DATA = Path("artifacts/en-us-vocab-context-002")
SHARDS = Path("artifacts/en-us-vocab-context-002-shards-01")
ADMISSION = Path("runs/en-us-vocab-context-002-admission.json")
CYCLE = Path("runs/lev-joint001-v04-cycle")
BASE = Path("runs/lev-joint001-v04-vocabulary")
ROOT = Path("runs/lev-joint001-v04-context-supplement")
PYTHON = ["uv", "run", "--extra", "modal", "python"]


def verify_admission():
    admission = read(ADMISSION)
    if admission["status"] != "data_admitted_pending_selected_model_and_calibration":
        raise ValueError("context data not admitted")
    expected = {
        str(DATA / "manifest.json"): admission["dataset_manifest_sha256"],
        str(SHARDS / "manifest.json"): admission["shards_manifest_sha256"],
        **admission["evidence"],
    }
    for path, sha in expected.items():
        if digest(path) != sha:
            raise ValueError(f"admitted evidence changed: {path}")
    plan = verify_shards(SHARDS)
    if plan["parent_manifest_sha256"] != digest(DATA / "manifest.json"):
        raise ValueError("context shards belong to another dataset")
    return plan


def wait_for_base():
    while not (BASE / "result.json").exists():
        with (BASE / "controller.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError("Vocabulary controller exited without a result")
        time.sleep(30)
    result = read(BASE / "result.json")
    return result["status"] == "complete"


def execute(name, command):
    write(ROOT / f"{name}.started.json", dict(command=command))
    with (ROOT / f"{name}.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    write(ROOT / f"{name}.finished.json", dict(returncode=result.returncode))
    if result.returncode:
        raise RuntimeError(f"{name} failed; retained without automatic retry")


def main(execute_jobs=False):
    plan = verify_admission()
    if not execute_jobs:
        print(f"Admission verified: {len(plan['shards'])} shards; no model execution")
        return
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Waiting for completed base vocabulary evaluation", flush=True)
        if not wait_for_base():
            write(
                ROOT / "result.json",
                dict(status="predecessor-not-complete", final_goal_complete=False),
            )
            return
        plan = verify_admission()
        cycle = read(CYCLE / "result.json")
        if cycle["status"] != "context-cycle-complete":
            raise ValueError("selected calibrated cycle is incomplete")
        calibration = Path(cycle["final_run"]) / "calibration-fit/calibration.json"
        write(
            ROOT / "plan.json",
            dict(
                admission_sha256=digest(ADMISSION),
                selection_sha256=digest(CYCLE / "selection.json"),
                calibration_sha256=digest(calibration),
                checkpoint=cycle["checkpoint"],
                concurrency=1,
                retries=0,
                threshold=0.5,
                base_result_sha256=digest(BASE / "result.json"),
            ),
        )
        outputs = []
        for spec in plan["shards"]:
            index = spec["index"]
            name = f"lev-context002-v04-shard-{index:02d}-01"
            out = Path("runs") / ("modal-" + name)
            execute(
                name,
                PYTHON
                + [
                    "-m",
                    "lev.vocabulary_jobs",
                    "run",
                    "--shards",
                    str(SHARDS),
                    "--selection",
                    str(CYCLE / "selection.json"),
                    "--calibration",
                    str(calibration),
                    "--config",
                    "configs/lev-4b-v04.toml",
                    "--kev",
                    "/Users/saltbo/Develop/oss/kev",
                    "--run-id",
                    name,
                    "--checkpoint",
                    cycle["checkpoint"],
                    "--shard",
                    str(index),
                    "--execute",
                ],
            )
            execute(
                name + "-download",
                PYTHON
                + [
                    "scripts/download_modal_run.py",
                    "--run-id",
                    name,
                    "--out",
                    str(out),
                ],
            )
            outputs.extend(["--run", str(out)])
        report_path = ROOT / "word-report.json"
        execute(
            "merge",
            PYTHON
            + [
                "-m",
                "lev.vocabulary_jobs",
                "merge",
                "--dataset",
                str(DATA),
                "--shards",
                str(SHARDS),
                "--out",
                str(report_path),
            ]
            + outputs,
        )
        report = read(report_path)
        write(
            ROOT / "result.json",
            dict(
                status=report["status"],
                overall=report["overall"],
                word_report=str(report_path),
                final_goal_complete=False,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    main(parser.parse_args().execute)
