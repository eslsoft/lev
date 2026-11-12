"""Wait for a development-selected calibrated model, then evaluate every fixed shard.

Runs at most two GPU jobs concurrently. Failed or interrupted submissions are
preserved and never automatically resubmitted under a different experiment id.
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
from pathlib import Path
import subprocess
import time

from lev.io import digest, read, write

CYCLE = Path("runs/lev-joint001-v03-cycle")
ROOT = Path("runs/lev-joint001-v03-vocabulary")
SHARDS = Path("artifacts/en-us-vocab-35000-v3-shards-01")


def main(version="v03"):
    if version not in ("v03", "v04"):
        raise ValueError("unknown registered experiment")
    global ROOT, CYCLE
    ROOT = Path(f"runs/lev-joint001-{version}-vocabulary")
    CYCLE = Path(f"runs/lev-joint001-{version}-cycle")
    ROOT.mkdir(exist_ok=True)
    own_lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(
        "Waiting for context cycle and frozen calibration; vocabulary model scores remain unopened.",
        flush=True,
    )
    while not (CYCLE / "result.json").exists():
        with (CYCLE / "controller.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError(
                    "Context controller exited without a result; inspect its preserved logs before proceeding"
                )
        time.sleep(30)
    result = read(CYCLE / "result.json")
    if result["status"] != "context-cycle-complete":
        write(
            ROOT / "result.json",
            dict(
                status="awaiting-next-development-candidate",
                context_result=result,
                final_goal_complete=False,
            ),
        )
        print(
            "Context gate was not met; vocabulary evaluation not launched.", flush=True
        )
        return
    admission = read("runs/en-us-vocab-35000-v3-tokenization.json")
    if admission["status"] != "passed" or admission["failures"]:
        raise ValueError("Full vocabulary tokenization admission did not pass")
    plan = read(SHARDS / "manifest.json")
    calibration = Path(result["final_run"]) / "calibration-fit/calibration.json"
    write(
        ROOT / "plan.json",
        dict(
            shards_manifest_sha256=digest(SHARDS / "manifest.json"),
            selection_sha256=digest(CYCLE / "selection.json"),
            calibration_sha256=digest(calibration),
            concurrency=2,
            shards=len(plan["shards"]),
            retries=0,
            threshold=0.5,
        ),
    )

    def run_shard(spec):
        i = spec["index"]
        name = f"lev-vocab35000-{version}-shard-{i:02d}-01"
        marker = ROOT / (name + ".started.json")
        command = [
            "uv",
            "run",
            "--extra",
            "modal",
            "python",
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
            f"configs/lev-4b-{version}.toml",
            "--kev",
            "/Users/saltbo/Develop/oss/kev",
            "--run-id",
            name,
            "--checkpoint",
            result["checkpoint"],
            "--shard",
            str(i),
            "--execute",
        ]
        write(marker, dict(command=command))
        print("Starting shard", i, flush=True)
        with (ROOT / (name + ".log")).open("x") as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        write(ROOT / (name + ".finished.json"), dict(returncode=status.returncode))
        if status.returncode:
            return dict(shard=i, status="failed", run_id=name)
        out = Path("runs") / ("modal-" + name)
        command = [
            "uv",
            "run",
            "--extra",
            "modal",
            "python",
            "scripts/download_modal_run.py",
            "--run-id",
            name,
            "--out",
            str(out),
        ]
        with (ROOT / (name + "-download.log")).open("x") as log:
            downloaded = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return dict(
            shard=i,
            status="succeeded" if downloaded.returncode == 0 else "download-failed",
            run_id=name,
            local=str(out),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run_shard, plan["shards"]))
    write(ROOT / "shard-outcomes.json", outcomes)
    if any(r["status"] != "succeeded" for r in outcomes):
        write(
            ROOT / "result.json",
            dict(
                status="incomplete-shards", outcomes=outcomes, final_goal_complete=False
            ),
        )
        raise RuntimeError("Some shards failed; no retry-and-pick merge")
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        "-m",
        "lev.vocabulary_jobs",
        "merge",
        "--dataset",
        "artifacts/en-us-vocab-35000-v3",
        "--shards",
        str(SHARDS),
        "--out",
        str(ROOT / "word-report.json"),
    ]
    for outcome in outcomes:
        command.extend(["--run", outcome["local"]])
    with (ROOT / "merge.log").open("x") as log:
        merged = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if merged.returncode:
        raise RuntimeError("Vocabulary merge failed")
    report = read(ROOT / "word-report.json")
    write(
        ROOT / "result.json",
        dict(
            status=report["status"],
            overall=report["overall"],
            word_report=str(ROOT / "word-report.json"),
            final_goal_complete=False,
            note="Measurement completed; accuracy and task coverage must be assessed against the continuing model goal.",
        ),
    )
    print(
        "All vocabulary shards scored and word report written; not a claim of full word mastery.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v03", "v04"), default="v03")
    main(parser.parse_args().version)
