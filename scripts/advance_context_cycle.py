"""Advance the authorized v0.3 cycle; never resubmit a recorded stage or select on test."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from lev.dataset import bundle
from lev.io import digest, fingerprint, read, write
from lev.report import summarize

ROOT = Path("runs/lev-joint001-v03-cycle")
DATA = Path("artifacts/en-joint-001")
OLD = Path("artifacts/en-general-005")
CONFIG = "configs/lev-4b-v03.toml"
SMOKE = "lev-joint001-v03-smoke-01"
TRAIN = "lev-joint001-v03-train-01"
INCUMBENT = "/lev-store/lev-general004-v02-train-01/result/checkpoint"
# Attempt 01 was preempted during inference; its failure and partial outputs remain.
# Attempt 02 repeats the fixed population, never selecting between attempt scores.
INCUMBENT_DEV = "lev-joint001-v02-dev-02"
CANDIDATE = f"/lev-store/{TRAIN}/result/checkpoint"


def gate(new_candidate, new_incumbent, retention_candidate, retention_incumbent):
    reports = [new_candidate, new_incumbent, retention_candidate, retention_incumbent]

    def rate(report):
        return report["decisions"]["accuracy_all_requested"]["rate"]

    kind = "passage_correction_acceptability"
    return (
        all(r["decisions"]["missing"] == 0 for r in reports)
        and rate(new_candidate) > rate(new_incumbent)
        and new_candidate["by_kind"][kind]["accuracy_all_requested"]["rate"]
        >= new_incumbent["by_kind"][kind]["accuracy_all_requested"]["rate"]
        and rate(retention_candidate) >= rate(retention_incumbent) - 0.01
    )


def main(version="v03"):
    import modal

    if version not in ("v03", "v04"):
        raise ValueError("unknown registered experiment")
    root = Path(f"runs/lev-joint001-{version}-cycle")
    config_path = f"configs/lev-4b-{version}.toml"
    train = f"lev-joint001-{version}-train-01"
    if version == "v04":
        recovery = read(root / "preemption-recovery-01.json")
        if (
            recovery["failed_run"] != train
            or recovery["replacement_run"] != "lev-joint001-v04-train-02"
            or recovery["config_sha256"] != digest(config_path)
            or recovery["dataset_sha256"] != digest(DATA / "manifest.json")
            or read(Path(recovery["failed_artifacts"]) / "modal-status.json")["status"]
            != "failed"
            or recovery["platform_log_sha256"] != digest(recovery["platform_log"])
        ):
            raise ValueError("registered preemption recovery evidence differs")
        train = recovery["replacement_run"]
    candidate_checkpoint = f"/lev-store/{train}/result/checkpoint"
    # Stage paths are per experiment; historical results are never overwritten.
    global ROOT, CONFIG, TRAIN, CANDIDATE
    ROOT, CONFIG, TRAIN, CANDIDATE = root, config_path, train, candidate_checkpoint
    ROOT.mkdir(exist_ok=True)
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, "LEV_KEV_PATH": "/Users/saltbo/Develop/oss/kev"}
    volume = modal.Volume.from_name("lev-experiments")

    def wait_remote(run, seconds):
        deadline = time.monotonic() + seconds
        while True:
            try:
                status = json.loads(
                    b"".join(volume.read_file(run + "/modal-status.json"))
                )
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("No final remote status: " + run)
                time.sleep(30)
                continue
            if status["status"] != "succeeded":
                raise RuntimeError(f"{run}: {status}")
            return status

    def stage(name, command, remote=None, timeout=14400):
        start, finish = (
            ROOT / (name + ".started.json"),
            ROOT / (name + ".finished.json"),
        )
        if finish.exists():
            if read(finish)["returncode"] != 0:
                raise RuntimeError("Recorded failed stage; no automatic retry: " + name)
            return
        if start.exists():
            if remote:
                wait_remote(remote, timeout)
                write(finish, dict(returncode=0, recovered_from_remote_status=True))
                return
            raise RuntimeError(
                "Interrupted local stage requires evidence inspection: " + name
            )
        write(start, dict(command=command, remote_run=remote))
        print("Starting", name, flush=True)
        with (ROOT / (name + ".log")).open("x") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        write(finish, dict(returncode=result.returncode))
        if result.returncode:
            raise RuntimeError("Failed stage: " + name)
        print("Completed", name, flush=True)

    def download(run):
        out = Path("runs") / ("modal-" + run)
        stage(
            "download-" + run,
            [
                "uv",
                "run",
                "--extra",
                "modal",
                "python",
                "scripts/download_modal_run.py",
                "--run-id",
                run,
                "--out",
                str(out),
            ],
        )
        return out

    def remote(
        operation, run, dataset=DATA, checkpoint=None, timeout=14400, comparisons=()
    ):
        command = [
            "uv",
            "run",
            "--extra",
            "modal",
            "python",
            "-m",
            "lev.modal_runner",
            operation,
            "--dataset",
            str(dataset),
            "--config",
            CONFIG,
            "--run-id",
            run,
            "--timeout",
            str(timeout),
            "--execute",
        ]
        if checkpoint:
            command += ["--checkpoint", checkpoint]
        for model in comparisons:
            command += ["--compare-checkpoint", model]
        stage(run, command, remote=run, timeout=timeout)
        return download(run)

    if version == "v03":
        print(
            "Waiting for existing long-state smoke; no duplicate submission.",
            flush=True,
        )
        wait_remote(SMOKE, 3600)
        smoke = download(SMOKE)
        stage(
            "verify-smoke",
            [
                "uv",
                "run",
                "--extra",
                "modal",
                "python",
                "scripts/verify_training_run.py",
                "--run",
                str(smoke),
                "--dataset",
                "artifacts/en-joint-001-smoke",
                "--out",
                str(ROOT / "smoke-verification.json"),
            ],
        )
    else:
        from lev.workflow import config

        previous = config("configs/lev-4b-v03.toml")
        current = config(CONFIG)
        previous.pop("model_name")
        current.pop("model_name")
        assert previous["training"].pop("epochs") == 1
        assert current["training"].pop("epochs") == 2
        if current != previous:
            raise ValueError(
                "v0.4 resource admission only covers an epoch-count change"
            )
        verified = read("runs/lev-joint001-v03-cycle/training-verification.json")
        if verified["status"] != "passed" or verified["dataset_sha256"] != digest(
            DATA / "manifest.json"
        ):
            raise ValueError("previous full-data resource admission failed")
        print(
            "Reusing successful full-data resource admission; only epoch count changes.",
            flush=True,
        )
    trained = remote("train", TRAIN)
    stage(
        "verify-train",
        [
            "uv",
            "run",
            "python",
            "scripts/verify_training_run.py",
            "--run",
            str(trained),
            "--dataset",
            str(DATA),
            "--out",
            str(ROOT / "training-verification.json"),
        ],
    )
    if version == "v03":
        incumbent = remote(
            "benchmark", INCUMBENT_DEV, checkpoint=INCUMBENT, timeout=3600
        )
    else:
        incumbent = Path("runs/modal-" + INCUMBENT_DEV)
        if read(incumbent / "result/status.json")["status"] != "succeeded":
            raise ValueError("incumbent evaluation did not succeed")
        if digest(incumbent / "dataset/development.jsonl") != digest(
            DATA / "development.jsonl"
        ):
            raise ValueError("incumbent population differs")
        if read(incumbent / "result/job.json")["checkpoint"]["files"] != fingerprint(
            "runs/modal-lev-general004-v02-train-01/result/checkpoint"
        ):
            raise ValueError("incumbent checkpoint differs")
    candidate = remote(
        "benchmark",
        f"lev-joint001-{version}-dev-01",
        checkpoint=CANDIDATE,
        timeout=3600,
    )
    retention = remote(
        "benchmark",
        f"lev-general005-{version}-retention-01",
        dataset=OLD,
        checkpoint=CANDIDATE,
        timeout=3600,
    )
    prior = Path("runs/modal-lev-general004-v02-dev-01")
    if digest(prior / "dataset/development.jsonl") != digest(OLD / "development.jsonl"):
        raise ValueError(
            "Historical development differs; cannot reuse incumbent evidence"
        )
    if read(prior / "result/status.json")["status"] != "succeeded":
        raise ValueError("Historical incumbent evaluation failed")
    if read(prior / "result/job.json")["checkpoint"]["files"] != fingerprint(
        "runs/modal-lev-general004-v02-train-01/result/checkpoint"
    ):
        raise ValueError("Historical development checkpoint differs from incumbent")
    rows, _ = bundle(DATA)
    oldrows, _ = bundle(OLD)

    def report(directory, records):
        return summarize(
            [r for r in records if r["split"] == "development"],
            read(directory / "result/native/rows.json"),
            0.5,
        )

    reports = dict(
        candidate=report(candidate, rows),
        incumbent=report(incumbent, rows),
        retention_candidate=report(retention, oldrows),
        retention_incumbent=report(prior, oldrows),
    )
    decision = ROOT / "selection.json"
    if not decision.exists():
        selected = gate(
            *[
                reports[k]
                for k in [
                    "candidate",
                    "incumbent",
                    "retention_candidate",
                    "retention_incumbent",
                ]
            ]
        )
        write(
            decision,
            dict(
                candidate_selected=selected,
                reports=reports,
                checkpoint=CANDIDATE,
                checkpoint_files=fingerprint(trained / "result/checkpoint"),
                dataset_sha256=digest(DATA / "manifest.json"),
                protocol_sha256=digest(f"docs/experiments/en-joint-001-{version}.md"),
                test_scored=False,
                rule="New total accuracy improves, passage accuracy does not decline, historical total decline <= 1 percentage point, no missing requests.",
            ),
        )
    if not read(decision)["candidate_selected"]:
        write(
            ROOT / "result.json",
            dict(
                status="needs-next-development-iteration",
                test_opened=False,
                final_goal_complete=False,
            ),
        )
        print(
            "Gate not met. All results preserved; continuing goal needs next development iteration.",
            flush=True,
        )
        return
    final = remote(
        "finalize",
        f"lev-joint001-{version}-final-01",
        checkpoint=CANDIDATE,
        timeout=3600,
        comparisons=(INCUMBENT,),
    )
    stage(
        "verify-final",
        [
            "uv",
            "run",
            "python",
            "scripts/verify_training_run.py",
            "--run",
            str(trained),
            "--dataset",
            str(DATA),
            "--benchmark",
            str(final / "test"),
            "--out",
            str(ROOT / "final-verification.json"),
        ],
    )
    write(
        ROOT / "result.json",
        dict(
            status="context-cycle-complete",
            checkpoint=CANDIDATE,
            final_run=str(final),
            vocabulary_35000_evaluation_pending=True,
            final_goal_complete=False,
        ),
    )
    print(
        "Context cycle complete; full vocabulary evaluation remains required.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v03", "v04"), default="v03")
    main(parser.parse_args().version)
