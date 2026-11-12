"""Finish the authorized fixed v0.2 cycle; model selection precedes test access."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from lev.io import digest, read, write


def select_candidate(reports):
    candidate, incumbent = reports["lev02"], reports["lev01"]
    return (
        candidate["decisions"]["missing"] == 0
        and candidate["macro_accuracy"] > incumbent["macro_accuracy"]
        and candidate["by_kind"]["grammaticality"]["accuracy_all_requested"]["rate"]
        >= incumbent["by_kind"]["grammaticality"]["accuracy_all_requested"]["rate"]
    )


def main():
    import modal

    p = argparse.ArgumentParser()
    p.add_argument("--train-run", default="lev-general004-v02-train-01")
    a = p.parse_args()
    run = a.train_run
    dataset = "artifacts/en-general-004"
    config = "configs/lev-4b-v02.toml"
    root = Path("runs")
    local = root / ("modal-" + run)
    dev_run = "lev-general004-v02-dev-01"
    final_run = "lev-general004-v02-final-01"
    volume = modal.Volume.from_name("lev-experiments")
    deadline = time.monotonic() + 10800
    print(
        "Waiting for the existing training run; no duplicate training submission.",
        flush=True,
    )
    while True:
        try:
            status = json.loads(b"".join(volume.read_file(run + "/modal-status.json")))
        except FileNotFoundError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "No completed training status before watcher deadline"
                )
            time.sleep(30)
            continue
        if status["status"] != "succeeded":
            raise RuntimeError(f"Training did not succeed: {status}")
        break
    env = os.environ.copy()
    env["LEV_KEV_PATH"] = "/Users/saltbo/Develop/oss/kev"

    def call(*args):
        print("Executing:", " ".join(args), flush=True)
        subprocess.run(list(args), check=True, env=env)

    call(
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        "scripts/download_modal_run.py",
        "--run-id",
        run,
        "--out",
        str(local),
    )
    call(
        "uv",
        "run",
        "python",
        "scripts/verify_training_run.py",
        "--run",
        str(local),
        "--dataset",
        dataset,
        "--out",
        str(root / "en-general004-v02-training-verification.json"),
    )
    checkpoint = "/lev-store/" + run + "/result/checkpoint"
    call(
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        "-m",
        "lev.modal_runner",
        "benchmark",
        "--dataset",
        dataset,
        "--config",
        config,
        "--run-id",
        dev_run,
        "--checkpoint",
        checkpoint,
        "--timeout",
        "1800",
        "--execute",
    )
    call(
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        "scripts/download_modal_run.py",
        "--run-id",
        dev_run,
        "--out",
        str(root / ("modal-" + dev_run)),
    )
    comparison = root / "en-general004-candidate-comparison.json"
    call(
        "uv",
        "run",
        "python",
        "scripts/compare_generalization.py",
        "--dataset",
        dataset,
        "--run",
        "kev=runs/modal-lev-general004-baseline-dev-01/result",
        "--run",
        "lev01=runs/modal-lev-general004-incumbent-dev-01/result",
        "--run",
        "lev02=" + str(root / ("modal-" + dev_run) / "result"),
        "--out",
        str(comparison),
    )
    report = read(comparison)
    selected = select_candidate(report["reports"])
    decision = dict(
        candidate_selected=selected,
        selected_checkpoint=checkpoint if selected else None,
        decided_at=datetime.now(timezone.utc).isoformat(),
        comparison_sha256=digest(comparison),
        dataset_sha256=digest(Path(dataset) / "manifest.json"),
        test_scored=False,
        rule="Candidate macro development accuracy strictly improves on incumbent, with no grammar accuracy decline and no missing outputs.",
        status="selected-before-test"
        if selected
        else "needs-next-development-iteration",
    )
    write(root / "en-general004-v02-selection.json", decision)
    if not selected:
        print(
            "Candidate did not meet the fixed development gate. Blind test remains unscored.",
            flush=True,
        )
        return
    call(
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        "-m",
        "lev.modal_runner",
        "finalize",
        "--dataset",
        dataset,
        "--config",
        config,
        "--run-id",
        final_run,
        "--checkpoint",
        checkpoint,
        "--compare-checkpoint",
        "jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        "--compare-checkpoint",
        "/lev-store/lev-general-20261021-01/result/checkpoint",
        "--timeout",
        "3600",
        "--execute",
    )
    destination = root / ("modal-" + final_run)
    call(
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        "scripts/download_modal_run.py",
        "--run-id",
        final_run,
        "--out",
        str(destination),
    )
    call(
        "uv",
        "run",
        "python",
        "scripts/compare_generalization.py",
        "--dataset",
        dataset,
        "--split",
        "test",
        "--run",
        "kev=" + str(destination / "comparison-0-test"),
        "--run",
        "lev01=" + str(destination / "comparison-1-test"),
        "--run",
        "lev02=" + str(destination / "test"),
        "--out",
        str(root / "en-general004-final-comparison.json"),
    )
    call(
        "uv",
        "run",
        "python",
        "scripts/verify_training_run.py",
        "--run",
        str(local),
        "--dataset",
        dataset,
        "--benchmark",
        str(destination / "test"),
        "--out",
        str(root / "en-general004-v02-final-verification.json"),
    )
    print(
        "Fixed cycle completed; inspect the preserved reports before writing the model card.",
        flush=True,
    )


if __name__ == "__main__":
    main()
