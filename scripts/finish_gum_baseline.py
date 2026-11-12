"""Download the existing registered baseline once and verify its report."""

import fcntl
import json
from pathlib import Path
import subprocess
import time

from lev.io import digest, read, write

ROOT = Path("runs/lev-joint005-morphology-development-cycle")


def main():
    import modal

    with (ROOT / "finish-controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(ROOT / "finish-plan.json")
        for path, expected in plan["inputs"].items():
            if digest(path) != expected:
                raise ValueError("finishing input changed: " + path)
        registration = read(ROOT / "registration.json")
        volume = modal.Volume.from_name("lev-experiments")
        print("Waiting for registered baseline; no model submission", flush=True)
        while True:
            try:
                status = json.loads(b"".join(volume.read_file(registration["baseline_run_id"] + "/modal-status.json")))
                break
            except FileNotFoundError:
                time.sleep(30)
        write(ROOT / "baseline-remote-terminal-status.json", status)
        for stage in plan["stages"]:
            name = stage["name"]
            if name == "report" and (status["status"] != "succeeded" or status["returncode"] != 0):
                raise RuntimeError("Baseline failed; downloaded evidence retained, no report")
            marker = ROOT / (name + ".started.json")
            if marker.exists():
                raise RuntimeError("Stage already recorded; inspect before resuming: " + name)
            write(marker, stage)
            print("Starting " + name, flush=True)
            with (ROOT / (name + ".log")).open("x") as log:
                result = subprocess.run(stage["command"], stdout=log, stderr=subprocess.STDOUT)
            write(ROOT / (name + ".finished.json"), dict(returncode=result.returncode))
            if result.returncode:
                raise RuntimeError("Stage failed: " + name)
        write(ROOT / "finish-result.json", dict(status="baseline_downloaded_and_reported", goal_complete=False))


if __name__ == "__main__":
    main()
