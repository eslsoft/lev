"""Verify downloaded execution evidence and checkpoint content; never synthesize it."""

import argparse
import math
from pathlib import Path

from lev.dataset import bundle
from lev.io import digest, fingerprint, read, write


def verify(run, dataset, out, benchmark=None):
    records, manifest = bundle(dataset)
    job = read(run / "result/job.json")
    checkpoint = run / "result/checkpoint"
    metrics = read(checkpoint / "training_metrics.json")
    native_config = read(checkpoint / "training_config.json")
    model = read(checkpoint / "lev-model.json")
    assert read(run / "modal-status.json")["status"] == "succeeded"
    assert read(run / "result/status.json") == {"returncode": 0, "status": "succeeded"}
    assert job["operation"] == "train" and job["split"] == "train"
    assert (
        job["dataset_sha256"]
        == digest(dataset / "manifest.json")
        == model["dataset_sha256"]
    )
    assert model["research_only"] == manifest["research_only"]
    cfg = job["config"]["training"]
    n = sum(r["split"] == "train" for r in records)
    expected = n * cfg["epochs"]
    assert metrics["records_seen"] == metrics["requested_records"] == expected
    assert metrics["rejected_records"] == metrics["truncated_records"] == 0
    assert metrics["optimizer_steps"] == cfg["epochs"] * math.ceil(
        n / (cfg["batch"] * cfg["accum"])
    )
    assert metrics["forward_tokens"] > 0 and metrics["wall_seconds"] > 0
    assert metrics["device"] == "cuda" and metrics["dtype"] == "bf16"
    for epoch in metrics["grad_norm"]:
        assert math.isfinite(epoch["mean"]) and math.isfinite(epoch["max"])
    for k, v in cfg.items():
        assert native_config["args"][k] == v
    files = fingerprint(checkpoint)
    if benchmark:
        eval_job = read(benchmark / "job.json")
        assert read(benchmark / "status.json")["status"] == "succeeded"
        assert eval_job["checkpoint"]["files"] == files
        assert eval_job["dataset_sha256"] == job["dataset_sha256"]
    write(
        out,
        dict(
            status="passed",
            dataset_sha256=job["dataset_sha256"],
            checkpoint_files=files,
            processed_requests=expected,
            optimizer_steps=metrics["optimizer_steps"],
            wall_seconds=metrics["wall_seconds"],
            research_only=model["research_only"],
            benchmark_verified=str(benchmark) if benchmark else None,
        ),
    )
    print(
        f"Verified {expected} processed requests, {metrics['optimizer_steps']} steps; complete checkpoint fingerprints retained."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--benchmark", type=Path)
    a = p.parse_args()
    verify(a.run, a.dataset, a.out, a.benchmark)
