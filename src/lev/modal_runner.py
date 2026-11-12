"""Optional Modal transport for the existing Lev train/benchmark workflow."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import tarfile
from pathlib import Path

from .dataset import bundle
from .io import digest, read, write
from .report import summarize
from .upstream import checkout, contract
from .workflow import config, execute, model_identity, plan

ROOT = Path(__file__).resolve().parents[2]


def snapshot_kev(source, destination, commit):
    """Export committed objects without copying credentials, hooks or local files."""
    checkout(source, commit)
    destination = Path(destination)
    archive = destination.parent / "kev.bundle"
    subprocess.run(
        ["git", "-C", str(source), "bundle", "create", str(archive), "HEAD"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "-q", str(archive), "HEAD"], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "-q", "--detach", commit],
        check=True,
    )
    checkout(destination, commit)


def preflight(args):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.run_id):
        raise ValueError(
            "run-id must be a simple unique name (letters, digits, _ or -)"
        )
    if not 1 <= args.timeout <= 86400:
        raise ValueError("timeout must be between 1 and 86400 seconds")
    cfg = config(args.config)
    checkpoint = args.checkpoint or cfg["init_from"]
    if args.operation in ("train", "experiment") and args.checkpoint:
        raise ValueError("training initialization belongs in the pinned config")
    comparisons = getattr(args, "compare_checkpoint", []) or []
    if comparisons and args.operation != "finalize":
        raise ValueError("comparison checkpoints are only used by finalize")
    for model in [checkpoint, *comparisons]:
        if model.startswith("/lev-store/"):
            if not re.fullmatch(
                r"/lev-store/[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}/result/checkpoint", model
            ):
                raise ValueError("checkpoint must name a prior run's result/checkpoint")
        elif "hub" not in model_identity(model):
            raise ValueError(
                "use a pinned Hub checkpoint or a /lev-store run checkpoint"
            )
    _, manifest = bundle(args.dataset)
    if manifest["example_only"]:
        raise ValueError("Modal jobs require a real dataset, not example-only data")
    if not args.kev:
        raise ValueError("set LEV_KEV_PATH or pass --kev for an existing checkout")
    upstream = checkout(args.kev, cfg["kev_commit"])
    contract(args.kev)
    return {
        "operation": args.operation,
        "checkpoint": checkpoint,
        "comparison_checkpoints": comparisons,
        "run_id": args.run_id,
        "gpu": args.gpu,
        "timeout": args.timeout,
        "volume": args.volume,
        "dataset_sha256": digest(Path(args.dataset) / "manifest.json"),
        "config_sha256": digest(args.config),
        "upstream": upstream,
        "research_only": manifest.get("research_only", False),
        "threshold": 0.5,
        "resources": {"cpu": 4, "memory_mib": [32768, 65536], "kev_python": "3.13.8"},
    }


def run_job(request):
    """Runs inside Modal. All original artifacts and failures live on the Volume."""
    import modal

    volume = modal.Volume.from_name(request["volume"])
    volume.reload()
    root = Path("/lev-store") / request["run_id"]
    root.mkdir(exist_ok=False)
    try:
        write(root / "modal-job.json", request)
        shutil.copytree("/lev-input/dataset", root / "dataset")
        shutil.copyfile("/lev-input/config.toml", root / "config.toml")
        if digest(root / "dataset/manifest.json") != request["dataset_sha256"]:
            raise ValueError("uploaded dataset differs from local preflight")
        if digest(root / "config.toml") != request["config_sha256"]:
            raise ValueError("uploaded config differs from local preflight")
        args = argparse.Namespace(
            command=request["operation"],
            config=root / "config.toml",
            kev="/opt/kev",
            out=root / "result",
            dataset=root / "dataset",
            device="cuda",
            checkpoint=request["checkpoint"],
            split="development",
            calibration=None,
        )
        volume.commit()
        if request["operation"] in ("experiment", "finalize"):
            result = experiment(args, root, request, volume)
        else:
            job = plan(args)
            result = execute(job)
        write(root / "modal-status.json", result)
        return {"path": str(root), **result}
    except BaseException as error:
        write(root / "modal-status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        volume.commit()


def experiment(args, root, request, volume):
    """Execute frozen stages; finalize scores a model already selected on development."""
    records, manifest = bundle(args.dataset)
    baseline = request["checkpoint"]
    checkpoint = str(root / "result/checkpoint")
    reports = {}
    stages = [
        ("baseline", "benchmark", "development", baseline, None),
        ("result", "train", None, None, None),
        ("development", "benchmark", "development", checkpoint, None),
        ("calibration-read", "benchmark", "calibration", checkpoint, None),
        ("calibration-fit", "calibrate", None, None, None),
        (
            "test",
            "benchmark",
            "test",
            checkpoint,
            str(root / "calibration-fit/calibration.json"),
        ),
        ("baseline-test", "benchmark", "test", baseline, None),
    ]
    if request["operation"] == "finalize":
        checkpoint = request["checkpoint"]
        stages = [
            ("calibration-read", "benchmark", "calibration", checkpoint, None),
            ("calibration-fit", "calibrate", None, None, None),
            (
                "test",
                "benchmark",
                "test",
                checkpoint,
                str(root / "calibration-fit/calibration.json"),
            ),
        ]
        stages.extend(
            (f"comparison-{i}-test", "benchmark", "test", model, None)
            for i, model in enumerate(request["comparison_checkpoints"])
        )
    for name, operation, split, model, calibration in stages:
        stage = argparse.Namespace(**vars(args))
        stage.command, stage.out = operation, root / name
        stage.split, stage.checkpoint, stage.calibration = split, model, calibration
        stage.benchmark = str(root / "calibration-read")
        print(f"Starting {name}", flush=True)
        execute(plan(stage))
        if operation == "benchmark":
            report = summarize(
                [r for r in records if r["split"] == split],
                read(root / name / "native/rows.json"),
                request["threshold"],
            )
            report.update(
                research_only=manifest.get("research_only", False), split=split
            )
            write(root / name / "lev-report.json", report)
            reports[name] = report["decisions"]
            if report["decisions"]["missing"]:
                raise RuntimeError(
                    f"{name}: incomplete predictions, see preserved report"
                )
        volume.commit()
        print(f"Completed {name}", flush=True)
    result = {
        "status": "succeeded",
        "reports": reports,
        "research_only": manifest.get("research_only", False),
        "threshold": request["threshold"],
        "checkpoint": checkpoint,
    }
    write(root / "experiment.json", result)
    return result


def snapshot_archive(snapshot, path):
    # Modal directory mounts omit .git; an explicit archive preserves verification metadata.
    def portable_owner(info):
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with tarfile.open(path, "w:gz", compresslevel=1) as archive:
        archive.add(snapshot, arcname=".", filter=portable_owner)
    return path


def build_app(args, snapshot, dataset, config_path, *, job_function=None):
    """Construct the SDK graph without contacting Modal or allocating a GPU."""
    import modal
    from lev.modal_runner import run_job as remote_job

    archive = snapshot_archive(snapshot, Path(snapshot).parent / "kev-snapshot.tar.gz")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_pip_install("uv==0.9.2")
        .env({"UV_PYTHON": "3.13.8"})
        .add_local_file(
            Path(snapshot) / "pyproject.toml", "/opt/kev/pyproject.toml", copy=True
        )
        .add_local_file(Path(snapshot) / "uv.lock", "/opt/kev/uv.lock", copy=True)
        .run_commands(
            "uv sync --project /opt/kev --frozen --no-dev --no-install-project"
        )
        .add_local_file(archive, "/opt/kev-snapshot.tar.gz", copy=True)
        .run_commands(
            "tar --no-same-owner -xzf /opt/kev-snapshot.tar.gz -C /opt/kev && rm /opt/kev-snapshot.tar.gz",
            "git -C /opt/kev rev-parse HEAD",
            "uv sync --project /opt/kev --frozen --no-dev",
        )
        .env(
            {
                "HF_HOME": "/lev-cache/huggingface",
                "PYTHONPATH": "/opt/lev/src",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .add_local_dir(ROOT / "src", "/opt/lev/src")
        .add_local_dir(dataset, "/lev-input/dataset")
        .add_local_file(Path(config_path).resolve(), "/lev-input/config.toml")
    )
    app = modal.App("lev-training")
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)
    cache = modal.Volume.from_name("lev-model-cache", create_if_missing=True)
    worker = app.function(
        image=image,
        gpu=args.gpu,
        cpu=4,
        memory=(32768, 65536),
        timeout=args.timeout,
        volumes={"/lev-store": volume, "/lev-cache": cache},
        retries=0,
        max_containers=1,
    )(job_function or remote_job)
    return app, worker


def modal_output():
    import modal

    return modal.enable_output()


def launch(args, request):
    cfg = config(args.config)
    with tempfile.TemporaryDirectory(prefix="lev-modal-") as temp:
        snapshot = Path(temp) / "kev"
        snapshot_kev(args.kev, snapshot, cfg["kev_commit"])
        dataset = Path(temp) / "dataset"
        dataset.mkdir()
        _, manifest = bundle(args.dataset)
        for name in ("manifest.json", *manifest["files"]):
            shutil.copyfile(Path(args.dataset) / name, dataset / name)
        bundle(dataset)
        app, worker = build_app(args, snapshot, dataset, args.config)
        with modal_output():
            with app.run(detach=True):
                print(json.dumps(worker.remote(request), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("train", "benchmark", "experiment", "finalize")
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint",
        help="benchmark: pinned Hub ID or /lev-store/RUN/result/checkpoint",
    )
    parser.add_argument(
        "--compare-checkpoint",
        action="append",
        default=[],
        help="finalize: fixed test comparator, repeat for multiple models",
    )
    parser.add_argument("--config", default="configs/lev-4b.toml")
    parser.add_argument("--kev", default=os.environ.get("LEV_KEV_PATH"))
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--volume", default="lev-experiments")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request = preflight(args)
    if args.execute:
        launch(args, request)
    else:
        print(json.dumps({"dry_run": True, **request}, indent=2))


if __name__ == "__main__":
    main()
