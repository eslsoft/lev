"""Prepare reproducible jobs; only explicit execution starts model work."""

import os
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .dataset import bundle, fields
from .io import digest, fingerprint, read, write
from .upstream import checkout, contract

TRAIN_OPTIONS = {
    "epochs",
    "lr",
    "batch",
    "accum",
    "lora",
    "head_dim",
    "checkpointing",
    "max_state",
    "seed",
}


def config(path):
    cfg = tomllib.loads(Path(path).read_text())
    fields(
        cfg,
        ("model_name", "kev_commit", "base", "base_revision", "init_from", "training"),
    )
    if not re.fullmatch(r"lev-[a-z0-9.-]+", cfg["model_name"]):
        raise ValueError("model_name must be a versioned lev-* name")
    for pin in (
        cfg["kev_commit"],
        cfg["base_revision"],
        cfg["init_from"].partition("@")[2],
    ):
        if not re.fullmatch(r"[a-f0-9]{40}", pin):
            raise ValueError(
                "Kev, base and init checkpoint must be pinned by full commit SHA"
            )
    fields(cfg["training"], TRAIN_OPTIONS)
    for name in TRAIN_OPTIONS - {"lr"}:
        value = cfg["training"][name]
        if type(value) is not int or value < (
            0 if name in {"seed", "checkpointing"} else 1
        ):
            raise ValueError(f"invalid training integer: {name}")
    if not 0 < cfg["training"]["lr"] < 1 or cfg["training"]["checkpointing"] not in (
        0,
        1,
    ):
        raise ValueError("invalid learning rate/checkpointing")
    if not 384 <= cfg["training"]["max_state"] <= 65536:
        raise ValueError("max_state outside Kev's training range")
    return cfg


def model_identity(checkpoint):
    if Path(checkpoint).is_dir():
        return {
            "path": str(Path(checkpoint).resolve()),
            "files": fingerprint(checkpoint),
        }
    if not re.fullmatch(r"[\w.-]+/[\w.-]+@[a-f0-9]{40}", checkpoint):
        raise ValueError(
            "checkpoint must be a local directory or a Hub ID pinned by full SHA"
        )
    return {"hub": checkpoint}


def benchmark_info(path):
    path = Path(path)
    info = read(path / "job.json")
    if (
        info["operation"] != "benchmark"
        or read(path / "status.json")["status"] != "succeeded"
    ):
        raise ValueError("requires a successful benchmark job")
    if digest(Path(info["dataset"]) / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("benchmark dataset manifest changed")
    bundle(info["dataset"])
    return info


def check_calibration(path, identity):
    cal = read(path)
    saved = cal["checkpoint"]
    # Local paths are provenance; the complete checkpoint content binds calibration.
    same = (
        saved["files"] == identity["files"]
        if set(saved) == set(identity) == {"path", "files"}
        else saved == identity
    )
    if not same:
        raise ValueError("calibration belongs to a different checkpoint")
    t = cal["temperature"]
    if type(t) not in (int, float) or not 0 < t < float("inf"):
        raise ValueError("invalid calibration temperature")
    return cal


def plan(args):
    cfg = config(args.config)
    kev_path = args.kev or os.environ.get("LEV_KEV_PATH")
    if not kev_path:
        raise ValueError(
            "set LEV_KEV_PATH or pass --kev pointing to your existing checkout"
        )
    upstream = checkout(kev_path, cfg["kev_commit"])
    contract(upstream["path"])
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError(f"choose a new output directory: {out}")
    job = {
        "operation": args.command,
        "model_name": cfg["model_name"],
        "upstream": upstream,
        "config": cfg,
        "config_sha256": digest(args.config),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lev_source": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    native = out / ("checkpoint" if args.command == "train" else "native")
    if args.command in ("train", "benchmark"):
        _, manifest = bundle(args.dataset)
        dataset = Path(args.dataset).resolve()
        job.update(
            dataset=str(dataset),
            dataset_sha256=digest(dataset / "manifest.json"),
            example_only=manifest["example_only"],
            research_only=manifest.get("research_only", False),
        )
    prefix = ["uv", "run", "--project", upstream["path"], "--frozen", "--no-dev"]
    env = {}
    if args.command == "train":
        if manifest["example_only"]:
            raise ValueError("example-only dataset: training refused")
        command = [
            *prefix,
            "python",
            "-m",
            "kev.train",
            "--data",
            str(dataset / "train.jsonl"),
            "--base",
            cfg["base"],
            "--base_revision",
            cfg["base_revision"],
            "--init_from",
            cfg["init_from"],
            "--out",
            str(native),
            "--device",
            args.device,
            "--dtype",
            "bf16" if args.device == "cuda" else "fp32",
        ]
        for key, value in cfg["training"].items():
            command += ["--" + key, str(value)]
        job["split"] = "train"
    elif args.command == "benchmark":
        identity = model_identity(args.checkpoint)
        job.update(checkpoint=identity, split=args.split)
        if args.calibration:
            cal = check_calibration(args.calibration, identity)
            if cal["dataset_sha256"] != job["dataset_sha256"]:
                raise ValueError(
                    "use calibration and test partitions of the same frozen dataset"
                )
            env["KEV_TEMPERATURE"] = str(cal["temperature"])
            job["calibration_sha256"] = digest(args.calibration)
        else:
            # Do not accidentally inherit an unrelated caller's temperature override.
            job["calibration_sha256"] = None
        command = [
            *prefix,
            "python",
            "-m",
            "lev.bridge",
            "benchmark",
            "--run",
            identity.get("path", identity.get("hub")),
            "--data",
            str(dataset / f"{args.split}.jsonl"),
            "--out",
            str(native),
            "--device",
            args.device,
            "--max-state",
            str(cfg["training"]["max_state"]),
        ]
    elif args.command == "calibrate":
        info = benchmark_info(args.benchmark)
        if info["split"] != "calibration" or info["example_only"]:
            raise ValueError(
                "fit only on a real calibration partition, never development/test/examples"
            )
        if info["upstream"]["commit"] != upstream["commit"]:
            raise ValueError("benchmark used a different Kev revision")
        job.update(
            benchmark=str(Path(args.benchmark).resolve()),
            dataset_sha256=info["dataset_sha256"],
            checkpoint=info["checkpoint"],
        )
        command = [
            *prefix,
            "python",
            "-m",
            "lev.bridge",
            "calibrate",
            "--benchmark",
            job["benchmark"],
            "--out",
            str(out / "calibration.json"),
        ]
    else:
        identity = model_identity(args.checkpoint)
        if "path" not in identity:
            raise ValueError("serve a locally trained Lev checkpoint")
        model = read(Path(identity["path"]) / "lev-model.json")
        if model["name"] != cfg["model_name"]:
            raise ValueError("checkpoint Lev name does not match config")
        cal = check_calibration(args.calibration, identity)
        env["KEV_TEMPERATURE"] = str(cal["temperature"])
        job.update(checkpoint=identity, calibration_sha256=digest(args.calibration))
        command = [
            *prefix,
            "--extra",
            "serve",
            "python",
            "-m",
            "lev.bridge",
            "serve",
            "--run",
            identity["path"],
            "--name",
            cfg["model_name"],
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--device",
            args.device,
            "--max-state",
            str(cfg["training"]["max_state"]),
        ]
    job.update(command=command, env=env, out=str(out))
    return job


def execute(job):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    env = os.environ.copy()
    env.pop("KEV_TEMPERATURE", None)
    env.update(job["env"])
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    status = {"status": "failed"}
    try:
        if env.get("KEV_TRUNCATE_STATES", "0") != "0":
            raise ValueError("Lev runs do not allow input truncation")
        # Preserve the exact command's output and nonzero exit code. No retries or fallback model.
        with (out / "run.log").open("x") as log:
            process = subprocess.run(
                job["command"],
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(
                f"{job['operation']} failed ({process.returncode}); see {out / 'run.log'}"
            )
        if job["operation"] == "train":
            fingerprint(out / "checkpoint")
            write(
                out / "checkpoint" / "lev-model.json",
                {
                    "name": job["model_name"],
                    "status": "trained-not-validated",
                    "research_only": job.get("research_only", False),
                    "upstream": job["upstream"],
                    "dataset_sha256": job["dataset_sha256"],
                    "config": job["config"],
                },
            )
        status["status"] = "succeeded"
    except BaseException as error:
        status["error"] = str(error)
        raise
    finally:
        write(out / "status.json", status)
    return status
