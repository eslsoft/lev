"""Encoder decision model: a ModernBERT backbone with a two-way true/false head.

The Kev path in `workflow.py` trains a 4B decoder with fp32 frozen weights, which
needs ~20 GB before a single activation. A true/false judgement over a short
passage does not: the frozen packs in this repository have a median state of a
few dozen tokens and a 2.3k-token worst case, so a 150M encoder answers the same
question inside a 16 GB card.

This module is additive and self-contained. It reads the same frozen dataset
directory, writes the same `job.json` / `status.json` / `run.log` evidence, and
emits the same `native/rows.json` contract, so `lev report` scores it with
unchanged code and the same denominators. Nothing here calls Kev or needs
`LEV_KEV_PATH`.

Standard library only. The torch work lives in `lev.encoder_train` behind the
`encoder` extra, so the default Lev environment keeps its empty dependency set.
"""

import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .dataset import bundle, fields
from .io import digest, encoded, write

MODEL = re.compile(r"lev-[a-z0-9.-]+")
SHA = re.compile(r"[a-f0-9]{40}")
TRAIN_KEYS = {
    "epochs",
    "lr",
    "batch",
    "accum",
    "max_tokens",
    "dtype",
    "seed",
    "warmup",
}
DTYPES = ("fp32", "fp16", "bf16")
TOP = ("model_name", "backbone", "backbone_revision", "max_len", "threshold", "training")


def config(path):
    """Parse and strictly validate an encoder config; unknown keys are errors."""
    cfg = tomllib.loads(Path(path).read_text())
    fields(cfg, TOP)
    if not MODEL.fullmatch(cfg["model_name"]):
        raise ValueError("model_name must be a versioned lev-* name")
    if not isinstance(cfg["backbone"], str) or not re.fullmatch(
        r"[\w.-]+/[\w.-]+", cfg["backbone"]
    ):
        raise ValueError("backbone must be a Hub ID")
    if not SHA.fullmatch(cfg["backbone_revision"]):
        raise ValueError("backbone_revision must pin the full commit SHA")
    if type(cfg["max_len"]) is not int or not 384 <= cfg["max_len"] <= 8192:
        raise ValueError("max_len must be an integer in [384, 8192]")
    # The decision boundary is declared in the config, so job.json records the
    # exact threshold that produced a run instead of an implicit 0.5.
    if type(cfg["threshold"]) is not float or not 0 < cfg["threshold"] < 1:
        raise ValueError("threshold must be a float in (0, 1)")
    fields(cfg["training"], TRAIN_KEYS)
    for name in ("epochs", "batch", "accum", "max_tokens", "seed"):
        value = cfg["training"][name]
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"invalid training integer: {name}")
    if cfg["training"]["dtype"] not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    lr, warmup = cfg["training"]["lr"], cfg["training"]["warmup"]
    if type(lr) is not float or not 0 < lr < 1:
        raise ValueError("lr must be a positive float below 1")
    if type(warmup) is not float or not 0 <= warmup < 1:
        raise ValueError("warmup must be a float in [0, 1)")
    return cfg


def state_text(state):
    """Render a record's state deterministically: strings verbatim, objects as sorted lines."""
    if isinstance(state, str):
        return state
    if not isinstance(state, dict):
        raise ValueError("state must be a string or an object")
    lines = []
    for key in sorted(state):
        value = state[key]
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        elif value is None or isinstance(value, (dict, list, bool, int, float)):
            lines.append(f"{key}: {encoded(value)}")
        else:
            raise ValueError(
                f"state.{key} must be JSON-serialisable, got {type(value).__name__}"
            )
    return "\n".join(lines)


def examples(records):
    """One training pair per (request, question) in a frozen Kev export.

    `instructions` already carries the claim, so the model sees exactly the text
    Kev's `api_request` projection would send: state plus question, no label, no
    reviewer, no evidence, no `_meta`.
    """
    out = []
    for record in records:
        meta = record["_meta"]
        state = state_text(record["state"])
        for qid in sorted(record["questions"]):
            question = record["questions"][qid]
            if question["type"] != "noul":
                raise ValueError(
                    f"{meta['id']}/{qid}: only noul questions are supported"
                )
            label = question["label"]
            if type(label) is not bool:
                raise ValueError(f"{meta['id']}/{qid}: label must be a JSON boolean")
            out.append(
                {
                    "id": meta["id"],
                    "question": qid,
                    "text": state + "\n\n" + question["instructions"],
                    "label": int(label),
                    "kind": question["src"],
                    "group": meta.get("group_id"),
                    "source": meta.get("source"),
                    "variant": meta.get("variant"),
                }
            )
    if not out:
        raise ValueError("no questions in the export")
    return out


def fingerprint(directory):
    """Content identity of an encoder checkpoint directory."""
    directory = Path(directory)
    if not (directory / "lev-encoder.json").is_file():
        raise ValueError(f"not an encoder checkpoint: {directory}")
    return {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def plan(args):
    """Build the job record and the exact subprocess command. No side effects."""
    cfg = config(args.config)
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError(f"choose a new output directory: {out}")
    _, manifest = bundle(args.dataset)
    dataset = Path(args.dataset).resolve()
    if args.command == "encode":
        split = "train"
        if manifest["example_only"]:
            raise ValueError("example-only dataset: training refused")
    else:
        split = args.split
    native = out / ("checkpoint" if args.command == "encode" else "native")
    job = {
        # `report` gates on operation == "benchmark"; `engine` records the truth.
        "operation": "train" if args.command == "encode" else "benchmark",
        "engine": "encoder",
        "model_name": cfg["model_name"],
        "config": cfg,
        "config_sha256": digest(args.config),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": digest(dataset / "manifest.json"),
        "split": split,
        "example_only": manifest["example_only"],
        "research_only": manifest.get("research_only", False),
        "lev_source": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "out": str(out),
    }
    if args.command == "encode":
        command = [
            sys.executable,
            "-m",
            "lev.encoder_train",
            "train",
            "--data",
            str(dataset / "train.jsonl"),
            "--out",
            str(native),
            "--model",
            cfg["backbone"],
            "--revision",
            cfg["backbone_revision"],
            "--max-len",
            str(cfg["max_len"]),
            "--device",
            args.device,
        ]
        for key, value in cfg["training"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
    else:
        identity = args.checkpoint
        if not Path(identity).is_dir():
            raise ValueError(
                "--checkpoint must be a local encoder checkpoint directory"
            )
        job["checkpoint"] = {
            "path": str(Path(identity).resolve()),
            "files": fingerprint(identity),
        }
        job["calibration_sha256"] = None
        command = [
            sys.executable,
            "-m",
            "lev.encoder_train",
            "predict",
            "--data",
            str(dataset / f"{split}.jsonl"),
            "--checkpoint",
            str(Path(identity).resolve()),
            "--out",
            str(native / "rows.json"),
            "--max-len",
            str(cfg["max_len"]),
            "--device",
            args.device,
        ]
    job["command"] = command
    return job


def execute(job):
    """Run the job, preserving stdout/stderr and the real exit code. No retries."""
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONHASHSEED"] = str(job["config"]["training"]["seed"])
    status: dict[str, object] = {"status": "failed"}
    try:
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
                    "engine": "encoder",
                    "status": "trained-not-validated",
                    "research_only": job["research_only"],
                    "backbone": {
                        "repo": job["config"]["backbone"],
                        "revision": job["config"]["backbone_revision"],
                    },
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
